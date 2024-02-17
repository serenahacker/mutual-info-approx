
from builders import SET_MODEL_BUILDERS
from trainer import Trainer, StatisticalDistanceTrainer, DonskerVaradhanTrainer, DonskerVaradhanMITrainer#, DonskerVaradhanTrainer2
from datasets.distributions import CorrelatedGaussianGenerator, GaussianGenerator, NFGenerator, StandardGaussianGenerator, CorrelatedGaussianGenerator2, LabelledGaussianGenerator, RandomEncoderGenerator, ProtectedDatasetGenerator
from models.set import MultiSetTransformerEncoder, MultiSetTransformerEncoderDecoder
from utils import kl_mc, kl_mc_mixture, mi_corr_gaussian, kl_knn, kraskov_mi1

import torch.nn as nn
import torch
import math



class Task():
    trainer_cls=Trainer
    def __init__(self, args):
        self.args = args

    def build_model(self):
        return SET_MODEL_BUILDERS[self.args.model](self.args)
    
    def build_dataset(self):
        pass
    
    def build_training_args(self):
        train_args = {
            'batch_size': self.args.batch_size,
            'grad_steps': self.args.grad_steps,
            'data_kwargs': {'set_size': self.args.set_size}
        }
        eval_args = {
            'batch_size': self.args.batch_size,
            'data_kwargs': {'set_size': self.args.set_size}
        }
        return train_args, eval_args

    def build_trainer_kwargs(self):
        trainer_kwargs = {
            'eval_every': self.args.eval_every,
            'save_every': self.args.save_every,
            'ss_schedule': self.args.ss_schedule
        }
        return trainer_kwargs
    
    def build_trainer(self, model, optimizer, scheduler, train_dataset, val_dataset, test_dataset, device, logger, checkpoint_dir=None):
        train_args, eval_args = self.build_training_args()
        trainer_kwargs = self.build_trainer_kwargs()
        trainer = self.trainer_cls(model, optimizer, train_dataset, val_dataset, test_dataset, 
            train_args, eval_args, device, logger=logger, scheduler=scheduler, checkpoint_dir=checkpoint_dir, **trainer_kwargs)
        return trainer

#
#   Statistical Distance Tasks
#

LOSSES = {
    'l1': nn.L1Loss,
    'mse': nn.MSELoss
}

class StatisticalDistanceTask(Task):
    trainer_cls = StatisticalDistanceTrainer

    def build_model(self):
        self.args.input_size = self.args.n
        return super().build_model()

    def build_training_args(self):
        sample_kwargs = {
            'set_size': self.args.set_size, 
        }
        if self.args.dataset == 'gmm':
            sample_kwargs['nu']=5
            sample_kwargs['mu0']=0
            sample_kwargs['s0']=0.3

        if self.args.equi and self.args.vardim:
            dim_range = math.ceil(self.args.n/8)
            sample_kwargs['dims'] = (max(2,self.args.n-dim_range),self.args.n+dim_range)
        else:
            sample_kwargs['n'] = self.args.n

        train_args = {
            'batch_size': self.args.batch_size,
            'grad_steps': self.args.grad_steps,
            'sample_kwargs': sample_kwargs,
            'label_kwargs': {},
            'clip': getattr(self.args, 'clip', -1)
        }
        eval_args = {
            'batch_size': self.args.batch_size,
            'sample_kwargs': sample_kwargs,
            'label_kwargs': {}
        }
        return train_args, eval_args


class KLTask(StatisticalDistanceTask):
    def build_dataset(self):
        if self.args.dataset == 'gmm':
            generator = GaussianGenerator(num_outputs=2, variable_dim=self.args.equi, return_params=True, mixture=True)
        elif self.args.dataset == 'nf':
            generator = NFGenerator(32, 2, num_outputs=2, use_maf=False, variable_dim=self.args.equi, return_params=True)
        else:
            raise NotImplementedError("gmm or nf")
        return generator, None, None

    def build_training_args(self):
        train_args, eval_args = super().build_training_args()
        train_args['normalize'] = 'whiten'
        eval_args['normalize'] = 'whiten'
        return train_args, eval_args
    
    def build_trainer_kwargs(self):
        trainer_kwargs = {
            'eval_every': self.args.eval_every,
            'save_every': self.args.save_every,
            'label_fct': kl_mc,
            'exact_loss': True,
            'criterion': nn.L1Loss(),
            'baselines': {'knn': kl_knn}
        }
        if getattr(self.args, 'criterion', None) is not None:
            trainer_kwargs['criterion'] = LOSSES[self.args.criterion]
        return trainer_kwargs
        
class MITask(StatisticalDistanceTask):
    def build_dataset(self):
        generator = CorrelatedGaussianGenerator(return_params=True, variable_dim=self.args.equi)
        return generator, generator, None

    def build_training_args(self):
        train_args, eval_args = super().build_training_args()
        train_args['normalize'] = 'none'
        eval_args['normalize'] = 'none'
        return train_args, eval_args
    
    def build_trainer_kwargs(self):
        trainer_kwargs = {
            'eval_every': self.args.eval_every,
            'save_every': self.args.save_every,
            'label_fct': mi_corr_gaussian,
            'exact_loss': True,
            'criterion': nn.MSELoss(),
            'baselines': {'kraskov':kraskov_mi1}
        }
        if getattr(self.args, 'criterion', None) is not None:
            trainer_kwargs['criterion'] = LOSSES[self.args.criterion]
        return trainer_kwargs


class DVTask(StatisticalDistanceTask):
    trainer_cls=DonskerVaradhanTrainer

    def build_dataset(self):
        if self.args.dataset == 'gmm':
            generator = GaussianGenerator(num_outputs=2, variable_dim=self.args.equi, return_params=True, mixture=True)
        elif self.args.dataset == 'nf':
            generator = NFGenerator(32, 2, num_outputs=2, use_maf=False, variable_dim=self.args.equi, return_params=True)
        elif self.args.dataset == 'corr':
            generator = CorrelatedGaussianGenerator2(return_params=True, variable_dim=self.args.equi, max_rho=self.args.max_rho)
        else:
            raise NotImplementedError("gmm or nf")
        return generator, generator, None

    def build_training_args(self):
        train_args, eval_args = super().build_training_args()
        if self.args.split_inputs:
            train_args['sample_kwargs']['sample_groups']=2

        train_args['normalize'] = 'whiten'
        eval_args['normalize'] = 'whiten'
        return train_args, eval_args
    
    def build_trainer_kwargs(self):
        trainer_kwargs = {
            'eval_every': self.args.eval_every,
            'save_every': self.args.save_every,
            'label_fct': kl_mc,
            'criterion': nn.L1Loss(),
            'split_inputs': self.args.split_inputs,
            'mode': 'kl'
        }
        if getattr(self.args, 'criterion', None) is not None:
            trainer_kwargs['criterion'] = LOSSES[self.args.criterion]
        return trainer_kwargs
    
    def _build_model_mst(self):
        model_kwargs={
            'ln':self.args.layer_norm,
            'remove_diag':False,
            'num_blocks':self.args.num_blocks,
            'num_heads':self.args.num_heads,
            'dropout':self.args.dropout,
            'equi':self.args.equi,
            'decoder_layers': self.args.decoder_layers,
            'merge': 'concat',
            'weight_sharing': 'sym',     #IMPORTANT
        }
        set_model = MultiSetTransformerEncoder(self.args.n, self.args.n, self.args.latent_size, self.args.hidden_size, 1, **model_kwargs)
        return set_model
    
    def _build_model_encdec(self):
        model_kwargs={
            'ln':self.args.layer_norm,
            'remove_diag':False,
            'enc_blocks':self.args.enc_blocks,
            'dec_blocks':self.args.dec_blocks,
            'num_heads':self.args.num_heads,
            'dropout':self.args.dropout,
            'equi':self.args.equi,
            'output_layers': self.args.decoder_layers,
            'merge': 'concat',
            'decoder_self_attn': self.args.decoder_self_attn
        }
        n = self.args.n * 2 if self.args.dataset == 'corr'else self.args.n
        set_model = MultiSetTransformerEncoderDecoder(n, n, self.args.latent_size, self.args.hidden_size, 1, **model_kwargs)
        return set_model
    
    def build_model(self):
        return self._build_model_encdec()



class DVMITask(StatisticalDistanceTask):
    trainer_cls=DonskerVaradhanMITrainer

    def build_dataset(self):
        if self.args.dataset == 'corr':
            generator = CorrelatedGaussianGenerator(return_params=True, variable_dim=self.args.equi, max_rho=self.args.max_rho)
        elif self.args.dataset == 'mixture':
            generator = LabelledGaussianGenerator(return_params=True, variable_dim=self.args.equi)
        elif self.args.dataset == 'adult':
            generator = ProtectedDatasetGenerator.from_adult(return_params=True)
        elif self.args.dataset == 'adult-rand':
            model_kwargs={
                'in_features': 102,
                'hidden_dim': 100,
                'activation': nn.ReLU(),
            }
            generator = RandomEncoderGenerator.from_adult(model_kwargs, return_params=True, variable_dim=self.args.equi)
        else:
            raise NotImplementedError("corr or mixture")
        return generator, generator, None

    def build_training_args(self):
        train_args, eval_args = super().build_training_args()
      
        train_args['sample_kwargs']['sample_groups']=2 if self.args.split_inputs else 1
        eval_args['sample_kwargs']['sample_groups']=2 if self.args.split_inputs else 1
        train_args['normalize'] = self.args.normalize
        eval_args['normalize'] = self.args.normalize
        return train_args, eval_args
    
    def build_trainer_kwargs(self):

        trainer_kwargs = {
            'eval_every': self.args.eval_every,
            'save_every': self.args.save_every,
            'criterion': nn.L1Loss(),
            'estimate_size': getattr(self.args, 'estimate_size', -1),
            'scale': getattr(self.args, 'scale', 'none'),
            'eps': getattr(self.args, 'eps', 1e-8),
            'model_type': self.args.dv_model,
            'split_inputs': self.args.split_inputs
        }
        if self.args.dataset == 'corr':
            trainer_kwargs['x_marginal'] = StandardGaussianGenerator()
            trainer_kwargs['y_marginal'] = StandardGaussianGenerator()
            trainer_kwargs['sample_marg'] = getattr(self.args, 'sample_marg', True)
            trainer_kwargs['label_fct'] = mi_corr_gaussian
        elif self.args.dataset == 'mixture':
            trainer_kwargs['x_marginal'] = None
            trainer_kwargs['y_marginal'] = None
            trainer_kwargs['sample_marg'] = False
            trainer_kwargs['label_fct'] = kl_mc_mixture
        elif self.args.dataset == 'adult':
            trainer_kwargs['x_marginal'] = None
            trainer_kwargs['y_marginal'] = None
            trainer_kwargs['sample_marg'] = False
            trainer_kwargs['label_fct'] = None
        elif self.args.dataset == 'adult-rand':
            trainer_kwargs['x_marginal'] = None
            trainer_kwargs['y_marginal'] = None
            trainer_kwargs['sample_marg'] = False
            trainer_kwargs['label_fct'] = None

        if getattr(self.args, 'criterion', None) is not None:
            trainer_kwargs['criterion'] = LOSSES[self.args.criterion]
        return trainer_kwargs
    
    def _build_model_mst(self):
        model_kwargs={
            'ln':self.args.layer_norm,
            'remove_diag':False,
            'num_blocks':self.args.num_blocks,
            'num_heads':self.args.num_heads,
            'dropout':self.args.dropout,
            'equi':self.args.equi,
            'decoder_layers': self.args.decoder_layers,
            'merge': 'concat',
            'weight_sharing': 'sym',     #IMPORTANT?? Not sure if necessary or not for MI but probably helpful
            'merge_output_sets': True
        }
        if self.args.dataset == 'corr':
            x_size, y_size = self.args.n, self.args.n
        elif self.args.dataset == 'mixture' or self.args.dataset == 'adult-rand':
            x_size, y_size = self.args.n, 1
        elif self.args.dataset == 'adult':
            x_size, y_size = 102, 1
        set_model = MultiSetTransformerEncoder(x_size, y_size, self.args.latent_size, self.args.hidden_size, 1, **model_kwargs)
        return set_model

    def _build_model_encdec(self):
        model_kwargs={
            'ln':self.args.layer_norm,
            'remove_diag':False,
            'enc_blocks':self.args.enc_blocks,
            'dec_blocks':self.args.dec_blocks,
            'num_heads':self.args.num_heads,
            'dropout':self.args.dropout,
            'equi':self.args.equi,
            'output_layers': self.args.decoder_layers,
            'merge': 'concat',
            'decoder_self_attn': self.args.decoder_self_attn
        }
        if self.args.dataset == 'corr':
            input_size = self.args.n * 2
        elif self.args.dataset == 'mixture' or self.args.dataset == 'adult-rand':
            input_size = self.args.n + 1
        elif self.args.dataset == 'adult':
            input_size = 102 + 1
        set_model = MultiSetTransformerEncoderDecoder(input_size, input_size, self.args.latent_size, self.args.hidden_size, 1, **model_kwargs)
        return set_model

    def build_model(self):
        if self.args.dv_model == 'mst':
            return self._build_model_mst()
        else:
            return self._build_model_encdec()



#
#   DV2
#
'''
class DVTask2(StatisticalDistanceTask):
    trainer_cls=DonskerVaradhanTrainer2

    def build_dataset(self):
        if self.args.dataset == 'gmm':
            generator = GaussianGenerator(num_outputs=2, variable_dim=self.args.equi, return_params=True, mixture=True)
        elif self.args.dataset == 'nf':
            generator = NFGenerator(32, 2, num_outputs=2, use_maf=False, variable_dim=self.args.equi, return_params=True)
        elif self.args.dataset == 'corr':
            generator = CorrelatedGaussianGenerator(return_params=True, variable_dim=self.args.equi, max_rho=self.args.max_rho)
        elif self.args.dataset == 'corr2':
            generator = CorrelatedGaussianGenerator2(return_params=True, variable_dim=self.args.equi, max_rho=self.args.max_rho)
        else:
            raise NotImplementedError("gmm or nf")
        return generator, generator, None

    def build_training_args(self):
        train_args, eval_args = super().build_training_args()
        #if self.args.split_inputs:
            #train_args['sample_kwargs']['sample_groups']=2

        train_args['normalize'] = 'whiten'
        eval_args['normalize'] = 'whiten'
        return train_args, eval_args
    
    def build_trainer_kwargs(self):
        trainer_kwargs = {
            'eval_every': self.args.eval_every,
            'save_every': self.args.save_every,
            'criterion': nn.L1Loss(),
            'split_inputs': False,
            'dataset': self.args.dataset,
            'model_type': 'mst',
            'estimate_size': self.args.estimate_size,
            'model_type': self.args.dv_model
        }
        if self.args.dataset == 'corr':
            trainer_kwargs['mode'] = 'mi-kl'
            trainer_kwargs['label_fct'] = mi_corr_gaussian
        elif self.args.dataset == 'corr2':
            trainer_kwargs['mode'] = 'mi-kl'
            trainer_kwargs['label_fct'] = kl_mc
        else:
            trainer_kwargs['mode'] = 'kl'
            trainer_kwargs['label_fct'] = kl_mc   

        if getattr(self.args, 'criterion', None) is not None:
            trainer_kwargs['criterion'] = LOSSES[self.args.criterion]
        return trainer_kwargs
    
    def _build_model_mst(self):
        model_kwargs={
            'ln':self.args.layer_norm,
            'remove_diag':False,
            'num_blocks':self.args.num_blocks,
            'num_heads':self.args.num_heads,
            'dropout':self.args.dropout,
            'equi':self.args.equi,
            'decoder_layers': self.args.decoder_layers,
            'merge': 'concat',
            'weight_sharing': 'sym',     #IMPORTANT
        }
        n = self.args.n * 2 if self.args.dataset == 'corr'else self.args.n
        set_model = MultiSetTransformerEncoder(n, n, self.args.latent_size, self.args.hidden_size, 1, **model_kwargs)
        return set_model
    
    def _build_model_encdec(self):
        model_kwargs={
            'ln':self.args.layer_norm,
            'remove_diag':False,
            'enc_blocks':self.args.enc_blocks,
            'dec_blocks':self.args.dec_blocks,
            'num_heads':self.args.num_heads,
            'dropout':self.args.dropout,
            'equi':self.args.equi,
            'output_layers': self.args.decoder_layers,
            'merge': 'concat',
            'decoder_self_attn': self.args.decoder_self_attn
        }
        set_model = MultiSetTransformerEncoderDecoder(self.args.n*2, self.args.n*2, self.args.latent_size, self.args.hidden_size, 1, **model_kwargs)
        return set_model
    
    
    def build_model(self):
        if self.args.dv_model == 'encdec':
            return self._build_model_encdec()
        else:
            return self._build_model_mst()
'''


TASKS = {
    'stat/KL': KLTask,
    'stat/MI': MITask,
    'stat/DV': DVTask,
    'stat/DV-MI': DVMITask,
    #'stat/DV2': DVTask2
}

