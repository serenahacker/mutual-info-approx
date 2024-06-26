import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import tqdm
import os

import wandb

from utils import whiten_split, batched_cov, batched_shuffle

SS_SCHEDULE_15=[{'set_size':(1,5), 'steps':20000}, {'set_size':(3,10), 'steps':5000}, {'set_size':(8,15), 'steps':5000}]
SS_SCHEDULE_30=[{'set_size':(1,5), 'steps':20000}, {'set_size':(3,10), 'steps':5000}, {'set_size':(8,15), 'steps':5000}, {'set_size':(10,30), 'steps':5000}]
SS_SCHEDULE_50=[{'set_size':(1,5), 'steps':20000}, {'set_size':(3,10), 'steps':5000}, {'set_size':(8,15), 'steps':5000}, {'set_size':(10,30), 'steps':5000}, {'set_size':(25,50), 'steps':10000}]
SS_SCHEDULE_75=[{'set_size':(1,5), 'steps':20000}, {'set_size':(3,10), 'steps':5000}, {'set_size':(8,15), 'steps':5000}, {'set_size':(10,30), 'steps':5000}, {'set_size':(25,50), 'steps':5000}, {'set_size':(50,75), 'steps':5000}]
SS_SCHEDULES={15:SS_SCHEDULE_15, 30:SS_SCHEDULE_30, 75:SS_SCHEDULE_75}

class SetSizeScheduler():
    def __init__(self, schedule, step_mult=1):
        self.schedule=schedule
        self.N = sum([entry['steps'] for entry in schedule])
        self.step_mult=step_mult

    def get_set_size(self, iter_id):
        if iter_id >= 0:    #return last set size for iter_id -1
            step=0
            for entry in self.schedule:
                step += entry['steps']
                if self.step_mult * iter_id < step:
                    return entry['set_size']
        return self.schedule[-1]['set_size']    #fallback for now

class Trainer():
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, logger=None,
            eval_every=500, save_every=2000, criterion=nn.BCEWithLogitsLoss(), scheduler=None, checkpoint_dir=None, ss_schedule=-1):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_args = train_args
        self.eval_args = eval_args
        self.device = device
        self.logger = logger
        self.eval_every = eval_every
        self.save_every = save_every
        self.criterion = criterion
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.ss_schedule = SetSizeScheduler(SS_SCHEDULES[ss_schedule]) if ss_schedule > 0 else None

    def save_checkpoint(self, step, metrics):
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(), 
            'scheduler': self.scheduler, 
            'step': step, 
            'metrics': metrics
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
        torch.save(save_dict, checkpoint_path)

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
        load_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(load_dict['model'])
        self.optimizer.load_state_dict(load_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(load_dict['scheduler'])
        step, metrics = load_dict['step'], load_dict['metrics']
        return step, metrics
    
    def train(self, train_steps, val_steps, test_steps):
        all_metrics={
            'train/loss': [],
        }

        def _log(name, value, step=-1):
            if name not in all_metrics:
                all_metrics[name] = []
            all_metrics[name].append(value)
            if self.logger is not None and step >= 0:
                self.logger.add_scalar(name, value, step)
            wandb.log({name: value}, step)

        initial_step=0

        if self.checkpoint_dir is not None:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
                if os.path.exists(checkpoint_path):
                    initial_step, metrics = self.load_checkpoint()

        avg_loss = 0
        # loss_fct = nn.BCEWithLogitsLoss()
        for i in tqdm.tqdm(range(initial_step, train_steps)):
            if self.ss_schedule is not None:
                set_size = self.ss_schedule.get_set_size(i)
                self.train_args['data_kwargs']['set_size'] = set_size
                self.eval_args['data_kwargs']['set_size'] = set_size

            loss = self.train_step(i, train_steps, self.train_dataset)
            
            _log('train/loss', loss, i)

            if i > initial_step:
                if self.eval_every > 0 and val_steps > 0 and self.val_dataset is not None and i % self.eval_every == 0:
                    val_metrics = self.evaluate(val_steps, self.val_dataset)
                    for k, v in val_metrics.items():
                        key = "val/"+k
                        _log(key, v, i)

                if self.checkpoint_dir is not None and i % self.save_every == 0:
                    self.save_checkpoint(i, all_metrics)
            

        if self.test_dataset is not None:
            test_metrics = evaluate(test_steps, self.test_dataset)
            for k, v in test_metrics.items():
                key = "test/"+k
                _log(key, v, -1)


        return all_metrics
    
    def train_step(self, i, steps, dataset):
        args = self.train_args
        (X,Y), target = dataset(args['batch_size'], **args['data_kwargs'])

        out = self.model(X.to(self.device),Y.to(self.device))
        loss = self.criterion(out.squeeze(-1), target.to(self.device))
        loss.backward()

        if (i+1) % args['grad_steps'] == 0 or i == (steps - 1):
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def evaluate(self, steps, dataset):
        args = self.eval_args
        n_correct = 0
        with torch.no_grad():
            for i in range(steps):
                (X,Y), target = self.train_dataset(args['batch_size'], **args['data_kwargs'])
                out = self.model(X.to(self.device),Y.to(self.device)).squeeze(-1)
                n_correct += torch.eq((out > 0), target.to(self.device)).sum().item()
        
        return {'acc': n_correct / (args['batch_size'] * steps)}


class StatisticalDistanceTrainer(Trainer):
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, criterion, 
            label_fct, exact_loss, baselines, logger=None, save_every=2000, eval_every=500, scheduler=None, 
            checkpoint_dir=None, ss_schedule=-1):
        super().__init__(model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device,
            save_every=save_every, eval_every=eval_every, criterion=criterion, scheduler=scheduler, logger=logger,
            checkpoint_dir=checkpoint_dir, ss_schedule=ss_schedule)
        self.label_fct = label_fct
        self.exact_loss = exact_loss
        self.baselines = baselines
    
    def train_step(self, i, steps, dataset):
        args = self.train_args
        if self.exact_loss:
            X, theta = dataset(args['batch_size'], **args['sample_kwargs'])
            labels = self.label_fct(*theta, X=X[0], **args['label_kwargs']).squeeze(-1)
        else:
            X = dataset(args['batch_size'], **args['sample_kwargs'])
            if args['normalize'] == 'scale-linear':
                X, avg_norm = normalize_sets(*X)
            labels = self.label_fct(*X, **args['label_kwargs'])
        if args['normalize'] == 'scale-inv':
            X, avg_norm = normalize_sets(*X)
        elif args['normalize'] == 'whiten':
            X = whiten_split(*X)
        
        out = self.model(*X).squeeze(-1)

        loss = self.criterion(out, labels)
        loss.backward()

        if (i+1) % args['grad_steps'] == 0 or i == (steps - 1):
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def evaluate(self, steps, dataset):
        args = self.eval_args
        model_losses=[]
        baseline_losses={baseline:[] for baseline in self.baselines.keys()}
        with torch.no_grad():
            for i in range(steps):
                if self.exact_loss:
                    X, theta = dataset(args['batch_size'], **args['sample_kwargs'])
                    labels = self.label_fct(*theta, X=X[0], **args['label_kwargs']).squeeze(-1)
                else:
                    X = dataset(args['batch_size'], **args['sample_kwargs'])
                    labels = self.label_fct(*X, **args['label_kwargs'])
                
                for baseline_name, baseline_fct in self.baselines.items():
                    baseline_out = baseline_fct(*X).squeeze(-1)
                    baseline_loss = self.criterion(baseline_out, labels)
                    baseline_losses[baseline_name].append(baseline_loss.item())

                if not self.exact_loss and args['normalize'] == 'scale-linear':
                    X, avg_norm = normalize_sets(*X)
                    labels = self.label_fct(*X, **args['label_kwargs'])

                if args['normalize'] == 'scale-inv':
                    X, avg_norm = normalize_sets(*X)
                elif args['normalize'] == 'whiten':
                    X = whiten_split(*X)
                
                out = self.model(*X).squeeze(-1)
                loss = self.criterion(out, labels)
                model_losses.append(loss.item())

        metrics = {'loss': sum(model_losses)/len(model_losses)}
        for baseline_name, baseline_losses in baseline_losses.items():
            key = baseline_name + '/loss'
            metrics[key] = sum(baseline_losses)/len(baseline_losses)
        return metrics

    def _forward(self, *X):
        return self.model(*X).squeeze(-1)


import math

class DonskerVaradhanTrainer(Trainer):
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, criterion, label_fct, 
            logger=None, save_every=2000, eval_every=500, scheduler=None, checkpoint_dir=None, ss_schedule=-1, split_inputs=True, mode='kl'):
        super().__init__(model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, logger=logger,
            save_every=save_every, criterion=criterion, scheduler=scheduler, checkpoint_dir=checkpoint_dir, ss_schedule=ss_schedule)
        self.label_fct = label_fct
        self.split_inputs = split_inputs
        self.mode=mode

    @staticmethod
    def _KL_estimate(X, Y):
        return X.sum(dim=1)/X.size(1) - Y.logsumexp(dim=1) + math.log(Y.size(1))

    def _forward_KL(self, X, Y):
        if self.split_inputs:
            X0,X1 = X.chunk(2, dim=1)
            Y0,Y1 = Y.chunk(2, dim=1)
            X_out, Y_out = self.model(X0, Y0, X1, Y1)
        else:
            X_out, Y_out = self.model(X, Y, X, Y)

        return self._KL_estimate(X_out, Y_out)

    def _forward_MI(self, X, Y):
        X0,X1 = X.chunk(2, dim=1)
        Y0,Y1 = Y.chunk(2, dim=1)

        Z1 = self.model(X0, Y0)
        Z2 = self.model(X0, Y1)

        return self._KL_estimate(Z1, Z2)

    def _forward_MI_KL(self, X, Y):
        
        Xs = X.chunk(6, dim=1)
        Ys = Y.chunk(6, dim=1)

        Z_joint1 = torch.cat([Xs[0],Ys[0]], dim=-1)
        Z_marginal1 = torch.cat([Xs[1],Ys[4]], dim=-1)

        Z_joint2 = torch.cat([Xs[2],Ys[2]], dim=-1)
        Z_marginal2 = torch.cat([Xs[3],Ys[5]], dim=-1)

        Z_joint_out, Z_marginal_out = self.model(Z_joint1, Z_marginal1, Z_joint2, Z_marginal2)

        return self._KL_estimate(Z_joint_out, Z_marginal_out)

    def _forward(self, X, Y):
        if self.mode == 'kl':
            return self._forward_KL(X, Y)
        elif self.mode == 'mi':
            return self._forward_MI(X, Y)
        elif self.mode == 'mi-kl':
            return self._forward_MI_KL(X, Y)
        else:
            raise NotImplementedError("")

    
    def train_step(self, i, steps, dataset):
        args = self.train_args
        (X,Y), _ = dataset(args['batch_size'], **args['sample_kwargs'])
        if args['normalize'] == 'whiten':
            X,Y = whiten_split(X,Y)

        X, Y = X.to(self.device),Y.to(self.device)
        
        d_out = self._forward(X,Y)

        loss = -1* d_out.mean()
        loss.backward()

        if args['clip'] > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args['clip'])

        if (i+1) % args['grad_steps'] == 0 or i == (steps - 1):
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def _eval(self, steps, dataset, set_size):
        args = self.eval_args
        sample_kwargs = {k:v for k,v in args['sample_kwargs'].items() if k != "set_size"}
        avg_loss = 0
        avg_diff = 0
        with torch.no_grad():
            for i in range(steps):
                (X,Y), theta = self.train_dataset(args['batch_size'], set_size=(set_size, set_size+1), **sample_kwargs)
                d_true = self.label_fct(*theta, X=X, **args['label_kwargs']).squeeze(-1)
                if args['normalize'] == 'whiten':
                    X,Y = whiten_split(X,Y)
                
                X, Y = X.to(self.device),Y.to(self.device)
                
                d_out = self._forward(X,Y)

                avg_loss += self.criterion(d_out, d_true)
                avg_diff += (d_out - d_true).mean().item()
            avg_loss /= steps
            avg_diff /= steps
        
        return avg_loss, avg_diff
        

    def evaluate(self, steps, dataset):
        #set_sizes = (100, 300, 1000)
        metrics={}
        #for ss in set_sizes:
        avg_loss_ss, avg_diff_ss = self._eval(steps, dataset, 300)
        metrics["criterion"] = avg_loss_ss
        metrics["signed-diff"] = avg_diff_ss
        
        return metrics





class DonskerVaradhanMITrainer(Trainer):
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, criterion, label_fct, 
            x_marginal, y_marginal, logger=None, save_every=2000, eval_every=500, scheduler=None, checkpoint_dir=None, ss_schedule=-1,
            sample_marg=True, estimate_size=-1, scale='none', eps=1e-6, model_type='mst', split_inputs=True):
        super().__init__(model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, logger=logger,
            save_every=save_every, eval_every=eval_every, criterion=criterion, scheduler=scheduler, checkpoint_dir=checkpoint_dir, ss_schedule=ss_schedule)
        self.label_fct = label_fct
        self.x_marginal = x_marginal
        self.y_marginal = y_marginal
        self.sample_marg = sample_marg
        self.estimate_size = estimate_size
        self.scale = scale
        self.eps=eps
        self.split_inputs = split_inputs
        self.model_type = model_type

    @staticmethod
    def _KL_estimate(X, Y):
        return X.sum(dim=1)/X.size(1) - Y.logsumexp(dim=1) + math.log(Y.size(1))


    def _forward(self, X, Y):
        if self.split_inputs:
            X0, X1 = X.chunk(2, dim=1)
            Y0, Y1 = Y.chunk(2, dim=1)
        else:
            X0, X1 = X, X
            Y0, Y1 = Y, Y

        if self.sample_marg:
            marginal_kwargs={
                'batch_size': X0.size(0),
                'n': X0.size(-1),
                'n_samples': X0.size(1),
                'sample_groups': 2
            }
            X2, X3 = self.x_marginal(**marginal_kwargs).to(self.device).chunk(2, dim=1)
            Y2, Y3 = self.y_marginal(**marginal_kwargs).to(self.device).chunk(2, dim=1)
        else:
            Y2 = batched_shuffle(Y0)
            X2 = X0
            if self.estimate_size > 0:
                N = (X1.size(1) // self.estimate_size) * self.estimate_size
                X1 = X1[:,:N]#rearrange(X[:,:N], 'b (e n) d -> (b e) n d', e=self.estimate_size) #X[:, :N].view(-1, self.estimate_size, *X.size()[2:])
                Y1 = Y1[:,:N] #Y[:, :N].view(-1, self.estimate_size, *Y.size()[2:])
                Y3 = batched_shuffle(rearrange(Y1, 'b (e n) d -> (b e) n d', e=self.estimate_size))
                Y3 = rearrange(Y3, '(b e) n d -> b (e n) d', e=self.estimate_size)
            else:
                Y3 = batched_shuffle(Y1)
            X3 = X1

        if self.model_type == 'encdec':
            Z_joint1 = torch.cat([X0,Y0], dim=-1)
            Z_marginal1 = torch.cat([X2,Y2], dim=-1)

            Z_joint2 = torch.cat([X1,Y1], dim=-1)
            Z_marginal2 = torch.cat([X3,Y3], dim=-1)

            Z_joint_out, Z_marginal_out = self.model(Z_joint1, Z_marginal1, Z_joint2, Z_marginal2)
        else:
            Z_joint1 = torch.cat([X,Y], dim=-1)
            Z_marginal1 = torch.cat([X,Y], dim=-1)

            Z_joint_out, Z_marginal_out = self.model(Z_joint1, Z_marginal1)

        if (not self.sample_marg) and self.estimate_size > 0:
            Z_joint_out = rearrange(Z_joint_out, 'b (e n) -> (b e) n', e=self.estimate_size)
            Z_marginal_out = rearrange(Z_marginal_out, 'b (e n) -> (b e) n', e=self.estimate_size)
        
        if self.scale == 'log':
            Z_joint_out = torch.log(torch.abs(Z_joint_out))
            Z_marginal_out = torch.log(torch.abs(Z_marginal_out))
        elif self.scale == 'logcov':
            cov_joint = batched_cov(Z_joint1)
            cov_marg = batched_cov(Z_marginal1)
            log_scale_factor = (cov_marg.logdet() - cov_joint.logdet()).unsqueeze(-1) * 1./2
            Z_joint_out = torch.log(F.sigmoid(Z_joint_out) + self.eps) + log_scale_factor
            Z_marginal_out = torch.log(F.sigmoid(Z_marginal_out) + self.eps) + log_scale_factor

        out = self._KL_estimate(Z_joint_out, Z_marginal_out)

        if (not self.sample_marg) and self.estimate_size > 0:
            out = rearrange(out, '(b e) -> b e', e=self.estimate_size)
            out = out.mean(dim=1, keepdim=True)

        return out
    
    def train_step(self, i, steps, dataset):
        args = self.train_args
        (X,Y), _ = dataset(args['batch_size'], **args['sample_kwargs'])
        if args['normalize'] == 'whiten':
            X,Y = whiten_split(X,Y)

        X, Y = X.to(self.device),Y.to(self.device)
        
        d_out = self._forward(X, Y)

        loss = -1* d_out.mean()
        loss.backward()

        if (i+1) % args['grad_steps'] == 0 or i == (steps - 1):
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def _eval(self, steps, dataset, set_size):
        args = self.eval_args
        sample_kwargs = {k:v for k,v in args['sample_kwargs'].items() if k != "set_size"}
        avg_loss = 0
        avg_diff = 0
        with torch.no_grad():
            for i in range(steps):
                (X,Y), theta = dataset(args['batch_size'], set_size=(set_size, set_size+1), **sample_kwargs)
                d_true = self.label_fct(*theta, X=(X,Y), **args['label_kwargs']).squeeze(-1)
                if args['normalize'] == 'whiten':
                    X,Y = whiten_split(X,Y)
                
                X, Y = X.to(self.device),Y.to(self.device)
                
                d_out = self._forward(X,Y)

                avg_loss += self.criterion(d_out, d_true)
                avg_diff += (d_out - d_true).mean().item()
            avg_loss /= steps
            avg_diff /= steps
        
        return avg_loss, avg_diff
        

    def evaluate(self, steps, dataset):
        if self.label_fct is None:
            return {}
        set_sizes = (300,)#(100, 300, 800)
        metrics={}
        for ss in set_sizes:
            avg_loss_ss, avg_diff_ss = self._eval(steps, dataset, ss)
            metrics["criterion_%d" % ss] = avg_loss_ss
            metrics["signed-diff_%d" % ss] = avg_diff_ss
        
        return metrics

#
#   DV2
#
'''
from utils import batched_shuffle

class DonskerVaradhanTrainer2(Trainer):
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, criterion, label_fct, 
            logger=None, save_every=2000, eval_every=500, scheduler=None, checkpoint_dir=None, ss_schedule=-1, split_inputs=True, 
            mode='kl', model_type='encdec', dataset='corr', estimate_size=-1):
        super().__init__(model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, logger=logger,
            save_every=save_every, criterion=criterion, scheduler=scheduler, checkpoint_dir=checkpoint_dir, ss_schedule=ss_schedule)
        self.label_fct = label_fct
        self.split_inputs = split_inputs
        self.mode = mode
        self.model_type = model_type
        self.dataset = dataset
        self.estimate_size=estimate_size

    @staticmethod
    def _KL_estimate(X, Y):
        return X.sum(dim=1)/X.size(1) - Y.logsumexp(dim=1) + math.log(Y.size(1))

    def _forward_KL(self, X, Y):

        X_out, Y_out = self.model(X, Y)

        return self._KL_estimate(X_out, Y_out)

#    def _forward_MI(self, X, Y):
#        X0,X1 = X.chunk(2, dim=1)
#        Y0,Y1 = Y.chunk(2, dim=1)

#        Z1 = self.model(X0, Y0)
#        Z2 = self.model(X0, Y1)

#        return self._KL_estimate(Z1, Z2)

    def _forward_MI_KL(self, X, Y):
        Y_marginal = Y[:,torch.randperm(Y.size(1))]
        P = torch.cat([X,Y], dim=-1)
        Q = torch.cat([X,Y_marginal], dim=-1)

        if self.model_type == 'mst':
            Z_joint= torch.cat([X,Y], dim=-1)
            Z_marginal = torch.cat([X,Y_marginal], dim=-1)
            Z_joint_out, Z_marginal_out = self.model(Z_joint, Z_marginal)
            out = self._KL_estimate(Z_joint_out, Z_marginal_out)
        else:
            if self.estimate_size > 0:
                N = (X.size(1) // self.estimate_size) * self.estimate_size
                X = X[:,:N]#rearrange(X[:,:N], 'b (e n) d -> (b e) n d', e=self.estimate_size) #X[:, :N].view(-1, self.estimate_size, *X.size()[2:])
                Y = Y[:,:N] #Y[:, :N].view(-1, self.estimate_size, *Y.size()[2:])
                Y_marginal = batched_shuffle(rearrange(Y, 'b (e n) d -> (b e) n d', e=self.estimate_size))
                Y_marginal = rearrange(Y_marginal, '(b e) n d -> b (e n) d', e=self.estimate_size)
            else:
                Y_marginal = batched_shuffle(Y)

            Z_joint= torch.cat([X,Y], dim=-1)
            Z_marginal = torch.cat([X,Y_marginal], dim=-1)
            Z_joint_out, Z_marginal_out = self.model(P, Q, Z_joint, Z_marginal)
            if self.estimate_size > 0:
                Z_joint_out = rearrange(Z_joint_out, 'b (e n) -> (b e) n', e=self.estimate_size)
                Z_marginal_out = rearrange(Z_marginal_out, 'b (e n) -> (b e) n', e=self.estimate_size)

            out = self._KL_estimate(Z_joint_out, Z_marginal_out)
            if self.estimate_size > 0:
                out = rearrange(out, '(b e) -> b e', e=self.estimate_size)
                out = out.mean(dim=1, keepdim=True)

        return out

    def _forward(self, X, Y):
        if self.mode == 'kl':
            return self._forward_KL(X, Y)
        elif self.mode == 'mi':
            return self._forward_MI(X, Y)
        elif self.mode == 'mi-kl':
            return self._forward_MI_KL(X, Y)
        else:
            raise NotImplementedError("")

    
    def train_step(self, i, steps, dataset):
        args = self.train_args
        (X,Y), _ = dataset(args['batch_size'], **args['sample_kwargs'])
        if args['normalize'] == 'whiten':
            X,Y = whiten_split(X,Y)

        X, Y = X.to(self.device),Y.to(self.device)
        
        d_out = self._forward(X,Y)

        loss = -1* d_out.mean()
        loss.backward()

        if args['clip'] > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args['clip'])

        if (i+1) % args['grad_steps'] == 0 or i == (steps - 1):
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def _eval(self, steps, dataset, set_size):
        args = self.eval_args
        sample_kwargs = {k:v for k,v in args['sample_kwargs'].items() if k != "set_size"}
        avg_loss = 0
        avg_diff = 0
        with torch.no_grad():
            for i in range(steps):
                (X,Y), theta = self.train_dataset(args['batch_size'], set_size=(set_size, set_size+1), **sample_kwargs)
                d_true = self.label_fct(*theta, X=X, **args['label_kwargs']).squeeze(-1)
                if args['normalize'] == 'whiten':
                    X,Y = whiten_split(X,Y)
                
                X, Y = X.to(self.device),Y.to(self.device)
                
                d_out = self._forward(X,Y)

                avg_loss += self.criterion(d_out, d_true)
                avg_diff += (d_out - d_true).mean().item()
            avg_loss /= steps
            avg_diff /= steps
        
        return avg_loss, avg_diff
        

    def evaluate(self, steps, dataset):
        #set_sizes = (100, 300, 1000)
        metrics={}
        #for ss in set_sizes:
        avg_loss_ss, avg_diff_ss = self._eval(steps, dataset, 300)
        metrics["criterion"] = avg_loss_ss
        metrics["signed-diff"] = avg_diff_ss
        
        return metrics

'''


