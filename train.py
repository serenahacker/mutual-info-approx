from models2 import MultiSetTransformer, PINE
from models import MultiSetTransformer1, EquiMultiSetTransformer1
from utils import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
import os
import shutil
import glob
import tqdm
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--target', type=str, default='wasserstein')
    parser.add_argument('--model', type=str, default='csab')
    parser.add_argument('--data', type=str, default='gmm')
    parser.add_argument('--normalize', type=str, choices=('none', 'scale', 'whiten'))
    #parser.add_argument('--norm_in', action='store_true')
    #parser.add_argument('--norm_out', action='store_true')
    parser.add_argument('--scaleinv', action='store_true')
    parser.add_argument('--checkpoint_dir', type=str, default="/checkpoint/kaselby")
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--scaling', type=float, default=0.5)
    parser.add_argument('--blur', type=float, default=0.05)
    parser.add_argument('--equi', action='store_true')
    parser.add_argument('--num_inds', type=int, default=-1)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=120000)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--old_model', action='store_true')
    return parser.parse_args()



def evaluate(model, generator, label_fct, exact_loss=False, batch_size=64, sample_kwargs={}, label_kwargs={}, criterion=nn.L1Loss(), steps=5000, normalize=False, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    model_losses = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(steps)):
            if exact_loss:
                X, theta = generator(batch_size, **sample_kwargs)
                labels = label_fct(*theta, X=X[0], **label_kwargs).squeeze(-1)
            else:
                X = generator(batch_size, **sample_kwargs)
                labels = label_fct(*X, **label_kwargs)
            if normalize == 'scale':
                Xnorm, avg_norm = normalize_sets(*X)
                out = model(*Xnorm).squeeze(-1)
                out *= avg_norm.squeeze(-1).squeeze(-1)
            elif normalize == 'whiten':
                Xnorm = whiten_split(*X)
                out = model(*Xnorm).squeeze(-1)
            else:
                out = model(*X).squeeze(-1)
            model_loss = criterion(out, labels)
            model_losses.append(model_loss.item())
    return sum(model_losses)/len(model_losses)

def train(model, sample_fct, label_fct, baselines={}, exact_loss=False, criterion=nn.L1Loss(), batch_size=64, steps=3000, lr=1e-5, 
    checkpoint_dir=None, output_dir=None, save_every=1000, sample_kwargs={}, label_kwargs={}, normalize='none'):
    #model.train(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    initial_step=1
    if checkpoint_dir is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                load_dict = torch.load(checkpoint_path)
                model, optimizer, initial_step, losses = load_dict['model'], load_dict['optimizer'], load_dict['step'], load_dict['losses']

    for i in tqdm.tqdm(range(initial_step,steps+1)):
        optimizer.zero_grad()
        if exact_loss:
            X, theta = sample_fct(batch_size, **sample_kwargs)
            #if use_cuda:
                #X = [x.cuda() for x in X]
                #theta = [t.cuda() for t in theta]
            if normalize == 'scale':
                X, avg_norm = normalize_sets(*X)
            labels = label_fct(*theta, X=X[0], **label_kwargs).squeeze(-1)
        else:
            X = sample_fct(batch_size, **sample_kwargs)
            #if use_cuda:
                #X = [x.cuda() for x in X]
            if normalize == 'scale':
                X, avg_norm = normalize_sets(*X)
            labels = label_fct(*X, **label_kwargs)
        if normalize == 'whiten':
            X = whiten_split(*X)
        loss = criterion(model(*X).squeeze(-1), labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % save_every == 0 and checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            torch.save({'model':model,'optimizer':optimizer, 'step': i, 'losses':losses}, checkpoint_path)

    seed = torch.randint(100, (1,)).item()
    '''
    model_loss = evaluate(model, sample_fct, label_fct, exact_loss=exact_loss, 
        batch_size=batch_size, label_kwargs=label_kwargs, sample_kwargs=sample_kwargs, criterion=criterion, 
        steps=500, normalize=normalize, seed=seed)
    baseline_losses = {}
    for baseline, baseline_fct in baselines.items():
        baseline_losses[baseline] = evaluate(baseline_fct, sample_fct, label_fct, exact_loss=exact_loss, 
            batch_size=batch_size, label_kwargs=label_kwargs, sample_kwargs=sample_kwargs, criterion=criterion, 
            steps=500, normalize=normalize, seed=seed)
    '''

    torch.save(model._modules['module'], os.path.join(output_dir,"model.pt"))  
    torch.save({'losses':losses}, os.path.join(output_dir,"logs.pt"))   

    return losses


if __name__ == '__main__':
    args = parse_args()
    run_dir = os.path.join("runs", args.run_name)
    '''if os.path.exists(run_dir):
        if args.overwrite:
            shutil.rmtree(run_dir)
        else:
            raise Exception("Folder exists and overwrite is set to false.")'''
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    device = torch.device("cuda:0")

    if not args.old_model:
        DIM=32
        if args.model == 'csab':
            model_kwargs={
                'ln':True,
                'remove_diag':True,
                'num_blocks':args.num_blocks,
                'equi':args.equi, 
                'output_size':1,
                'num_heads':args.num_heads,
                'num_inds':args.num_inds,
                'dropout':args.dropout
            }
            if args.equi:
                model_kwargs['input_size'] = 1
                model_kwargs['latent_size'] = 32
                model_kwargs['hidden_size'] = 48
            else:
                model_kwargs['input_size'] = DIM
                model_kwargs['latent_size'] = 256
                model_kwargs['hidden_size'] = 384
            model=MultiSetTransformer(**model_kwargs).to(device)
        elif args.model == 'pine':
            model = PINE(DIM, 32, 16, 2, 384, 1).to(device)
        else:
            raise NotImplementedError()
    else:
        DIM=32
        model_kwargs={'ln':True, 'remove_diag':True, 'num_blocks':2, 'num_heads':4}
        if args.equi:
            model = EquiMultiSetTransformer1(1,1,dim_hidden=32, **model_kwargs).to(device)
        else:
            model = MultiSetTransformer1(32, 1,1,dim_hidden=256, **model_kwargs).to(device)

    batch_size=args.batch_size
    steps=args.steps

    if torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        print("Let's use", n_gpus, "GPUs!")
        model = nn.DataParallel(model)
        batch_size *= n_gpus
        steps = int(steps/n_gpus)
    
    sample_kwargs={}
    if args.equi:
        sample_kwargs['dims'] = (24,40)
    else:
        sample_kwargs['n'] = DIM

    if args.target == 'w1':
        sample_kwargs['set_size'] = (10,150)
        label_fct = wasserstein
        label_kwargs={'scaling':0.98, 'blur':0.001}
        baselines={'sinkhorn_default':wasserstein}
        exact_loss=False
        lr = 5e-3
        criterion=nn.MSELoss()
        mixture=True
    elif args.target == 'w2':
        sample_kwargs['set_size'] = (10,150)
        label_fct = wasserstein2_gaussian
        label_kwargs={}
        baselines={'sinkhorn_default':wasserstein2}
        exact_loss=True
        lr = 1e-3
        criterion=nn.MSELoss()
        mixture=False
    elif args.target == 'w1_exact':
        sample_kwargs['set_size'] = (10,150)
        label_fct = wasserstein_mc
        label_kwargs={}
        baselines={'sinkhorn_default':wasserstein2}
        exact_loss=True
        lr = 1e-3
        criterion=nn.MSELoss()
        mixture=False
    elif args.target == 'kl':
        sample_kwargs['set_size'] = (200,300)
        label_fct = kl_mc
        label_kwargs={}
        baselines={'knn':kl_knn}
        exact_loss=True
        lr = 1e-5
        sample_kwargs['nu']=5
        sample_kwargs['mu0']=0.5
        sample_kwargs['s0']=0.5
        criterion=nn.L1Loss()
        mixture=True
        batch_size = int(batch_size/4)
        if args.equi:
            sample_kwargs['dims'] = (2,4)
        else:
            DIM=2
            sample_kwargs['n'] = 2

    if args.data == 'gmm':
        generator = GaussianGenerator(num_outputs=2, scaleinv=args.scaleinv, variable_dim=args.equi, return_params=exact_loss, mixture=mixture)
    elif args.data == 'nf':
        generator = NFGenerator(32, 2, num_outputs=2, use_maf=False, variable_dim=args.equi, return_params=exact_loss)
    else:
        raise NotImplementedError("nf or gmm")

    losses = train(model, generator, label_fct, baselines=baselines, checkpoint_dir=os.path.join(args.checkpoint_dir, args.checkpoint_name), \
        exact_loss=exact_loss, output_dir=run_dir, criterion=criterion, steps=steps, lr=lr, batch_size=batch_size, \
        sample_kwargs=sample_kwargs, label_kwargs=label_kwargs, normalize=args.normalize)




'''
d=2
hs=32
nh=4
ln=True
n_blocks=2
model= EquiEncoder(hs, n_blocks, nh, ln).cuda()
losses=train(model, generate_gaussian_nd, wasserstein, criterion=nn.MSELoss(), steps=20000, lr=1e-3, n=2, set_size=(50,75))
'''


