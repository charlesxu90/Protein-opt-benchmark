from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
import random
import os, time
import multiprocessing as mp
import warnings
from src.optimize import BayesianOptimization, BO_ARGS
import src.objectives as objectives
import src.utils as utils

'''
Script to repdouce all of the active learning simulations on GB1 and TrpB datasets. Launches optimization runs as separate processes.
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--name", type=str, default="GB1")
    parser.add_argument("--encoding", type=str, default="onehot")
    parser.add_argument("--mtype", type=str, default="DNN_ENSEMBLE", help='model type, in "BOOSTING_ENSEMBLE", "GP_BOTORCH", "DNN_ENSEMBLE", "DKL_BOTORCH"')
    parser.add_argument("--acq_fn", type=str, default="UCB", help="model type, in 'GREEDY', 'UCB', 'TS'")
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--n_pseudorand_init", type=int, default=96)
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--output_path", type=str, default='results/5x96_simulations/')
    parser.add_argument("--run", type=int, default=1, help="run number")
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--kernel", type=str, default="RBF", choices=["RBF"])
    parser.add_argument("--xi", type=float, default=4, help="trade-off parameter for the UCB acquisition function")
    parser.add_argument("--activation", type=str, default="lrelu")
    parser.add_argument("--min_noise", type=float, default=1e-6)
    parser.add_argument("--train_iter", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--verbose", type=int, default=2)

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    protein = args.name
    encoding = args.encoding
    device = args.device
    print(device)

    obj = objectives.Combo(protein, encoding)

    obj_fn = obj.objective
    domain = obj.get_domain()
    ymax = obj.get_max()
    disc_X = obj.get_points()[0]
    disc_y = obj.get_points()[1]
    batch_size = args.batch_size #number of samples to query in each round of active learning

    n_pseudorand_init = args.n_pseudorand_init #number of initial random samples
    budget = args.budget #total number of samples to query, not including random initializations

    try:
        mp.set_start_method('spawn')
    except:
        print('Context already set.')
    
    # make dir to hold tensors
    path = args.output_path
    subdir = path + protein + '/' + encoding + '/'
    os.makedirs(subdir, exist_ok=True)
    os.system('cp ' + __file__ + ' ' + subdir) #save the script that generated the results
    print('Script stored.')

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #do random search baseline
    start_x, start_y, start_indices = utils.samp_discrete(n_pseudorand_init, obj, seed)
    if budget != 0:
        _, randy, rand_indices = utils.samp_discrete(budget, obj, seed)
        rand_indices = torch.cat((start_indices, rand_indices), 0)
    else:
        rand_indices = start_indices

    torch.save(rand_indices, subdir + 'Random_' + str(seed + 1) + 'indices.pt')
    print('Random search done.')

    kernel=args.kernel #kernel must be radial basis function, only applies to GP_BOTORCH and DKL_BOTORCH
    mtype = args.mtype #model type, in "BOOSTING_ENSEMBLE", "GP_BOTORCH", "DNN_ENSEMBLE", "DKL_BOTORCH"
    acq_fn = args.acq_fn #acquisition function, in "GREEDY", "UCB", "TS":        
    dropout=args.dropout #dropout rate, only applies to neural networks models (DNN_ENSEMBLE and DKL_BOTORCH)

    if mtype == 'GP_BOTORCH' and 'ESM2' in encoding:
        lr = 1e-1
    else:
        lr = 1e-3
    
    num_simult_jobs = 1 #number of simulations to run in parallel

    #set the architecture of the neural network
    if 'DNN' in mtype and 'ENSEMBLE' in mtype:
        if 'onehot' in encoding:
            arc  = [domain[0].size(-1), 30, 30, 1]
        elif 'AA' in encoding:
            arc  = [domain[0].size(-1), 8, 8, 1]
        elif 'georgiev' in encoding:
            arc  = [domain[0].size(-1), 30, 30, 1]
        elif 'ESM2' in encoding:
            arc  = [domain[0].size(-1), 500, 150, 50, 1] 
    elif 'GP' in mtype:
        arc = [domain[0].size(-1), 1]
    elif 'DKL' in mtype:
        if 'onehot' in encoding:
            arc  = [domain[0].size(-1), 30, 30, 1]
        elif 'AA' in encoding:
            arc  = [domain[0].size(-1), 8, 8, 1]
        elif 'georgiev' in encoding:
            arc  = [domain[0].size(-1), 30, 30, 1]
        else:
            arc  = [domain[0].size(-1), 500, 150, 50, 1]
    else:
        arc = [domain[0].size(-1), 1]

    #filename
    fname = f"{mtype}-{acq_fn}-{dropout}-{kernel}-{arc[-2:]}-{args.run}"
    sim_args = BO_ARGS(
            mtype=mtype,
            kernel=kernel,
            acq_fn=acq_fn,
            xi=args.xi, #xi term, only applies to UCB
            architecture=arc,
            activation=args.activation,
            min_noise=args.min_noise,
            trainlr=lr,
            train_iter=args.train_iter,
            dropout=dropout,
            mcdropout=0,
            verbose=args.verbose,
            bb_fn=obj_fn,
            domain=domain,
            disc_X=disc_X,
            disc_y=disc_y,
            noise_std=0,
            n_rand_init=0, #additional random inits
            budget=budget,
            query_cost=1,
            queries_x=start_x,
            queries_y=start_y,
            indices=start_indices,
            savedir=subdir+fname,
            batch_size = batch_size
        )

    BayesianOptimization.run(sim_args, seed)
    
    print('Tensors will be saved in {}'.format(subdir))
