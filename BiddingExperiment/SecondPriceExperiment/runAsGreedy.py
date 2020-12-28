#!/usr/bin/env python
import numpy as np
import pickle
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import sys
import pandas as pd
# import torch.optim as optim
from FwFM_Network import FwFM_ForCELoss, FwFM_Logits
from Utils import CreateDataset, runCodeGreedy, createTrValTestData

if __name__ == '__main__':
    current_directory = os.getcwd()
    path_to_read = os.path.join(current_directory, 'DataForPred/')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('index_bid_mult', type=int, help = "Index for bid multiplier")
    parser.add_argument('index_seed', type=int, help = "Payment per Conversion")
    args = parser.parse_args()

    possible_bid_mults = [0.25 + 0.05 * i for i in range(26)]
    np.random.seed(12369)
    possible_seeds = np.random.choice(np.arange(1000000) + 1, size = 100, replace = False)

    bid_mult = possible_bid_mults[int(args.index_bid_mult)]
    seed_to_use = possible_seeds[int(args.index_seed)]

    mult_for_cost = 100
    num_of_indxs_to_use = 128

    X_Train, X_Val, X_Test, extra_train, extra_val, extra_test, target , q, lbs, camps_identifiers, num_advs, _ = \
        createTrValTestData(path_to_read, mult_for_cost, seed_to_use, num_of_indxs_to_use)

    ## Pytorch parameters to read the best network found before

    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    size_vocab = int((torch.max(X_Train) + 1).item())
    # size_vocab = np.sum(maximums)
    num_embs = X_Train.size()[1]

    ## Otherwise

    size_embs = 10
    useMask = False
    mask_to_use = None
    scale_grad_by_freq = True
    use_sigm = True

    net = 0
    if use_sigm:
        net = FwFM_Logits(size_vocab, size_embs, num_embs, mask_to_use = mask_to_use,\
            scale_grad_by_freq = scale_grad_by_freq).to(device)
    else:
        net = FwFM_ForCELoss(size_vocab, size_embs, num_embs, mask_to_use = mask_to_use,\
            scale_grad_by_freq = scale_grad_by_freq).to(device)


    # ## Load a Network

    # In[13]:

    path_to_readModel = os.path.join(os.getcwd(), 'BestModel/')
    files  = os.listdir(path_to_readModel)
    filesEndingInPt = []
    for fileName in files:
        if fileName.endswith(".pt"):
            filesEndingInPt.append(fileName)
    filesEndingInPt
    if len(filesEndingInPt) >1:
        print('There is more than one *.pt file in '+str(path_to_readModel))
        print('Please precise the model you want to use')
        sys.exit()
    elif len(files) ==0:
        print('There is no *.pt files in '+str(path_to_readModel))
        print('Please add at least one model in the folder')
        sys.exit()
    net.load_state_dict(torch.load(path_to_readModel + filesEndingInPt[0], map_location = device))

    # At least for the first iteration, we assume that all advertisers have the same dual multipliers.
    check_times_list = [1000, 2500, 5000, 10000, 20000, 50000]


    # In[ ]:
    torch.manual_seed(23)
    np.random.seed(673)

    sys.stdout.flush()

    spending, profit, bids_all, winners_all, ite_when_budget_vio = runCodeGreedy(X_Test, \
        extra_test, target, q, net, lbs, bid_mult, num_advs, check_times_list, \
        num_of_indxs_to_use, use_sigm, camps_identifiers)

    path_to_read = os.path.join(current_directory, 'ResultsGr/')
    if not os.path.exists(path_to_read):
        os.makedirs(path_to_read)

    run_name = str(int(args.index_bid_mult)) + '_' + str(int(args.index_seed))

    torch.save(spending, path_to_read + 'spending_real_run_' + run_name + '.pt')
    torch.save(profit, path_to_read + 'profit_real_run_' + run_name + '.pt')
    torch.save(bids_all, path_to_read + 'bids_all_' + run_name + '.pt')
    torch.save(winners_all, path_to_read + 'winners_all_' + run_name + '.pt')
    pickle.dump(ite_when_budget_vio, open(path_to_read + 'ite_when_budget_vio_' + run_name + '.p',"wb"))