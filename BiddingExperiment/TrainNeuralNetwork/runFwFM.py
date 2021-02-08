#!/usr/bin/env python
def RunExperiment(net, train_loader, train_loader_oversampled, val_loader, \
    val_loader_oversampled, Y_tr_np, Y_val_np, epochs, torchOptim, lossFnTr, lossFnVal, \
    results_directory, name, lossSigm, device, saveNetDict, scheduler, tol):
    losses = train(net, train_loader, train_loader_oversampled, val_loader, \
        val_loader_oversampled, Y_tr_np, Y_val_np, torchOptim, lossFnTr, lossFnVal, 0, epochs,\
        name, device, lossSigm, checkForDiv = True, pathToSave = results_directory,\
        saveNetDict = saveNetDict, scheduler = scheduler, tol = tol)
    np.savetxt(results_directory + name + '.txt', np.array(losses), fmt='%s')

if __name__ == '__main__':
    import numpy as np
    import pickle
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data.sampler import WeightedRandomSampler
    import os
    import time
    import sys
    from RunCode import train
    import torch.optim as optim
    from FwFM_Network import FwFM_Logits, FwFM_ForCELoss
    from Utils import CreateDataset

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('indexStep', type=int, help = "Instance and nabla to use")
    parser.add_argument('indexEmb', type=int, help = "Instance and nabla to use")
    parser.add_argument('indexMask', type=int, help = "Instance and nabla to use")
    parser.add_argument('indexScale', type=int, help = "Instance and nabla to use")
    parser.add_argument('indexLoss', type=int, help = "Instance and nabla to use")
    # parser.add_argument('indexReWeight', type=int, help = "Instance and nabla to use")
    args = parser.parse_args()

    possibleStepSizes = [1, 0.1, 0.01, 0.001, 0.0001]
    possibleEmbSizes = [5, 10, 25]
    possibleUseMask = [False, True]
    possibleScale_grad_by_freq = [False, True]

    stepSize = possibleStepSizes[int(args.indexStep)]
    size_embs = possibleEmbSizes[int(args.indexEmb)]
    useMask = possibleUseMask[int(args.indexMask)]
    scale_grad = possibleScale_grad_by_freq[int(args.indexScale)]
    # indRW = int(args.indexReWeight)

    lossSigm = False
    if int(args.indexLoss) == 1:
        lossSigm = True

    lossToUse = [[nn.CrossEntropyLoss().to(device), \
        nn.CrossEntropyLoss(reduction = 'sum').to(device)],\
        [nn.BCEWithLogitsLoss().to(device), \
        nn.BCEWithLogitsLoss(reduction = 'sum').to(device)]]

    lossFnTr, lossFnVal = lossToUse[int(args.indexLoss)]

    name = 'embSize_' + str(int(args.indexEmb)) + '_indStep_' + str(int(args.indexStep)) + \
    '_useMask_' + str(int(args.indexMask)) + '_scaleGrad_' + str(int(args.indexScale)) + \
    '_indLoss_' + str(int(args.indexLoss))

    print('Running '+ str(name))

    current_directory = os.getcwd()
    pathToRead = os.path.join(current_directory, 'DataForPred/')

    X_TrainNp= pickle.load(open(pathToRead+'X_Train.p', "rb"))
    X_ValNp = pickle.load(open(pathToRead+'X_Val.p', "rb"))
    # X_TestNp = pickle.load(open(pathToRead+'X_Test.p', "rb"))

    ## Get the matrices in the right format for the neural network.
    num_campaigns = int(np.max(X_TrainNp,axis=0)[1]+1)
    maximums  = np.max(X_TrainNp, axis =0) +1
    count = 0 
    for i in range(1, np.shape(X_TrainNp)[1]):
        X_TrainNp[:,i] += maximums[i-1] + count
        X_ValNp[:,i] += maximums[i-1] + count
        # X_TestNp[:,i] += maximums[i-1] + count
        count += maximums[i-1]
        
    X_Train= torch.tensor(X_TrainNp, dtype = torch.long)
    X_Val = torch.tensor(X_ValNp, dtype = torch.long)
    # X_Test = torch.tensor(X_TestNp, dtype = torch.long)

    # BCEWithLogitsLoss uses float, while CrossEntropyLoss uses long.
    possible_types_Y = [torch.long, torch.float]
    type_for_Y = possible_types_Y[int(args.indexLoss)]
    Y_Train = torch.tensor(pickle.load(open(pathToRead+'Y_Train.p', "rb")), dtype = type_for_Y)
    Y_Val = torch.tensor(pickle.load(open(pathToRead+'Y_Val.p', "rb")), dtype = type_for_Y)

    Y_tr_np = Y_Train.cpu().numpy()
    Y_val_np = Y_Val.cpu().numpy()

    # Y_Test = torch.tensor(pickle.load(open(pathToRead+'Y_Test.p', "rb")), dtype = type_for_Y)

    epochs = 25
    batch_sizeTr = 256
    batch_sizeVal = 512
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    ## Let us do the weights for the oversampling
    count_class_0 = torch.sum(Y_Train == 0.0).item()
    count_class_1 = torch.sum(Y_Train == 1.0).item()

    weights = torch.ones(2, dtype = torch.float)
    weights_norm = torch.ones(2, dtype = torch.float)
    weights[0] = 1./(count_class_0 + 0.0)
    weights[1] = 1./(count_class_1 + 0.0)
    weights_norm[0] = weights[0]/ torch.sum(weights)
    weights_norm[1] = weights[0]/ torch.sum(weights)

    weights_train = weights_norm[Y_Train.long()]
    weights_val = weights_norm[Y_Val.long()]
    sampler_tr = WeightedRandomSampler(weights_train, len(weights_train))
    sampler_val = WeightedRandomSampler(weights_val, len(weights_val))

    

    train_loader_oversampled = torch.utils.data.DataLoader(CreateDataset(X_Train, Y_Train),\
                        batch_size = batch_sizeTr, sampler=sampler_tr, **kwargs)
    train_loader = torch.utils.data.DataLoader(CreateDataset(X_Train, Y_Train),\
                        batch_size = batch_sizeVal, shuffle=True, **kwargs)

    val_loader_oversampled = torch.utils.data.DataLoader(CreateDataset(X_Val, Y_Val),\
                        batch_size = batch_sizeTr, sampler = sampler_val, **kwargs)
    val_loader = torch.utils.data.DataLoader(CreateDataset(X_Val, Y_Val),\
                        batch_size = batch_sizeVal, shuffle=False, **kwargs)

    size_vocab = int((torch.max(X_Train) + 1).item())
    num_embs = X_Train.size()[1]

    m = torch.ones(num_embs, num_embs).to(device)

    for i in torch.arange(num_embs):
        for j in range(0, i+1):
            m[i,j] = 0

    mask_to_use = ((m == 0).nonzero()).to(device)

    if not useMask:
        mask_to_use = None

    
    net = 0
    if int(args.indexLoss) == 1:
        net = FwFM_Logits(size_vocab, size_embs, num_embs, mask_to_use = mask_to_use, scale_grad_by_freq = scale_grad).to(device)
    else:
        net = FwFM_ForCELoss(size_vocab, size_embs, num_embs, mask_to_use = mask_to_use, scale_grad_by_freq = scale_grad).to(device)

    opt = optim.Adam(net.parameters(), lr = stepSize, betas=(0.9, 0.999))
    scheduler = None

    results_directory = os.path.join(os.getcwd(), 'Results/')

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    start_time = time.time()
    saveNetDict = True
    tol = 1000000
    RunExperiment(net, train_loader, train_loader_oversampled, val_loader, \
        val_loader_oversampled, Y_tr_np, Y_val_np, epochs, opt, lossFnTr, lossFnVal,
        results_directory, name, lossSigm, device, saveNetDict, scheduler, tol)
    print('Running ' + str(name) + ' took: '+str(time.time() - start_time) + ' secs.')