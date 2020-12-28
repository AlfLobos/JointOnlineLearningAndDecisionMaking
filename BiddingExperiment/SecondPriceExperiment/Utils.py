import torch
from torch.utils.data import Dataset
import sys
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
import time
import sys

class CreateDataset(Dataset):
    """
    Creates a Dataset to be later used in our DataLoader functions/
    """
    def __init__(self, X_Mat, Y_Vec):
        self.X_Mat = X_Mat 
        self.Y_Vec = Y_Vec

    def __len__(self):
        return self.X_Mat.size()[0]

    def __getitem__(self, idx):
        return [self.X_Mat[idx], self.Y_Vec[idx]]

def createTrValTestData(path_to_read, mult_for_cost, seed_to_use, num_of_indxs_to_use):
    """
    Args
    ------
    path_to_read: str
        Path where the datais stored
    mult_for_cost: float
        Multiplier to modify the Greedy heuristic
    seed_to_use: int 
        Seed to fix for obtaining the q vector.
    num_of_indxs_to_use: int
        Number of auctions in the bidder could potentially participate at each iteration
    
    Returns:
    ------
    X_Train: torch.tensor(torch.long) (2-d, num_points_train, num_features)
        Train feature matrix
    X_Val: torch.tensor(torch.long) (2-d, num_points_val, num_features)
        Validation feature matrix
    X_Test: torch.tensor(torch.long) (2-d, num_points_test, num_features)
        Test feature matrix
    extra_train: torch.tensor(torch.float) (2-d, num_points_train, 4)
        Tensor that has if a conversion occured, the highest competing bid,
        the cost-per-order (defined in the Criteo dataset, but not used here), 
        and the day of the week. The only column we use is 'cost' which 
        corresponds to column 1
    extra_val: torch.tensor(torch.float) (2-d, num_points_val, 4)
        Tensor that has if a conversion occured, the highest competing bid,
        the cost-per-order (defined in the Criteo dataset, but not used here), 
        and the day of the week. The only column we use is 'cost' which 
        corresponds to column 1
    extra_test: torch.tensor(torch.float) (2-d, num_points_test, 4)
        Tensor that has if a conversion occured, the highest competing bid,
        the cost-per-order (defined in the Criteo dataset, but not used here), 
        and the day of the week. The only column we use is 'cost' which 
        corresponds to column 1
    target: torch.tensor(torch.float) (1-d, num_advs)
        Target budget for each tensor for each iteration of the algorithm (is the 
        'b' vector from the paper)
    q: torch.tensor(torch.float) (1-d, num_advs)
        Price that each advertiser is willing to pay to the bidder per conversion.
    lbs: torch.tensor(torch.float) (1-d, num_advs)
        Percentage of the target budget that the advertisers would like to spend 
        (alpha on the notation of the paper)
    camps_identifiers: torch.tensor(torch.long) (1-d, num_advs)
        Indexes which represents the campaigns inside the embedding function.
    num_advs: int
        Total number of advertisers
    avg_cost_per_conv: np.array(float) (1-d, num_advs)
        Average cost per conv for each advertiser in the order we use
        the advertisers
    """
    Y_Train = torch.tensor(pickle.load(open(path_to_read+'Y_Train.p', "rb")), dtype = torch.float)
    Y_Val = torch.tensor(pickle.load(open(path_to_read+'Y_Val.p', "rb")), dtype = torch.float)
    Y_Test = torch.tensor(pickle.load(open(path_to_read+'Y_Test.p', "rb")), dtype = torch.float)

    X_train_np= pickle.load(open(path_to_read+'X_Train.p', "rb"))
    X_val_np = pickle.load(open(path_to_read+'X_Val.p', "rb"))
    X_test_np = pickle.load(open(path_to_read+'X_Test.p', "rb"))

    # 'conversion','cost', 'cpo', 'dayMonth'
    extra_train = torch.tensor(pickle.load(open(path_to_read+'Extra_Train.p', "rb")), dtype = torch.float)
    extra_val = torch.tensor(pickle.load(open(path_to_read+'Extra_Val.p', "rb")), dtype = torch.float)
    extra_test = torch.tensor(pickle.load(open(path_to_read+'Extra_Test.p', "rb")), dtype = torch.float)

    ## We are simply scaling the cost values
    ## 'conversion', 'cost', 'cpo', 'dayMonth'
    extra_train[:,1] *= 100 
    extra_val[:,1] *= 100
    extra_test[:,1] *= 100

    ## Get the matrices in the right format for the neural network.
    num_advs = int(np.max(X_train_np, axis = 0)[1]+1)
    maximums  = np.max(X_train_np, axis = 0) +1
    count = 0 
    for i in range(1, np.shape(X_train_np)[1]):
        X_train_np[:,i] += maximums[i-1] + count
        X_val_np[:,i] += maximums[i-1] + count
        X_test_np[:,i] += maximums[i-1] + count
        count += maximums[i-1]
        
    X_Train= torch.tensor(X_train_np, dtype = torch.long)
    X_Val = torch.tensor(X_val_np, dtype = torch.long)
    X_Test = torch.tensor(X_test_np, dtype = torch.long)

    ## We get the total cost per capaign in the test set which will work as our total budgets.
    # 'dayWeek', 'advertiser', 'click', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6','cat8', 'cat9'
    # dfTest = pd.DataFrame({'advertiser': X_Test[:, 1], 'dayMonth': np.array(extra_test[:, 3]),\
    #     'cost': np.array(extra_test[:, 1]), 'cpo': np.array(extra_test[:, 2]), 'conversion': np.array(Y_Test)})
    dfTest = pd.DataFrame({'advertiser': np.array(X_Test[:, 1]), 'conversion': np.array(extra_test[:, 0]), \
                       'cost': np.array(extra_test[:, 1]), 'cpo': np.array(extra_test[:, 2])})
    
    sum_by_advertiser = dfTest.groupby('advertiser')['cost', 'conversion'].sum()

    camps_identifiers = torch.tensor(np.array(list(sum_by_advertiser.index)).astype(int), dtype = torch.long)

    tot_cost_by_camp = np.array(sum_by_advertiser['cost'])
    conv_by_advertiser = np.array(sum_by_advertiser['conversion'])

    avg_cost_per_conv = tot_cost_by_camp/conv_by_advertiser

    # Create q vector. Is important to fix the seed here to be consistent when comparing to the 
    # greedy policy.
    np.random.seed(seed_to_use)
    q = torch.tensor(avg_cost_per_conv, dtype = torch.float) \
        * torch.tensor(np.random.uniform(low = 0.5, high = 1.5, size = num_advs), dtype = torch.float)

    del X_train_np
    del X_val_np
    del X_test_np
    del count
    del dfTest

    ## Let us now define the budgets and payment per conversion

    # The target budgets are obtained proportionally from the total budgets.
    target = torch.tensor(tot_cost_by_camp *(num_of_indxs_to_use/(X_Test.size()[0]+0.0)),\
                dtype = torch.float)
    ## We desire all advertisers to spend at least 95% of their budget.
    lbs = torch.ones(num_advs) * 0.95

    return X_Train, X_Val, X_Test, extra_train, extra_val, extra_test, target , q, \
        lbs, camps_identifiers, num_advs, avg_cost_per_conv

def retBidsAndCamps2nd(slice_X_mat, len_indexes, num_advs, net, q, one_minus_blam, depleted_camps,
    use_sigm, sigm_or_softMax, camp_ids):
    """
    Returns the bids and the advertisers we bid on behalf of.
    Args:
    --------
    slice_X_mat: torch.tensor(torch.long) (2-d, (len_indexes x num_features))
        Tensor to process per row to obtaining the prob. of conversion for all campigns
        for each datapoint
    len_indexes: int
        Number of datapoints to process per batch
    num_advs: int
        Number of advertisers
    net: Pytorch model
        Pre-trained Pytorch model which we assume that receives data in the format
        slice_X_mat.
    one_minus_blam: torch.tensor(torch.float) (1-d)
        Vector representing the value of 1 - lambda.
    depleted_camps: torch.tensor(torch.bool)
        Each coordinate represents a advertiser, if true a advertiser has not exhausted
        its budget yet, otherwise it has.
    use_sigm: bool
        If true the network uses sigmoid to obtain the probabilities, otherwise
        softmax
    sigm_or_softMax: torch.nn.functional
        Either  nn.Sigmoid() or nn.Softmax(dim = 1)
    camp_ids: torch.tensor(torch.long) (1-d)
        Has the indexes which represent the different advertisers on the embedding vectors. 
        The indexes are in order in which we process the advertisers.
    
    Returns
    --------
    rik_winners: torch.tensor(torch.float)
    """
    riks = torch.zeros(num_advs, len_indexes, dtype = torch.float)
    with torch.no_grad():
        count = 0
        for row_to_use in slice_X_mat:
            toUse = row_to_use.repeat(num_advs,1)
            toUse[:,1] = camp_ids
            output = net(toUse)
            if use_sigm:
                ## In this option we should have sigm_or_softMax = nn.Sigmoid()
                riks[:, count] = torch.squeeze(sigm_or_softMax(output)) * q
            else:
                ## In this option we should have sigm_or_softMax = nn.SoftMax(dim =1)
                riks[:, count] = torch.squeeze(sigm_or_softMax(output)[:,1]) * q
            count +=1
        # Set as zero the revenue for all depleted advertisers
        for i, bool_elem in enumerate(depleted_camps):
            if bool_elem:
                riks[i,:] *= 0.0
    bids2nd, winners = torch.max((riks.t() * one_minus_blam).t(), dim =0)
    rik_winners = riks[winners, torch.arange(len_indexes)]
    return rik_winners, bids2nd, winners

def retBidsAndCampsGreedy(slice_X_mat, len_indexes, num_advs, net, q, bid_mult, depleted_camps,
    use_sigm, sigm_or_softMax, camp_ids):
    """
    Returns the bids and the advertisers we bid on behalf of.
    Args:
    --------
    slice_X_mat: torch.tensor(torch.long) (2-d, (len_indexes x num_features))
        Tensor to process per row to obtaining the prob. of conversion for all campigns
        for each datapoint
    len_indexes: int
        Number of datapoints to process per batch
    num_advs: int
        Number of advertisers
    net: Pytorch model
        Pre-trained Pytorch model which we assume that receives data in the format
        slice_X_mat.
    bid_mult: torch.float >0
        Multiplier to modify all bids to be submitted.
    depleted_camps: torch.tensor(torch.bool)
        Each coordinate represents a advertiser, if true a advertiser has exhausted
        its budget, otherwise it hasn't.
    use_sigm: bool
        If true the network uses sigmoid to obtain the probabilities, otherwise
        softmax
    sigm_or_softMax: torch.nn.functional
        Either  nn.Sigmoid() or nn.Softmax(dim = 1)
    camp_ids: torch.tensor(torch.long) (1-d)
        Has the indexes which represent the different advertisers on the embedding vectors. 
        The indexes are in order in which we process the advertisers.
    
    Returns
    --------
    rik_winners: torch.tensor(torch.float)
    """
    riks = torch.zeros(num_advs, len_indexes, dtype = torch.float)
    with torch.no_grad():
        count = 0
        for row_to_use in slice_X_mat:
            toUse = row_to_use.repeat(num_advs,1)
            toUse[:,1] = camp_ids
            output = net(toUse)
            if use_sigm:
                ## In this option we should have sigm_or_softMax = nn.Sigmoid()
                riks[:, count] = torch.squeeze(sigm_or_softMax(output)) * q
            else:
                ## In this option we should have sigm_or_softMax = nn.SoftMax(dim =1)
                riks[:, count] = torch.squeeze(sigm_or_softMax(output)[:,1]) * q
            count +=1
        # Set as zero the revenue for all depleted advertisers
        for i, bool_elem in enumerate(depleted_camps):
            if bool_elem:
                riks[i,:] *= 0.0
    bids_greedy, winners = torch.max((riks.t() * bid_mult).t(), dim =0)
    rik_winners = riks[winners, torch.arange(len_indexes)]
    return rik_winners, bids_greedy, winners

def spendAndRevenuePerAdvertiser(win_camps, rik_win_camps, bids, cost_to_use, num_advs):
    """
    Args
    -----
    win_camps: torch.tensor(torch.long) (1-d)
        advertisers to bid on behalf of in order for the given batch
    rik_win_camps: torch.tensor(torch.float) (1-d)
        Revenue the advertisers would pay in case their corresponding 
        bid wins the auction
    bids: torch.tensor(torch.float) (1-d)
        Bids that the DSP submits to the auctions
    cost_to_use: torch.tensor(torch.float) (1-d)
        Second price for the different auctions
    num_advs: int
        Number of total advertisers (regardless if depleted or not).
    """
    ## We only bid when the dual revenue is greater  
    auctionWon = (cost_to_use <= bids).float()
    did_we_bid = (bids >= 0.0).float()
    spend = torch.zeros(num_advs)
    profit = torch.zeros(num_advs)
    ## The spend is for the advertisers, and the profit for the DSP
    spend.index_add_(0, win_camps, rik_win_camps * did_we_bid * auctionWon)
    profit.index_add_(0, win_camps, (rik_win_camps - cost_to_use ) * did_we_bid * auctionWon)
    return spend, profit

def subgNoSpendInvolved(blam, lbs, target):
    return target*((blam>=0.0).float()) + target*lbs*((blam<0.0).float())

def runCodeDualMet(X, extras, target, q, net, lbs, initBlam, num_advs, check_times_list, lenghtOfIte, 
    step_size, use_sigm, camp_ids):
    """
    Args
    -------
    X: torch.tensor(torch.long) (2-d, num_points x num_features)
        Feature matrix to use
    extras: torch.tensor(torch.float) (2-d, num_points x 3)
        Tensor matrix with the column representing 'conversion', 'cost', 
        'cpo', 'dayMonth'. We only use the 'cost' column.
    target: torch.tensor(torch.float) (1-d, num_advertisers)
        Target budget for each advertiser  (total budget for all advertisers is
        target * numIte)
    q: torch.tensor(torch.float) (1-d, num_advertisers)
        Avergae value of a user converting for each advertiser
    net: Pytorch model
        Pre-trained Pytorch model which we assume that receives data in the format 
        of the X tensor.
    lbs: torch.tensor(torch.float) (1-d, num_advertisers)
        Minimum targe budget percentage to be spent per advertiser
    initBlam: torch.tensor(torch.float) (1-d, num_advertisers)
        Initial value for the dual variables
    num_advs: int
        Number of total advertisers (regardless if depleted or not).
    check_times_list: List[int]
        Iteration numbers in which we print the tota running time.
    lenghtOfIte: int
        Number of auctions to perform per iteration
    step_size: float
        Step size value to use for the subgradient step
    use_sigm: bool
        If true the network uses sigmoid to obtain the probabilities, otherwise
        softmax
    camp_ids: torch.tensor(torch.long) (1-d)
        Has the indexes which represent the different advertisers on the embedding vectors. 
        The indexes are in order in which we process the advertisers.
    """
    sigm_or_softMax = nn.Sigmoid()
    if not use_sigm:
        sigm_or_softMax = nn.Softmax(dim = 1)
    blam = initBlam.clone()
    totPoints = X.size()[0]
    costVec = extras[:,1]
    start_time = time.time()
    numIte = int(totPoints/lenghtOfIte)
    maxBudgets = target * numIte
    bidsAll = torch.zeros(numIte * lenghtOfIte)
    winnersAll = torch.zeros(numIte * lenghtOfIte)
    spending = torch.zeros(num_advs, numIte, dtype = torch.float)
    profit = torch.zeros(num_advs, numIte, dtype = torch.float)
    allBlams = torch.zeros(num_advs, numIte)

    curr_tot_spending = torch.zeros(num_advs)

    first_time_exc_budget = True

    ite_first_time_budget_exc = -1

    # One way of creating a boolean tensor containing only False elements
    depleted_camps = (torch.ones(num_advs) < 0)

    for i in range(0, numIte):
        one_minus_blam = 1 - blam
        rik_winners, bids, winners = retBidsAndCamps2nd(X[lenghtOfIte*i:lenghtOfIte*(i+1)], 
            lenghtOfIte, num_advs, net, q, one_minus_blam, depleted_camps, use_sigm, 
            sigm_or_softMax, camp_ids)

        spendAux, profitAux = spendAndRevenuePerAdvertiser(winners, rik_winners, bids, 
            costVec[lenghtOfIte*i:lenghtOfIte*(i+1)], num_advs)

        spending[:,i] = spendAux
        profit[:,i] = profitAux
        allBlams[:,i] = blam.clone()
        bidsAll[lenghtOfIte * i : lenghtOfIte * (i+1)] = bids[:]
        winnersAll[lenghtOfIte * i : lenghtOfIte * (i+1)] = winners[:]

        gradp = subgNoSpendInvolved(blam, lbs, target)
        blam -= step_size * (gradp - spendAux)

        curr_tot_spending += spendAux

        depleted_camps = (curr_tot_spending >= maxBudgets)

        if (torch.sum(depleted_camps) > 0) and first_time_exc_budget:
            # First time budget is exceeded
            ite_first_time_budget_exc = i
            first_time_exc_budget =  False
        if i in check_times_list:
            print('Iteration '+str(i)+', time so far '+str(time.time() - start_time) + ' seconds')
            sys.stdout.flush()
    return allBlams, spending, profit, bidsAll, winnersAll, ite_first_time_budget_exc

def runCodeGreedy(X, extras, target, q, net, lbs, bid_mult, num_advs, check_times_list, 
    lenghtOfIte, use_sigm, camp_ids):
    """
    Args
    -------
    X: torch.tensor(torch.long) (2-d, num_points x num_features)
        Feature matrix to use
    extras: torch.tensor(torch.float) (2-d, num_points x 3)
        Tensor matrix with the column representing 'conversion', 'cost', 
        'cpo', 'dayMonth'. We only use the 'cost' column.
    target: torch.tensor(torch.float) (1-d, num_advertisers)
        Target budget for each advertiser  (total budget for all advertisers is
        target * numIte)
    q: torch.tensor(torch.float) (1-d, num_advertisers)
        Avergae value of a user converting for each advertiser
    net: Pytorch model
        Pre-trained Pytorch model which we assume that receives data in the format 
        of the X tensor.
    lbs: torch.tensor(torch.float) (1-d, num_advertisers)
        Minimum targe budget percentage to be spent per advertiser
    initBlam: torch.tensor(torch.float) (1-d, num_advertisers)
        Initial value for the dual variables
    num_advs: int
        Number of total advertisers (regardless if depleted or not).
    check_times_list: List[int]
        Iteration numbers in which we print the tota running time.
    lenghtOfIte: int
        Number of auctions to perform per iteration
    step_size: float
        Step size value to use for the subgradient step
    use_sigm: bool
        If true the network uses sigmoid to obtain the probabilities, otherwise
        softmax
    camp_ids: torch.tensor(torch.long) (1-d)
        Has the indexes which represent the different advertisers on the embedding vectors. 
        The indexes are in order in which we process the advertisers.
    """
    sigm_or_softMax = nn.Sigmoid()
    if not use_sigm:
        sigm_or_softMax = nn.Softmax(dim = 1)
    totPoints = X.size()[0]
    costVec = extras[:,1]
    start_time = time.time()
    numIte = int(totPoints/lenghtOfIte)
    maxBudgets = target * numIte
    bidsAll = torch.zeros(numIte * lenghtOfIte)
    winnersAll = torch.zeros(numIte * lenghtOfIte)
    spending = torch.zeros(num_advs, numIte, dtype = torch.float)
    profit = torch.zeros(num_advs, numIte, dtype = torch.float)

    curr_tot_spending = torch.zeros(num_advs)
    first_time_exc_budget = True
    ite_first_time_budget_exc = -1
    # One way of creating a boolean tensor containing only False elements
    depleted_camps = (torch.ones(num_advs) < 0)

    for i in range(0, numIte):
        rik_winners, bids, winners = retBidsAndCampsGreedy(X[lenghtOfIte*i:lenghtOfIte*(i+1)], 
            lenghtOfIte, num_advs, net, q, bid_mult, depleted_camps, use_sigm, 
            sigm_or_softMax, camp_ids)

        spendAux, profitAux = spendAndRevenuePerAdvertiser(winners, rik_winners, bids, 
            costVec[lenghtOfIte*i:lenghtOfIte*(i+1)], num_advs)

        spending[:,i] = spendAux
        profit[:,i] = profitAux
        bidsAll[lenghtOfIte * i : lenghtOfIte * (i+1)] = bids[:]
        winnersAll[lenghtOfIte * i : lenghtOfIte * (i+1)] = winners[:]

        curr_tot_spending += spendAux
        depleted_camps = (curr_tot_spending >= maxBudgets)
        if (torch.sum(depleted_camps) > 0) and first_time_exc_budget:
            # First time budget is exceeded
            ite_first_time_budget_exc = i
            first_time_exc_budget =  False
        if i in check_times_list:
            print('Iteration '+str(i)+', time so far '+str(time.time() - start_time) + ' seconds')
            sys.stdout.flush()
    return spending, profit, bidsAll, winnersAll, ite_first_time_budget_exc
