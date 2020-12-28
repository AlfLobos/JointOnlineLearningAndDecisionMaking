import torch
from torch.utils.data import Dataset
import sys
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import time
from torch.utils.data import Dataset, DataLoader
import time
import os
import sys


def get_q(num_advs, avg_cost_per_conv, possible_seeds, index_to_use):
    """
    Returns the q vector used at the given simulation

    Args
    ------
    num_advs: int
        Number of advertisers
    avg_cost_per_conv: np.array(float) (1-d, num_advs)
        Average cost per conv for each advertiser in the order we use
        the advertisers
    possible_seeds: List[int]
        Vector containing the seeds used at each simulation
    index_to_use: int
        Index to use in possible_seeds to extract the correct seed. 
        In out case is equivalent to sim_number.
    
    Returns:
    ------
    q: torch.tensor(torch.float) (1-d, num_advs)
        Tensor representing the price each advertiser would pay per conversion
    
    """

    np.random.seed(possible_seeds[index_to_use])
    return torch.tensor(avg_cost_per_conv, dtype = torch.float) \
        * torch.tensor(np.random.uniform(low = 0.5, high = 1.5, size = num_advs), dtype = torch.float)


def get_total_measures_per_run_camp(index_step_or_bid, num_of_sims, numIte, 
    num_advs, max_budget, pathToReadResults):
    """
    Args
    ------
    index_step_or_bid: int
        Index repreenting the step size of bid multiplier that we want to retrive information for
    num_of_sims: int
        Number of total simulations run (assume the first starts with 0).
    num_advs: int
        Number of advertisers.
    max_budget: torch.tensor(torch.float)
        Maximum budget that each advertiser was allowed to spend.
    pathToReadResults: string
        Path to where the data is stored.
    
    Returns
    ------
    profit_mat: np.ndarray(float) (2-d, num_of_sims, num_advs)
        It has the total profit per (simulation, advertiser)
    spend_mat: np.ndarray(float) (2-d, num_of_sims, num_advs)
        It has the total spending per (simulation, advertiser)
    """
    # Matrices to save the results
    profit_mat = np.zeros((num_of_sims, num_advs))
    spend_mat = np.zeros((num_of_sims, num_advs))
    vio_mat = np.zeros((num_of_sims, num_advs))
    ites_budgets_excs = []
    for sim in range(num_of_sims):
        # Ending for the files with given index_step_or_bid and simulation number
        ending_pt = str(index_step_or_bid) + '_' + str(sim) + '.pt'
        ending_p = str(index_step_or_bid) + '_' + str(sim) + '.p'
        # Read data for this run
        profit_tensor = \
            torch.load(os.path.join(pathToReadResults, 'profit_real_run_' + ending_pt))
        spend_tensor = \
            torch.load(os.path.join(pathToReadResults, 'spending_real_run_' + ending_pt))
        profit_mat[sim,:] = torch.sum(profit_tensor, dim = 1).numpy()
        spend_mat[sim,:] = torch.sum(spend_tensor, dim = 1).numpy()
        vio_mat[sim, :] = get_first_vio_per_adv(spend_tensor.numpy(), max_budget, num_advs, numIte)
        ite_budget_exceeded = pickle.load(open(os.path.join(pathToReadResults, 'ite_when_budget_vio_' + ending_p), "rb"))
        ites_budgets_excs.append(ite_budget_exceeded)
    return profit_mat, spend_mat, vio_mat, ites_budgets_excs


def create_df_from_dict(dict_to_read, x_col_name, y_col_name):
    """
    Fill it up
    """
    names_keys = list(dict_to_read.keys())
    concat_y_vector = np.concatenate([dict_to_read[key_name] for key_name in names_keys])
    concat_x_list = []
    for key_name in names_keys:
        concat_x_list.extend([key_name for i in range(len(dict_to_read[key_name]))])
    pd_to_ret = pd.DataFrame({y_col_name: concat_y_vector, x_col_name: concat_x_list}) 
    return pd_to_ret

def conv_to_dict_w_arrays(dict_to_read, val_to_change, changed_valued):
    keys_names = list(dict_to_read.keys())
    dict_to_ret = {}
    for key_name in keys_names:
        aux_array = [val if val != val_to_change else val_to_change for val in dict_to_read[key_name]]
        dict_to_ret[key_name] = np.array(aux_array)
    return dict_to_ret

def get_first_vio_per_adv(spend_mat, max_budget, num_advs, numIte):
    cum_sum_mat = np.cumsum(spend_mat, axis = 1)
    array_to_ret = np.zeros(num_advs)
    for i in range(num_advs):
        exc_budget_vector = cum_sum_mat[i,:] >= max_budget[i]
        if np.sum(exc_budget_vector) < 1:
            array_to_ret[i] =  numIte
        else:
            array_to_ret[i] = np.argmax(exc_budget_vector)
    return array_to_ret