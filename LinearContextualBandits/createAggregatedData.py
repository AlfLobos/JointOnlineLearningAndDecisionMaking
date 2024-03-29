#!/usr/bin/env python
import numpy as np
import pandas as pd
import time as time
import os
import pickle as pkl


def get_avg_results(dict_FoldersPerExperiment, exp_name, methods_name, rho, max_iteration, path_data_root):
    """
    Given an experiment name, it returns average quantities for that experiment over the simulations run.

    Args
    -------
    dict_FoldersPerExperiment: Dict[str, Lit[str]]
        This dictionary contains as the value all the names of the experiment folders without the simulation
        number, and as keys a list of folder names containing all simulations that were run for that experiment.
    exp_name: str
        Experiment name without the simulation number (it should exist as a a key in dict_FoldersPerExperiment).
    methods_name: List[str]
        List of methods names.
    rho: float
        Cost of performing an action.
    max_iteration: int
        Maximum iteration run
    path_data_root: str
        Path where the experiments subfolders are.

    Return
    -------
    dict_to_ret: Dict[str, any]
        It has stored the average over all simulations of the quantities 'lams', 'rewards_over_time',
        'remaining_budget_over_time', 'actions_taken', 'total_actions_taken', 'total_reward' over the execution 
        of the algorithm. It also has the average of the quantities 'best_offline' and 
        'total_reward'.
    """
    dict_to_ret = {}
    dict_to_ret['best_offline'] = []
    dict_to_ret['num_of_sims'] = 0
    for metNum, methodName in enumerate(methods_name):
        dict_of_method = {}
        # dict_of_method['lams'] = np.zeros(max_iteration + 1)
        dict_of_method['rewards_over_time'] = np.zeros(max_iteration)
        dict_of_method['remaining_budget_over_time'] = np.zeros(max_iteration)
        dict_of_method['total_actions_taken'] = []
        dict_of_method['total_reward'] = []
        num_sim_verification = 0
        for folder_name in dict_FoldersPerExperiment[exp_name]:
            variable_budget = max_iteration
            full_folder_path = path_data_root + folder_name + '/'
            all_data_dict = pkl.load(open(os.path.join(full_folder_path, 'all_data.p'), "rb"))
            # lams_sim = all_data_dict['lams' + '_' + methodName]
            vec_y_sim = all_data_dict['y_ts' + '_' + methodName]
            vec_rew_sim = all_data_dict['rewards' + '_' + methodName]

            rew_on_time = np.zeros(max_iteration)
            budget_on_time = np.zeros(max_iteration)
            count_for_rew = 0
            for i, y_curr in enumerate(vec_y_sim):
                if y_curr == 1:
                    rew_on_time[i] = vec_rew_sim[count_for_rew]
                    count_for_rew += 1
                    variable_budget -= rho
                budget_on_time[i] = variable_budget
            
            # dict_of_method['lams'] += lams_sim
            dict_of_method['rewards_over_time'] += rew_on_time
            dict_of_method['remaining_budget_over_time'] += budget_on_time
            dict_of_method['total_actions_taken'].append(np.sum(vec_y_sim))
            dict_of_method['total_reward'].append(np.sum(vec_rew_sim))

            num_sim_verification += 1

            if metNum == 0:
                dict_to_ret['num_of_sims'] += 1
                dict_to_ret['best_offline'].append(all_data_dict['bestOffline'])

        if num_sim_verification != dict_to_ret['num_of_sims']:
            print('Number of simulations do not coincide')
        
            
        # dict_of_method['lams'] /= dict_to_ret['num_of_sims']
        dict_of_method['rewards_over_time'] /= dict_to_ret['num_of_sims']
        dict_of_method['remaining_budget_over_time'] /= dict_to_ret['num_of_sims']
        dict_to_ret[methodName] = dict_of_method

    return dict_to_ret


def get_agg_results(dict_FoldersPerExperiment, exp_names, methods_name, listTs, rho, path_data_root):
    avg_results_all = {}
    for exp_name in exp_names:
        start_time = time.time()
        max_iteration = listTs[int(exp_name[0])]
        dict_experiment = get_avg_results(dict_FoldersPerExperiment, exp_name,\
                                   methods_name, rho, max_iteration, path_data_root)
        dict_to_save_for_exp = {}
        for method in methods_name:
            dict_to_save_for_exp[method+'_'+'avg_tot_reward'] = \
                np.average(np.array(dict_experiment[method]['total_reward']))
            dict_to_save_for_exp[method+'_'+'rewards_over_time'] = \
                dict_experiment[method]['rewards_over_time'][:]
        
        dict_to_save_for_exp['avg_best_offline'] = np.average(np.array(dict_experiment['best_offline']))
        avg_results_all[exp_name] = dict_to_save_for_exp
        print(exp_name + ' took: ' + str(time.time() - start_time) + ' secs.')
        del dict_experiment
    return avg_results_all

if __name__ == '__main__':
    path_data_root = '/Volumes/disk2s2/OnlineSetting/ResultsICML/'
    all_directories = os.listdir(path_data_root)

    dict_FoldersPerExperiment = {}

    for folderName in all_directories:
        start_pos = folderName.find('_',1)
        exp_name = folderName[(start_pos+1):]
        if exp_name not in dict_FoldersPerExperiment:
            dict_FoldersPerExperiment[exp_name] = []
        dict_FoldersPerExperiment[exp_name].append(folderName)

    try:
        # We got an extra folder called 'Store' when we run the experiments.
        del dict_FoldersPerExperiment['Store']
    except:
        pass

    methods_name = ['MatrixApp-subg', 'ThompsonSampling-subg', 'RidgeReg-subg',
        'RidgeRegPlusRandomness-subg', 'KnownThetaAst-subg', 'FixTheta-subg']

    all_exps_names = list(dict_FoldersPerExperiment.keys())
    dict_exps_per_comb = {}

    num_of_d_n_confs = 8
    rho_used = 4

    for i in range(num_of_d_n_confs):
        dict_exps_per_comb[i] = []
    for exp_name in all_exps_names:
        dict_exps_per_comb[int(exp_name[4])].append(exp_name)

    path_to_save_agg_data = '/Volumes/disk2s2/OnlineSetting/AggregatedResults/'

    for i in range(0, num_of_d_n_confs):
        avgResults =  get_agg_results(dict_FoldersPerExperiment, dict_exps_per_comb[i], methods_name, \
                                    [1000, 5000, 10000], rho_used, path_data_root)
        pkl.dump(avgResults, \
            open(path_to_save_agg_data + 'comb_' + str(i) + '.p', "wb"))
