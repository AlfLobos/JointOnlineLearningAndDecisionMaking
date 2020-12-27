#!/usr/bin/env python
import os
import numpy as np

def ReturnListsOrderedByIndex(rand_indxs_split, all_run_files, indexToSearch, possibleIndexValues):
    """
    Returns a list of format List[List[str]] with each sublist being all the run files that 
    will be included in a task file. The run files are ordered depending on the attribute 
    obtained when doing a_run_file_example[-indexToSearch] for which we assume that all
    its unique elements are in possibleIndexValues

    Args
    ------
    rand_indxs_split: List[np.array(int)]
        List of arrays each of which contains a partition of ints. 
    all_run_files: List[str]
        List containing string with path + file_name + args configiuration parameters.
        It contains the whole sequence of python code that we want to run.
    indexToSearch: int
        Position in which one parameter of interest appear in an average element in 
        all_run_files. In the case that we run it correspond to the index of 
        the number of maximum iterations to use.
    possibleIndexValues: List[int]
        Different values that the position retrieved using indexToSearch can take.
    
    Return
    ------
    list_run_files_to_ret: List[List[str]]
        list that contains len(rand_indxs_split) lists of strings. Each list contains
        string eachof them representing path + file_name + args configiuration parameters.
    """
    list_run_files_to_ret = []
    for i in range(len(rand_indxs_split)):
        aux_dict = {}
        for pos_index in possibleIndexValues:
            aux_dict[pos_index] = []
        for indexInList in rand_indxs_split[i]:
            a_run_file_ex = all_run_files[indexInList]
            aux_dict[a_run_file_ex[-indexToSearch]].append(a_run_file_ex)
        list_run_files_to_ret.append([])
        for pos_index in possibleIndexValues:
            list_run_files_to_ret[i] += aux_dict[pos_index]
    return list_run_files_to_ret


if __name__ == '__main__':
    path  = 'Path where linConWithCons.py will be run\'
    name_of_python_file = 'linConWithCons.py'
    num_sim = 100
    possible_max_iteration = [1000, 5000, 10000]
    # We use only the first index of rho in our experiments.
    possible_rho = [2, 4, 8]
    possible_d_n_conf = [[5,10], [10,5], [5,5], [10,10], [25,50], [50,25], [25,25], [50,50]]
    possible_bd_on_revenue_error = [0.0, 0.1, 0.5]
    possible_bd_on_unc_per_row = [0.0, 0.1]

    all_run_files = [path + name_of_python_file+ ' ' + str(sim) + ' ' + str(ind_T) + ' ' + str(ind_rho) + \
                    ' ' + str(ind_Comb) + ' ' + str(ind_RevE) + ' ' + str(ind_UncPRow) + '\n'
                        for sim in range(num_sim)
                        for ind_T in range(len(possible_max_iteration))
                        for ind_rho in [1]
                        for ind_Comb in [0, 1, 2, 3, 4, 5, 6, 7]
                        for ind_RevE  in [0, 1, 2]
                        for ind_UncPRow in [0, 1] ]

    rand_indxs = np.random.choice(np.arange(len(all_run_files)), len(all_run_files), replace = False)
    rand_indxs_split = np.array_split(rand_indxs,4)

    run_files_format_to_save = ReturnListsOrderedByIndex(rand_indxs_split, all_run_files, 10, ['0', '1', '2'])

    for expNum in range(len(rand_indxs_split)):
        fileToWrite = open('expWoutNegIntercept'+'_'+str(expNum),"w")
        fileToWrite.writelines(run_files_format_to_save[expNum])
        fileToWrite.close()