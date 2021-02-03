#!/usr/bin/env python

import numpy as np
import os
import time
import argparse
from utils_run import run_an_experiment, save_procedure

if __name__ == '__main__':

    # Common Parameters
    num_sim = 100
    possible_max_iteration = [1000, 5000, 10000]
    # We use only the first index of Rho in our experiments.
    possible_rho = [2, 4, 8]
    possible_d_n_conf = [[5,10], [10,5], [5,5], [10,10], [25,50], [50,25], [25,25], [50,50]]
    possible_bd_on_revenue_error = [0.0, 0.1, 0.5]
    possible_bd_on_unc_per_row = [0.0, 0.1]

    thetaMets = ['RidgeReg', 'RidgeRegPlusRandomness', 'MatrixApp', 'ThompsonSampling', 'KnownThetaAst', 'FixTheta']
    ## In the future I could add other mirror Descent methods
    mirrorDescMets = ['subg']

    ## The part about adding a negative intercept is not done in this file.
    ## Other fixed parameters

    alpha_b = 0.5
    b = 1
    bd_unc_ridge = 0.3
    alphaForRidge = 0.001

    initLam = 0

    np.random.seed(3678)
    ## Up to 5 seeds per simulations
    seedsMat = np.random.choice(10000000, (num_sim,2))

    parser = argparse.ArgumentParser()
    parser.add_argument('sim', type=int, help = "Sim to run")
    parser.add_argument('T', type=int,  help = "Length of the simulation")
    parser.add_argument("rho", type=int, help = "Rho to use")
    parser.add_argument('comb', type=int, help = "Combination of num_vec and size_vec to use")
    parser.add_argument('revE'      , type=int, help = "Bd on Revenue Error")
    parser.add_argument('uncRow', type=int, help = "Bd on uncertainty per Row for each row of W")

    args = parser.parse_args()
    indexSim = int(args.sim)
    indexT = int(args.T)
    indexRho = int(args.rho)
    indexComb =  int(args.comb)
    indexRevE = int(args.revE)
    indexUncRow =  int(args.uncRow)

    sim = indexSim
    T = possible_max_iteration[indexT]
    rho = possible_rho[indexRho]
    num_vec, size_vec = possible_d_n_conf[indexComb]
    bd_on_revenue_error = possible_bd_on_revenue_error[indexRevE]
    bd_on_unc_per_row = possible_bd_on_unc_per_row[indexUncRow]


    initTheta = np.ones(size_vec)/np.sqrt(size_vec)
    seedsToUse = seedsMat[sim]
    barC = rho
    eta = 0.5/np.sqrt(T)
    thres = int(np.sqrt(T)/2)
    nu = bd_on_revenue_error * np.sqrt(size_vec * np.log(T))/10
    if bd_on_revenue_error == 0:
        nu = 0.1

    start_time = time.time()
    infoToRet, theta_ast, W_real, allRandInRev, bestOffline = run_an_experiment(T, b, alpha_b, num_vec, size_vec, nu, initLam,\
        initTheta, eta, rho, thres, mirrorDescMets, thetaMets, alphaForRidge, bd_on_revenue_error, \
        bd_unc_ridge, bd_on_unc_per_row, seedsToUse, barC)

    end_time = time.time()

    listIndexes = [indexSim, indexT, indexRho, indexComb, indexRevE, indexUncRow]

    midPartOfName = ''
    for ind in listIndexes:
        midPartOfName += str(ind) + '_'

    print('Running '+midPartOfName+' took: '+str(end_time-start_time)+' secs.')

    parent_folder_to_save = os.path.join(os.getcwd(), 'ResultsICML')

    if not os.path.exists(parent_folder_to_save):
        os.makedirs(parent_folder_to_save)

    save_procedure(infoToRet, theta_ast, bestOffline, allRandInRev, parent_folder_to_save, listIndexes)
