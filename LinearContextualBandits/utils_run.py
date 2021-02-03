import numpy as np
# np.random.default_rng = np.random.default_rng

from sklearn.linear_model import Ridge
import time
import pickle
import os


def update_M_and_Minverse(M, invM, w_t):
    midNom = np.matmul(np.expand_dims(w_t, axis=1), np.expand_dims(w_t, axis=0))
    nom = np.matmul(np.matmul(invM, midNom),invM)
    denom = 1 + np.dot(np.dot(w_t, invM), w_t)
    return M + midNom, invM - nom/denom

class Generate_theta():
    def __init__(self, dtUpToT):
        self.method_name = dtUpToT['ThetaUpdMethod']
        self.thres = dtUpToT['thres']
        self.M = dtUpToT['M']
        self.invM = dtUpToT['invM']
        self.sumActRew = 0
        self.thetaHat = dtUpToT['thetas'][-1]
        
        self.ridgeObj = dtUpToT['ridgeRegObject']
        # self.rng = dtUpToT['rng']
        self.nu = dtUpToT['nu']
            
    def update_data(self, dtUpToT):
        reward =  dtUpToT['rewards'][-1]
        w_selec = dtUpToT['w_selected'][-1]
        self.M, self.invM = \
            update_M_and_Minverse(self.M, self.invM, dtUpToT['w_selected'][-1])
        dtUpToT['M'], dtUpToT['invM'] = self.M, self.invM
        self.sumActRew +=  w_selec * reward
        self.thetaHat = np.dot(self.invM , self.sumActRew)
        
    def get_theta(self, numTimesCons, dtUpToT, uncVecForRidge = None):
        method_name = dtUpToT['ThetaUpdMethod']
        thres = dtUpToT['thres']
        if (method_name == 'RidgeReg' or method_name == 'RidgeRegPlusRandomness') and numTimesCons < thres:
            return self.thetaHat
        else:            
            if method_name == 'RidgeReg' or method_name == 'RidgeRegPlusRandomness':
                thetaJustRidge = 0
                ## This was a trick to make the ridge regression method faster
                # if len(dtUpToT['w_selected'])<= 1000:
                dtUpToT['ridgeRegObject'].fit(dtUpToT['w_selected'], dtUpToT['rewards'])
                thetaJustRidge = dtUpToT['ridgeRegObject'].coef_[:]
                # else:
                #     indexes = np.random.choice(len(dtUpToT['w_selected']), 1000, replace=False)
                #     w_selToUse = [dtUpToT['w_selected'][i] for i in indexes]
                #     rew_ToUse = [dtUpToT['rewards'][i] for i in indexes]
                #     dtUpToT['ridgeRegObject'].fit(w_selToUse, rew_ToUse)
                #     thetaJustRidge = dtUpToT['ridgeRegObject'].coef_[:]
                #     thetaJustRidge = 0.99 * dtUpToT['thetas'][-1] + 0.01 * thetaJustRidge
                if method_name == 'RidgeReg':
                    return thetaJustRidge
                else:
                    extRand = uncVecForRidge/np.sqrt(numTimesCons)
                    return thetaJustRidge + extRand
            elif method_name == 'MatrixApp':
                return self.thetaHat
            elif method_name == 'ThompsonSampling':
                nu = dtUpToT['nu']
                return np.squeeze(np.random.multivariate_normal(self.thetaHat, nu*nu*self.invM, 1))                

def solve_dual_problem(matW, theta, lam, rho):
    k_ast = np.argmax(np.dot(matW, theta))
    z = np.zeros(np.shape(matW)[0])
    dotMax = np.dot(matW[k_ast], theta)
    if dotMax - lam * rho >0:
        z[k_ast] = 1
        return dotMax, z, matW[k_ast], 1, k_ast
    else:
        return dotMax, z, matW[k_ast], 0, k_ast

def subg_dual(rho, binActTaken, lam, b, alpha_b):
    if lam >=0:
        return -rho * binActTaken + b
    else:
        return -rho * binActTaken + alpha_b
    
def mirror_descent(lam_old, alpha_b, tilde_g, eta, nameOfMethod = 'subg'):
    if nameOfMethod == 'subg':
        lamBefProj = lam_old - eta * tilde_g
    if alpha_b is not None:
        return lamBefProj
    else:
        return np.maximum(lamBefProj, 0)

def best_offline_solution(theta_ast, W_all, rho, b, alpha_b, T):
    bestValues = np.array([np.max(np.dot(W_all[i], theta_ast))  for i in range(len(W_all))])
    sortedValues  = (np.sort(bestValues))[::-1]
    minActs, maxActs = int(np.ceil(alpha_b *T/rho)), int(np.floor(b*T/rho))
    onlyBestMaxActs = (sortedValues[:maxActs])
    vec_cumul_sum = np.cumsum(onlyBestMaxActs)
    if minActs<maxActs:
        # Expected case.
        return np.max(vec_cumul_sum[minActs:])
    else:
        print("Caution, we have lower/upper bounds ({0}, {1})".format(minActs, maxActs) )
        return np.max(vec_cumul_sum[-1])

def createW_theta(num_vec, size_vec):
    W_real, theta_ast = 0, 0
    theta_ast = np.random.rand(size_vec) - 0.5
    theta_ast /= np.sqrt(np.dot(theta_ast, theta_ast))
    W_real = np.random.rand(num_vec, size_vec) - 0.5
    l2_norm = np.sqrt((W_real * W_real).sum(axis=1))
    for i in range(num_vec):
        W_real[i] /=  l2_norm[i]
    return theta_ast, W_real

def create_rands(num_vec, size_vec, max_iter):
    randTensorW = np.zeros((max_iter, num_vec, size_vec))
    randForRidge = np.zeros((max_iter, size_vec))
    randForRev = (np.random.rand(max_iter) - 0.5) * 2
    for i in range(max_iter):
        randTensorW[i,:,:] = (np.random.rand(num_vec, size_vec) - 0.5) * 2
        randForRidge[i,:] = (np.random.rand(size_vec) - 0.5) * 2
    return randTensorW, randForRidge, randForRev

def general_algorithm(b, alpha_b, theta_ast, barC, eta, rho, initT, dtUpToT , thetaGen,\
                     Ws, RandVecForRidge, vectUncRev):
    theta  = dtUpToT['thetas'][0][:]
    lam = dtUpToT['lams'][0]
    finT = initT + len(Ws)

    budgetLeft = dtUpToT['budgetLeft']
    
    methodTheta = dtUpToT['ThetaUpdMethod']
    
    numTimesCons = dtUpToT['numTimesCons']

    for t in range(initT, finT):

        ## Necessary Randomness 
        randForRidge = RandVecForRidge[t - initT]
        uncRev = vectUncRev[t - initT]
        
        ## Step 1. Obtain theta
        if methodTheta == 'KnownThetaAst':
            theta = theta_ast[:]
        elif methodTheta == 'FixTheta':
            theta = dtUpToT['thetas'][0][:]
        else:
            if numTimesCons > 0:
                theta = thetaGen.get_theta(numTimesCons, dtUpToT, uncVecForRidge = randForRidge)

        ## Step 2. Receive W
        matW = Ws[t - initT]   
        
        
        ## Step 3. Obtain primal Value
        dot_t, z_t, w_t, y_t, k_t = solve_dual_problem(matW, theta, lam, rho)
        dot_ast_ts_ast, z_t_ast, w_t_ast, y_t_ast, k_t_ast = solve_dual_problem(matW, theta_ast, lam, rho)
        numTimesCons += y_t

        ## 'dot_ts', 'dot_ast_ts', 'dot_ts_ast', 'dot_ast_ts_ast'
        ### Save info of dual problem using theta_t
        dtUpToT['z_ts'].append(z_t)
        dtUpToT['y_ts'].append(y_t)
        dtUpToT['k_ts'].append(k_t)
        ### Save info of dual problem using theta_ast
        dtUpToT['z_t_asts'].append(z_t_ast)
        dtUpToT['y_ts_ast'].append(y_t_ast)
        dtUpToT['k_ts_ast'].append(k_t_ast)

        ## All possible dots Combinations
        dtUpToT['dot_ts'].append(dot_t)
        dtUpToT['dot_ast_ts'].append(np.dot(w_t_ast, theta))
        dtUpToT['dot_ts_ast'].append(np.dot(w_t, theta_ast))
        dtUpToT['dot_ast_ts_ast'].append(dot_ast_ts_ast)
        
        ## ## Step 4. and 5. Stochastic Subgradient and Mirror Descent
        
        tilde_g = subg_dual(rho, y_t, lam, b, alpha_b)
#         print('tilde_g: '+str(tilde_g))
        lam = mirror_descent(lam, alpha_b, tilde_g, eta, dtUpToT['MDescMethod'])
#         print('lam: '+str(lam))
        dtUpToT['lams'].append(lam)
        
        if y_t == 1:    
            dtUpToT['thetas'].append(theta)
            dtUpToT['w_selected'].append(w_t)
            dtUpToT['w_selected_ast'].append(w_t_ast)
            dtUpToT['numTimesCons'] += 1
            
            ## Step 6. Budget consumption and observe revenue
            budgetLeft -= rho
            dtUpToT['budgetLeft'] -= rho
            dtUpToT['rewards'].append(np.dot(w_t, theta_ast) + uncRev)
            dtUpToT['rewards_ast'].append(np.dot(w_t_ast, theta_ast) + uncRev)
            
            ## Step 7. Break Condition
            if budgetLeft < barC:
                break
                
            ## Step 8. Update theta data
            if methodTheta not in ['KnownThetaAst', 'FixTheta']:
                thetaGen.update_data(dtUpToT)


def run_an_experiment(T, b, alpha_b, num_vec, size_vec, nu, initLam, initTheta, eta, rho, thres, \
             mirrorDescMets, thetaMets, alphaForRidge, bd_on_revenue_error, bd_unc_ridge, \
             bd_on_unc_per_row, seedsToUse, barC):
    """
    T: int
        Maximum iteration to run.
    b: target budget per iteration
        T * b gies the total budget
    alpha_b: float
        Lower bound for the target to be spent at each period, in terms of the paper is alpha *b 
    num_vec: int
        The 'd' coordinate in the paper adn can also be thought as the number of vectors
        we can elect from at each iteration
    size_vec: int
        The 'n' coordinate in the paper and it corresponds to the size of the vectors
        mentioned above.
    nu: float
        Patrameter for the Thompson Sampling method
    initLam: float
        Initial lambda, or dual variable, to use.
    initTheta: np.array(float) (1-d, size_vec)
        Initial value for the theta parameter
    eta: float
        Step size for the mirror descent method used (only subgradient descent is implemented)
    rho: float
        Cost of performing an action
    thres: int
        The Ridge Regression and Ridge R. plus uncertainty execute the Matrix Method for the first 
        'thres' iterations before executing the corresponding methods.  
        Number of actions to use the Matrix Method before starting to use the Ridge Regression
        method with and without uncertainty. 
    mirrorDescMets: list[str]
        List of Mirror descent methods to execute. Currently only subgradient descent is 
        implemented
    thetaMets: List[str]
        List of methods to try, example Ridge Regression, Matrix Method, etc.
    alphaForRidge: float
        Regularization parameter to use when running ridge regression.
    bd_on_revenue_error: float
        Bound the uniform that is added a error to the revenue function (every time we add to the 
        revenue function an i.i.d. Uniform(-0.1,0.1) * bd_on_revenue_error).
    bd_unc_ridge: float
        Bound the uncertainty used in the Ridge Regression Plus Uncertainty method
    bd_on_unc_per_row: float
        Bound on the uncertainty that is added elementwise to W at each iteration.
    seedsToUse: List[int]
        Seeds use for reproducibility
    barC: float
        Constant defined in the paper, and is used to make sure that the methods do not exceed the 
        total budget. For this problem, barC = rho.

    Return 
    -------
    infoToRet: Dict[str,any]
        All information collected for the experiment, including dual variables, revenue obtained, 
        action taken, etc.
    theta_ast: np.array(float)
        theta^* value from the paper. Is the real theta^* value that the methods try to learn.
    W_real: np.ndarray(float) 
        Real W matrix over which uncertainty is added depending on the experiment characteristics.
    vectUncRev: np.array(float) (1-d, T)
        T i.i.d. Uiform(-1,1) used for adding uncertainty to the revenue term.
    bestOffline: float
        Value of best offline solution possible if all the uncertainty  were known in advance.
    
    """
    
    infoToRet = {}

    np.random.seed(seedsToUse[0])

    theta_ast, W_real = createW_theta(num_vec, size_vec)  
    randTensorW, randForRidge, randForRev = create_rands(num_vec, size_vec, T)
    Ws = [W_real + randTensorW[i] * bd_on_unc_per_row for i in range(T)]
    RandVecsForRidge = [randForRidge[i][:] * bd_unc_ridge for i in range(T)]
    vectUncRev = randForRev * bd_on_revenue_error
    del randTensorW
    del randForRidge
    
    bestOffline = best_offline_solution(theta_ast, Ws, rho, b, alpha_b, T)

    allCombThetaMirDescMet = [(thetaMet, mDesMet) for thetaMet in thetaMets for mDesMet in mirrorDescMets]

    for pair in allCombThetaMirDescMet:
        thetaMet, mDesMet = pair[0], pair[1]
        ## Fix Seed2

        ## "Data Up To Period t"
        dtUpToT = {}

        ## Create Random Generator for Sampling Gaussians and Ridge Object

        np.random.seed(seedsToUse[1])
        # dtUpToT['rng'] =  np.random.default_rng()
        dtUpToT['ridgeRegObject'] = Ridge(alpha = alphaForRidge, fit_intercept = 0.0)
        dtUpToT['budgetLeft'] = b * T
        dtUpToT['numTimesCons'] = 0
        dtUpToT['lams'] = [initLam]
        dtUpToT['thetas'] = [initTheta]
        dtUpToT['nu'] = nu
        dtUpToT['thres'] = thres
        dtUpToT['M'] = np.diag(np.ones(size_vec))
        dtUpToT['invM'] = np.diag(np.ones(size_vec))
        dtUpToT['ThetaUpdMethod'] = thetaMet
        dtUpToT['MDescMethod'] = mDesMet
        
        ## Populate dtUpToT
        
        ## Everything related to "Data Up To Period T".
        namesWithEmptyLists = ['dot_ts', 'dot_ast_ts', 'dot_ts_ast', 'dot_ast_ts_ast', 'rewards',\
            'rewards_ast', 'w_selected', 'w_selected_ast', 'k_ts', 'k_ts_ast', 'y_ts', 'y_ts_ast', \
            'z_ts', 'z_t_asts']
        for name in namesWithEmptyLists:
            dtUpToT[name] = []
        
        dtUpToT['theta_ast'] = theta_ast[:]
        dtUpToT['W_real'] = W_real[:]

        ## Theta Generator

        thetaGen = 0
        if dtUpToT['ThetaUpdMethod'] not in ['KnownThetaAst', 'FixTheta']:
            thetaGen = Generate_theta(dtUpToT)

        general_algorithm(b, alpha_b, theta_ast, barC, eta, rho, 0, dtUpToT , thetaGen,\
                    Ws, RandVecsForRidge, vectUncRev)
        del dtUpToT['ridgeRegObject']
        infoToRet[thetaMet+'-'+mDesMet] = dtUpToT
    return infoToRet, theta_ast, W_real, vectUncRev, bestOffline

def save_procedure(dictFull, theta_ast, bestOffline, allRandInRev, parent_folder_to_save, listIndexes):
    midPartOfName = ""
    for ind in listIndexes:
        midPartOfName += '_'+str(ind)
    folderToSave = parent_folder_to_save + '/' + midPartOfName
    if not os.path.exists(parent_folder_to_save + '/' + midPartOfName):
        os.makedirs(parent_folder_to_save + '/' + midPartOfName)

    dict_all_data = {}
    dict_all_data['bestOffline'] =  bestOffline

    for name in dictFull.keys():
        dict_all_data['lams' + '_' + name] =  np.array(dictFull[name]['lams'])
        dict_all_data['y_ts' + '_' + name] =  np.array(dictFull[name]['y_ts'], dtype = int)
        dict_all_data['rewards' + '_' + name] =  np.array(dictFull[name]['rewards'], dtype = float)

    pickle.dump(dict_all_data, open(os.path.join(folderToSave, 'all_data.p'), "wb"))
