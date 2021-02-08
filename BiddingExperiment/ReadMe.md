# Bidding Experiment

We split this experiment code into three folders. We briefly explain the goal and the final output of running the codes in each folder. As a word of caution, this experiment saves approximately 50 Gb on raw results. 
1. DataForPred. A folder containing the codes for creating the data used in this experiment. The output of the codes contained here is a folder called "DataForPred" which contains the data for the experiment (both for the conversion prediction architecture and running our methodology). 
We assume you have downloaded the dataset from https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/ and unzip it before running any code.
2. TrainNeuralNetwork. It has the network architecture we tried and all the necessary codes to train and validate it. The final output of the codes in this folder is the architecture's parameters with the highest AUC. We store the best architecture parameters in a Pickle file in the folder 'BestModel'.
3. SecondPriceExperiment.  Here we assume you have already run the codes from the previous two folders. The codes in this folder run both our methodology and the heuristic (described in detail in the supplementary materials). The final output of running the codes in this folder is Jupyter Notebook file in which we output the tables and graphs shown in the paper


Files per Folder.

1. CreateData

utils_data.py. Utilities file containing all necessary functions to run create_data.py.
create_data.py. Main file of the folder. As input, it needs the path where  'criteo_attribution_dataset.tsv.gz' is saved and the path to store the data.


2. TrainNeuralNetwork

FwFM_Network.py. File containing the Pytorch architectures tried.
Utils.py. Utilities file for this experiment.
RunCode.py. Contains the train and validation functions. 
runFwFM.py. Runs one setup of the network. 
create_tasks.py.  Creates eight files, each with 15 lines each representing an experiment setup to be run. 

3. SecondPriceExperiment

Utils.py. Utilities file used for running this experiment.
Utils_res.py. Utilities file used for analyzing the results.
runAsSecPrice.py. File to run our methodology, assuming indexes for step-size and simulation will be passed when run.
runAsGreedy. File to run the heuristic assuming indexes for $\gamma$ and simulation will be passed when run.
create_tasks_dual_method.py. It creates four files. Each file has 225 lines, each line representing a pair of step-size and simulation to run using our methodology. 
create_tasks_greedy.py. It creates ten files.  Each file has 260 lines, each line representing a pair of gamma and simulation to run using the greedy heuristic. 
