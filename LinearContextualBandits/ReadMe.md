# Linear Contextual Bandits Experiment.

This folder contains the following files.

## createTaskFiles.py

It creates four task files that we use parallelize our work.

## utils_run.py

A general utility file containing function used to run the Linear Contextual Bandits experiment.

## linConWithCons.py

File that runs the linear contextual experiment when receiving a set of arguments using argsparser. All the different runs configuration of these file are created in createTaskFiles.py. When running all these tasks approximately 17Gb of data will be stored.

## createAggregatedData.py

linConWithCons.py creates a different folder for each run. createAggregatedData.py groups the runs that belong to the same experiment and obtain average data about the revenue, actions, and other information. The output of this file are dictionaries, one per configuration of (d,n) tried. 

## AnalyzeExperiments.ipynb

Jupyter notebook that uses the output of 'createAggregatedData.py' to obt the graphs and tables shown in the paper.

