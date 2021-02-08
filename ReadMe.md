# Joint Online Learning and Decision Making via Dual Mirror Descent.

This GitHub contains the code from the paper "Joint Online Learning and Decision Making via Dual Mirror Descent" submitted to ICML 2021. Each experiment is contained in its own folder.

We run most of the code for both experiments using a Linux cluster we had access to (details omitted for anonymity). We parallelize the code by experiment settings and simulations tried. The files we use to create the different runs are included in each folder. The Linear Contextual Bandit experiment creates the does not need  a data creation step before running the experiment, while the bidding experiment does. A pre-requisite for the bidding experiment is to download the dataset from https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/ and unzip it. The Linear Contextual Bandits stores aproximately 5.7 Gb of raw data while the bidding experiment stores approximately 50 Gb. ~

In terms of code/libraries used:
1. Python >= 3.6
2. Pytorch 
3. Numpy/Pandas/Seaborn/Matplotlib
4. Sklearn
5. Jupyter Notebook for analzing results (only this step uses Jupyter Notebooks).
