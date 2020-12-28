import pandas as pd
from copy import deepcopy
import pickle
from operator import itemgetter
import numpy as np
from utils_data import *

if __name__ == '__main__':
    # Change this line to the pathw where the file criteo_attribution_dataset.tsv.gz is.
    pathToGZIPWithCriteoData = '/Users/alobos/Documents/CriteoDataset/criteo_attribution_dataset/'
    print('Making a dataframe out of criteo_attribution_dataset.tsv.gz')
    df_from_Criteo = pd.read_csv(pathToGZIPWithCriteoData+'criteo_attribution_dataset.tsv.gz', sep='\t', compression='gzip')

    # DayMonth is obtained in the same way as done in the Jupyter-Notebook given by Criteo.
    df_from_Criteo['dayMonth'] = np.floor(df_from_Criteo.timestamp / 86400.).astype(int)
    df_from_Criteo['dayWeek'] = [int(day) % 7 for day in list(df_from_Criteo['dayMonth'])]

    print('df_from_Criteo.head(10)')
    print(df_from_Criteo.head(10))
    print()

    ## Columns we use for our experiment
    cat_cols_wout_day = \
        ['campaign', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9']
        # Columns that contain categorical data
    cat_cols_w_day = ['dayWeek', 'dayMonth'] + cat_cols_wout_day
    ## Columns that have either binary or real data.
    noncat_cols = ['click', 'conversion', 'cost', 'cpo']

    map_attr_to_indxs = createMapColsAttrToIndxs(df_from_Criteo, cat_cols_w_day)



    print('Using the dataframe read from Criteo, create a dataframe', end=' ')
    print('with the columns that we need and using indexes instead of categories.')
    df_all = CreateDfForDynamic(df_from_Criteo, map_attr_to_indxs, cat_cols_w_day, noncat_cols)

    del df_from_Criteo
    
    df_all['KeepData'] = np.ones(len(df_all['cat1']))

    print('df_all.columns: '+str(df_all.columns))

    ## Minimum number of times that Campaign should appear in the 
    thresToUse = 10000

    remCampLessThan = RetRemoveCampsWithLessThan(df_all[["campaign", "KeepData"]], \
        (df_all.loc[df_all['dayMonth'] <= 20])[["campaign", "KeepData"]], \
        (df_all.loc[df_all['dayMonth'] > 20])[["campaign", "KeepData"]], thres = thresToUse)

    binCheck = np.array(itemgetter(*list(df_all['campaign']))(remCampLessThan))
    df_all['KeepData'] -= binCheck

    print('Rows Before Removing Small Campaigns '+str(len(df_all['campaign'])))

    dfWoutSmallCamp = df_all[df_all['KeepData'] > 0]

    print()
    print('After '+str(len(dfWoutSmallCamp['campaign'])))

    dfWoutSmallCamp.index = range(len(dfWoutSmallCamp['campaign']))

    ## Days start at 0. Train goes from day 0 to 20
    ## Let's remove all rows in the Val and Test Sets With Attributes that do not appear in the Train Set 
    mapToRemove = createMapUnique(dfWoutSmallCamp,\
        deepcopy(dfWoutSmallCamp[dfWoutSmallCamp['dayMonth'] <=20]), cat_cols_wout_day)

    
    ## The following part is to remove rows in the Validation/Test that use attributes that do not appear in 
    ## Train. 

    rowsToRemove = np.zeros(len(dfWoutSmallCamp[dfWoutSmallCamp['dayMonth'] >20]['cat1']))
    lastRowDay20 = len(dfWoutSmallCamp[dfWoutSmallCamp['dayMonth'] <=20]['cat1'])
    print('lastRowDay20: '+str(lastRowDay20))

    for name in cat_cols_wout_day:
        binCheck = itemgetter(*list(dfWoutSmallCamp[dfWoutSmallCamp['dayMonth'] >20][name]))(mapToRemove[name])
        rowsToRemove += np.array(binCheck)

    dfWoutSmallCamp['RemoveData'] = 0.0
    lastRowDay20 = len(dfWoutSmallCamp[dfWoutSmallCamp['dayMonth'] <=20]['dayMonth'])
    print('lastRowDay20: '+str(lastRowDay20))
    dfWoutSmallCamp.loc[lastRowDay20:,'RemoveData'] += rowsToRemove

    df_final = dfWoutSmallCamp[dfWoutSmallCamp['RemoveData'] <= 0.0]
    df_final.index =  range(len(df_final['campaign']))
    del dfWoutSmallCamp

    ## Let's split the dataframe in Train (dayMonth <= 20), Validation (dayMonth = 21), 
    ## Test (dayMonth >= 22)

    origRowsInTrain = len((df_all[df_all['dayMonth'] <= 20])['dayWeek'])
    origRowsInVal = len((df_all[(df_all['dayMonth'] > 20) & (df_all['dayMonth']  <= 22) ])['dayWeek'])
    origRowsInTest = len((df_all[df_all['dayMonth'] >= 23])['dayWeek'])

    dfFirstThreeWeeks = df_final[df_final['dayMonth'] <= 20]
    dfNextTwoDays = df_final[(df_final['dayMonth'] > 20) & (df_final['dayMonth'] <= 22)]
    dfLastWeek = df_final[df_final['dayMonth'] >= 23]

    rowsInTrain = len(dfFirstThreeWeeks['dayWeek'])
    rowsInVal = len(dfNextTwoDays['dayWeek'])
    rowsInTest = len(dfLastWeek['dayWeek'])

    # Print some statistics
    print('Original Rows In Train '+str(origRowsInTrain)+', After Cleaning '+str(rowsInTrain))
    print('Original Rows In Val '+str(origRowsInVal)+', After Cleaning '+str(rowsInVal))
    print('Original Rows In Test '+str(origRowsInTest)+', After Cleaning '+str(rowsInTest))
    print('Original Rows ' +str(len(df_all['dayWeek']))+', After Cleaning '+str(len(df_final['dayWeek'])))

    print('Different Attributes Per Column')
    for colName in df_all.columns:
        print(str(colName)+': Original '+str(len(df_all[colName].unique()))+', Now '+str(len(df_final[colName].unique())))

    for name in cat_cols_wout_day:
        if len(list(set(dfNextTwoDays[name]) - set(dfFirstThreeWeeks[name].unique())))>0:
            print('Problem With Column '+str(name)+' in Validation Dataset')
        if len(list(set(dfLastWeek[name]) - set(dfFirstThreeWeeks[name].unique())))>0:
            print('Problem With Column '+str(name)+' in Test Dataset')
    
    del df_all

    ## Change the dataset to use consecutive numbers (this is for the model trained in Pytorch)

    mapFeatInv = turnMapAround(map_attr_to_indxs)

    oldToNewAux, mapFeatInvFinal, df_Train, df_Val, df_Test =  ChangeAttrNumbers(df_final, dfFirstThreeWeeks,\
    dfNextTwoDays, dfLastWeek, mapFeatInv, cat_cols_wout_day)

    del dfFirstThreeWeeks
    del dfNextTwoDays
    del dfLastWeek

    ## Save the data

    pathToSave = '/Users/alobos/Documents/PhD/ICML2021_submission_code/BiddingExperiment/CreateData/DataForPred/'
    if not os.path.exists(pathToSave):
        os.makedirs(pathToSave)

    pickle.dump(df_Train[['dayWeek', 'campaign', 'cat1', 'cat2', 'cat3', 'cat4',\
        'cat5', 'cat6', 'cat7','cat8', 'cat9']].to_numpy(), open(pathToSave+'X_Train.p', "wb"))
    pickle.dump(df_Val[['dayWeek', 'campaign', 'cat1', 'cat2', 'cat3', 'cat4',\
        'cat5', 'cat6', 'cat7', 'cat8', 'cat9']].to_numpy(), open(pathToSave+'X_Val.p', "wb"))
    pickle.dump(df_Test[['dayWeek', 'campaign', 'cat1', 'cat2', 'cat3', 'cat4',\
        'cat5', 'cat6', 'cat7', 'cat8', 'cat9']].to_numpy(), open(pathToSave+'X_Test.p', "wb"))
    pickle.dump(np.squeeze(df_Train[['conversion']].to_numpy()), open(pathToSave+'Y_Train.p', "wb"))
    pickle.dump(np.squeeze(df_Val[['conversion']].to_numpy()), open(pathToSave+'Y_Val.p', "wb"))
    pickle.dump(np.squeeze(df_Test[['conversion']].to_numpy()), open(pathToSave+'Y_Test.p', "wb"))
    pickle.dump(df_Train[['conversion', 'cost', 'cpo', 'dayMonth']].to_numpy(), open(pathToSave+'Extra_Train.p', "wb"))
    pickle.dump(df_Val[['conversion', 'cost', 'cpo', 'dayMonth']].to_numpy(), open(pathToSave+'Extra_Val.p', "wb"))
    pickle.dump(df_Test[['conversion', 'cost', 'cpo', 'dayMonth']].to_numpy(), open(pathToSave+'Extra_Test.p', "wb"))
    pickle.dump(mapFeatInvFinal, open(pathToSave + 'mapAllAttr.p', "wb"))