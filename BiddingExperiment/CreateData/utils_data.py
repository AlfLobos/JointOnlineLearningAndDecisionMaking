import pickle
import pandas as pd
import numpy as np
from operator import itemgetter
from copy import deepcopy
import os

def createMapColsAttrToIndxs(df, categoriesToSearchForUnique):
    """
    This function returns a map<column_name(string),map<atribute_name(int),index(int)>>, where the 'column_name' are
    taken from categoriesToSearchForUnique. For each column_name we searhc the unique elements on df and map each
    element to an int. 
    """
    mapToRet = {}
    for name in categoriesToSearchForUnique:
        mapToRet[name] = {}
        allAttr = list(pd.unique(df[name]))
        for i, attr in enumerate(allAttr):
            mapToRet[name][attr] = i
    return mapToRet

def CreateDfForDynamic(df, map_attr_to_indxs, listColumnsCat, listColumsBinOrReal):
    toRet = deepcopy(df[[*listColumnsCat, *listColumsBinOrReal]])
    for name in listColumnsCat:
        print(name)
        toRet.loc[:,name] = list(itemgetter(*list(toRet[name]))(map_attr_to_indxs[name]))
    return toRet


def RetRemoveCampsWithLessThan(dfFull, dfTrain, dfRest, thres = 1000):
    """
    This function returns a dictionary that has each camapign as a key, and a value of 1
    if the campaign should be removed and 0 otherwise. The criterion to keep a campaign
    is if it appears at least thres times on both dfTrain and dfRest. 
    """
    camps = dfFull['campaign'].unique()
    campsInTrain= list(dfTrain['campaign'].unique())
    campsInRest= list(dfRest['campaign'].unique())
    sumCampsTrain = dfTrain[["campaign", "KeepData"]].groupby('campaign')['KeepData'].sum()
    sumCampsRest = dfRest[["campaign", "KeepData"]].groupby('campaign')['KeepData'].sum()
    mapToRemove = {}
    for i in camps:
        if (i in campsInRest) and  (i in campsInTrain):
            if (sumCampsTrain[i] >= thres) and (sumCampsRest[i] >= thres):
                mapToRemove[i] = 0
            else:
                mapToRemove[i] = 1
        else:
            mapToRemove[i] = 1
    return mapToRemove

def createMapUnique(df, df2, colsOfInt):
    """
    This function assumes df2 to be a subset of df and returns a 
    map<column_name(string),map<attribute_number(int),{0,1}>> where each column_name is a member of colsOfInt. 
    For each column_name the map[column_name] has as keys *list(df[column_name].unique()) and as values
    a 0 if the attribute appears in df2 and 1 otherwise.
    """
    mapDif = {}
    print('Finding Differences')
    for name in colsOfInt:
        print(name, end = ', ')
        uniqElems = list(df2[name].unique())
        mapDif[name] = list(set(list(df[name].unique())) - set(uniqElems))
    mapToRet = {}
    for name in colsOfInt:
        mapToRet[name] = {}
        for num in list(df[name].unique()):
            mapToRet[name][int(num)] = 0
        for num in mapDif[name]:
            mapToRet[name][int(num)] = 1
    return mapToRet

def turnMapAround(mapFeatures):
    """
    As the name implies, this function change the change the values for the keys and viceversa.
    This is done for each keyParentLevel on a map<keyParentLevel, map<key,value>>.
    """
    toRet = {}
    for key in mapFeatures.keys():
        toRet[key] = {}
        for key2 in mapFeatures[key].keys():
            toRet[key][mapFeatures[key][key2]] = key2
    return toRet

def ChangeAttrNumbers(df, df1, df2, df3, mapFeatInv, colsOfInt):
    oldToNew = {}
    mapFeatInvFinal = {}
    df_Train =  deepcopy(df1)
    df_Val =  deepcopy(df2)
    df_Test = deepcopy(df3)
    for name in colsOfInt:
        print(name)
        mapFeatInvFinal[name] = {}
        uniqElemsInShort = df[name].unique()
        oldToNew[name] = {}
        for i, elem in enumerate(uniqElemsInShort):
            oldToNew[name][elem] = i
            mapFeatInvFinal[name][i] = mapFeatInv[name][elem]
        df_Train[name] = np.array(list(itemgetter(*list(df1[name]))(oldToNew[name])))
        df_Val[name] = np.array(list(itemgetter(*list(df2[name]))(oldToNew[name])))
        df_Test[name] = np.array(list(itemgetter(*list(df3[name]))(oldToNew[name])))
    return oldToNew, mapFeatInvFinal, df_Train, df_Val, df_Test