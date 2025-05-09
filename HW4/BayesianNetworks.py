# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from functools import reduce


## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs

    return factorTable


## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)

    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i, 'probs'] = sum(a == (i + 1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j, 'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable



## Join of two factors
## Factor1, Factor2: two factor tables
##
## Should return a factor table that is the join of factor 1 and 2.
## You can assume that the join of two factors is a valid operation.
## Hint: You can look up pd.merge for mergin two factors
def joinFactors(Factor1, Factor2):
    col_val_1, col_val_2 = list(Factor1.columns.values), list(Factor2.columns.values)
    same_col = list(set(col_val_1).intersection(set(col_val_2))-set(['probs']))
    if not same_col:
        # cross join
        factorTable = pd.merge(Factor1, Factor2, how='cross')
    else:
        # inner join
        factorTable = pd.merge(Factor1, Factor2, on=same_col)
    factorTable['probs'] = factorTable['probs_x'] * factorTable['probs_y']
    factorTable = factorTable.drop(columns=['probs_x', 'probs_y'], axis=1)
    return factorTable


## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the left side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    remain_col = list(set(factorTable.columns.values)-set(['probs', hiddenVar]))
    result_factor = factorTable.groupby(remain_col).sum()
    result_factor = result_factor.reset_index()
    # remove hiddenVar column
    result_factor = result_factor.drop([hiddenVar], axis=1)
    return result_factor


## Update BayesNet for a set of evidence variables
## bayesnet: a list of factor and factor tables in dataframe format
## evidenceVars: a list of variable names in the evidence list
## evidenceVals: a list of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesnet, evidenceVars, evidenceVals):
    result_net = []
    for factor in bayesnet:
        for i in range(len(evidenceVars)):
            if evidenceVars[i] in list(factor.columns.values):
                factor = factor[factor[evidenceVars[i]] == evidenceVals[i]]
        result_net.append(factor)
    return result_net


## Run inference on a Bayesian network
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVars: a list of variable names to be marginalized
## evidenceVars: a list of variable names in the evidence list
## evidenceVals: a list of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using
## join and marginalization of the sets of variables.
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesnet, hiddenVars, evidenceVars, evidenceVals):
    # try to update bayesnet
    if evidenceVars:
        bayesnet = evidenceUpdateNet(bayesnet, evidenceVars, evidenceVals)
    bayesnet = marginalizeNetworkVariables(bayesnet, hiddenVars)
    result_factor = bayesnet[0]
    for i in range(1, len(bayesnet)):
        result_factor = joinFactors(result_factor, bayesnet[i])
    # normalize
    result_factor['probs'] = result_factor['probs'] / result_factor['probs'].sum()
    return result_factor


## you can add other functions as you wish.

## This function takes a Bayesian network, bayesNet,
## and marginalizes out a list of variables hiddenVar.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
    for var in hiddenVar:
        new_net, factors_to_marginalize = [], []
        for factor in bayesNet:
            if var in list(factor.columns.values):
                factors_to_marginalize.append(factor)
            else:
                new_net.append(factor)
        if len(factors_to_marginalize) > 0:
            factor = factors_to_marginalize[0]
            for i in range(1, len(factors_to_marginalize)):
                factor = joinFactors(factor, factors_to_marginalize[i])
            factor = marginalizeFactor(factor, var)
            new_net.append(factor)
        bayesNet = new_net
    return bayesNet
