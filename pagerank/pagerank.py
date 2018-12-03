import os
import sys
import math

import numpy as np
import pandas

# Generalized matrix operations:

def extractNodes(matrix):
    nodes = set()
    for colKey in matrix:
        nodes.add(colKey)
    for rowKey in matrix.T:
        nodes.add(rowKey)
    return nodes

def makeSquare(matrix, keys, default=0.0):
    matrix = matrix.copy()
    
    def insertMissingColumns(matrix):
        for key in keys:
            if not key in matrix:
                matrix[key] = pandas.Series(default, index=matrix.index)
        return matrix

    matrix = insertMissingColumns(matrix) # insert missing columns
    matrix = insertMissingColumns(matrix.T).T # insert missing rows

    return matrix.fillna(default)

def ensureRowsPositive(matrix):
    matrix = matrix.T
    for colKey in matrix:
        if matrix[colKey].sum() == 0.0:
            matrix[colKey] = pandas.Series(np.ones(len(matrix[colKey])), index=matrix.index)
    return matrix.T

def normalizeRows(matrix):
    return matrix.div(matrix.sum(axis=1), axis=0)

def euclideanNorm(series):
    return math.sqrt(series.dot(series))

# PageRank specific functionality:

def startState(nodes):
    if len(nodes) == 0: raise ValueError("There must be at least one node.")
    startProb = 1.0 / float(len(nodes))
    return pandas.Series({node : startProb for node in nodes})

def integrateRandomSurfer(nodes, transitionProbs, rsp):
    alpha = 1.0 / float(len(nodes)) * rsp
    return transitionProbs.copy().multiply(1.0 - rsp) + alpha

def powerIteration(transitionWeights, rsp=0.15, epsilon=0.00001, maxIterations=1000):
    # Clerical work:
    transitionWeights = pandas.DataFrame(transitionWeights)
    # print(transitionWeights.head())
    nodes = extractNodes(transitionWeights)
    transitionWeights = makeSquare(transitionWeights, nodes, default=0.0)
    # print(transitionWeights.head())
    transitionWeights = ensureRowsPositive(transitionWeights)
    print(transitionWeights.head())
    # Setup:
    state = startState(nodes)
    transitionProbs = normalizeRows(transitionWeights)
    print(transitionProbs)
    transitionProbs = integrateRandomSurfer(nodes, transitionProbs, rsp)
    print(transitionProbs)
    # Power iteration:
    for iteration in range(maxIterations):
        oldState = state.copy()
        state = state.dot(transitionProbs)
        delta = state - oldState
        if euclideanNorm(delta) < epsilon:
            break

    return state

def pagerank_caitien(transitionWeights, rsp=0.15, epsilon=0.00001, maxIterations=1000,alpha=0.85):
    transitionWeights = pandas.DataFrame(transitionWeights)
    # print(transitionWeights.head())
    nodes = extractNodes(transitionWeights)
    nodes = list(nodes)
    transitionWeights = makeSquare(transitionWeights, nodes, default=0.0)
    # print(transitionWeights.head())
    # transitionWeights = ensureRowsPositive(transitionWeights)
    print(transitionWeights.head())
    # Setup:
    transitionProbs = normalizeRows(transitionWeights)
    transitionProbs =  transitionWeights.div(transitionWeights.sum(axis=1), axis=0)
    print(transitionProbs)
    # transitionProbs = integrateRandomSurfer(nodes, transitionProbs, rsp)
    print(transitionProbs)
    state = startState(nodes)
    n = len(nodes)
    sink = np.array(transitionWeights.sum(1))==0
    print(sink)
    for iteracter in range(maxIterations):
        oldState = state.copy()
        # state[0] = 2
        for i in range(n):
            Ii = np.array(transitionProbs[nodes[i]])
            # account for sink states
            Si = sink / float(n)
            # account for teleportation to state i
            Ti = np.ones(n) / float(n)
            state[i] = oldState.dot(Ii*alpha + Si*alpha + Ti*(1-alpha))
        # print(state.get_values())
        delta = state - oldState
        if euclideanNorm(delta) < epsilon:
            break
        # break
    print(iteracter)
    return state

from scipy.sparse import csc_matrix

def pageRank(G, s = .85, maxerr = .001):
    """
    Computes the pagerank for each of the n states.
    Used in webpage ranking and text summarization using unweighted
    or weighted transitions respectively.
    Args
    ----------
    G: matrix representing state transitions
       Gij can be a boolean or non negative real number representing the
       transition weight from state i to j.
    Kwargs
    ----------
    s: probability of following a transition. 1-s probability of teleporting
       to another state. Defaults to 0.85
    maxerr: if the sum of pageranks between iterations is bellow this we will
            have converged. Defaults to 0.001
    """
    n = G.shape[0]
    print(G)
    # transform G into markov matrix M
    M = csc_matrix(G,dtype=np.float)
    rsums = np.array(M.sum(1))[:,0]
    ri, ci = M.nonzero()
    M.data /= rsums[ri]
    print(M.toarray())
    # bool array of sink states
    sink = rsums==0
    print(sink)
    # Compute pagerank r until we converge
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r-ro)) > maxerr:
        ro = r.copy()
        # calculate each pagerank at a time
        for i in range(0,n):
            # inlinks of state i
            Ii = np.array(M[:,i].todense())[:,0]
            # account for sink states
            Si = sink / float(n)
            # account for teleportation to state i
            Ti = np.ones(n) / float(n)

            r[i] = ro.dot( Ii*s + Si*s + Ti*(1-s) )

    # return normalized pagerank
    return r/sum(r)




def test(transitionWeights):
    # Example extracted from 'Introduction to Information Retrieval'
    # G = np.array([[0,0,1,0,0,0,0],
    #               [0,1,1,0,0,0,0],
    #               [1,0,1,1,0,0,0],
    #               [0,0,0,1,1,0,0],
    #               [0,0,0,0,0,0,1],
    #               [0,0,0,0,0,1,1],
    #               [0,0,0,1,1,0,1]])
    transitionWeights = pandas.DataFrame(transitionWeights)
    # print(transitionWeights.head())
    nodes = extractNodes(transitionWeights)
    nodes = list(nodes)
    transitionWeights = makeSquare(transitionWeights, nodes, default=0.0)
    G = np.array(transitionWeights)
    print(G)
    print(pageRank(G,s=.85))

def convert_matrix(m):
    n = len(m)
    return {str(i):{str(j):m[j][i] for j in range(n)} for i in range(n)}
if __name__ == '__main__':
    weights = {
        'a':{'b':1,'c':1,'d':1,'e':2},
        'b':{'a':2},
        'c':{'d':1},
        'd':{'a':2}
    }
    G = np.array([[0,0,1,0,0,0,0],
                  [0,1,1,0,0,0,0],
                  [1,0,1,1,0,0,0],
                  [0,0,0,1,1,0,0],
                  [0,0,0,0,0,0,1],
                  [0,0,0,0,0,1,1],
                  [0,0,0,1,1,0,1]])
    weights = convert_matrix(G)
    state = powerIteration(weights)
    print(state)
    state = pagerank_caitien(weights)
    print(state)
    state = pageRank(G)
    print(state)
