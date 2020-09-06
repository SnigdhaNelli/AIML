import numpy as np
import sys

def valIteration():  # TASK 1

    epsilon = 1e-16
    large_negative = -1e+40
    # filename = "data/mdp/mdpfile01.txt"
    filename = sys.argv[1]

    mdpFile = open(filename).readlines()
    tot_lines = len(mdpFile)
    transition_lines = tot_lines - 5

    arr = []
    for i in range(3):
        arr.append(int(mdpFile[i].rstrip('\n').split(' ')[1]))

    numStates, numActions, start = arr
    discount = float(mdpFile[tot_lines-1].rstrip('\n').split(' ')[2])
    end_list = map(int, mdpFile[3].rstrip('\n').split(' ')[1:])

    def stopping_criterion(v1, v2):
        ans = True
        for i in range(len(v1)):
            if abs(v1[i] - v2[i]) > epsilon:
                ans = False
                break
        return ans

    def findNext(valIter, valOrAction):
        valIterTemp = np.zeros((numStates, numActions))
        valIterTempNeg = np.full((numStates, numActions), large_negative)    
        
        for i in range(transition_lines):
            transition = mdpFile[i+4].rstrip('\n').split(' ')      
            valIterTemp[int(transition[1])][int(transition[2])] += float(transition[5]) * (discount * valIter[int(transition[3])] + float(transition[4]))
            valIterTempNeg[int(transition[1])][int(transition[2])] = valIterTemp[int(transition[1])][int(transition[2])]

        if valOrAction == 0:
            return np.max(valIterTempNeg, 1)  ## valIterNext

        return np.argmax(valIterTempNeg, 1) ## optimal action

    def valueIterFunction():
        valIter = np.zeros(numStates)
        t = 0
        while True:
            valIterNext = findNext(valIter, 0)

            for i in range(numStates):
                if valIterNext[i] == large_negative:
                    valIterNext[i] = 0.0
            

            if stopping_criterion(valIter, valIterNext):
                return valIterNext, t+1

            valIter = valIterNext
            t += 1

        return valIter, t

    def optimal_policy(v):
        return findNext(v, 1)



    V, t = valueIterFunction()
    A = optimal_policy(V)
    for i in range(numStates):
        if i in end_list:
            A[i] = -1
        print V[i], A[i]
    print "iterations", t

valIteration()