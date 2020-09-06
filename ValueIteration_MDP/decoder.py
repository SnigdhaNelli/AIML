import sys
import numpy as np
np.random.seed(0)

gridfile = sys.argv[1]
policyFileName = sys.argv[2]
prob = float(sys.argv[3])

mazeFile = open(gridfile).readlines()
mazeFile = [map(int, x.rstrip('\n').split(' ')) for x in mazeFile]
mazeFileFlatten = np.array(mazeFile).flatten()

start_state = -1
end_state = -1

policyFile = open(policyFileName).readlines()
policyFileActions = [int(policyFile[i].rstrip('\n').split(' ')[1]) for i in range(len(policyFile)-1)]
mazeLen = int(pow(len(policyFileActions),.5))
S = len(policyFileActions)

for i in range(len(policyFileActions)):
    if mazeFileFlatten[i] == 2:
        start_state = i
    if mazeFileFlatten[i] == 3:
        end_state = i
    if (start_state >= 0 and end_state >= 0):
        break

decodeMap = {0: 'N', 1: 'W', 2: 'E', 3: 'S'}

def findNextMaze(s, a):
    map1 = {0: s-mazeLen, 1: s-1, 2: s+1, 3: s+mazeLen}  # write according to decodeMap
    return map1[a]

def findValidMoves(s):

    valDir = [0, 0, 0, 0]


    if s == end_state or mazeFileFlatten[s] == 1:
        return valDir


    if s-mazeLen >= 0:  # North
        if mazeFileFlatten[s-mazeLen] != 1:
            valDir[0] = 1

    if s-1 >= 0 and ((s-1)/mazeLen == s/mazeLen):  # West
        if mazeFileFlatten[s-1] != 1:
            valDir[1] = 1


    if s+1 < S and ((s+1)/mazeLen == s/mazeLen):  # East
        if mazeFileFlatten[s+1] != 1:
            valDir[2] = 1

    if s+mazeLen < S:  # South
        if mazeFileFlatten[s+mazeLen] != 1:
            valDir[3] = 1

    return valDir

def findProbDistr(v, a):
    dist = [0.0, 0.0, 0.0, 0.0]
    for i in range(len(v)):
        if v[i] == 0:
            continue
        if i == a:
            dist[i] = prob + (1-prob)/sum(v)
        else:
            dist[i] = (1-prob)/sum(v)

    return dist

# print policyFileActions
next_state = start_state

steps = 0

while next_state != end_state:
    validMoves = findValidMoves(next_state)
    optimal_action = policyFileActions[next_state]
    probDistr = findProbDistr(validMoves, optimal_action)
    probable_action = np.random.choice(4, 1, p= probDistr)[0]
    print(decodeMap[probable_action]),
    next_state = findNextMaze(next_state, probable_action)
    steps += 1

# print '\n', steps
