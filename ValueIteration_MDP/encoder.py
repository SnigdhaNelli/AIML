import numpy as np
import sys

filename = sys.argv[1]
prob = float(sys.argv[2])

mazeFile = open(filename).readlines()
mazeFile = [map(int, x.rstrip('\n').split(' ')) for x in mazeFile]
mazeLen = len(mazeFile)

mazeFileFlatten = np.array(mazeFile).flatten()

S = mazeLen**2
A = 4  # N0 E1 S2 W3
gamma = 1
start = -1
end = -1

for i in range(S):
    if mazeFileFlatten[i] == 2:
        start = i
    if mazeFileFlatten[i] == 3:
        end = i
    if (start >= 0 and end >= 0):
        break

print "numStates", S
print "numActions", A
print "start", start
print "end", end

def findNextMaze(s, a):
    map1 = {0: s-mazeLen, 1: s-1, 2: s+1, 3: s+mazeLen}  # write according to decodeMap
    return map1[a]

for s in range(S):
    if s == end or mazeFileFlatten[s] == 1:
        continue

    valDir = [0, 0, 0, 0]

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

    totValid = sum(valDir)

    for a in range(A):
        if valDir[a] == 0:
            continue
        print "transition", s, a, findNextMaze(s, a), -1, prob + (1-prob)/totValid

        if prob != 1:
            for j in range(A):
                if valDir[j] != 0 and j != a:
                    print "transition", s, a, findNextMaze(s, j), -1, (1-prob)/totValid
        

print "discount ", gamma
