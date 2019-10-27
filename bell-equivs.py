import numpy
from scipy.special import comb

DIMS = 4 # aka d
SUBSET_SIZE = 7 # aka k

numSubsets = int(comb(DIMS*DIMS, SUBSET_SIZE))

# TODO note that this list gets altered and used by sortSubsets and fillBucket
# so you can only call sortSubsets once per run
# not terribly elegant, i'd prefer more dilineation between whats global/local
usedList = [0] * numSubsets

# sorts sets of k qudit bell states into LELM equivalence classes
# returns a list of equivalence classes
def sortSubsets():
    # list to be filled with equivalence class buckets
    bucketList = []

    # iterate through all possible subsets
    for i in range(numSubsets):
        # subset already sorted -- move on
        if usedList[i] == 1: 
            continue
        else:
            # create and populate a new equivalence class
            newBucket=[i]
            usedList[i] = 1
            fillBucket(newBucket, newBucket)
            bucketList.append(newBucket)

    return bucketList

# helper for sortSubsets
# recursively finds all subsets generated from the 
# subsets in indicesToExplore. adds these subsets to bucket
def fillBucket(bucket, indicesToExplore):
    nextIndicesToExplore = []
    for index in indicesToExplore:
        # go through the generators for each state

        for gen in gens:
            
            newSubset = numpy.matmul(gen, indexToSubset(index))

            newIndex = subsetToIndex(newSubset)
            # if newIndex hasn't been marked, then add to bucket the same bucket and plan to explore this new state
            if usedList[newIndex]==0:
                bucket.append(newIndex)
                nextIndicesToExplore.append(newIndex)
                usedList[newIndex] = 1

    if len(nextIndicesToExplore) > 0:
        fillBucket(bucket, nextIndicesToExplore)


# makes the generators ic, ip, sc, sp
def makeGenerators():
    d = DIMS
    # column-shifting matrices
    # eg, colShift[0] shifts column 0 down by 1
    colShift = []

    for n in range(d):
        # make cn: shift everything in column n down by one
        cn = []
        # build the matrix row-by-row
        for i in range(d*d):
            # each row is all 0s except for one well-placed 1
            row = [0]*(d*d)
            if i % d == n:
                colNum = (i-d) % (d*d)
            else:
                colNum = i
            row[colNum] = 1
            cn.append(row)
        colShift.append(cn)

    # row-shifting matrices
    # eg, rowShift[0] shifts row 0 to the right by 1
    rowShift = []

    for n in range(d):
        # make rn: shift everything in row n to the right by one
        rn = []
        # build the matrix row-by-row
        for i in range(d*d):
            # each row is all 0s except for one well-placed 1
            row = [0]*(d*d)
            if i >= d*n and i < d*(n+1):
                colNum = ((i-1) % d) + int(i/d)*d
            else:
                colNum = i
            row[colNum] = 1
            rn.append(row)
        rowShift.append(rn)

    # contrust ic: shift every column down
    ic = colShift[0]
    for i in range(1,d):
        ic = numpy.matmul(ic, colShift[i])

    # contrust ip: shift every row to the right
    ip = rowShift[0]
    for i in range(1,d):
        ip = numpy.matmul(ip, rowShift[i])

    # contrust sc: shift column n down n times
    # presumes d is at least 2 because... come on, guys
    sc = colShift[1]
    for i in range(2,d):
        for j in range(i):
            sc = numpy.matmul(sc, colShift[i])

    # contrust sp: shift row n to the right n times
    # presumes d is at least 2 because... come on, guys
    sp = rowShift[1]
    for i in range(2,d):
        for j in range(i):
            sp = numpy.matmul(sp, rowShift[i])

    return [ic,ip,sc,sp]
    

# for a set of k d-dimensional bell states
# given a subset's index, 0 to (d^2 choose k)-1,
# return the corresponding subset of k bell states
def indexToSubset(i):
    n = DIMS * DIMS - 1
    k = SUBSET_SIZE - 1

    subset = [0] * (n+1)

    for pos in range(len(subset)):
        nChooseK = comb(n, k,exact=True) # avoid doing this computation twice
        if i < nChooseK:
            subset[pos] = 1
            k = k-1
        else:
            i = i-nChooseK
        n = n-1

    return subset


# given a set of k d-dimensional bell states,
# return the corresponding index from 0 to (d^2 choose k)-1
def subsetToIndex(subset):
    n = DIMS * DIMS - 1
    k = SUBSET_SIZE - 1

    i = 0

    for pos in range(len(subset)):
        if subset[pos] == 0:
            i = i + comb(n,k,exact=True)
        else:
            k = k-1
        n = n-1
    
    return i


# given a list of equivalence classes (e.g. the output of sortSubsets),
# prints a tictactoe diagram of 1 representative from each class        
def showRepresentatives(bucketList):
    print("displaying representatives from "+str(len(bucketList))+" equivalence classes")

    for i in range(len(bucketList)):
        tictactoe(indexToSubset(bucketList[i][0]))


# given a d^2-vector representing a set of bell states,
# print the tic-tac-toe diagram
# please don't give me a vector whose length isnt a perfect square :/
def tictactoe(v):
    d = int(numpy.sqrt(len(v)))
    s = ""
    for c in range(0,d):
        for p in range(0,d):
            # 1 if psi_c^p is in this set, 0 if not
            statePresent = v[c*d + p]
            if statePresent == 1:
                s += " X "
            else:
                s += " - "
        s += '\n'
    print(s)


gens = makeGenerators()
buckList = sortSubsets()
showRepresentatives(buckList)
