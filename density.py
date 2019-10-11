import numpy
import math


##    sample vectors written in std basis
#     00 01 02 03 10 11 12 13 20 21 22 23 30 31 32 33
v1 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
v2 = [0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 0, 0, 0, 0,-1, 0]
v3 = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
v4 = [0, 0, 0, 1,-1j,0, 0, 0, 0,-1, 0, 0, 0, 0,1j, 0]

##     hyperentangled stuff written in bell basis (top bottom)
#      00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
h1  = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
h2  = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
h3  = [0,1-1j,0,1+1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
h4  = [0,1+1j,0,1-1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
h5  = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,-1, 0]
h6  = [0, 0, 0, 0,-1, 0,-1, 0, 0, 0, 0, 0, 1, 0,-1, 0] 
h7  = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,-1j, 0,1j]
h8  = [0, 0, 0, 0, 0,-1, 0,-1, 0, 0, 0, 0, 0,-1j, 0,1j]
h9  = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
h10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
h11 = [0, 0, 0, 0, 0, 0, 0, 0,0,-1+1j,0,-1-1j, 0, 0, 0, 0]
h12 = [0, 0, 0, 0, 0, 0, 0, 0,0,-1-1j,0,-1+1j, 0, 0, 0, 0]
h13 = [0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 1, 0, 1, 0] 
h14 = [0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0,-1, 0,-1, 0] 
h15 = [0, 0, 0, 0, 0,1j, 0,-1j,0, 0, 0, 0, 0,-1, 0,-1]
h16 = [0, 0, 0, 0, 0,1j, 0,-1j,0, 0, 0, 0, 0, 1, 0, 1]

## (the important 7) hyperentangled states in the std basis
## we can distinguish one each of hstd1, hstd2 ... hstd7
#         00 01 02 03 10 11 12 13 20 21 22 23 30 31 32 33
hstd1a = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
hstd1b = [1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1, 0, 0, 0, 0,-1]
hstd1c = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1]
hstd1d = [1, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1]
hstd2a = [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
hstd2b = [0, 0, 1, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 1, 0, 0]
hstd3a = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
hstd3b = [0, 0, 0, 1, 0, 0,-1, 0, 0,-1, 0, 0, 1, 0, 0, 0]
hstd4a = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
hstd4b = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1, 0]
hstd5a = [0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0]
hstd5b = [0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 1, 0]
hstd6a = [0, 0, 1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0,-1, 0, 0]
hstd6b = [0, 0, 1, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 1, 0, 0]
hstd7a = [0, 0, 0, 1, 0, 0, 1, 0, 0,-1, 0, 0,-1, 0, 0, 0]
hstd7b = [0, 0, 0, 1, 0, 0,-1, 0, 0, 1, 0, 0,-1, 0, 0, 0]

hyperSets = [[hstd1a,hstd1b,hstd1c,hstd1d],
            [hstd2a,hstd2b],[hstd3a,hstd3b],
            [hstd4a,hstd4b],[hstd5a,hstd5b],
            [hstd6a,hstd6b],[hstd7a,hstd7b]]

## qu4it bell states in std basis
#       00 01 02 03 10 11 12 13 20 21 22 23 30 31 32 33
## c p
psi00 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
psi01 = [1, 0, 0, 0, 0,1j, 0, 0, 0, 0,-1, 0, 0, 0,0,-1j]
psi02 = [1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1, 0, 0, 0, 0,-1]
psi03 = [1, 0, 0, 0, 0,-1j,0, 0, 0, 0,-1, 0, 0, 0, 0,1j]
psi10 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]
psi11 = [0, 1, 0, 0, 0, 0,1j, 0, 0, 0, 0,-1,-1j,0, 0, 0]
psi12 = [0, 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0]
psi13 = [0, 1, 0, 0, 0, 0,-1j,0, 0, 0, 0,-1,1j, 0, 0, 0]
psi20 = [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
psi21 = [0, 0, 1, 0, 0, 0, 0,1j,-1, 0, 0, 0, 0,-1j,0, 0]
psi22 = [0, 0, 1, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0,-1, 0, 0]
psi23 = [0, 0, 1, 0, 0, 0,0,-1j,-1, 0, 0, 0, 0,1j, 0, 0]
psi30 = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
psi31 = [0, 0, 0, 1,1j, 0, 0, 0, 0,-1, 0, 0, 0, 0,-1j,0]
psi32 = [0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 0, 0, 0, 0,-1, 0]
psi33 = [0, 0, 0, 1,-1j,0, 0, 0, 0,-1, 0, 0, 0, 0,1j, 0]

bellStates = [psi00,psi01,psi02,psi03,
              psi10,psi11,psi12,psi13,
              psi20,psi21,psi22,psi23,
              psi30,psi31,psi32,psi33]

##################### test a hyperentangled -> bell transformation #####################

lynnTransform1 =   [[1,1,0,0],   # 0 -> 0 + 1
                   [1,-1,0,0],   # 1 -> 0 - 1
                   [0,0,1,1],    # 2 -> 2 + 3
                   [0,0,1,-1]]   # 3 -> 2 - 3

lynnTransform2 =   [[1,0,1,0],   # 0 -> 0 + 2
                   [0,1,0,1],    # 1 -> 1 + 3
                   [1,0,-1,0],   # 2 -> 0 - 2
                   [0,1,0,-1]]   # 3 -> 1 - 3

tommyTransform1 =  [[1,1,1,1],      # 0 -> 0 +  1 +  2 +  3
                   [1,1j,-1,-1j],   # 1 -> 0 + i1 -  2 - i3
                   [1,-1,1,-1],     # 2 -> 0 -  1 +  2 -  3
                   [1,-1j,-1,1j]]   # 3 -> 0 - i1 -  2 + i3

tommyTransform2 =  [[1,1,1,1],      # 0 -> 0 + 1 + 2 + 3
                   [1,1,-1,-1],     # 1 -> 0 + 1 - 2 - 3
                   [1,-1,-1,1],     # 2 -> 0 - 1 - 2 + 3
                   [1,-1,1,-1]]     # 3 -> 0 - 1 + 2 - 3

identity =         [[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]]

w  = -1/2 + numpy.sqrt(3)/2 * (1j)
w2 = -1/2 - numpy.sqrt(3)/2 * (1j)
d3unity =          [[1,1,1,0],
                    [1,w,w2,0],
                    [1,w2,w,0],
                    [0,0,0,1]]

transes = [lynnTransform1,lynnTransform2,tommyTransform1,tommyTransform2,identity,d3unity]

def testTransformsCrazy():
    transL =  [[-1,-1,-1,-1],
                [-1,-1,-1,-1],
                [-1,-1,-1,-1],
                [-1,-1,-1,-1]]
    transR =  [[-1,-1,-1,-1],
                [-1,-1,-1,-1],
                [-1,-1,-1,-1],
                [-1,-1,-1,-1]]
    
    NUM_TRANSFORMS = 43046721 # 3^16
    for i in range(200):
        for j in range(200):
            transResult = testTransformHelper(transL,transR)
             # if you found more than 3 bell states, lmk
            if transResult[0] > 3: print(transResult[1])
            transR = incTransform(transR)
        transL = incTransform(transL)


    

# a transform is like a number. increment moves us to the next one
def incTransform(trans):
    # initially, we're carrying a 1 bc we need to add 1
    carry = 1
    # positions are labelled 0-15, top left to lower right
    pos = 15
    while(carry != 0):
        row = int(pos/4)
        col = pos % 4
        if (trans[row][col] == -1):
            trans[row][col] = 0
            carry = 0
        elif (trans[row][col] == 0): 
            trans[row][col] = 1
            carry = 0
        elif (trans[row][col] == 1):
            trans[row][col] = -1
            carry = 1
        pos = pos-1
    return trans

def testTransforms():
    # test EVERY PAIR OF TRANSFORMS
    for transL in transes:
        for transR in transes:
            transResult = testTransformHelper(transL,transR)
            # if you found more than 3 bell states, lmk
            if transResult[0] > 3: print(transResult[1])

def testTransform(transL,transR):
    transResult = testTransformHelper(transL,transR)
    print(transResult[1])

# helper fxn for all the testTransform fxns
# returns (how many bell states there are, string showing results of transform)
def testTransformHelper(transL,transR):
        allStateStr = ""  # uhh sorry about the bad variable names...
        bellNum = 0
        # on EVERY HYPERENTANGLED STATE
        for set in hyperSets:
            for hyp in set:
                trans = transform(hyp,transL,transR)
                oneStateStr = prettyd4(trans)
                for bell in bellStates:
                    if bell == cleanCoeffs(trans): 
                        oneStateStr += "\thooray, bell state!"
                        bellNum += 1
                allStateStr += oneStateStr+"\n"
            allStateStr += "--------------------------------------------------------\n"
        return (bellNum, allStateStr)

def transform(hyp,leftKetTrans=lynnTransform1,rightKetTrans=lynnTransform1):
    # tells me how each single-particle ket transforms: 
    #leftKetTrans = lynnTransform
    #rightKetTrans = lynnTransform

    sum = [0] * 16

    # go through all the two-particles kets in hyp...
    for i in range(len(hyp)):
        if hyp[i] == 0: continue
        left = (int) (i/4)  # state of the left particle
        right = i % 4  # state of the right particle

        # transform the kets, and take their product
        prod = tensorProd(leftKetTrans[left],rightKetTrans[right])

        # add this to our final state
        for j in range(len(sum)):
            sum[j] += hyp[i] * prod[j]
    
    return sum

def tensorProd(left,right):
    ''' takes 2 single-particle states (BOTH vectors of len d) <-- pls don't fight me on this
        returns their product (vector of len d^2)
        ex: left:  |0> - |1> = [1,-1,0]
            right: |2>       = [0,0,1]
            out:   |02> - |12> = [0,0,1,0,0,-1,0,0,0]  '''
    d = len(left)
    prod = []
    
    for i in range(d):
        for j in range(d):
            prod.append(left[i]*right[j])
    
    return prod




##################### hyperentangled -> qu4it ############################

phip = [[1,0,0],[1,1,1]]
phim = [[1,0,0],[-1,1,1]]
psip = [[1,1,0],[1,0,1]]
psim = [[1,1,0],[-1,0,1]]

def bellCross(b1,b2):
    ''' converts a cross product of two qubit bells states to the corresponding superposition of qu4it bell states'''
    ''' b1, b2 are bell states, represented as [[coefficient, L, R],[coefficient, L, R]] (see above) '''
    prod = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # foil it out
    for ket1 in b1:
        for ket2 in b2:
            newCoeff = ket1[0] * ket2[0]
            newL = 2*ket1[1] + ket2[1]
            newR = 2*ket1[2] + ket2[2]
            # so the ket we've arrived at, in the d=4 std basis, is |newL, newR>
            # now we just add this ket to the product
            pos = 4*newL + newR
            prod[pos] = prod[pos]+newCoeff
    return stdToBell(prod)

##################### basis conversion ############################

M = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,-1, 0, 1, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0,1+1j, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0,1-1j, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0,-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
     

# takes you from the bell basis to the std basis
# this is also the conjugate transpose of Pbellstd
# columns are bell states written in std basis
# should be halved!
Pstdbell = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1,1j,-1,-1j],
            [1,1j,-1,-1j,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1,1j,-1,-1j,0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1,1j,-1,-1j,0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 1,-1],
            [1,-1, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1,-1j,-1,1j,0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1,-1j,-1,1j,0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1,-1j,-1,1j],
            [1,-1j,-1,1j,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


def bellToStd(b):
    ''' converts a bell basis vector to the standard basis '''
    return numpy.matmul(Pstdbell,b)

def stdToBell(s):
    ''' converts a std basis vector to the bell basis '''
    v = []
    for n in range(len(Pstdbell)):
        sum = 0
        for rowNum in range(len(Pstdbell)):
            sum += numpy.conjugate(Pstdbell[rowNum][n]) * s[rowNum]
        v += [sum]
    return v

##################### density matrix ############################

def trL(v):
    ''' gives the left trace of the density matrix corresponding to the state v
        v should be in the d=3 standard basis
        returns as a list containing the rows as lists'''
    ''' this is how you check if its fully entangled '''
    mat = []
    for r in range(4):
        row = []
        for c in range(4):
            el = 0
            for k in range(4):
                el += v[4*k+r] * numpy.conjugate(v[4*k+c])
            row += [el]
        mat += [row]
    return mat

################## prettying up outputs #########################

def prettyMatrix(m):
    ''' makes a string of the matrix m with complex entries, but pretty '''
    ''' this is useful for when ur using the trL density stuff '''
    out = ""
    for row in m:
        for z in row:
            add = ""
            if (z==0): add += '0'
            elif (numpy.imag(z)==0): add += str(numpy.real(z))
            elif (numpy.real(z)==0): add += str(numpy.imag(z))
            else: add += str(z)
            while len(add) < 5:
                add += " "
            out += add
        out += '\n'
    return out

def prettyd4(v):
    ''' makes a string of v, a vector of len 16 representing a d=4 2-particle state,
        in the std basis '''
    out = ""

    # make it extra pretty by dividing out common factors
    cleanV = cleanCoeffs(v)
    for i in range(len(v)):
        if cleanV[i] == 0: continue
        right = i % 4
        left = (int) (i/4)
        coeffString = ""
        
        coeff = cleanV[i]
        if coeff == 1: coeffString = " + "
        elif coeff == -1: coeffString = " - "
        else: coeffString = " + "+str(coeff) 

        out += coeffString+"|"+str(left)+str(right)+">"
    return out

def cleanCoeffs(v):
    gcf = 0
    for i in range(len(v)):
        gcf = math.gcd(gcf, int(v[i].real))
        gcf = math.gcd(gcf, int(v[i].imag))
    if gcf == 0: return v

    out = []
    for i in range(len(v)):
        coeff = v[i].real/gcf + (v[i].imag/gcf)*1j
        out.append(coeff)
    return out



        