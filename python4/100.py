import numpy as np


#A[0, 0]  A[0, 1]  A[0, 2]  A[0, 3]
#A[1, 0]  A[1, 1]  A[1, 2]  A[1, 3]
#A[2, 0]  A[2, 1]  A[2, 2]  A[2, 3]
#A[3, 0]  A[3, 1]  A[3, 2]  A[3, 3]

#sayı değeri

#A[n, m]

#tüm satır
#A[n]


#sütünlar(o dan başlıyor)

#A[:, m]


#u @ v



def isSingular(A):
    B = np.array(A, dtype=np.float_)
    try:
        fixRowZero(B)
        fixRowOne(B)
        fixRowTwo(B)
        fixRowThree(B)
    except MatrixIsSingular:
        return True
    return False

class MatrixIsSingular(Exception): pass


def fixRowZero(A):
    if A[0, 0] == 0:
        A[0] = A[0] + A[1]
    if A[0, 0] == 0:
        A[0] = A[0] + A[2]
    if A[0, 0] == 0:
        A[0] = A[0] + A[3]
    if A[0, 0] == 0:
        raise MatrixIsSingular()
    A[0] = A[0] / A[0, 0]
    return A

def fixRowOne(A):
    A[1] = A[1] - A[1, 0] * A[0]
    if A[1, 1] == 0:
        A[1] = A[1] + A[2]
        A[1] = A[1] - A[1, 0] * A[0]
    if A[1, 1] == 0:
        A[1] = A[1] + A[3]
        A[1] = A[1] - A[1, 0] * A[0]
    if A[1, 1] == 0:
        raise MatrixIsSingular()
    A[1] = A[1] / A[1, 1]
    return A


def fixRowTwo(A):
    A[2] = A[2] - A[2, 0] * A[0]
    A[2] = A[2] - A[2, 1] * A[1]

    if A[2, 2] == 0:
        A[2] = A[2] + A[3]
        A[2] = A[2] - A[2, 0] * A[0]
        A[2] = A[2] - A[2, 1] * A[1]

    if A[2, 2] == 0:
        raise MatrixIsSingular()
    A[2] = A[2] / A[2, 2]
    return A


def fixRowThree(A):
    A[3] = A[3] - A[3, 0] * A[0]
    A[3] = A[3] - A[3, 1] * A[1]
    A[3] = A[3] - A[3, 2] * A[2]
    if A[3, 3] == 0:
        raise MatrixIsSingular()
    A[3] = A[3] / A[3, 3]
    return A


import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14


def gsBasis4(A):
    B = np.array(A, dtype=np.float_)
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]
    if la.norm(B[:, 1]) > verySmallNumber:
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else:
        B[:, 1] = np.zeros_like(B[:, 1])


    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1]

    if la.norm(B[:, 2]) > verySmallNumber:
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else:
        B[:, 2] = np.zeros_like(B[:, 2])


    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2]


    if la.norm(B[:, 3]) > verySmallNumber:
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else:
        B[:, 3] = np.zeros_like(B[:, 3])

    return B


def gsBasis(A):
    B = np.array(A, dtype=np.float_)
    for i in range(B.shape[1]):
        for j in range(i):
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]
        if la.norm(B[:, i]) > verySmallNumber:
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else:
            B[:, i] = np.zeros_like(B[:, i])

    return B


# This function uses the Gram-schmidt process to calculate the dimension
def dimensions(A):
    return np.sum(la.norm(gsBasis(A), axis=0))

import numpy as np
from numpy.linalg import norm, inv
from numpy import transpose
from readonly.bearNecessities import *


def build_reflection_matrix(bearBasis) :
    E = gsBasis(bearBasis)
    TE = np.array([[1, 0],
                   [0, -1]])
    T = E@TE@transpose(E)
    return T


%matplotlib inline
import matplotlib.pyplot as plt

bearBasis = np.array(
    [[1,   -1],
     [1.5, 2]])
T = build_reflection_matrix(bearBasis)

reflected_bear_white_fur = T @ bear_white_fur
reflected_bear_black_fur = T @ bear_black_fur
reflected_bear_face = T @ bear_face

ax = draw_mirror(bearBasis)

ax.fill(bear_white_fur[0], bear_white_fur[1], color=bear_white, zorder=1)
ax.fill(bear_black_fur[0], bear_black_fur[1], color=bear_black, zorder=2)
ax.plot(bear_face[0], bear_face[1], color=bear_white, zorder=3)

ax.fill(reflected_bear_white_fur[0], reflected_bear_white_fur[1], color=bear_white, zorder=1)
ax.fill(reflected_bear_black_fur[0], reflected_bear_black_fur[1], color=bear_black, zorder=2)
ax.plot(reflected_bear_face[0], reflected_bear_face[1], color=bear_white, zorder=3);