# Import arima.py here
import numpy as np
import torch
import random
import math
from arima import Arima

import configs
configs.init()
'''
INFO:
    a) creates time series for AR orders from armin to armax and MA orders from mamin to mamax.
    b) roots of AR / MA polynomials are checked using eigenvalue methods, i.e. finding the eigenvalues of companion matrix. See Numerical Recipies in C - The Art of Scientific Computing, 2nd Edition, Chapter 9.5 Roots of Polynomials, page 375 Eigenvalue methods. See function 'checkCompanionMtrx'.
    c) Time series generated are stationary and invertible. The length of smallest root should be outside of unit circle. The boundary is set as an arbitrary small number 1.001, for numerical stability.
    d) Coefficients are selected based on algorithm by Beadle, E.R. and Djuric, P.M. in "Uniform random parameter generation of stable minimum-phase real ARMA(p,q) process". If the stationary/invertible condition is not statisfied, the process is repeated until stationary/invertible condition is satisfied.
    e) use multithreading to generate time series to speed up the process.
'''


class Dataset:
    def __init__(self):
        self.id = 0
        self.isregenerating = False
        self.armin = 0
        self.armax = 9
        self.mamin = 0
        self.mamax = 9
        self.lenmin = 1000
        self.lenmax = 1000
        self.batchsize = 100  # Initially 50. Set to 600 to reduce fluctuations
        self.nthreads = 7
        self._categories = None
        self._nexamples = None
        self._order = None
        self._size = None
        self._remainder = None
        self._pool = None
        self.all_data = {}

        # isARmodel to be defined in other scripts (Remember to edit this line)
        self.isARmodel = True

    def size(self):
        return self._size

    def checkCompanionMtrx(self, coefs):
        '''
        Using Numpy, check roots of AR and MA polynomials
        Input: Numpy array 
        Output: Boolean, min root
        '''
        # Add constant term (1) before finding roots
        coefs = np.insert(coefs, len(coefs), 1)
        output = False
        roots = np.roots(coefs)

        minroot = np.abs(roots).min()

        if minroot > 1.001:
            output = True

        return output, minroot

    # Functions to draw coefficients uniformly
    # Based on paper Uniform Random Parameter Generation of Stable Minimum-Phase Real ARMA(p,q) Processes by Beadle, E.R. and Djuric, P.M, IEEE Signal Processing Letters, 4(9), 259-261.

    def J(self, x, k):
        '''
        Domain of J, i.e. x, is in [-1,1]. Upon simplifying equation 12.
        '''
        if k % 2 == 0:
            output = (x**2 - 1) ** (k/2 - 1) * (x + 1)
        else:
            output = (x**2 - 1) ** ((k - 1)/2)
        return np.abs(output)

    def JArea(self, x, k):
        '''
        Function to integrate
        '''
        if k % 2 == 0:
            n = k/2 - 1
        else:
            n = (k - 1)/2

        return math.sqrt(math.pi) * math.exp(math.lgamma(1+n) - math.lgamma(1.5+n))

    def genNumJ(self, k):
        '''
        generate random number from J(x) using rejection-sampling method.
        choose g(x) to be Uniform[-1,1]. Thus g(x) = 1/2.
        notice that J(x) <= to 1. Thus, J(x) / g(x) <= 2. We can choose M = 2 for example
        if u <= J(x) / Mg(x) then accept x else repeat Step 1 - 3
        The rejection method would be:
            Step 1 - generate x from g(x)
            Step 2 - generate u from Uniform[0,1]
            Step 3 - if u <= J(x) / Mg(x), which is equal to J(x).
        Increasing M improves the sampling closer to J(x).
        '''
        output = 0
        totalArea = self.JArea(1, k)
        Mg = 2 * 0.5
        while True:
            u = random.uniform(0, 1)
            xtemp = random.uniform(-1, 1)  # Uniform(-1, 1)
            if u * Mg <= (self.J(xtemp, k) / totalArea):
                output = xtemp
                break
        return output

    def generateUnifCoefs(self, k):
        '''
        Generate coefficients. See synthesis procedure and equation 2.
        Given order k >= 1, returns coefficients.
        This is going to be the reverse, i.e. coefficient of n-th order is stored in 1st entry of coefs.
        '''
        tempcoefs = np.zeros(k)
        tempcoefs[0] = self.genNumJ(1)
        prevVals = np.copy(tempcoefs)

        if k >= 2:
            for i in range(1, k):
                tempcoefs[i] = self.genNumJ(i+1)
                for j in range(i):
                    tempcoefs[j] = prevVals[j] + tempcoefs[i] * prevVals[i-j-1]

                prevVals = np.copy(tempcoefs)

        return prevVals[::-1]

    def randcoef(self, ord_ar, ord_ma):
        '''
        Generated random coefficients
        '''
        # Generate AR coefficients
        minrootAR = 0
        if ord_ar > 0:
            rootcheck = False
            while not rootcheck:
                nAR = self.generateUnifCoefs(ord_ar)
                rootcheck, minrootAR = self.checkCompanionMtrx(nAR)

            AR = -1 * nAR
        else:
            AR = []

        # Generate MA coefficients
        minrootMA = 0
        if ord_ma > 0:
            rootcheck = False
            while not rootcheck:
                nMA = self.generateUnifCoefs(ord_ma)
                rootcheck, minrootMA = self.checkCompanionMtrx(nMA)

            MA = nMA
        else:
            MA = []
        return {'ar': AR, 'ma': MA, 'minroot': minrootAR}

    def init(self):
        if self.isARmodel:
            self._categories = self.armax - self.armin + 1
        else:
            self._categories = self.mamax - self.mamin + 1

    def generate(self):
        '''
        Generate 600 time series examples 
        Input: None
        Output: 600 time series examples (ts_input) and its corresponding labels (ts_output)
        '''
        self.isregenerating = True
        # I fixed this at 1000 (Should change this to be a Class attribute)
        ts_len = 1000
        # Time series examples
        ts_input = torch.Tensor(
            self.batchsize, 1, ts_len, device=configs.device)
        ts_output = torch.LongTensor(
            self.batchsize, 1, device=configs.device)  # Ground truth labels

        # Initialize counter for batch indexing
        count = 0

        ord_range = self.armax - self.armin + 1
        ma_range = self.mamax - self.mamin + 1

        # Divide by 100 which is the total number of permutations between P and Q
        for i in range(self.batchsize//100):

            # Double nested for-loops for AR and MA orders
            for ord_ar in range(ord_range):
                #print('Permuting ord_ar={}'.format(ord_ar))

                for ord_ma in range(ma_range):
                    coef = self.randcoef(ord_ar, ord_ma)
                    _arima = Arima()
                    if ord_ar == 0:
                        control = {'burnin': 0}
                    else:
                        control = {
                            'burnin': 5*(ord_ar+ord_ma)+min(50000, math.ceil(10/math.log(coef['minroot'])))}

                    res = _arima.simulate2(ts_len, coef, control)

                    # Normalize res
                    res_mean = torch.mean(res)
                    res_std = torch.std(res)
                    norm_res = res.sub(res_mean).div(res_std)

                    ts_input[count] = norm_res.detach().clone()

                    if self.isARmodel:
                        ts_output[count] = torch.LongTensor([ord_ar])
                    else:
                        ts_output[count] = ord_ma

                    count += 1

        self.isregenerating = False
        #print('Generated data')
        return {'input': ts_input, 'output': ts_output}
