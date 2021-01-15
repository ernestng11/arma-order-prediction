'''
INFO:
a) Generate ARMA time series for a given size and AR and MA coefficients

ARGS:

- 'n'               : size of the time series
- 'coefs'           : table with coeffcients
  - 'ar'            : sequence of AR terms
  - 'ma'            : sequence of MA terms
- 'control'         : table with control variables
  - 'repeats'       : number of times the sequence is simulated, default is 1
  - 'burnin'        : number of steps to simulate before series is recorded; default is 1000
  - 'distribution'  : a function to generate random numbers; default is 'torch.normal'
  
RETURN:
  - if control.repeats is 1, a Tensor if dimension 'n' is returned
  - if control.repeats is >1, a Tensor of dimension 'repeats x n' is returned.
'''
import torch
import numpy as np

import configs
configs.init()


class Arima():
    def __init__(self):
        self.repeats = 1

    def simulate2(self, n, coefs, control):

        res = torch.Tensor(n, device=configs.device)
        '''
        # Check if coefs is non-empty
        if coefs['ar'] and coefs['ma']:
            next
        else:
            return
        '''

        # print(coefs)
        repeats = 1
        burnin = control['burnin']  # or 1000

        # Generate white noise from Normal Distribution
        ts_wn = torch.normal(mean=0, std=1, size=(
            self.repeats, len(coefs['ma']) + burnin + n)).unsqueeze(0)
        # Shape of ts_wn: [1, repeats=1, len(coefs['ma']) + burnin + n - 1)]

        # MA part
        if len(coefs['ma']) > 0:
            # Add 1 in front of MA sequence for constant epsilon term
            coefs['ma'] = np.insert(coefs['ma'], len(coefs['ma']), 1)
            MA = torch.Tensor(coefs['ma']).reshape(1, 1, -1)

            # Shape of MA: [1, 1, len(coefs['ma'])]
            ts_ma = torch.nn.functional.conv1d(
                input=ts_wn, weight=MA).squeeze(0)

        else:
            ts_ma = ts_wn.squeeze(0)

        # Whole time series equation
        if len(coefs['ar']) > 0:

            AR = torch.Tensor(coefs['ar']).reshape(1, 1, -1)
            # print(AR.size())
            # Shape of AR: [1, 1, len(coefs['ar'])]

            ts_ar = torch.zeros(self.repeats, len(coefs['ar']) + burnin + n)

            for j in range(burnin+n):

                # Apply AR over columns j to j+len(coefs['ar']) in ts_ar
                conv_temp = torch.nn.functional.conv1d(input=ts_ar.narrow(
                    1, j, len(coefs['ar'])).unsqueeze(0), weight=AR)

                ts_ar.narrow(1, j+len(coefs['ar']),
                             1).unsqueeze(0).copy_(conv_temp)

                # Add MA component

                add_temp = ts_ar.narrow(
                    1, j+len(coefs['ar']), 1).add(ts_ma.narrow(1, j, 1))

                ts_ar.narrow(1, j+len(coefs['ar']), 1).copy_(add_temp)
        else:
            ts_ar = ts_ma

        # Take only the last n number of numbers in ts_ar

        res.copy_(ts_ar.narrow(1, burnin, n).squeeze(0))

        return res
