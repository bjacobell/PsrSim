import numpy as np 
import matplotlib.pyplot as plt 
import os 
import glob 
from src import dmbeat
from src.psrsigsim.signal.fb_signal import FilterBankSignal
from src.psrsigsim.pulsar.profiles import GaussProfile
from src.psrsigsim.pulsar.pulsar import Pulsar
from src.psrsigsim.ism.ism import ISM

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

def apply_spec_idx(data, freq,
           spec_idx=-1.7, ref_freq=400,
           lft=False, beta=1.8, fc=70,
           bpl=False, spec_idx_2=-1.59, fb=1400):
    '''
    Given an input array `data` labeled by input vector `freq`, 
    apply a spectral index to modulate the amplitude of `data` over frequency.

    Here we assume that the first dimension of data is frequency.
    There is no assumption as to the other dimensions; 
    `data` can be 2D (freq, time) or 3D (freq, pol, time).

    lft: Does the pulsar spectrum have a low-frequency turnover (LFT)? If so, set to True.
    beta: Used only if LFT = True to set the smoothness of the turnover; see Jankowski+18.
    fc: Used only if LFT = True to set the turnover frequency [MHz].

    By default, spec_idx=-1.7 and ref_freq=400 (MHz), as is appropriate for
    PSR J0630-2834 per Jankowski+18.

    TODO: Simulate log-parabolic spectra.
    '''
    gains = np.array([(f/ref_freq)**spec_idx for f in freq])

    if lft == True:
        turnover_gains = np.array([np.exp((spec_idx/beta) * (f/fc)**(-1*beta)) for f in freq])
        gains *= turnover_gains
    if bpl == True:
        gains = []
        for f in freq:
            if f < fb:
                gains.append((f/ref_freq)**spec_idx)
            else:
                gains.append(((f/ref_freq)**spec_idx_2) * (fb/ref_freq)**(spec_idx-spec_idx_2))
        gains = np.array(gains)
    
    data2 = data * gains[:,None]
    return data2