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
           spec_idx=-1.7, ref_freq=400):
    '''
    Given an input array `data` labeled by input vector `freq`, 
    apply a spectral index to modulate the amplitude of `data` over frequency.

    Here we assume that the first dimension of data is frequency.
    There is no assumption as to the other dimensions; 
    `data` can be 2D (freq, time) or 3D (freq, pol, time).

    By default, spec_idx=-1.7 and ref_freq=400 (MHz), as is appropriate for
    PSR J0630-2834 per Jankowski+18.

    TODO: Simulate broken power laws and low-frequency turnovers.
    '''
    gains = np.array([(f/ref_freq)**spec_idx for f in freq])
    data2 = data * gains[:,None]
    return data2