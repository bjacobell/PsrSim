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

def rm_sim(data, freq,
           linpol=0.6, cirpol=0., polangle=45,
           RM=45.78):
    '''
    Simulates rotation measure effects for given linear
    and circular polarizations and polarization angle.

    data = dynamic spectrum-like data
        (first dimension is assumed to be frequency,
        as is typical for pulsar data)
    freq = frequencies corresponding to the dynamic spectrum
    linpol = linear polarization fraction
    cirpol = circular polarization fraction (typically ~0 for pulsars)
    polangle = polarization angle [deg]
    RM = rotation measure [rad m^-2]

    TODO: Enable phase-dependent polarization angle.
    TODO: Consider whether it would be better to be data-agnostic and 
    let the Stokes be multiplied into data at the end.
    '''

    I = data
    pL = linpol # linear polarization fraction
    pV = cirpol # circular polarization fraction
    L = I * pL # linearly polarized light
    psi = np.deg2rad(polangle) # polarization angle
    # Q = L * np.cos(2*psi)
    # U = L * np.sin(2*psi)
    V = I * pV

    lam = 3e8/freq 
    RM = RM # rad/m^2, 45.78 for J0630-2834
    psivec = RM*lam**2 + psi

    Q = L * np.cos(2*psivec)[:,None]
    U = L * np.sin(2*psivec)[:,None]

    full_stokes = np.array([I,Q,U,V])

    return full_stokes

