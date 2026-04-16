import numpy as np 
import matplotlib.pyplot as plt 
import os 
import glob 
import psrsigsim as pss

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

def downbin(a, phases, fbin=0.122, tbin=9.6636):
    nf = (a.dat_freq[-1].value-a.dat_freq[0].value)/fbin+1
    nt = a.tobs.value/tbin
    nfpb = len(a.data)/nf
    ntpb = len(a.data[0])/nt
    print(nf, nfpb, nt, ntpb)
    arr = a.data.reshape(int(round(nf)), int(round(nfpb)), int(round(nt)), int(round(ntpb)))
    arr = np.average(arr, axis=3)
    arr = np.average(arr, axis=1)
    return arr

def dmbeat(fstart, fstop, df, dt, tobs,
           P=1.2, D=0.05, DM=34, psr='J0630-2834'):
    '''
    Generates dynamic spectrum of flux values for a pulsar with 
    period P, duty cycle D, and dispersion measure DM
    by downsampling a more high-res spectrum to given 
    fsamp and tsamp values.

    fstart = lower frequency bound of dynamic spectrum [MHz]
    fstop = higher frequency bound of dynamic spectrum [MHz]
    df = frequency channel width [MHz]
    dt = time bin width [seconds]
    tobs = time length of observation [seconds]
    P = pulsar period [seconds]
    D = pulsar duty cycle (can be calculated as pulse width divided by pulse period)
    DM = pulsar dispersion measure [pc cm^-3]
    psr = pulsar name

    TODO: Enable non-Gaussian pulse profiles.
    '''

    dt_hr = dt / 8000 # high-res time sampling
    df_hr = df / 2 # high-res freq sampling

    cf = (fstart+fstop)/2 # center observing frequency, MHz
    print(f'Center frequency is {cf} MHz.')
    BW = fstop-fstart # observation bandwidth, MHz
    print(f'Total bandwidth is {BW} MHz.')
    nf = int(BW//df_hr) # number of frequency channels
    print(f'There will be {nf} frequency channels.')

    print(f'Sample rate is {1e-6/dt_hr} MHz.')

    # Use PsrSigSim for generating a signal in a high-res dynamic spectrum
    signal_1 = pss.signal.FilterBankSignal(fcent = cf, bandwidth = BW, Nsubband=nf, sample_rate=1e-6/dt_hr, fold = False)
    # Currently Gaussian profile
    # Note that 'width' of GaussProfile is in units of pulse phase, not seconds
    gauss_prof = pss.pulsar.GaussProfile(peak = 0.5, width = D, amp = 1.0)
    gauss_prof.init_profiles(2048, Nchan = 1)
    Smean = 10.0 # mean flux of pulsar in Jy; not currently used since we do not pass the Pulsar object through a telescope
    pulsar_1 = pss.pulsar.Pulsar(P, Smean, profiles = gauss_prof, name = psr)

    # Disperse
    ism_1 = pss.ism.ISM()
    pulsar_1.make_pulses(signal_1, tobs = tobs)
    ism_1.disperse(signal_1, DM)

    phases = np.linspace(0, tobs/P, len(signal_1.data[0,:]))

    a = downbin(signal_1, phases, fbin=df, tbin=dt)
    dmbeat_dynspec = a.T     # oriented like waterfall plot
    return dmbeat_dynspec