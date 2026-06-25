import numpy as np 
from astropy.coordinates import SkyCoord
from astropy import units as u
import scintools 
from scintools.scint_sim import Simulation

from mwprop.nemod.NE2025 import ne2025

import time

def calc_params(RA='06 30 00', DEC='-28 34 00', DM=34.425, cf=150, veff=77.3):
    c = SkyCoord(f'06 30 00 -28 34 00', unit=(u.hourangle, u.deg))
    cg = c.galactic

    Dk,Dv,Du,Dd = ne2025(ldeg=cg.l.deg, bdeg=cg.b.deg, dmd=DM, ndir=1, dmd_only=False)

    sbw1GHz = Dv['SBW']
    print(sbw1GHz) # MHz
    sbw150MHz = sbw1GHz * (cf/1000)**(22/5)
    print(sbw150MHz)
    scintime1GHz = Dv['SCINTIME']
    print(scintime1GHz) # s
    scintime150MHz = scintime1GHz * (cf/1000)**(6/5) * (veff/100)**(-1)
    print(scintime150MHz, f'sec at {cf} MHz')

    mb2 = 0.773 * (cf/sbw150MHz)**(5/6)
    print(mb2)

    screendist = Dv['DEFFSM2'] # kpc

    # mb2 = 0.773 * (nu_c / delnu_d)**(5/6)

    screendist_m = screendist * 3.086e16 * 1e3 # screendist in kpc
    lam = 3e8 / cf / 1e6 # assuming cf in MHz
    print(lam, 'm wavelength')
    wavenumber = 2*np.pi/lam
    rfresnel_m = np.sqrt(screendist_m/wavenumber)
    print(rfresnel_m/1e3, 'km Fresnel scale')

    return mb2, rfresnel_m, screendist_m

def scint_sim(mb2, rfresnel_m, screendist_m, veff=77.3, f_start=140, f_stop=160, dt=10, nmin=60, seed=3):
    rfresnel = 1 # per DR: since the simulation is done in 'Fresnel units', this is the unit length and should always be 1
    rfresnel_km = rfresnel_m/1e3 # in physical units of meters: 100,000 km
    ar = 1  # axial ratio of anisotropy
    psi = 0  # angle of velocity relative to the x-axis

    #dsvec = np.array([0.001])
    ds_km = dt * veff
    ds = ds_km / rfresnel_km
    #dsvec_km = dsvec * rfresnel_km
    #dtvec = dsvec_km / veff
    print('ds', ds)

    lf = f_start
    rf = f_stop
    cf = (lf+rf)/2
    bw = rf-lf
    frac_bw = bw/cf

    dlam = frac_bw  # fractional bandwidth
    freq = cf # center frequency
    ns = round(5*(1/ds))  # The size of the simulation in spatial steps. Total length in refractive scales is ns * ds
    # it seems like I can make the scintillation time smaller by tuning down ns
    # this is because total length in refractive scales is ns * ds
    nf = int(bw/(0.122))  # The number of frequency channels across the dlam fractional bandwidth

    # seed = 3  # The seed to generate the random phase screen, for reproducible experiments

    # these settings produce ~antisymmetric dynamic spectra
    # pass rectangular parameters nx, ny to save runtime

    start = time.time()
    sim = Simulation(mb2=mb2, rf=rfresnel, ar=ar, 
                        ds=ds, 
                        psi=psi, dlam=dlam,
                        inner=0.0001,
                        #dx=ds, dy=0.2*ds,
                        ns=ns,
                        #nx=nx,
                        #ny=ns, 
                        nf=nf, seed=seed,
                        freq=freq, 
                        dt=dt)
    end = time.time()
    print('ds',ds,'ns',ns,'time', end-start)
    dspec = sim.spi
    sim.plot_intensity()
    ntsamp = nmin*60 // dt
    return dspec.T[:,:ntsamp], ns*ds*rfresnel_km/veff * ntsamp/dspec.T.shape[1]

