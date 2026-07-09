# PsrSim
Pulsar simulation code for HERA.

### Running these notebooks
This code draws from scintools (Reardon et al. 2020; https://github.com/danielreardon/scintools), PsrSigSim (NANOGrav; https://github.com/PsrSigSim/PsrSigSim), and NE2025 (Ocker & Cordes 2026; https://github.com/stella-ocker/mwprop). As described in ./src/psrsigsim/README.md, various version conflicts arise when trying to pull all these packages together for simultaneous analysis. Accordingly, we include functions relevant for our analysis from PsrSigSim, with light modification to mitigate missing dependencies, in ./src/psrsigsim.

We use the following commands to build an environment for ./full_demo_notebook.ipynb and similar notebooks in this repo:

```
conda create --name scint python=3.9.23
conda activate scint
git clone https://github.com/danielreardon/scintools 
cd scintools
pip install .
pip install mwprop
conda install nbformat
conda install mpmath
```

This code is currently being expanded to incorporate beam models to better simulate how pulsars should look in HERA-like visibility data. We simulate visibilities using the software fftvis (Cox et al. 2025; https://github.com/tyler-a-cox/fftvis), as shown in the ./fftvis_test.ipynb notebook, but this is currently handled in a separate environment.
