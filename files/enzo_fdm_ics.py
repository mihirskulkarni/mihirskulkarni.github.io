"""
Usage: python enzo_fdm_ics.py dirname.
"""

import h5py as h5
import numpy as np
#from scipy import *
import sys
import re
# Read Baryon density
dirname = sys.argv[1]
if dirname[-1] == '/':
    dirname = dirname[:-1]
f = h5.File('./'+dirname+'/GridFDMDensity', 'r')
dataset = f['GridFDMDensity']
dens = np.array(dataset[0])

parfile = open(dirname+'/parameter_file.txt','r')
for line in parfile.readlines():
    if re.search('^CosmologyOmegaMatterNow',line, re.I):
        OmegaMatterNow = float(line.split()[2]) 
    if re.search('^CosmologyComovingBoxSize', line, re.I):
        BoxLength = float(line.split()[2])      # in Mpc/h
    if re.search('^FDMmass', line, re.I):
        mass_unit = float(line.split()[2])
    if re.search('^CosmologyInitialRedshift', line, re.I):
        InitialRedshift = float(line.split()[2])
    if re.search('^CosmologyHubbleConstantNow', line, re.I):
        HubbleConstantNow = float(line.split()[2])
 

#BoxLength = 1.0
hbar = 1.05457266e-27
#mass_unit = 1.0
mass = 1e-22*1.6021772e-12/(2.99792458e10**2)*mass_unit 
print("omegam0", OmegaMatterNow)
print("mass_unit",mass_unit)
print("Boxsize",BoxLength)

#InitialRedshift = 100.
#HubbleConstantNow = 0.704
#OmegaMatterNow = 0.268

a0 = 1./(1+InitialRedshift)
InitialTime = 0.81651316219217
LengthUnits = 3.085678e24*BoxLength/HubbleConstantNow/(1 + InitialRedshift)
TimeUnits = 2.519445e17/np.sqrt(OmegaMatterNow)/HubbleConstantNow/(1 + InitialRedshift)**1.5
acoef = 1.5**(1./3.)*a0
coef = hbar/mass*TimeUnits/(LengthUnits**2)
print(coef)

# solve poisson equation for theta
N = dens.shape[-1]
# use fft to solve theta
LHS = -2./3.*(dens-1)/InitialTime/coef

klhs = np.fft.fftn(LHS)
G1d = N * np.fft.fftfreq(N) * 2*np.pi
kx, ky, kz = np.meshgrid(G1d, G1d, G1d, indexing='ij')

G2 = kx**2 + ky**2 + kz**2
G2[0,0,0] = 1.

thetak = klhs/(-G2)
thetak[0,0,0] = 0

# Making sure that dens is positive
dens = np.where(dens <= 0, 0.01, dens)

print('smallest dens value = {:.3f}'.format(dens.min()))

theta = np.real(np.fft.ifftn(thetak))
#calculate wave function
repsi = np.sqrt(dens)*np.cos(theta)
impsi = np.sqrt(dens)*np.sin(theta)

if (np.iscomplexobj(repsi) or np.iscomplexobj(impsi)):
    print('dtype of repsi {:s}'.format(str(repsi.dtype)))
    print('dtype of impsi {:s}'.format(str(impsi.dtype)))
    raise ValueError("RePsi or ImPsi is not a float array.") 

#repsi = sqrt(dens)
#impsi = sqrt(dens)*0
# write out to new file
f1 = h5.File('./{:s}/GridRePsi'.format(dirname), 'w')
new_dens = f1.create_dataset('GridRePsi', data=repsi)
for a in dataset.attrs.keys():
    new_dens.attrs.create(a, dataset.attrs[a])
f1.close()

f1 = h5.File('./{:s}/GridImPsi'.format(dirname), 'w')
new_dens = f1.create_dataset('GridImPsi', data=impsi)
for a in dataset.attrs.keys():
    new_dens.attrs.create(a, dataset.attrs[a])
f1.close()
f.close()
