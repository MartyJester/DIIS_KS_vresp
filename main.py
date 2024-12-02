#!/usr/bin/python
from importlib.machinery import SourceFileLoader
OneDKSSCE = SourceFileLoader('OneDKSSCE', 'OneDKSSCE.py').load_module()
import numpy as np
import scipy as sp
from scipy import linalg, special
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import config
config.update("jax_enable_x64", True)
from tensorflow_probability.substrates import jax as tfp # The scaled error function is not yet implemented in jax
import pyscf
from pyscf import gto, scf
import time
from IPython.display import clear_output

np.set_printoptions(precision=3)

### Set up Grid and miscellanea parameters to be kept constant throughout the computation

nbPts = 300
b = 0.6

## Select Interaction
# wint = OneDKSSCE.softCoulomb
wint = OneDKSSCE.softCoulomb

## Parameters for external potential
R = 10
L = 8*(R+10)

## Number of particles and initial guess function for rho
N = 2
def LiH(x, weights, a_param, b_param, R):
    vals = jnp.square(np.sqrt(b_param/2)*jnp.exp(- b_param/2 * jnp.abs(x + R/2))) + \
    jnp.square(jnp.sqrt(a_param/2)*jnp.exp(- a_param/2 * jnp.abs(x - R/2)))
    vals = vals/jnp.sum(weights*vals)
    return vals

## Grid
x = np.linspace(-L/2, L/2, nbPts)

## Set up one-body part of hamiltonian
kin = OneDKSSCE.kinetic(nbPts,x[1]-x[0],1)
delta_invsqr = 1/(x[1]-x[0])**2
kin[0, 0] = kin[-1,-1] = 2*delta_invsqr
delta = x[1]-x[0]
weights = delta*np.ones(nbPts)
weights = jnp.array(weights)

vext = np.diag(OneDKSSCE.vExtMol_reduced(x, [-R,R], wint, [2.25,0.7]))
h1 = kin + vext

# InitialGuess:
rho = N*LiH(x, weights,1.1,1,R)

## 2-body part
Hartree_pot = np.diag(OneDKSSCE.HartreePotential(x, OneDKSSCE.softCoulomb, 0.6, rho*weights))
vsce = OneDKSSCE.compute_kssce_potential_gauge_zero(rho, weights, x, N, wint, b)
vsce_cond = OneDKSSCE.compute_vcondsce_potential_gauge_zero(rho, weights, x, N, wint, b)
vHxc = Hartree_pot
### HARTREE POTENTIAL VERIFIED ON 16/02/2024


## Construction initial hamiltonian
H = h1 + 0.5*np.diag(vHxc)

## Test quantities are properly initialized

print(H.shape)
plt.plot(x,vsce,label='vsce')
plt.plot(x,vHxc,label='vHxc selected')
plt.plot(x,rho,label='density')
plt.legend()
plt.show()
print("Shape of rho:")
print(rho.shape)

plt.plot(x,rho, color = 'r', linestyle = '-',label = 'Density')
plt.plot(x,np.diag(vext), color = 'b', linestyle = '-',label = 'Vext')
plt.grid()
plt.legend()
plt.show()

##### WITH DIIS - NEW ERROR DEFINITION #####
rhos = []
etas = []
energies = []
# Lambda = 0.1
shift = 0
DIIS_Space = 5
error_list = []
hamiltonian_list = []

# orbital_matrix = np.vstack((orbital_matrix, np.zeros(nbPts)))
for i in range(0, 100):
    ground_state_orbital_old = np.sqrt(rho / 2)

    rho_old = 2 * np.abs(ground_state_orbital_old) ** 2

    eigenvalues, eigenvectors = sp.linalg.eigh(H)

    ground_state_energy = eigenvalues[0]
    ground_state_orbital = eigenvectors[:, 0]

    # Compute the integral of |\phi|^2 (they are not normalized on the grid yet)
    # norm = np.sum(delta * np.abs(ground_state_orbital) ** 2)
    # # Normalize the ground state
    # ground_state_orbital = ground_state_orbital / np.sqrt(norm)
    rho = 2 * np.abs(ground_state_orbital) ** 2
    #     Hartree_pot = np.diag(OneDKSSCE.HartreePotential(x, OneDKSSCE.softCoulomb, 0.6, rho*weights))
    #     vsce = OneDKSSCE.compute_kssce_potential_gauge_zero(rho, weights, x, N, wint, b)
    #     vsce_cond = OneDKSSCE.compute_vcondsce_potential_gauge_zero(rho, weights, x, N, wint, b)
    vHxc = np.diag(OneDKSSCE.HartreePotential(x, OneDKSSCE.softCoulomb, 0.6, rho * weights))
    H = h1 + 0.5*vHxc
    # Density Matrix
    D = 2 * np.outer(ground_state_orbital, ground_state_orbital)
    error = (H @ D) - (D @ H)

    error_list.insert(0, error)
    hamiltonian_list.insert(0, H)

    nDIIS = np.min([DIIS_Space, len(error_list)])
    H_list = hamiltonian_list[:nDIIS]
    # H_list = np.stack(H_list)
    B = np.zeros((nDIIS + 1, nDIIS + 1))

    ### FOR LOOP VERIFIED ON 20 FEB
    # for i in range(0, nDIIS + 1):
    #     for j in range(0, nDIIS + 1):
    #         print(i)
    #         print(j)
    #         if i < nDIIS and j < nDIIS:
    #             #                 B[i,j] = np.dot(error_matrix[i,:],error_matrix[j,:])
    #             B[i, j] = np.sum(error_list[i] * error_list[j])
    #         elif i < nDIIS and j == nDIIS:
    #             B[i, j] = -1
    #         elif i == nDIIS and j < nDIIS:
    #             B[i, j] = -1
    #         else:
    #             B[i, j] = 0
    for i in range(0, nDIIS + 1):
        for j in range(0, nDIIS + 1):
            print(i)
            print(j)
            if i < nDIIS and j < nDIIS:
                #                 B[i,j] = np.dot(error_matrix[i,:],error_matrix[j,:])
                B[i, j] = np.trace(np.matmul(error_list[i], error_list[j].T))
            elif i < nDIIS and j == nDIIS:
                B[i, j] = -1
            elif i == nDIIS and j < nDIIS:
                B[i, j] = -1
            else:
                B[i, j] = 0

    print("matrix is")
    print(B)
    X = np.zeros(nDIIS + 1)
    X[-1] = -1
    w = np.linalg.solve(B, X)
    zeros_matrix = np.zeros_like(H)
    for i in range(0, len(w) - 1):
        zeros_matrix += w[i] * H_list[i]
    H = zeros_matrix