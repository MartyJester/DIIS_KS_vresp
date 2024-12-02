#!/usr/bin/python

# This code is a 1D implementation of KS-SCE based on pySCF
# Author: Antoine Marie <antoine.marie@ens-lyon.fr>

import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
from jax import jit, vmap, jacfwd, lax, jacrev
from functools import partial
import jax.numpy as jnp
import jax.lax as lax
from tensorflow_probability.substrates import jax as tfp # The scaled error function is not yet implemented in jax

import sys
sys.path.append('/home/antoinem/pyscf')
import pyscf
from pyscf import gto, scf

### Computational tools
x =3
def band_eig(h, s):
    ''' Band diagonal solver for eigenvalue problem. Takes s because it has to,
    but does not actually solve the generalized problem! '''

    nbdiag = len([i for i in h[0] if i != 0])

    diags = np.zeros((nbdiag, len(h)))
    for i in range(nbdiag):
        diags[i,:len(h)-i] = np.diag(h , k = i)
        assert np.allclose(np.diag(h,-i), np.diag(h,i))

    #Banddiagonal solver, solves for all eigenvectors, not ideal
    # e, c = scipy.linalg.eig_banded(diags, lower=True, select='i', select_range=(0,1))
    e, c = scipy.linalg.eigh(h, subset_by_index = [0,4])
    print(e)

    #Fix phase conventions
    idx = np.argmax(abs(c.real), axis=0)
    c[:,c[idx,np.arange(len(e))].real<0] *= -1

    return e, c

### Building the one-body hamiltonian

def kinetic(n, delta, order):
    ''' Return the matrix representation of the laplacian on a grid with n points equally spaced by a distance delta. One can choose between 1st, 2nd and 3rd order approximation for the derivative on the grid. '''
    listcoeff = [[1,-1/2],[5/4,-2/3,1/24],[49/36, -3/4, 3/40, -1/180]]
    coeff = listcoeff[order - 1]
    delta_invsqr = 1/delta**2

    kinOp = np.diag(coeff[0]*np.ones(n)*delta_invsqr)
    kinOp[0, 0] = kinOp[-1,-1] = 2*coeff[0]*delta_invsqr

    for i in range(1,len(coeff)):
        kinOp += np.diag(coeff[i]*np.ones(n-i)*delta_invsqr, k=-i)
        kinOp += np.diag(coeff[i]*np.ones(n-i)*delta_invsqr, k=+i)
    return kinOp

def vExtMol(listPos, listR, listZ, wint, b):
    ''' This function return the value of the external potential corresponding to nuclear at position listR with charge listZ. '''
    assert len(listR) == len(listZ)

    vExt = np.zeros(len(listPos))
    for i in range(len(listR)):
        vExt += - listZ[i]*wint(np.abs(listPos-listR[i]), b)
    return vExt

def vExtMol_reduced(listPos, listR, wint, b):
    ''' This function return the value of the external potential corresponding to nuclear at position listR with charge listZ. '''
    assert len(listR) == len(b)

    vExt = np.zeros(len(listPos))
    for i in range(len(listR)):
        vExt += - wint(np.abs(listPos-listR[i]), b[i])
    return vExt

def vExtRand(x,V0,sigma):
    ''' This function return the value of the random potential at the positions listPos. The parameter of the random potential are taken from filename. '''
    L = x[-1] - x[0]
    file_centers = open("centers.dat","r")
    file_amp = open("ranampl.dat","r")
    centers = []
    ampl =[]
    for line in file_centers:
        centers.append(float(line)*L/256)
    for line in file_amp:
        ampl.append(float(line))
    vrand = [np.sum([ampl[j]*V0*np.exp(-(x[i]-centers[j])**2/sigma**2) for j in range(len(ampl))]) for i in range(len(x))]

    return vrand


### Interaction functions

def softCoulomb(x, b, D=False):
    ''' Compute the soft Coulomb interaction with parameter b for a distance x. If D=True it returns the value of the derivative of the interaction instead. '''
    if not D:
        interaction = 1/jnp.sqrt(b + x**2)

    else:
        interaction = - x/(b + x**2)**(3/2)

    return interaction

def absCoulomb(x, b, D=False):
    ''' Compute the "absolute" Coulomb interaction with parameter b for a distance x. If D=True it returns the value of the derivative of the interaction instead. '''
    if not D:
        interaction = 1/(b + jnp.abs(x))

    else:
        interaction = -1/(b + x)**2*np.sign(x)

    return interaction

def wireCoulomb(x, b, D=False):
    ''' Compute the thin wire Coulomb interaction with parameter b for a distance x. If D=True it returns the value of the derivative of the interaction instead. '''
    if not D:
        interaction = np.sqrt(np.pi)*tfp.math.erfcx(x/(2*b))/(2*b)

    else:
        xb = x/(2*b)
        interaction = -1/(2*b**2) + np.sqrt(np.pi)*xb*tfp.math.erfcx(xb)/(2*b**2)

    return interaction

### SCE tools


#We designate N as a static argument, so it recompiles for every value of N
#This is necessary because it is used as the stop index in jnp.arange
#And it determines the size of the output array
@partial(jit, static_argnums=(3))
def compute_all_comotion_functions(rho, weights, x, N):
  '''Compute all co-motion functions with indices i=1 to N, with weighted density
  weighted_rho, coordinates x, and particle number N'''
  #Compute cumulant
  Ne = jnp.cumsum(weights*rho)

  #Use vmap to calculate them all in parallel
  compute_comotion_function = lambda i: jnp.interp(Ne+i-1-jnp.heaviside(Ne-N+i-1, 0.)*N,
                                                   Ne, x, left=x[0], right=x[-1])
  return vmap(compute_comotion_function, in_axes=(0), out_axes=(0))(jnp.arange(1,N+1))

@partial(jit, static_argnums=(3,4))
def compute_kssce_potential(rho, weights, x, N, wint, b):
    ''' Compute the SCE potential for the density rho, with integration weights, coordinates x, and particle number N. '''

    comotion_functions = compute_all_comotion_functions(rho, weights, x, N)
    potential = jnp.cumsum(jnp.ravel(jnp.sign(x-comotion_functions[1:])*weights*wint(jnp.abs(x-comotion_functions[1:]), b, D = True)))

    potential = potential - (potential[0]+potential[-1])/2

    functional_value = compute_functional(rho, weights, x, N, wint, b)

    potential = potential + functional_value/N - jnp.sum(potential*weights*rho)/N

    return potential

@partial(jit, static_argnums=(3,4))
def compute_kssce_potential_gauge_zero(rho, weights, x, N, wint, b):
    ''' Compute the SCE potential for the density rho, with integration weights, coordinates x, and particle number N. '''

    comotion_functions = compute_all_comotion_functions(rho, weights, x, N)
    potential = jnp.cumsum(jnp.ravel(jnp.sign(x-comotion_functions[1:])*weights*wint(jnp.abs(x-comotion_functions[1:]), b, D = True)))

    potential = potential - (potential[0]+potential[-1])/2

    return potential

@partial(jit, static_argnums=(3,4))
def compute_vcondsce_potential_gauge_zero(rho, weights, x, N, wint, b):
    ''' Compute the SCE potential for the density rho, with integration weights, coordinates x, and particle number N. '''

    comotion_functions = compute_all_comotion_functions(rho, weights, x, N)
    potential = jnp.ravel(wint(jnp.abs(x-comotion_functions[1:]), b, D = False))

    potential = potential - (potential[0]+potential[-1])/2

    return potential

@partial(jit, static_argnums=(3,4))
def compute_functional(rho, weights, x, N, wint, b):
  '''Compute the value of the functional for the weighted density weighted_rho,
  with coordinates x, and particle number N. Note the interaction
  1/sqrt(1+(x_1-x—2)^2)
  Average over non-flipped and flipped
  '''
  comotion_functions = compute_all_comotion_functions(rho, weights, x, N)
  comotion_functions_reverse = jnp.flip(compute_all_comotion_functions(jnp.flip(rho), jnp.flip(weights), jnp.flip(x), N), axis=1)
  return 1/4*jnp.sum(weights*rho/jnp.sqrt(1+(x - comotion_functions[1:])**2)) \
        + 1/4*jnp.sum(weights*rho/jnp.sqrt(1+(x - comotion_functions_reverse[1:])**2))


### LDA tools

def HartreePotential(x, wint, b, rho):
    ''' Build the Hartree Coulomb potential '''
    Nx = len(x)
    CoulombMat = np.empty((Nx,Nx))
    for i in range(Nx):
        for j in range(Nx):
            if 'wire' in str(wint):
                CoulombMat[i][j] = np.sqrt(np.pi)*scipy.special.erfcx(np.abs(x[i]-x[j])/(2*b))/(2*b)
            elif 'abs' in str(wint):
                CoulombMat[i][j] = 1/(b+np.abs(x[i]-x[j]))
            else:
                CoulombMat[i][j] = 1/(np.sqrt(b + (x[i]-x[j])**2))
            # else:
            #     print('Interaction not implemented! Using wireCoulomb')
            #     CoulombMat[i][j] = np.sqrt(np.pi)*scipy.special.erfcx(np.abs(x[i]-x[j])/(2*b))/(2*b)
    hartree_pot = np.dot(CoulombMat, rho)
    hartree_pot -= (hartree_pot[0]+hartree_pot[-1])*0.5

    return  np.diag(hartree_pot) 

# In this section we are going to compute the LDA exchange correlation energy as well as the LDA potential.

def LDA_correl_nrj(rs):
    ''' LDA correlation energy as a function of rs = 1/2rho '''
    A = 4.66; B = 2.092; C = 3.735; D = 23.63; E = 109.9
    gamma1 = 1.379; gamma2 = 1.837

    ec = - 0.5 * np.log(1 + D*rs + E*rs**gamma2) / (A + B*rs**gamma1 + C*rs**2)

    return ec

def LDA_correl_pot(rs):
    ''' LDA correlation potential as a function of rs = 1/2rho '''
    A = 4.66; B = 2.092; C = 3.735; D = 23.63; E = 109.9
    gamma1 = 1.379; gamma2 = 1.837
    dec = (- A + (gamma1 - 1)*B*rs**gamma1 + C*rs**2)*np.log(1 + D*rs + E*rs**gamma2)/((A + B*rs**gamma1 + C*rs**2)**2)  \
           - ( D*rs + gamma2*E*rs**gamma2 )/( 1 + D*rs + E*rs**gamma2 )/(A + B*rs**gamma1 + C*rs**2)
    potc = LDA_correl_nrj(rs)- rs * dec

    return potc

def g_func(x):
    ''' Interpolation to 14th order of the function g used to compute the LDA exchange energy '''
    x0 = 1.68548962
    g_of_x = np.where(x < x0, \
                      1.2113921675492336 + 0.13245425014709447*x**2 + 0.027601961140530002*x**4 + 0.005332833196863355*x**6 + 0.0008927499493768617*x**8 + 0.0001297097548558492*x**10 + 0.00001655591904610625*x**12 \
                      - np.log(x)*(1. + 0.16666666666666666*x**2 + 0.03333333333333333*x**4 + 0.005952380952380952*x**6 + 0.000925925925925926*x**8 + 0.00012626262626262626*x**10 + 0.000015262515262515263*x**12), \
                      4.615384615384615/x**14 - 1.0909090909090908/x**12 + 0.3333333333333333/x**10 - 0.14285714285714285/x**8 + 0.1/x**6 - 0.16666666666666666/x**4 + 2.7841639984158535/x \
                      - (1.2886078324507664 + np.log(x))/x**2
                      )
    return g_of_x

def LDA_exchange_nrj(rs):
    ''' LDA exchange energy as a function of rs = 1/2rho '''
    ex = - 0.25 * g_func(0.1 * np.pi / (0.5*rs)) / rs
    return ex

def dg_func(x):
    ''' Derivative of the function g '''
    x0 = 1.68548962
    dg_of_x = np.where(x<x0,
                       - 1./x + 0.09824183362752228*x + 0.07707451122878668*x**3 + 0.026044618228799173*x**5 + 0.006216073669088968*x**7 + 0.001170834922295863*x**9 + 0.00018340851329075983*x**11 \
                       - np.log(x)*(0.3333333333333333*x + 0.13333333333333333*x**3 + 0.03571428571428571*x**5 + 0.007407407407407408*x**7 + 0.0012626262626262627*x**9 + 0.00018315018315018315*x**11), \
                       - 64.61538461538461/x**15 + 13.090909090909092/x**13 - 3.3333333333333335/x**11 + 1.1428571428571428/x**9 - 0.6/x**7 + 0.6666666666666666/x**5 - 1./x**3 - 2.7841639984158535/x**2 \
                       + (2.*(1.2886078324507664 + np.log(x)))/x**3 \
                       )
    return dg_of_x

def LDA_exchange_pot(rs):
    ''' LDA exchange potential as a function of rs = 1/2rho '''
    ex = LDA_exchange_nrj(rs)
    dex = -ex/rs + 0.1*np.pi*dg_func(0.1 * np.pi / (0.5*rs))/(8*rs**3)
    potx = ex - rs*dex
    return potx

def LDA_pot(rho):
    ''' The functions computing the potential are defined in terms of r_s which is equal to 1/(2*rho)'''
    return(np.diag(LDA_correl_pot(0.5/rho) + LDA_exchange_pot(0.5/rho)))

### Build the two-body hamiltonian
# The different possibilities are the SCE potential as well as the Hartree, Hartree + Exact echange and LDA functionals.

def get_veffSCE(mol, density_matrix, *kwargs):
    '''Modified version of get_veff for use with SCE'''
    #Extract number of electrons
    N = sum(mol.nelec)
    #Density is here simply the diagonal of the density matrix
    rho = jnp.diag(jnp.array(density_matrix))/jnp.array(mol.weights)
    # functional_value, vHxc = compute_functional_derivative(rho, jnp.ones_like(rho), mol.x, N)
    # vHxc = compute_kssce_potential(rho, mol.weights, mol.x, N, mol.wint, mol.b)
    vHxc = compute_kssce_potential_gauge_zero(rho, mol.weights, mol.x, N, mol.wint, mol.b)

    return np.diag(vHxc)

def get_veffCondSCE(mol, density_matrix, *kwargs):
    '''Modified version of get_veff for use with SCE'''
    #Extract number of electrons
    N = sum(mol.nelec)
    #Density is here simply the diagonal of the density matrix
    rho = jnp.diag(jnp.array(density_matrix))/jnp.array(mol.weights)
    vHxc = compute_vcondsce_potential_gauge_zero(rho, mol.weights, mol.x, N, mol.wint, mol.b)

    return np.diag(vHxc)


def get_veffLDA(mol, density_matrix, *kwargs):
    '''Modified version of get_veff for use with LDA'''
    #Density is here simply the diagonal of the density matrix
    rho = np.diag(np.array(density_matrix))/np.array(mol.weights)
    # functional_value, vHxc = compute_functional_derivative(rho, jnp.ones_like(rho), mol.x, N)
    vLDA = LDA_pot(rho) + HartreePotential(mol.x, mol.wint, mol.b, np.diag(density_matrix))

    return vLDA

def get_veffHartree(mol, density_matrix, *kwargs):
    '''Modified version of get_veff for use with Hartree functional'''
    return HartreePotential(mol.x, mol.wint, mol.b, np.diag(density_matrix))

def get_veffEXX(mol, density_matrix, *kwargs):
    '''Modified version of get_veff for use with Hartree + Exact exchange functional'''
    return 0.5*HartreePotential(mol.x, mol.wint, mol.b, np.diag(density_matrix))

def get_veffNonInt(mol, density_matrix, *kwargs):
    ''' Return a matrix of zero for the Hxc potential '''
    Nx = len(mol.x)
    zeroMat = np.zeros((Nx,Nx))
    return zeroMat

### Compute energies

def energy_elec(dm=None, h1e=None, vhf=None):
    if dm is None: dm = myhf.make_rdm1()
    if h1e is None: h1e = myhf.get_hcore()
    if vhf is None: vhf = myhf.get_veff(myhf.mol, dm)
    e1 = np.einsum('ij,ji', h1e, dm)
    e_coul = np.einsum('ii,ii', vhf, dm)
    return e1 + e_coul, e_coul

@partial(jit, static_argnums=(2))
def hartree(x, rho, wint, b):
    ''' Compute the Hartree energy for a given list of positions and densities and an interaction wint with parameter b. '''
    #interaction_function = lambda x1, rho1, x2, rho2: rho1*rho2/jnp.sqrt(1+(x1-x2)**2)
    interaction_function = lambda x1, rho1, x2, rho2: rho1*rho2*wint(jnp.abs(x1 - x2), b)
    return 1/2*jnp.sum(vmap(vmap(interaction_function, in_axes=(0,0, None, None), out_axes=(0)),
        in_axes=(None, None, 0, 0), out_axes=(0))(x, rho, x, rho))


### Results

def print_results(mol, myhf, kin_mat, vext, L, a, plot, folder):
    ''' This function is used to print and store results at the end of the calculation. If plot is set to True it will also plot then densities etc. '''

    rdm1 = myhf.make_rdm1()
    rho = np.diag(np.array(rdm1))/np.array(mol.weights)
    # Plot the density and write its values in a file
    fig, ax = plt.subplots()
    ax.plot(mol.x, rho)
    ax.set(xlabel='x', ylabel='Density $\\rho(x)$',  title='Density')
    ax.grid()
    plt.savefig(folder + "/density_L" + str(L) + "V" + str(a) + ".pdf")
    if plot == True:
        plt.show()
    file = open(folder + "/density_L" + str(L) + "V" + str(a) + ".txt", "w")
    for line in range(len(rho)):
        file.write(str(mol.x[line]) + ' ' + str(rho[line]) + '\n')
    file.close()

    N = sum(mol.nelec)
    comotion_functions = compute_all_comotion_functions(rho, mol.weights, mol.x, N)
    vHxc = np.diag(myhf.get_veff(mol,rdm1))
    # vHxc = vHxc - (vHxc[0]+vHxc[-1])/2
    fig, ax = plt.subplots()
    ax.plot(mol.x, comotion_functions[0])
    ax.plot(mol.x, comotion_functions[1])
    ax.set(xlabel='x', ylabel='f',  title='Co-motion function')
    ax.grid()
    plt.savefig(folder + "/comotion_L" + str(L) + "V" + str(a) + ".pdf")
    if plot == True:
        plt.show()
    # Plot the potential and write its values in a file
    fig, ax = plt.subplots()
    ax.plot(mol.x, vHxc)
    ax.set(xlabel='x', ylabel='vHxc',  title='KS Potential')
    ax.grid()
    plt.savefig(folder + "/KSpotential_L" + str(L) + "V" + str(a) + ".pdf")
    if plot == True:
        plt.show()
    file = open(folder + "/KSpotential_L" + str(L) + "V" + str(a) + ".txt", "w")
    for line in range(len(rho)):
        file.write(str(mol.x[line]) + ' ' + str(vHxc[line]) + '\n')
    file.close()


    # Compute the different parts of the energy
    kin_nrj = np.einsum("ij, ij", rdm1, kin_mat)
    ext_nrj = np.einsum("ij, ij", rdm1, vext)
    onebody_nrj = np.einsum("ij, ij", rdm1, kin_mat + vext)
    print('Kinetic energy:', kin_nrj)
    print('External potential energy:', ext_nrj)
    print('One-body energy:', onebody_nrj)
    hartree_nrj = hartree(mol.x, np.diag(rdm1), mol.wint, mol.b)
    twobody_nrj = myhf.energy_elec(rdm1, mol.h1, myhf.get_veff(myhf.mol, rdm1))[1]
    print('Hartree:', hartree_nrj)
    print('Two-body energy:', twobody_nrj)
    print('Total:', onebody_nrj + twobody_nrj)
    file = open(folder + "/energies_L" + str(L) + "V" + str(a) + ".txt", "w")
    file.write(str(kin_nrj) + '\n' + str(ext_nrj) + '\n' + str(onebody_nrj) + '\n' + str(hartree_nrj) + '\n' + str(twobody_nrj) + '\n' + str(onebody_nrj + twobody_nrj))
    file.close()

    # Write the orbitals and their energies
    file_nrj = open(folder + "/orbital_energies_L" + str(L) + "V" + str(a) + ".txt", "w")
    file_nrj.write("\n".join([str(element) for element in myhf.mo_energy[:10]]))
    file_nrj.close()
    file_orb = open(folder + "/orbitals_L" + str(L) + "V" + str(a) + ".txt", "w")
    for line in range(len(rho)):
        file_orb.write(str(mol.x[line]) + ' ' )
        file_orb.write("  ".join([str(element) for element in myhf.mo_coeff[line,:10]]))
        file_orb.write('\n')
    file_orb.close()


##### MAIN #####
if __name__ == "__main__":

    loopPotentials = [0.]                                       #[0., 10., 25., 50., 75., 100.]
    loopInteractions = [25., 50., 100.]                           #[0.5, 1., 2.5, 10., 25.]

    for pot in loopPotentials:
        for int in loopInteractions:

            nbPts = 512
            b = 0.1
            wint = wireCoulomb
            L = int
            N = 2
            eps = 0.002
            a = pot
            V0 = a**2/L**2
            sigma = eps*L
            x = np.linspace(-L/2, L/2, nbPts)
            kin_mat = kinetic(nbPts,x[1]-x[0],1)
            delta_invsqr = 1/(x[1]-x[0])**2
            kin_mat[0, 0] = kin_mat[-1,-1] = 2*delta_invsqr
            vext = np.diag(vExtRand("/home/antoinem/PLR2/Python/", x, V0, sigma))
            h1 = kin_mat + vext
            mol = gto.M(verbose=4)
            mol.nelectron = N
            mol.incore_anyway = True
            mol.x = x
            mol.delta = mol.x[1]-mol.x[0]
            mol.delta_invsqr = 1/mol.delta**2
            mol.weights = mol.delta*np.ones(nbPts)
            mol.weights[0] = mol.weights[-1] = mol.delta/2
            mol.weights = np.array(mol.weights)
            mol.h1 = h1
            mol.wint = wint
            mol.b = b

            #initial guess
            # initfile = []
            # Linit = 25
            # with open('/home/antoinem/PLR2/Python/kssce_full/density_L25.0V75.0.txt') as f:
            #     for line in f:
            #         initfile.append([float(x) for x in line.split()])
            # initfile = np.array(initfile)
            # initguess = (Linit/L)*initfile[:,1]*mol.weights
            # initguess = np.diag(initguess)

            print('Starting SCF')
            time_start = time.perf_counter()
            myhf = scf.RHF(mol)
            myhf.chkfile = False
            myhf.get_hcore = lambda *args: mol.h1
            myhf.get_ovlp = lambda *args: np.eye(nbPts)
            myhf.get_veff = get_veffLDA
            myhf.eig = band_eig
            myhf.energy_elec = energy_elec
            myhf.init_guess =  '1e'
            myhf.max_cycle = 1000
            myhf.diis = True
            myhf.diis_space = 8
            myhf.chkfile = 'kssce.chk'
            myhf.damp = 0.5
            myhf.conv_tol = 1e-9
            myhf.diis_start_cycle = 2
            # myhf.level_shift = 0.1
            # scf.addons.dynamic_level_shift_(myhf, factor = 0.1)
            # myhf = scf.addons.frac_occ(myhf, tol=0.55)
            # myhf.kernel(initguess)
            myhf.kernel()
            time_elapsed = (time.perf_counter() - time_start)
            print('The computation time is', time_elapsed)

            print_results(mol, myhf, kin_mat, vext, L, a, True, "kslda")
