
"""
Script for calculating various spectra.

"""

import numpy as np
import scipy.sparse.linalg
from collections import OrderedDict
import sys,os
from mpi4py import MPI
import pickle
import time
import argparse
import h5py
# Local local stuff
from impurityModel.ed import spectra
from impurityModel.ed import finite
from impurityModel.ed.finite import c2i
from impurityModel.ed.average import k_B, thermal_average


def main(h0_filename,
         radial_filename,
         ls, nBaths, nValBaths,
         n0imps, doccs, dnTols, dnValBaths, dnConBaths,
         Fdd, Fpp, Fpd, Gpd,
         xi_2p, xi_3d, VP, chargeTransferCorrection,
         hField, hpField, nPsiMax, nChiMax,
         nPrintSlaterWeights, tolPrintOccupation,
         T, energy_cut,
         delta, delta_2, POS, deltaRIXS, deltaNIXS, XAS_projectors_filename, RIXS_projectors_filename, dc_method, energymesh, energylossmesh, energyrixsmesh):
    """
    First find the lowest eigenstates and then use them to calculate various spectra.

    Parameters
    ----------
    h0_filename : str
        Filename of the non-relativistic non-interacting Hamiltonian operator.
    radial_filename : str
        File name of file containing radial mesh and radial part of final
        and initial orbitals in the NIXS excitation process.
    ls : tuple
        Angular momenta of correlated orbitals.
    nBaths : tuple
        Number of bath states,
        for each angular momentum.
    nValBaths : tuple
        Number of valence bath states,
        for each angular momentum.
    n0imps : tuple
        Initial impurity occupation.
    doccs : float
        number of electrons according to RSPT
    dnTols : tuple
        Max devation from initial impurity occupation,
        for each angular momentum.
    dnValBaths : tuple
        Max number of electrons to leave valence bath orbitals,
        for each angular momentum.
    dnConBaths : tuple
        Max number of electrons to enter conduction bath orbitals,
        for each angular momentum.
    Fdd : tuple
        Slater-Condon parameters Fdd. This assumes d-orbitals.
    Fpp : tuple
        Slater-Condon parameters Fpp. This assumes p-orbitals.
    Fpd : tuple
        Slater-Condon parameters Fpd. This assumes p- and d-orbitals.
    Gpd : tuple
        Slater-Condon parameters Gpd. This assumes p- and d-orbitals.
    xi_2p : float
        SOC value for p-orbitals. This assumes p-orbitals.
    xi_3d : float
        SOC value for d-orbitals. This assumes d-orbitals.
    VP : float
        corelevel binding on 2P
  chargeTransferCorrection : float
        Double counting parameter
    hField : tuple
        Magnetic field.
    hpField: tuple
       Magnetic field on p states
    nPsiMax : int
        Maximum number of eigenstates to consider.
    nChiMax : int
        Maximum number of p states to consider
    nPrintSlaterWeights : int
        Printing parameter.
    tolPrintOccupation : float
        Printing parameter.
    T : float
        Temperature (Kelvin)
    energy_cut : float
        How many k_B*T above lowest eigenenergy to consider.
    delta : float
        Smearing, half width half maximum (HWHM). Due to short core-hole lifetime.
    delta_2 : float
        Smearing of L2 edge. Due to additional decay channel.
    POS : float
        Position where broadening for L2 should begin.
    deltaRIXS : float
        Smearing, half width half maximum (HWHM).
        Due to finite lifetime of excited states.
    deltaNIXS : float
        Smearing, half width half maximum (HWHM).
        Due to finite lifetime of excited states.
   XAS_projectors_filename: string
         File containing the XAS projectors, separated by an empty line
   RIXS_projectors_filename: string
         File containg the RIXS projectors, separated by an empty line


    """

    # MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0: t0 = time.time()

    # -- System information --
    nBaths = OrderedDict(zip(ls, nBaths))
    nValBaths = OrderedDict(zip(ls, nValBaths))

    # -- Basis occupation information --
    n0imps = OrderedDict(zip(ls, n0imps))
    dnTols = OrderedDict(zip(ls, dnTols))
    dnValBaths = OrderedDict(zip(ls, dnValBaths))
    dnConBaths = OrderedDict(zip(ls, dnConBaths))
    # -- Spectra information --
    # Energy cut in eV.
    energy_cut *= k_B*T
    # XAS parameters
    # Energy-mesh
    #w = np.linspace(-50, 50, 1000)
    w = np.linspace(energymesh[0], energymesh[1], energymesh[2])
    # Each element is a XAS polarization vector.
   # epsilons = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # [[0,0,1]]
    #epsilons = [[0, -1/np.sqrt(2), -1j/np.sqrt(2)], [0, 1/np.sqrt(2), -1j/np.sqrt(2)]]
    epsilons = [[ -1/np.sqrt(2), -1j/np.sqrt(2), 0], [ 1/np.sqrt(2), -1j/np.sqrt(2), 0],[0,0,1]]
    #epsilons = [[ -1/np.sqrt(2)*np.cos(np.pi/180*0), -1j/np.sqrt(2), -1/np.sqrt(2)*np.sin(np.pi/180*0)], [ 1/np.sqrt(2)*np.cos(np.pi/180*0), -1j/np.sqrt(2), 1/np.sqrt(2)*np.sin(np.pi/180*0)],[np.sin(np.pi/180*0),0,np.cos(np.pi/180*0)]]
    #epsilons = [[-1j/np.sqrt(2), 0, 1/np.sqrt(2)], [ -1j/np.sqrt(2), 0, -1/np.sqrt(2)]]
    # RIXS parameters
    # Polarization vectors, of in and outgoing photon.
    epsilonsRIXSin = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # [[0,0,1]]
    epsilonsRIXSout = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # [[0,0,1]]
    if (deltaRIXS > 0):
       #wIn = np.linspace(-25, 25, 1000)
       wIn= np.linspace(energyrixsmesh[0],energyrixsmesh[1],energyrixsmesh[2])
       if rank ==0:
           print("epsilonsRIXSin", epsilonsRIXSin)
           print("epsilonsRIXSout", epsilonsRIXSout)
    else:
       wIn = []
    #wLoss = np.linspace(-25, 25, 1000)
    wLoss = np.linspace(energylossmesh[0],energylossmesh[1],energylossmesh[2])
    # Read RIXS projectors from file
    XAS_projectors = None
    RIXS_projectors = None
    if XAS_projectors_filename:
            XAS_projectors = get_RIXS_projectors(XAS_projectors_filename)
            if rank == 0 : print ("XAS projectors")
            if rank == 0 : print (XAS_projectors)
    if RIXS_projectors_filename:
            RIXS_projectors = get_RIXS_projectors(RIXS_projectors_filename)
            if rank == 0 : print ("RIXS projectors")
            if rank == 0 : print (RIXS_projectors)
    # NIXS parameters
    qsNIXS = [2 * np.array([1, 1, 1]) / np.sqrt(3), 7 * np.array([1, 1, 1]) / np.sqrt(3)]
    # Angular momentum of final and initial orbitals in the NIXS excitation process.
    liNIXS,ljNIXS = 2, 2

    # -- Occupation restrictions for excited states --
    l = ls[1]
    restrictions = {}
    # Restriction on impurity orbitals
   
 
    indices = frozenset(c2i(nBaths, (l, s, m)) for s in range(2) for m in range(-l, l + 1))
    restrictions[indices] = (n0imps[l] - 1, n0imps[l] + dnTols[l] + 1)
    # Restriction on valence bath orbitals
    indices = []
 
    for b in range(nValBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (nValBaths[l] - dnValBaths[l], nValBaths[l])
    # Restriction on conduction bath orbitals
    indices = []
    for b in range(nValBaths[l], nBaths[l]):
        indices.append(c2i(nBaths, (l, b)))
    restrictions[frozenset(indices)] = (0, dnConBaths[l])

    # Read the radial part of correlated orbitals
    if (deltaNIXS > 0):
      # radialMesh, RiNIXS = np.loadtxt(radial_filename).T
      a=np.loadtxt(radial_filename).T
      if a.shape[0]==7:
        radialMesh = a[0]/1.8897259886
        RiNIXS = a[1]*np.sqrt(1.8897259886)/radialMesh
      else:
        radialMesh = a[0]
        RiNIXS = a[1]
    else:
       radialMesh = np.array([0])
       RiNIXS = np.array([0])
    RjNIXS = np.copy(RiNIXS)

    # Total number of spin-orbitals in the system
    n_spin_orbitals = sum(2 * (2 * ang + 1) + nBath for ang, nBath in nBaths.items())
    if rank == 0: print("#spin-orbitals:", n_spin_orbitals)
    if rank == 0: print("#hpField:", hpField)

    # Hamiltonian
    if rank == 0: print('Construct the Hamiltonian operator...')
    hOp = get_hamiltonian_operator(nBaths, nValBaths, [Fdd, Fpp, Fpd, Gpd],
                                   [xi_2p, xi_3d], VP,
                                   [n0imps, doccs, chargeTransferCorrection],
                                   hField, hpField, ls[1], ls[0],
                                   h0_filename,rank, dc_method)
    # Measure how many physical processes the Hamiltonian contains.
    if rank == 0: print('{:d} processes in the Hamiltonian.'.format(len(hOp)))
    # Many body basis for the ground state
    if rank == 0: print('Create basis...')
    basis = finite.get_basis(ls[1], nBaths, nValBaths, dnValBaths, dnConBaths,
                             dnTols, n0imps)
    if rank == 0: print('#basis states = {:d}'.format(len(basis)), flush=True)
    # Diagonalization of restricted active space Hamiltonian
    es, psis = finite.eigensystem(n_spin_orbitals, hOp, basis, nPsiMax, groundDiagMode='Lanczos')

    if rank == 0:
        print("time(ground_state) = {:.2f} seconds \n".format(time.time()-t0))
        t0 = time.time()

    # Calculate static expectation values
    finite.printThermalExpValues(ls[1], nBaths, es, psis)
    finite.printExpValues(ls[1], nBaths, es, psis)

    # Print Slater determinants and weights
    if rank == 0:
        print('Slater determinants/product states and correspoinding weights')
        weights = []
        for i, psi in enumerate(psis):
            print('Eigenstate {:d}.'.format(i))
            print('Consists of {:d} product states.'.format(len(psi)))
            ws = np.array([ abs(a)**2 for a in psi.values() ])
            s = np.array([ ps for ps in psi.keys() ])
            j = np.argsort(ws)
            ws = ws[j[-1::-1]]
            s = s[j[-1::-1]]
            weights.append(ws)
            if nPrintSlaterWeights > 0:
                print('Highest (product state) weights:')
                print(ws[:nPrintSlaterWeights])
                print('Corresponding product states:')
                print(s[:nPrintSlaterWeights])
                print('')

    # Calculate density matrix
    if rank == 0:
        print('Density matrix (in cubic harmonics basis):')
        for i, psi in enumerate(psis):
            print('Eigenstate {:d}'.format(i))
            n = finite.getDensityMatrixCubic(ls[1], nBaths, psi)
            print('#density matrix elements: {:d}'.format(len(n)))
            for e, ne in n.items():
                if abs(ne) > tolPrintOccupation:
                    if e[0] == e[1]:
                        print('Diagonal: (i,s) =',e[0],', occupation = {:7.2f}'.format(ne))
                    else:
                        print('Off-diagonal: (i,si), (j,sj) =',e,', {:7.2f}'.format(ne))
            print('')

    # Save some information to disk
    if rank == 0:
        # Most of the input parameters. Dictonaries can be stored in this file format.
        np.savez_compressed('data', ls=ls, nBaths=nBaths,
                            nValBaths=nValBaths,
                            n0imps=n0imps, doccs=doccs, dnTols=dnTols,
                            dnValBaths=dnValBaths, dnConBaths=dnConBaths,
                            Fdd=Fdd, Fpp=Fpp, Fpd=Fpd, Gpd=Gpd,
                            xi_2p=xi_2p, xi_3d=xi_3d, VP=VP,
                            chargeTransferCorrection=chargeTransferCorrection,
                            hField=hField,
                            hpField=hpField,
                            h0_filename=h0_filename,
                            nPsiMax=nPsiMax,
                            nChiMax=nChiMax,
                            T=T, energy_cut=energy_cut, delta=delta, delta_2=delta_2, POS=POS,
                            restrictions=restrictions,
                            epsilons=epsilons,
                            epsilonsRIXSin=epsilonsRIXSin,
                            epsilonsRIXSout=epsilonsRIXSout,
                            deltaRIXS=deltaRIXS,
                            deltaNIXS=deltaNIXS,
                            n_spin_orbitals=n_spin_orbitals,
                            hOp=hOp)
        # Save some of the arrays.
        # HDF5-format does not directly support dictonaries.
        h5f = h5py.File('spectra.h5','w')
        h5f.create_dataset('E',data=es)
        h5f.create_dataset('w',data=w)
        h5f.create_dataset('wIn',data=wIn)
        h5f.create_dataset('wLoss',data=wLoss)
        h5f.create_dataset('qsNIXS',data=qsNIXS)
        h5f.create_dataset('r',data=radialMesh)
        if (deltaNIXS > 0):
           h5f.create_dataset('RiNIXS',data=RiNIXS)
           h5f.create_dataset('RjNIXS',data=RjNIXS)
    else:
        h5f=None

    if rank == 0:
        print("time(expectation values) = {:.2f} seconds \n".format(time.time()-t0))

    # Consider from now on only eigenstates with low energy
    es = tuple( e for e in es if e - es[0] < energy_cut )
    psis = tuple( psis[i] for i in range(len(es)) )
    if rank == 0: print("Consider {:d} eigenstates for the spectra \n".format(len(es)))

    spectra.simulate_spectra(es, psis, hOp, T, w, delta, delta_2, POS, epsilons,
                             wLoss, deltaNIXS, qsNIXS, liNIXS, ljNIXS, RiNIXS, RjNIXS,
                             radialMesh, wIn, deltaRIXS, epsilonsRIXSin, epsilonsRIXSout,
                             restrictions, h5f, nBaths, ls[1], ls[0], XAS_projectors, RIXS_projectors)

    #print('Script finished for rank:', rank)


def get_hamiltonian_operator(nBaths, nValBaths, slaterCondon, SOCs, VP,
                             DCinfo, hField, hpField, ls1, ls0,
                             h0_filename,rank, dc_method):
    """
    Return the Hamiltonian, in operator form.

    Parameters
    ----------
    nBaths : dict
        Number of bath states for each angular momentum.
    nValBaths : dict
        Number of valence bath states for each angular momentum.
    slaterCondon : list
        List of Slater-Condon parameters.
    SOCs : list
        List of SOC parameters.
    DCinfo : list
        Contains information needed for the double counting energy.
    hField : list
        External magnetic field.
        Elements hx,hy,hz
    hpField : list
        magnetic field from the d-orbitals on the p orbitals
        Elements hpx, hpy, hpz
    h0_filename : str
        Filename of non-interacting, non-relativistic operator.
    VP : float
        Corebinding energy of 2p electrons
    l1 : int
        Orbital quantum number of the higher energy level
    l0 : int
        Orbital quantum number of the lower energy level

    Returns
    -------
    hOp : dict
        The Hamiltonian in operator form.
        tuple : complex,
        where each tuple describes a process of several steps.
        Each step is described by a tuple of the form: (i,'c') or (i,'a'),
        where i is a spin-orbital index.

    """
    # Divide up input parameters to more concrete variables
    Fdd, Fpp, Fpd, Gpd = slaterCondon
    xi_2p, xi_3d = SOCs
    n0imps, doccs, chargeTransferCorrection = DCinfo
    hx, hy, hz = hField
    hpx, hpy, hpz = hpField
    #print('TEST:', n0imps)
    # Calculate the U operator, in spherical harmonics basis.
    uOperator = finite.get2p3dSlaterCondonUop(ls1, Fdd=Fdd, Fpp=Fpp,
                                              Fpd=Fpd, Gpd=Gpd)

    # Add SOC, in spherical harmonics basis.
    SOC2pOperator = finite.getSOCop(xi_2p, ls0)
    SOC3dOperator = finite.getSOCop(xi_3d, ls1)

    # Double counting (DC) correction values.
    # MLFT DC
    if(dc_method == 'MLFT'):
       dc = finite.dc_MLFT(ls1, ls0, n3d_i=doccs, c=chargeTransferCorrection, Fdd=Fdd,
                        n2p_i=n0imps[ls0], Fpd=Fpd, Gpd=Gpd)
       eDCOperator = {}
    else:
       dc =finite.dc_FLL(ls1, ls0, n3d_i=doccs, c=chargeTransferCorrection, Fdd=Fdd,
                           n2p_i=n0imps[ls0], Fpd=Fpd, Gpd=Gpd)
       eDCOperator = {}
    for il, l in enumerate([ls1,ls0]):
        for s in range(2):
            for m in range(-l, l+1):
                eDCOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = -dc[il]

    #core level
    VPOperator={}
    l =ls0 
    for m in range(-l, l+1):
        for s in range(2):
            VPOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = VP
    # Magnetic field
    hHfieldOperator = {}
    l = ls1
    for m in range(-l, l+1):
        hHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] = hx*1/2.
        hHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] = hx*1/2.
        hHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] += -hy*1/2.*1j
        hHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] += hy*1/2.*1j
        for s in range(2):
            hHfieldOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = hz*1/2 if s==1 else -hz*1/2

    hpHfieldOperator = {}
    l = ls0
    for m in range(-l, l+1):
        hpHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] = hpx*1/2.
        hpHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] = hpx*1/2.
        hpHfieldOperator[(((l, 1, m), 'c'), ((l, 0, m), 'a'))] += -hpy*1/2.*1j
        hpHfieldOperator[(((l, 0, m), 'c'), ((l, 1, m), 'a'))] += hpy*1/2.*1j
        for s in range(2):
            hpHfieldOperator[(((l, s, m), 'c'), ((l, s, m), 'a'))] = hpz*1/2 if s==1 else -hpz*1/2

    # Read the non-relativistic non-interacting Hamiltonian operator from file.
    h0_operator = get_h0_operator(h0_filename, nBaths)

    # Add Hamiltonian terms to one operator.
    hOperator = finite.addOps([uOperator,
                               hHfieldOperator,
                               hpHfieldOperator,
                               VPOperator,
                               SOC2pOperator,
                               SOC3dOperator,
                               eDCOperator,
                               h0_operator])
    if (rank == 0): finite.printOp(nBaths,hOperator,"Local Hamiltonian: ") 

    # Convert spin-orbital and bath state indices to a single index notation.
    hOp = {}
    for process,value in hOperator.items():
        hOp[tuple((c2i(nBaths, spinOrb), action) for spinOrb, action in process)] = value
    return hOp


def get_h0_operator(h0_filename, nBaths):
    """
    Return h0 operator.

    Parameters
    ----------
    h0_filename : str
        Filename of non-interacting, non-relativistic operator.
    nBaths : dict
        Number of bath states for each angular momentum.

    Returns
    -------
    h0_operator : dict
        The non-relativistic non-interacting Hamiltonian in operator form.
        Hamiltonian describes 3d orbitals and bath orbitals.
        tuple : complex,
        where each tuple describes a process of two steps (annihilation and then creation).
        Each step is described by a tuple of the form:
        (spin_orb, 'c') or (spin_orb, 'a'),
        where spin_orb is a tuple of the form (l, s, m) or (l, b) or ((l_a, l_b), b).

    """
   # with open(h0_filename, 'rb') as handle:
   #     h0_operator = pickle.loads(handle.read())
    h0_operator = None
    if h0_filename.endswith(".dict"):
            h0_operator = read_h0_dict(h0_filename)
    else:
            with open(h0_filename, 'rb') as handle:
                    h0_operator = pickle.loads(handle.read())
    # Sanity check
    for process in h0_operator.keys():
        for event in process:
            if len(event[0]) == 2:
               try:
                  if (nBaths[event[0][0]] <= event[0][1]):
                     print("Error in h0!")
                     print(process)
                     print(event)
                     print(nBaths[event[0][0]])
                     print(event[0][1])
                  assert nBaths[event[0][0]] > event[0][1]
               except:
                  print("Error in h0!")
                  print(process)
                  print(event)
                  print(event[0][0])
                  print(event[0][1])
                  print(nBaths)
                  raise ValueError("ValueError when construting h0")

    return h0_operator


if __name__== "__main__":
    # Parse input parameters
    parser = argparse.ArgumentParser(description='Spectroscopy simulations')
    parser.add_argument('h0_filename', type=str,
                        help='Filename of non-interacting Hamiltonian.')
    parser.add_argument('radial_filename', type=str,
                        help='Filename of radial part of correlated orbitals.')
    parser.add_argument('--ls', type=int, nargs='+', default=[1, (2,3)],
                        help='Angular momenta of correlated orbitals.')
    parser.add_argument('--nBaths', type=int, nargs='+', default=[0, 10],
                        help='Number of bath states, for each angular momentum.')
    parser.add_argument('--nValBaths', type=int, nargs='+', default=[0, 10],
                        help='Number of valence bath states, for each angular momentum.')
    parser.add_argument('--n0imps', type=int, nargs='+', default=[6, 8],
                        help='Initial impurity occupation, for each angular momentum.')
    parser.add_argument('--doccs', type=float, default=-5.0,
                        help='Occupation of the d-states according to RSPT.')
    parser.add_argument('--dnTols', type=int, nargs='+', default=[0, 2],
                        help=('Max devation from initial impurity occupation, '
                              'for each angular momentum.'))
    parser.add_argument('--dnValBaths', type=int, nargs='+', default=[0, 2],
                        help=('Max number of electrons to leave valence bath orbitals, '
                              'for each angular momentum.'))
    parser.add_argument('--dnConBaths', type=int, nargs='+', default=[0, 0],
                        help=('Max number of electrons to enter conduction bath orbitals, '
                              'for each angular momentum.'))
    parser.add_argument('--Fdd', type=float, nargs='+', default=[7.5, 0., 9.9, 0., 6.6, 0., 0.],
                        help='Slater-Condon parameters Fdd. d-orbitals are assumed.')
    parser.add_argument('--Fpp', type=float, nargs='+', default=[0., 0., 0., 0., 0.],
                        help='Slater-Condon parameters Fpp. p-orbitals are assumed.')
    parser.add_argument('--Fpd', type=float, nargs='+', default=[8.9, 0., 6.8, 0., 0.],
                        help='Slater-Condon parameters Fpd. p- and d-orbitals are assumed.')
    parser.add_argument('--Gpd', type=float, nargs='+', default=[0., 5., 0., 2.8, 0., 0.],
                        help='Slater-Condon parameters Gpd. p- and d-orbitals are assumed.')
    parser.add_argument('--xi_2p', type=float, default=11.629,
                        help='SOC value for p-orbitals. p-orbitals are assumed.')
    parser.add_argument('--VP' , type=float, default=0.0,
                        help='corebinding energies from DFT')
    parser.add_argument('--xi_3d', type=float, default=0.096,
                        help='SOC value for d-orbitals. d-orbitals are assumed.')
    parser.add_argument('--chargeTransferCorrection', type=float, default=1.5,
                        help='Double counting parameter.')
    parser.add_argument('--hField', type=float, nargs='+', default=[0, 0, 0.0001],
                        help='Magnetic field. (h_x, h_y, h_z)')
    parser.add_argument('--hpField', type=float, nargs='+', default=[0, 0, 0.0001],
                        help='magnetic p-field. (h_px, h_py, h_pz')
    parser.add_argument('--nPsiMax', type=int, default=5,
                        help='Maximum number of eigenstates to consider.')
    parser.add_argument('--nChiMax', type=int, default=3,
                        help='Maximum number of eigenstates in p.')
    parser.add_argument('--nPrintSlaterWeights', type=int, default=3,
                        help='Printing parameter.')
    parser.add_argument('--tolPrintOccupation', type=float, default=0.5,
                        help='Printing parameter.')
    parser.add_argument('--T', type=float, default=300,
                        help='Temperature (Kelvin).')
    parser.add_argument('--energy_cut', type=float, default=10,
                        help='How many k_B*T above lowest eigenenergy to consider.')
    parser.add_argument('--delta', type=float, default=0.2,
                        help=('Smearing, half width half maximum (HWHM). '
                              'Due to short core-hole lifetime.'))
    parser.add_argument('--delta_2', type=float, default=0.01,
                        help=('additional smearing of L2 edge.'
                              'due to additional decay channel.'))
    parser.add_argument('--POS', type=float, default=6,
                        help=('Position where additional broadening for L2 should begin.'
                          'due to additional decay channel.'))
    parser.add_argument('--deltaRIXS', type=float, default=0.050,
                        help=('Smearing, half width half maximum (HWHM). '
                              'Due to finite lifetime of excited states.'))
    parser.add_argument('--deltaNIXS', type=float, default=-0.100,
                        help=('Smearing, half width half maximum (HWHM). '
                              'Due to finite lifetime of excited states.'))
    parser.add_argument('--XAS_projectors_filename', type=str, default=None,
                       help=('File containing    the XAS projectors. Separated by new lines.'))
    parser.add_argument('--RIXS_projectors_filename', type=str, default=None,
                        help=('File containing the RIXS projectors. Separated by newlines.'))
    parser.add_argument('--dc_method', type=str, default='MLFT',
                        help=('Double Counting method.'))
    parser.add_argument('--energymesh', type=int, nargs='+', default=[-25, 25, 1000],
                        help=('Energy mesh for plotting the XAS,XPS and PS.'))
    parser.add_argument('--energylossmesh', type=int, nargs='+', default=[-25, 25, 1000],
                        help=('Energy loss mesh for plotting the RIXS and NIXS'))
    parser.add_argument('--energyrixsmesh', type=int, nargs='+', default=[-25, 25, 1000],
                        help=('Energy mesh for plotting the XAS,XPS and PS.'))
    args = parser.parse_args()

    # Sanity checks
    assert len(args.ls) == len(args.nBaths)
    assert len(args.ls) == len(args.nValBaths)
    if args.doccs < 0:
        args.doccs=args.n0imps[1]
    for nBath, nValBath in zip(args.nBaths, args.nValBaths):
        assert nBath >= nValBath
    for ang, n0imp in zip(args.ls, args.n0imps):
        assert n0imp <= 2 * (2 * ang + 1)  # Full occupation
        assert n0imp >= 0
    assert len(args.Fdd) == 7
    assert len(args.Fpp) == 5
    assert len(args.Fpd) == 5
    assert len(args.Gpd) == 6
    assert len(args.hField) == 3
    assert len(args.hpField) == 3
    #comm = MPI.COMM_WORLD
    #rank = comm.rank
    #print("My rank: ",rank, flush=True)

    main(h0_filename=args.h0_filename,
         radial_filename=args.radial_filename,
         ls=tuple(args.ls), nBaths=tuple(args.nBaths),
         nValBaths=tuple(args.nValBaths), n0imps=tuple(args.n0imps),
         doccs=args.doccs,
         dnTols=tuple(args.dnTols), dnValBaths=tuple(args.dnValBaths),
         dnConBaths=tuple(args.dnConBaths),
         Fdd=tuple(args.Fdd), Fpp=tuple(args.Fpp),
         Fpd=tuple(args.Fpd), Gpd=tuple(args.Gpd),
         xi_2p=args.xi_2p, xi_3d=args.xi_3d, VP=args.VP,
         chargeTransferCorrection=args.chargeTransferCorrection,
         hField=tuple(args.hField), nPsiMax=args.nPsiMax,
         hpField=tuple(args.hpField), nChiMax=args.nChiMax,
         nPrintSlaterWeights=args.nPrintSlaterWeights,
         tolPrintOccupation=args.tolPrintOccupation,
         T=args.T, energy_cut=args.energy_cut,
         delta=args.delta, delta_2=args.delta_2, POS=args.POS, deltaRIXS=args.deltaRIXS, deltaNIXS=args.deltaNIXS,
         XAS_projectors_filename=args.XAS_projectors_filename,
         RIXS_projectors_filename=args.RIXS_projectors_filename, dc_method=args.dc_method,
         energymesh=args.energymesh, energylossmesh=args.energylossmesh, energyrixsmesh=args.energyrixsmesh)


