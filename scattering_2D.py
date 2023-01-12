import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from formfactors import f_params
import itertools
from collections import Counter

def two_theta2Qxy(wavelength, two_theta):
    return 4*np.pi*np.sin(np.deg2rad(two_theta/2))/wavelength


def export_results_csv(result, filename='results.csv', full=False, delimiter=';'):

    if full:
        headernames = ['Qxy_all', 'intensity_all', 'hk_all']
    elif not full:
        headernames = ['Qxy_unique', 'intensity_unique', 'multiplicity']

    with open(filename, 'w') as csvfile:
        for h in headernames:
            csvfile.write(h)
            csvfile.write(delimiter)
        csvfile.write('\n')

        for n in range(len(result[headernames[0]])):
            for h in headernames:
                csvfile.write(str(result[h][n]))
                csvfile.write(delimiter)
            csvfile.write('\n')



def get_reflexes(atoms, dummy=False, hs=range(-4, 5), ks=range(-4, 5), Qmin=1.0):
    """Calculates all reflexes of the atom objects within the range of hs and ks
    Input:
        atoms: ase atoms object
        dummy (bool): If true, then instead of real form factors, 
                      dummy form factors that are different for each site are used
        hs: miller h indices to screen
        ks: miller k indices to screen
        Qmin: minimal absolute value of Q-vector (Angstrom^-1)


    Returns:
        dict: dictionary containing all the results
    """

    a, b, _, _, _, gamma = atoms.cell.cellpar()
    
    fract_coord_list = atoms.get_scaled_positions()
    chem_sym = atoms.get_chemical_symbols()
    
    Qxy_reflexes = []
    intensities = []
    hk_list = []

    for hk in itertools.product(hs, ks):
        h, k = hk
        Fhkl = 0
        Qxy = calc_Qxy(a, b, gamma, h, k)

        for n, c_frac in enumerate(zip(chem_sym, fract_coord_list)):
            c, frac = c_frac
            if dummy:
                f= (1/(n+1))  # 1/(n+1) to make sites distiguishable
            else:
                f = form_factor(c, Qxy)
            Fhkl += np.exp(-2*np.pi*1j*(h*frac[0] + k*frac[1])) * f

        intensity = np.real(Fhkl * np.conj(Fhkl))  # calculate intensity

        if intensity>1e-5:
            if Qxy>Qmin:
                Qxy_reflexes.append(Qxy)
                intensities.append(intensity)
                hk_list.append([h, k])

    Qxys_unique = []
    mult = []
    hk_unique = []
    intensities_unique = []
    indlist = []
    for n, Qxy in enumerate(Qxy_reflexes):
        inds = np.where(abs(np.array(Qxy_reflexes)-Qxy)<=1e-5)[0]
        
        if n not in indlist:
            Qxys_unique.append(Qxy)
            intensities_unique.append(np.sum(np.array(intensities)[inds]))
            mult.append(len(inds))
            hk_unique.append(np.array(hk_list)[inds])

        indlist.extend(inds)


    result = dict()

    result['Qxy_all'], result['intensity_all'], result['hk_all'] = zip(*sorted(zip(Qxy_reflexes, np.array(intensities)/max(intensities_unique), hk_list)))
    result['Qxy_unique'], result['intensity_unique'], result['hk_unique'], result['multiplicity'] = zip(*sorted(zip(Qxys_unique, np.array(intensities_unique)/max(intensities_unique), hk_unique, mult)))

    return result


def plot_result_gixd(result, reflections_exp, Qrange=[1.0,3.2]):
    fig, ax = plt.subplots(1,1)
    ax.vlines(np.array(result['Qxy_unique']), 0, result['intensity_unique'], color='darkorange', linewidth=5, alpha=1)
    ax.vlines(np.array(reflections_exp), 0, 1, color='darkslategray', linestyle='--')

    ax.set_xlim(Qrange)
    ax.set_ylim([0,1])

    ax.set_xlabel('Q$_\mathrm{xy}$ ($\mathrm{\AA}^{-1}$)')
    ax.set_ylabel('Relative intensity')

    plt.legend(['Simulation', 'Experiment'])
    
    plt.tight_layout()

    return fig, ax

def calc_2D_recip_vecs(a1, a2):
    """
    Converts lattice vectors a1 and a2 into their reciprocal vectors
    """
    a1x, a1y = a1
    a2x, a2y = a2
    # calculation of reciprocal vectors
    A = np.array([[a2y, -a1y], [-a2x, a1x]])
    B = 2*np.pi/(a1x*a2y - a1y*a2x) * A
    
    b1 = np.array([B[0][0], B[1][0]])
    b2 = np.array([B[0][1], B[1][1]])

    return b1, b2

def calc_2D_vecs(a, b, gamma):
    """
    Calculates the 2D lattice vectors based on the cell parameters a, b, gamma
    """
    # real-space vectors
    a1x = a
    a1y = 0
    a2x = np.cos(np.deg2rad(gamma)) * b
    a2y = np.sin(np.deg2rad(gamma)) * b

    return [a1x, a1y], [a2x, a2y]

def calc_Qxy(a, b, gamma, h, k):
    """
    Converts 2D lattice information and miller indices h,k into
    the in-plane components of scattering vector Qxy.
    """
    
    # real-space vectors
    a1, a2 = calc_2D_vecs(a, b, gamma)
    
    # calculation of reciprocal vectors
    b1, b2 = calc_2D_recip_vecs(a1, a2)

    # calculaton of norm of scattering vector
    Qxy = np.linalg.norm(h*b1 + k*b2)

    return Qxy
    

def form_factor(atomsymol, qabs):
    """Calculates the form_factor:
    f(qabs) = sum_{i=1}^{4} a_i * (-b_i (qabs/4pi)^2) + c

    Args:
        atomsymol (str): Symbol of Atom
        qabs (float): Norm of scattering vector Qxy

    Returns:
        float : form factor
    """
    a_params = f_params[atomsymol]['a_params']
    b_params = f_params[atomsymol]['b_params']
    c = f_params[atomsymol]['c']
    f = 0
    for a, b in zip(a_params, b_params):
        f += a * np.exp(-b*(qabs/(4*np.pi))**2)
    return f + c