"""
pe.py contains a library of functions useful for
polyelectrolyte (pe) solutions. Functions specific
to polyelectrolyte solutions with salt are contained
in salt.py.
"""

import scipy.optimize
import numpy as np

# CONSTANTS
m3_2_L = 1E3
NA = 6.022E23



def calc_rho_solv(rho_p_list, rho_s_list, beads_2_M):
    """Computes the mol/L concentration of solvent assuming uniform bead size."""
    # calculates volume fractions of each component
    phi_p_arr = M_2_phi(rho_p_list, beads_2_M) # [vol/vol]
    phi_s_arr = M_2_phi(rho_s_list, beads_2_M) # [vol/vol]
    phi_solv_arr = 1 - 2*(phi_p_arr + phi_s_arr) # 2* for 2 components
    assert np.all(phi_solv_arr >= 0), 'negative phi_solv in calc_rho_solv.'

    return phi_2_M(phi_solv_arr, beads_2_M)


def eps_h2o(T):
    """Dielectric constant of water near 0 C (273 K). Eqn 4 of Adhikari et al.
    Macromol. (2019)."""
    return 87.740 - 0.40008*(T-273) + 9.398E-4*(T-273) - 1.410E-6*(T-273)


def get_beads_2_M(sigma, SI=False):
    """
    Computes conversion from beads/sigma^3 to mol/L.

    If SI True, sigma is expected in meters. Otherwise, in Angstroms.
    """
    # Avogadro's number [molecules/mol]
    NA = 6.022E23
    # conversion of meters^3 to L
    m3_2_L = 1E3
    # conversion of meters to Angstroms [A]
    A_2_m = 1E-10
    if not SI:
        sigma *= A_2_m # converts from A to m

    # conversion from beads/sigma^3 to mol/L (M)
    beads_2_M = (NA * sigma**3 * m3_2_L)**(-1)

    return beads_2_M


def get_naming_structure(N, lB, f=None, lB_dig=3):
    """Produces naming structure for folders."""
    # creates string for N specification
    if str(N) == '*':
        N_str = 'NA(*)NB(*)'
    else:
        N_str =  'NA({0:d})NB({0:d})'.format(N)

    # creates specific string for Bjerrum length specification
    if str(lB) == '*':
        lB_str = 'lB(*)'
    elif lB_dig == 3:
        lB_str = 'lB({0:.3f})'.format(lB)
    elif lB_dig == 2:
        lB_str = 'lB({0:.2f})'.format(lB)
    else:
        print('Please code option for lB with {0:d} digits past decimal'.format(lB_dig))

    # no charge fraction specified
    if f is None:
        f_str = ''
    elif str(f) == '*':
        f_str = 'f(*)'
    # charge fraction specified
    else:
        f_str = 'f({0:.3f})'.format(f)

    # creates naming structure
    naming_structure = N_str + lB_str + f_str

    return naming_structure


def lB_2_T(lB, T0=298, sigma=4E-10, ret_res=False):
    """Solves for temperature at given Bjerrum length under condition from Adhikari et al. 2019 that lB/l = 1.2 at 298 K."""
    def cond(T, lB, sigma=sigma):
        """condition function whose root gives the temperature T given Bjerrum length lB."""
        return lB_fn(T, sigma=sigma) - lB

    T = scipy.optimize.fsolve(cond, T0, args=(lB,))[0]

    if ret_res:
        res = np.abs(lB_fn(T, sigma=sigma) - lB)
        return T, res

    return T


def lB_2_T_arr(lB_arr, T_range, fix_eps=False, sigma=4E-10):
    """Same as lB_2_T but converts an array of Bjerrum lengths to a corresponding array of temperatures."""
    T_test_arr = np.linspace(*T_range, 100)
    lB_test_arr = np.array([lB_fn(T, sigma=sigma) for T in T_test_arr])
    T0_arr = np.interp(lB_arr, lB_test_arr, T_test_arr)
    T_arr = np.array([lB_2_T(lB, T0=T0_arr[i], sigma=sigma)  \
                        for i, lB in enumerate(lB_arr)])

    if fix_eps:
        e = 1.602E-19 # charge on electron [C]
        eps0 = 8.85E-12 # permittivity of free space [A^2.s^4 / m^3.kg]
        kB = 1.38065E-23 # Boltzmann constant [J/K]
        eps_h2o = 80
        T_arr = e**2 / (4*np.pi*eps0*eps_h2o*kB*lB_arr) / sigma

    return T_arr


def lB_fn(T, sigma=4E-10):
    """
    Computes the Bjerrum length based on Adhikari et al. 2019 Fig. 4
    (lB/l = 1.2 at 298 K).
    If sigma (characteristic length scale [m]) is provided, the Bjerrum length
    is computed explicitly.
    """
    e = 1.602E-19 # charge on electron [C]
    eps0 = 8.85E-12 # permittivity of free space [A^2.s^4 / m^3.kg]
    kB = 1.38065E-23 # Boltzmann constant [J/K]
    if sigma is None:
        return 2.78075E4 / (eps_h2o(T)*T)
    else:
        return e**2 / (4*np.pi*eps0*eps_h2o(T)*kB*T) / sigma


def M_2_phi(rho_list, beads_2_M):
    """Computes the volume fraction phi from a density in mol beads / L."""
    return np.array(rho_list) * (np.pi/6) / beads_2_M

def phi_2_M(phi_list, beads_2_M):
    """Computes the density in mol beads / L given volume fraction phi."""
    return np.array(phi_list) * beads_2_M / (np.pi/6)
