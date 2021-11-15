"""
salt.py contains a library of functions used for processing
results from the CCLS code for a polyelectrolyte solution with
salt that was developed by Pengfei Zhang of the Wang group.
Some guidance and suggestions were provided by Chris Balzer.
Author: Andy Ylitalo
Date created: 4/23/20
"""

import pe
import plot

import os
import numpy as np
import glob
import pandas as pd
import scipy.optimize


# CONSTANTS
NA = 6.022E23 # Avogadro's number, molecules / mol
m3_2_L = 1E3
K_2_C = 273.15 # conversion from Kelvin to Celsius (subtract this)
L_2_mL = 1000
m_2_A = 1E10 # multiplicative factor to convert from meters to Angstroms


def binodal_vary_f_data(data_folder, mu_salt_folder, rho_s_M, f_list,
                    T_cels=True, sigma=4E-10, ext='.PD', N=100,
                    T_range=[273.15, 373.15]):
    """
    Computes data to plot with plot.binodal_vary_f().
    """
    data_f = {}
    # converts mol/L to beads/sigma^3
    beads_2_M = pe.get_beads_2_M(sigma, SI=True)
    rho_s = rho_s_M / beads_2_M

    for f in f_list:
        # loads full data
        naming_structure = pe.get_naming_structure(N, '*', f=f)
        data = load_data(data_folder, naming_structure=naming_structure,
                                ext=ext)

        # gets data at fixed saltwater reservoir concentration
        _, _, T_arr, rho_p_I, rho_p_II, = get_fixed_conc_data(data,
                                mu_salt_folder, rho_s, sigma, ext=ext,
                                T_range=T_range, T_cels=T_cels)
        # stores data
        data_f[f] = T_arr, rho_p_I, rho_p_II

    return data_f


def binodal_vary_N_data(data_folder, mu_salt_folder, rho_s_M, N_list,
                        T_cels=True, sigma=4E-10, ext='.PD', f=1,
                        T_range=[273.15, 373.15]):
    """
    Gets data to plot with plot.binodal_vary_N()
    """
    data_N = {}
    # converts mol/L to beads/sigma^3
    beads_2_M = pe.get_beads_2_M(sigma, SI=True)
    rho_s = rho_s_M / beads_2_M

    for N in N_list:
        # loads full data
        naming_structure = pe.get_naming_structure(N, '*', f=f)
        data = load_data(data_folder, naming_structure=naming_structure,
                                ext=ext)
        # gets data at fixed saltwater reservoir concentration
        _, _, T_arr, rho_p_I, rho_p_II, = get_fixed_conc_data(data,
                                mu_salt_folder, rho_s, sigma, ext=ext,
                                T_range=T_range, T_cels=T_cels)
        data_N[N] = [T_arr, rho_p_I, rho_p_II]

    return data_N


def binodal_vary_sigma_data(data, mu_salt_folder, rho_s_M, sigma_list,
                        T_cels=True, N=100, ext='.PD', f=1,
                        T_range=[273.15, 373.15]):
    """
    Computes data to plot with `plot.binodal_vary_sigma()`.
    """
    data_sigma = {}
    for sigma in sigma_list:
        # converts mol/L to beads/sigma^3
        beads_2_M = pe.get_beads_2_M(sigma, SI=True)
        rho_s = rho_s_M / beads_2_M

        # gets data at fixed saltwater reservoir concentration
        _, _, T_arr, rho_p_I, rho_p_II, = get_fixed_conc_data(data,
                                mu_salt_folder, rho_s, sigma, ext=ext,
                                T_range=T_range, T_cels=T_cels)
        # stores data
        data_sigma[sigma] = [T_arr, rho_p_I, rho_p_II]

    return data_sigma


def compute_rho_s(df, rho_p, beads_2_M):
    """
    Computes the overall concentration of salt and volume fraction of
    supernatant alpha.
    """
    # multiplies polycation density by 2 to include polyanion density
    df['alpha'] = (rho_p/beads_2_M - 2*df['rhoPCII'].to_numpy(dtype=float)) / \
                    (2*df['rhoPCI'].to_numpy(dtype=float) - 2*df['rhoPCII'].to_numpy(dtype=float))
    df['rhoS'] = df['alpha'].to_numpy(dtype=float)*df['rhoCI'].to_numpy(dtype=float) + \
                    (1-df['alpha'].to_numpy(dtype=float))*df['rhoCII'].to_numpy(dtype=float)
    df.dropna(inplace=True)

    return df


def conv_ali_conc(df_conv, rho_p, rho_s):
    """
    Converts overall concentrations of polyelectrolyte and salt based on the
    conversion already computed in the dataframe.

    Parameters
    ----------
    df_conv : Pandas dataframe
        Dataframe of experimental measurements of complex coacervate phase
        coexistence, including converted values of Ali et al. data (computed
        within the dataframe using df_conv_ali_conc())
    rho_p, rho_s : float
        concentration of the desired compound [mol (solute)/L (solvent)]
        (rho_p is polyelectrolyte, rho_s is salt)

    Returns
    -------
    rho_p_conv, rho_s_conv : float
        Converted value of the given concentration [mol (solute)/L (solution)],
        rho_p_conv is polyelectrolyte, rho_s_conv is salt
    """
    cond = (df_conv['rho_p [M]'] == rho_p) & (df_conv['rho_s [M]'] == rho_s)
    rho_p_conv = df_conv['rho_p (conv) [M]'].loc[cond].iloc[0]
    rho_s_conv = df_conv['rho_s (conv) [M]'].loc[cond].iloc[0]

    return rho_p_conv, rho_s_conv


def df_conv_ali_conc(df_exp, mL_per_mol_pe=80, mw_salt=119, density_salt=2.76,
                    simple_red=1.0):
    """
    Converts overall polymer and salt concentrations reported in Ali et al.
    (2019) from mol (solute) / L (solvent) to mol (solute) / L (solution) based
    on Dr. Yuanchi Ma's comment that they reported the concentrations in the
    former units.

    Parameters
    ----------
    df_exp : Pandas dataframe
        Dataframe of experimental measurements with the following headers:
        rho_p [M], rho_s [M], T [C], rho_p^sup [M], sigma_p^sup [M],
        rho_p^co [M], sigma_p^co [M], first author
    mL_per_mol_pe : float, optional
        Volume [mL] added to an aqueous solution upon dissolving 1 mole of the
        polyelectrolyte. Dr. Yuanchi Ma reported 80 mL increase in volume of
        920 mL of H2O upon dissolving 1 mol of polyelectrolyte (PSS + PDADMA).
    mw_salt : float, optional
        Molecular weight of the salt. In this case, KBr was used, which has a
        molecular weight of 119 g/mL.
    density_salt : float, optional
        Mass density of the salt. In this case, KBr was used, which has a mass
        density of 2.76 g/mL.
    simple_red : float, optional
        If not 1, does not convert based on densities, but instead simply
        reduces the values by the factor given. Default is 1.

    Returns
    -------
    df_conv : Pandas dataframe
        Returns the same dataframe with converted values recorded in additional
        columns:
    """
    # extracts data from Ma et al.
    df_ma = df_exp.loc[df_exp['first author'] == 'ma']
    df_ali = df_exp.loc[df_exp['first author'] == 'ali']

    # corrects measurements from Ali et al (2019)
    rho_p_ali = df_ali['rho_p [M]'].to_numpy(dtype=float)
    rho_s_ali = df_ali['rho_s [M]'].to_numpy(dtype=float)

    if simple_red != 1:
        rho_p_ali_conv = simple_red*rho_p_ali
        rho_s_ali_conv = simple_red*rho_s_ali
    else:
        # computes volume of entire solution (Ali et al. just reported solvent
        # volume acc. to Dr. Yuanchi Ma)
        V = solv_2_soln_vol(rho_p_ali, rho_s_ali, mL_per_mol_pe=mL_per_mol_pe,
                                mw_salt=mw_salt, density_salt=density_salt)

        # converts concentrations from mol / L (solvent) to mol / L (solution)
        rho_p_ali_conv = rho_p_ali / V
        rho_s_ali_conv = rho_s_ali / V

    # stores corrected values separately
    df_ali['rho_p (conv) [M]'] = rho_p_ali_conv
    df_ali['rho_s (conv) [M]'] = rho_s_ali_conv
    # no conversions for Ma et al. data
    df_ma['rho_p (conv) [M]'] = df_ma['rho_p [M]']
    df_ma['rho_s (conv) [M]'] = df_ma['rho_s [M]']

    # recombines data and displays result
    df_conv = pd.concat([df_ali, df_ma])

    return df_conv


def extract_cp_data(data, beads_2_M, z_name, T_range=[273, 373], sigma=4E-10, i_cp=-1):
    """
    Extracts data for critical point (polymer, salt, and z coordinates).
    """
    data = np.asarray([[beads_2_M*2*df['rhoPCI'].iloc[i_cp],
                beads_2_M*df['rhoCI'].iloc[i_cp],
                df['BJ'].iloc[i_cp]] for df in data.values()])
    # converts Bjerrum length to temperature as z-coordinate
    if z_name == 'T [K]':
        lB_arr = data[:, 2]
        data[:, 2] = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma)
    elif z_name == 'T [C]':
        lB_arr = data[:, 2]
        data[:, 2] = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma) - K_2_C
    elif z_name != 'BJ':
        print('z_name not recognized. Expected ''T [K]'', ''T [C]'', or ''BJ''')

    return np.transpose(data)


def extract_df_mu_data(df_mu, z_name):
    """
    Extracts data from dataframe for binodal at constant chemical potential,
    salt reservoir concentration, etc. (just needs to be 2D).
    """
    # extracts data for plotting binodal at fixed salt concentration
    # phase I (supernatant)
    x_mu1 = 2*(df_mu['rhoPA'].loc[df_mu['phase']=='I'].to_numpy(dtype=float))
    y_mu1 = df_mu['rhoA'].loc[df_mu['phase']=='I'].to_numpy(dtype=float)
    z_mu1 = df_mu[z_name].loc[df_mu['phase']=='I'].to_numpy(dtype=float)
    # phase II (coacervate)
    x_mu2 = 2*(df_mu['rhoPA'].loc[df_mu['phase']=='II'].to_numpy(dtype=float))
    y_mu2 = df_mu['rhoA'].loc[df_mu['phase']=='II'].to_numpy(dtype=float)
    z_mu2 = df_mu[z_name].loc[df_mu['phase']=='II'].to_numpy(dtype=float)

    return (x_mu1, y_mu1, z_mu1, x_mu2, y_mu2, z_mu2)


def extract_df_slice_data(df_slice, z_name):
    """
    Extracts data from dataframe for subset of full 3D binodal for plotting.
    """
    # extracts data for plotting full binodal
    # phase I (supernatant)
    x1 = 2*(df_slice['rhoPA'].loc[df_slice['phase']=='I'].to_numpy(dtype=float))
    y1 = df_slice['rhoA'].loc[df_slice['phase']=='I'].to_numpy(dtype=float)
    z1 = df_slice[z_name].loc[df_slice['phase']=='I'].to_numpy(dtype=float)
    # phase II (coacervate)
    x2 = 2*(df_slice['rhoPA'].loc[df_slice['phase']=='II'].to_numpy(dtype=float))
    y2 = df_slice['rhoA'].loc[df_slice['phase']=='II'].to_numpy(dtype=float)
    z2 = df_slice[z_name].loc[df_slice['phase']=='II'].to_numpy(dtype=float)

    return (x1, y1, z1, x2, y2, z2)


def extract_fixed_comp(data, rho_p, rho_s, beads_2_M, z_name, T_range, sigma=4E-10):
    """
    Extracts and formats data for coexisting phases at different temperatures
    for a fixed overall composition of polymer (rho_p) and salt (rho_s), given
    in mol/L.
    """
    results = fixed_rho_total(data, rho_p, rho_s, beads_2_M)
    rho_PCI_list = results['rho_PCI']
    rho_PCII_list = results['rho_PCII']
    rho_CI_list = results['rho_CI']
    rho_CII_list = results['rho_CII']
    lB_arr = results['lB']
    alpha = results['alpha']
    # converts Bjerrum length to temperature as z-coordinate
    if z_name == 'T [K]':
        z_arr = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma)
    elif z_name == 'T [C]':
        z_arr = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma) - K_2_C
    elif z_name == 'BJ':
        z_arr = lB_arr
    else:
        print('z_name not recognized. Expected ''T [K]'', ''T [C]'', or ''BJ''')

    # group data into format x1, y1, z1, x2, y2, z2
    rho_p_I = 2*np.asarray(rho_PCI_list)
    rho_p_II = 2*np.asarray(rho_PCII_list)
    rho_s_I = np.asarray(rho_CI_list)
    rho_s_II = np.asarray(rho_CII_list)
    data_comp = (rho_p_I, rho_s_I, z_arr, rho_p_II, rho_s_II, z_arr)

    return data_comp


def extract_fixed_y(data, y_rough, fit_fixed_y=True):
    """
    Extracts data for a fixed z value (usually either Bjerrum length or T).
    """
    # splits up data
    x1_coll, y1_coll, z1_coll, x2_coll, y2_coll, z2_coll = data

    # initializes output
    x_y1 = []
    y_y1 = []
    z_y1 = []
    x_y2 = []
    y_y2 = []
    z_y2 = []

    # loops through z since data are organized in planes of constant z
    for z in z1_coll:
        x_z1, y_z1, z_z1, x_z2, y_z2, z_z2 = extract_fixed_z(data, z)
        # finds indices for points with closest y values to fixed value
        i_y1 = np.argmin(np.abs(y_z1 - y_rough))
        i_y2 = np.argmin(np.abs(y_z2 - y_rough))

        # adds points to output
        # supernatant
        x_y1 += [x_z1[i_y1]]
        y_y1 += [y_z1[i_y1]]
        z_y1 += [z_z1[i_y1]]
        # coacervate
        x_y2 += [x_z2[i_y2]]
        y_y2 += [y_z2[i_y2]]
        z_y2 += [z_z2[i_y2]]

    if fit_fixed_y:
        # quadratic fit results in too sharp a curve at the top--try linear
        # linear misses the contours -- try quadratic but skipping the top

        # fits x and z data to remove noise
        x2_fit = np.linspace(np.min(x_y1), np.max(x_y2), len(x_y1))
        a2, b2, c2 = np.polyfit(x_y2[:-5], z_y2[:-5], 2)
        z2_fit = a2*x2_fit**2 + b2*x2_fit + c2
        # sets values to fitted values
        x_y2 = x2_fit
        z_y2 = z2_fit

        # x values of supernatant are extremely small, so just set to zero
        x_y1 = np.zeros([len(x_y1)])
        y_y1 = y_rough*np.ones([len(x_y1)])

    return x_y1, y_y1, z_y1, x_y2, y_y2, z_y2


def extract_fixed_z(data, z_rough):
    """
    Extracts data for a fixed z value (usually either Bjerrum length or T).
    """
    x1_coll, y1_coll, z1_coll, x2_coll, y2_coll, z2_coll = data
    # finds temperature nearest desired temperature
    i_z = np.argmin(np.abs(z1_coll-z_rough))
    z = z1_coll[i_z]

    x_z1 = x1_coll[z1_coll==z]
    y_z1 = y1_coll[z1_coll==z]
    z_z1 = z1_coll[z1_coll==z]
    x_z2 = x2_coll[z2_coll==z]
    y_z2 = y2_coll[z2_coll==z]
    z_z2 = z2_coll[z2_coll==z]

    return (x_z1, y_z1, z_z1, x_z2, y_z2, z_z2)


def extract_outline_data(data_3d, fixed_y_vals, fixed_z_vals):
    """
    Extracts data for plotting outlines of binodal surface.
    """
    # initializes list of datasets
    data_outlines = []
    # collects data for fixed y (typically y = 0)
    for y in fixed_y_vals:
        data_y = extract_fixed_y(data_3d, y)
        data_outlines += [data_y]
    # collects data for fixed z (typically max and min temperature)
    for z in fixed_z_vals:
        data_z = extract_fixed_z(data_3d, z)
        x1, y1, z1, x2, y2, z2 = data_z

        # connects branches 1 and 2
        # gets point nearest critical point from coacervate branch
        i_2 = np.argmax(y2)
        # finds point from supernatant branch nearest previous point
        i_1 = np.argmin( (np.asarray(x1) - x2[i_2])**2 + \
                        (np.asarray(y1) - y2[i_2])**2 + \
                        (np.asarray(z1) - z2[i_2])**2 )
        # sets nearest point on coacervate branch to the value at the supernatant
        x2[i_2] = x1[i_1]
        y2[i_2] = y1[i_1]
        z2[i_2] = z1[i_1]

        # adds outline data to the list
        data_outlines += [data_z]

    # sets last values on outlines to the nearest values on z outlines
    for i, y in enumerate(fixed_y_vals):
        data_y = data_outlines[i]
        x_y1, y_y1, z_y1, x_y2, y_y2, z_y2 = data_y
        for j, z in enumerate(fixed_z_vals):
            data_z = data_outlines[len(fixed_y_vals) + j]
            x_z1, y_z1, z_z1, x_z2, y_z2, z_z2 = data_z

            # finds index of y data corresponding to fixed z
            i_z = np.argmin(np.abs(np.asarray(z_y1) - z))
            # finds index of z data corresponding to fixed y for supernatant
            # and coacervate
            j_z1 = np.argmin(np.abs(np.asarray(y_z1) - y))
            j_z2 = np.argmin(np.abs(np.asarray(y_z2) - y))
            # assigns points on fixed y outline to nearest points on fixed
            # z outline for a continuous outline
            x_y1[i_z] = x_z1[j_z1]
            y_y1[i_z] = y_z1[j_z1]
            z_y1[i_z] = z_z1[j_z1]
            x_y2[i_z] = x_z2[j_z2]
            y_y2[i_z] = y_z2[j_z2]
            z_y2[i_z] = z_z2[j_z2]

        # stores modified y data
        data_outlines[i] = (x_y1, y_y1, z_y1, x_y2, y_y2, z_y2)

    return data_outlines


def fixed_rho_total(data, rho_p, rho_s, beads_2_M):
    """
    Computes the concentrations of polycation (PC), cation (C), polyanion (PA),
    and anion (A) in the supernatant (I) and coacervate (II) phases for
    different Bjerrum lengths.

    Parameters
    ----------
    data : dictionary of Pandas dataframes
        Contains dataframes of data from liquid state
        theory calculations indexed by Bjerrum length.
        Dataframes have densities in beads/sigma^3.
    rho_p : float
        Average density of polymer (cation + anion) in
        both phases [mol/L]
    rho_s : float
        Average density of salt (just cation since 1 cation
        and 1 anion come from one KBr molecule) in both
        phases [mol/L]
    beads_2_M : float
        Multiplicative conversion to get from beads/sigma^3
        to moles of monomer/L.

    Returns
    -------
    results : dictionary of arrays/lists
        Contains:
        'alpha' : N-element list
            List of volume fractions of phase I [nondim].
        'galvani' : N-element list
            List of galvani potentials [???] #TODO figure out units
        'lB' : (Nx1) numpy array
            Array of Bjerrum non-dimensionalized by sigma (defined in definition
            of "data" dictionary).
        'rho_PAI', 'rho_PAII' : N-element lists
            List of concentrations of polyanion in phases I (supernatant) and
            II (coacervate) [mol/L]
        'rho_PCI', 'rho_PCII' : N-element lists
            List of concentrations of polycation in phases I (supernatant) and
            II (coacervate) [mol/L]
        'rho_AI', 'rho_AII' : N-element list
            List of concentrations of anion in phases I (supernatant) and
            II (coacervate) [mol/L]
        'rho_CI', 'rho_CII' : N-element list
            List of concentrations of cation in phases I (supernatant) and
            II (coacervate) [mol/L]

    """
    # initializes outputs
    results = {'alpha' : [], 'galvani' : [], 'lB' : [],
                'rho_AI' : [], 'rho_AII' : [], 'rho_PAI' : [], 'rho_PAII' : [],
                'rho_CI' : [], 'rho_CII' : [], 'rho_PCI' : [], 'rho_PCII' : [],
    }

    # computes coexistence at each Bjerrum length and stores results if physical
    for lB in data.keys():
        df = data[lB]
        df_s = compute_rho_s(df, rho_p, beads_2_M)
        # ensures that the total salt concentration is within the possible two-phase range
        if rho_s <= np.max(df_s['rhoS'])*beads_2_M and \
                                rho_s >= np.min(df_s['rhoS'])*beads_2_M:
            # finds the index of the dataframe that has the closest salt concentration to the given value
            diff_rho_s = np.abs(df_s['rhoS']*beads_2_M - rho_s)
            i_same_salt = np.argmin(diff_rho_s)
            alpha = df_s['alpha'].iloc[i_same_salt]
            # recomputes the volume fraction of supernatant more precisely using
            # interpolation
            alpha = np.interp(rho_s, df_s['rhoS']*beads_2_M,
                        df_s['alpha'].to_numpy(dtype='float64'))
            if alpha == 1:
                print('rho_s = {0:.64f}'.format(rho_s/beads_2_M))
                print('rho_p = {0:.64f}'.format(rho_p/beads_2_M))
                print('rhoPCI = {0:.64f}'.format(df['rhoPCI'].loc[i_same_salt]))
                print('rhoPCII = {0:.64f}'.format(df['rhoPCII'].loc[i_same_salt]))
                print('rhoCI = {0:.64f}'.format(df['rhoCI'].loc[i_same_salt]))
                print('rhoCII = {0:.64f}'.format(df['rhoCII'].loc[i_same_salt]))
                print(df.loc[i_same_salt])
            # ensures that the ratio of volume I to total volume is physical
            # (i.e., in the range [0,1])
            if alpha > 1 or alpha < 0:
                continue
            results['alpha'] += [alpha]
            results['lB'] += [lB]
            if 'Psi' in df_s.columns:
                results['galvani'] += [df_s['Psi'].iloc[i_same_salt]*beads_2_M]
            results['rho_AI'] += [df_s['rhoAI'].iloc[i_same_salt]*beads_2_M]
            results['rho_AII'] += [df_s['rhoAII'].iloc[i_same_salt]*beads_2_M]
            results['rho_PAI'] += [df_s['rhoPAI'].iloc[i_same_salt]*beads_2_M]
            results['rho_PAII'] += [df_s['rhoPAII'].iloc[i_same_salt]*beads_2_M]
            results['rho_CI'] += [df_s['rhoCI'].iloc[i_same_salt]*beads_2_M]
            results['rho_CII'] += [df_s['rhoCII'].iloc[i_same_salt]*beads_2_M]
            results['rho_PCI'] += [df_s['rhoPCI'].iloc[i_same_salt]*beads_2_M]
            results['rho_PCII'] += [df_s['rhoPCII'].iloc[i_same_salt]*beads_2_M]

    return results


def fixed_rho_total_legacy(data, rho_p, rho_s, beads_2_M):
    """
    *LEGACY*: only returns polycation/cation concentrations. Use updated version
    (`fixed_rho_total()`), which returns a dictionary of all concentrations.

    Computes the polycation concentration in the
    supernatant (I) and coacervate (II) phases for
    different Bjerrum lengths.

    Parameters
    ----------
    data : dictionary of Pandas dataframes
        Contains dataframes of data from liquid state
        theory calculations indexed by Bjerrum length.
        Dataframes have densities in beads/sigma^3.
    rho_p : float
        Average density of polymer (cation + anion) in
        both phases [mol/L]
    rho_s : float
        Average density of salt (just cation since 1 cation
        and 1 anion come from one KBr molecule) in both
        phases [mol/L]
    beads_2_M : float
        Multiplicative conversion to get from beads/sigma^3
        to moles of monomer/L.

    Returns
    -------
    lB_arr : (Nx1) numpy array
        Array of Bjerrum non-dimensionalized by sigma (defined in definition
        of "data" dictionary).
    rho_PCI_list : N-element list
        List of densities of polycation in phase I (supernatant) [mol/L]
    rho_PCII_list : N-element list
        List of densities of polycation in phase II (coacervate) [mol/L]
    alpha_list : N-element list (only returned if ret_alpha==True)
        List of volume fractions of phase I [nondim].
    """
    # initializes outputs
    lB_valid_list = []
    rho_PCI_list = []
    rho_PCII_list = []
    rho_CI_list = []
    rho_CII_list = []
    alpha_list = []

    # computes coexistence at each Bjerrum length and stores results if physical
    for lB in data.keys():
        df = data[lB]
        df_s = compute_rho_s(df, rho_p, beads_2_M)
        # ensures that the total salt concentration is within the possible two-phase range
        if rho_s <= np.max(df_s['rhoS'])*beads_2_M and \
                                rho_s >= np.min(df_s['rhoS'])*beads_2_M:
            # finds the index of the dataframe that has the closest salt concentration to the given value
            diff_rho_s = np.abs(df_s['rhoS']*beads_2_M - rho_s)
            i_same_salt = np.argmin(diff_rho_s)
            alpha = df_s['alpha'].iloc[i_same_salt]
            # recomputes the volume fraction of supernatant more precisely using
            # interpolation
            alpha = np.interp(rho_s, df_s['rhoS']*beads_2_M,
                        df_s['alpha'].to_numpy(dtype='float64'))
            if alpha == 1:
                print('rho_s = {0:.64f}'.format(rho_s/beads_2_M))
                print('rho_p = {0:.64f}'.format(rho_p/beads_2_M))
                print('rhoPCI = {0:.64f}'.format(df['rhoPCI'].loc[i_same_salt]))
                print('rhoPCII = {0:.64f}'.format(df['rhoPCII'].loc[i_same_salt]))
                print('rhoCI = {0:.64f}'.format(df['rhoCI'].loc[i_same_salt]))
                print('rhoCII = {0:.64f}'.format(df['rhoCII'].loc[i_same_salt]))
                print(df.loc[i_same_salt])
            # ensures that the ratio of volume I to total volume is physical
            # (i.e., in the range [0,1])
            if alpha > 1 or alpha < 0:
                continue
            lB_valid_list += [lB]
            rho_PCI_list += [df_s['rhoPCI'].iloc[i_same_salt]*beads_2_M]
            rho_PCII_list += [df_s['rhoPCII'].iloc[i_same_salt]*beads_2_M]
            rho_CI_list += [df_s['rhoCI'].iloc[i_same_salt]*beads_2_M]
            rho_CII_list += [df_s['rhoCII'].iloc[i_same_salt]*beads_2_M]
            alpha_list += [alpha]

    lB_arr = np.array(lB_valid_list)

    return rho_PCI_list, rho_PCII_list, rho_CI_list, rho_CII_list, lB_arr, alpha_list


def fixed_mu(mu, data, qty, comp='muAI', beads_2_M=1):
    """
    """
    return fixed_conc(mu*np.ones([len(data.keys())]), data, qty, comp=comp,
                      beads_2_M=beads_2_M)


def fixed_conc(mu_conc, data, qty, comp='muAI', beads_2_M=1, quiet=True):
    """
    Extracts the value of the requested quantity at the fixed value of the
    chemical potential provided for different Bjerrum lengths/temperatures.

    The dataframes must provide chemical potential values along with the
    densities or else the function will assert an error.

    Parameters
    ----------
    mu_conc : numpy array of floats
        Fixed chemical potential at each temperature/Bjerrum length
        [nondimensionalized by beta]
    data : dictionary of Pandas DataFrame
        Table of densities and chemical potentials indexed by Bjerrum length.
    qty : string
        The quantity from df to return. Options include 'rhoPC', 'rhoPA',
        'rhoC', and 'rhoA'.
    comp : string
        Component and phase's chemical potential to fix, e.g., muAI, muPI,
        muPCII, etc.

    Returns
    -------
    lB_arr : (N) numpy array of floats
        Bjerrum lengths non-dimensionalized by sigma, the hard-sphere diameter.
    qty_I_arr : (N) numpy array of of floats
        Value of the quantity requested in phase I (polyelectrolyte-poor).
    qty_II_arr : (N) numpy array of of floats
        Value of the quantity requested in phase II (polyelectrolyte-rich).

    """
    # which component's chemical potential should I fix? Is it the same in both phases
    # or does it differ by the Galvani potential?
    lB_arr = np.array(list(data.keys()))
    # initializes lists to store results
    lB_ret = []
    qty_I_list = []
    qty_II_list = []

    # finds entries with chemical potential nearest to the fixed value
    for i, lB in enumerate(lB_arr):
        # extracts dataframe with given value of Bjerrum length
        df = data[lB]
        # ensures that chemical potential values are provided in the dataframe
        assert any(col in comp for col in df.columns) , \
                'salt.fixed_conc() requires the dataframe' + \
                 ' df to include chemical potential mu for comp.'
        # extracts corresponding list of chemical potentials for the desired component and phase
        no_nans = np.logical_not( np.logical_or(np.isnan(df[comp].to_numpy(dtype=float)),
                     np.logical_or(np.isnan(df[qty+'I'].to_numpy(dtype=float)),
                                   np.isnan(df[qty+'II'].to_numpy(dtype=float)))))
        mu_arr = df.loc[no_nans, comp].to_numpy(dtype=float)
        qty_I_arr = df.loc[no_nans, qty+'I'].to_numpy(dtype=float)
        qty_II_arr = df.loc[no_nans, qty+'II'].to_numpy(dtype=float)

        # loads chemical potential from list
        try:
            mu = mu_conc[i]
        except IndexError:
            print("i out of bounds of mu_conc w len = {0:d} for lB = {1:.3f}".format(len(mu_conc), lB))

        # tries to interpolate value of quantity given chemical potential
        if (mu >= np.min(mu_arr)) and (mu <= np.max(mu_arr)):
            qty_I = np.interp(mu, mu_arr, qty_I_arr)
            qty_II = np.interp(mu, mu_arr, qty_II_arr)
        else:
            if not quiet:
                print('Chemical potential = {0:.3f} outside of range for Bjerrum length = {1:.3f}.'.format(mu, lB))
            continue

        # if successful, stores Bjerrum length and values of desired quantity in each phase
        lB_ret += [lB]
        qty_I_list += [qty_I]
        qty_II_list += [qty_II]

    # converts results to numpy arrays
    lB_ret = np.array(lB_ret)
    qty_I_arr = np.array(qty_I_list)*beads_2_M
    qty_II_arr = np.array(qty_II_list)*beads_2_M

    return lB_ret, qty_I_arr, qty_II_arr


def fixed_mu_T(beta_mu_ref, data, qty, T_ref=313, comp='muAI', sigma=4E-10):
    """
    Extracts the value of the requested quantity at the fixed value of the
    chemical potential, not scaled by beta (1/kT). This is in contrast to
    fixed_mu(), which fixes the chemical potential scaled by beta.

    The dataframes must provide chemical potential values along with the
    densities or else the function will assert an error.

    Parameters
    ----------
    beta_mu_ref : float
        Fixed chemical potential [nondimensionalized by beta at the reference temperature]
    data : dictionary of Pandas DataFrame
        Table of densities and chemical potentials indexed by Bjerrum length.
    qty : string
        The quantity from df to return. Currently just written for density
        'rho'.
    T_ref : float
        Reference temperature [K] at which beta_mu_ref is given
    comp : string
        Component and phase's chemical potential to fix, e.g., muAI, muPI,
        muPCII, etc.

    Returns
    -------
    lB_arr : (N) numpy array of floats
        Bjerrum lengths non-dimensionalized by sigma, the hard-sphere diameter.
    qty_I_arr : (N) numpy array of of floats
        Value of the quantity requested in phase I (polyelectrolyte-poor).
    qty_II_arr : (N) numpy array of of floats
        Value of the quantity requested in phase II (polyelectrolyte-rich).

    """
    # which component's chemical potential should I fix? Is it the same in both phases
    # or does it differ by the Galvani potential?
    lB_arr = np.array(list(data.keys()))
    # initializes lists to store results
    lB_ret = []
    qty_I_list = []
    qty_II_list = []
    # finds entries with chemical potential nearest to the fixed value
    for lB in lB_arr:
        # extracts dataframe with given value of Bjerrum length
        df = data[lB]

        # computes temperature
        T = pe.lB_2_T(lB, sigma=sigma)
        # scales beta mu (mu/kB*T) by current temperature
        beta_mu = beta_mu_ref*T_ref/T

        # ensures that chemical potential value are provided in the dataframe
        assert any(col in comp for col in df.columns) , \
                'salt.fixed_mu() requires the dataframe' + \
                 ' df to include chemical potential mu for comp.'
        # extracts corresponding list of chemical potentials for the desired component and phase
        no_nans = np.logical_not( \
                     np.logical_or(np.isnan(df[qty+'I'].to_numpy(dtype=float)),
                                   np.isnan(df[qty+'II'].to_numpy(dtype=float))))
        beta_mu_arr = df.loc[no_nans, comp].to_numpy(dtype=float)
        qty_I_arr = df.loc[no_nans, qty+'I'].to_numpy(dtype=float)
        qty_II_arr = df.loc[no_nans, qty+'II'].to_numpy(dtype=float)

        # tries to interpolate value of quantity given chemical potential
        if (beta_mu >= np.min(beta_mu_arr)) and (beta_mu <= np.max(beta_mu_arr)):
            qty_I = np.interp(beta_mu, beta_mu_arr, qty_I_arr)
            qty_II = np.interp(beta_mu, beta_mu_arr, qty_II_arr)
        else:
            print('Chemical potential = {0:.3f} beta outside of range for Bjerrum length = {1:.3f}.'.format(beta_mu, lB))
            continue

        # if successful, stores Bjerrum length and values of desired quantity in each phase
        lB_ret += [lB]
        qty_I_list += [qty_I]
        qty_II_list += [qty_II]

    # converts results to numpy arrays
    lB_ret = np.array(lB_ret)
    qty_I_arr = np.array(qty_I_list)
    qty_II_arr = np.array(qty_II_list)

    return lB_ret, qty_I_arr, qty_II_arr



def get_filepaths(data_folder, naming_structure='NA(*)NB(*)lB(*)', ext='.PD'):
    """
    Returns list of filepaths to datafiles for salt computations (.PD files).
    Loads data for all values of NA, NB (degree of polymerization of the chains)
    and lB (Bjerrum length).
    Parameters
    ----------
    data_folder : string
        Folder containing phase diagram data to load
    naming_structure : string
       Naming structure of data files (with '*' to indicate any string)
    ext : string
        Extension for files with data (includes period)

    Returns
    -------
    filepath_list : list of strings
        List of filepaths to data for phase diagrams at different N, lB
    """
    # collects list of folders with the given naming structure
    tmp = glob.glob(os.path.join(data_folder, naming_structure))

    # collects list of all filepaths to data files within the folders
    filepath_list = []
    for folder in tmp:
        try:
            _, _, files_contained = next(os.walk(folder))
        except:
            print('Could not walk through files in {0:s}'.format(folder))
        filepath_list += [os.path.join(folder, filename) for filename in files_contained if ext in filename]

    return filepath_list


def get_fixed_conc_data(data, mu_salt_folder, rho_s, sigma, ext='.PD',
                        z_name='T [K]', T_range=[273.15, 373.15], T_cels=False,
                        mu_naming_structure='NA(*)NB(*)lB(*'):
    """Gets data at fixed saltwater reservoir concentration."""
    df_mu = make_df_mu(data, mu_salt_folder, rho_s, T_range, sigma,
          naming_structure=mu_naming_structure, ext=ext)
    # computes binodal for given composition
    rho_p_I, rho_s_I, T_arr, \
    rho_p_II, rho_s_II, _ = extract_df_mu_data(df_mu, z_name)

    # computes temperature and identifies data within range
    liq_h2o = np.logical_and(T_arr >= T_range[0], T_arr <= T_range[1])
    # converts temperature from Kelvin to Celsius
    if T_cels:
        T_arr -= K_2_C

    return rho_s_I[liq_h2o], rho_s_II[liq_h2o], T_arr[liq_h2o], \
            rho_p_I[liq_h2o], rho_p_II[liq_h2o]


def get_mu_conc_lB(mu_salt_folder, rho_salt, beads_2_M=1,
                   naming_structure="NA(100)NB(100)lB(*)",
                   ext='PD', num_rows_hdr=2):
    """
    Gets the chemical potentials that match a given concentration for
    different temperatures.
    """
    filepath_list = get_filepaths(mu_salt_folder, naming_structure=naming_structure, ext=ext)

    # converts units of rho_salt to beads/sigma^3 if given in mol/L
    rho_salt /= beads_2_M

    # initializes lists to store results
    lB_list = []
    mu_list = []

    for filepath in filepath_list:
        _, lB = get_N_lB(filepath)

        df = pd.read_csv(filepath, header=num_rows_hdr, delim_whitespace=True)

        # extracts corresponding list of chemical potentials for the desired component and phase
        no_nans = np.logical_not( \
                     np.logical_or(np.isnan(df['rhoAI'].to_numpy(dtype=float)),
                                   np.isnan(df['muAI'].to_numpy(dtype=float))))
        mu_arr = df.loc[no_nans, 'muAI'].to_numpy(dtype=float)
        rhoAI_arr = df.loc[no_nans, 'rhoAI'].to_numpy(dtype=float)

        # tries to interpolate value of chemical potential given density of salt
        if (rho_salt >= np.min(rhoAI_arr)) and (rho_salt <= np.max(rhoAI_arr)):
            mu = np.interp(rho_salt, rhoAI_arr, mu_arr)
        else:
            print('rhoAI = {0:.3f} outside of range for Bjerrum length = {1:.3f}.'.format(rho_salt, lB))
            continue

        lB_list += [lB]
        mu_list += [mu]

    lB_arr = np.array(lB_list)
    mu_arr = np.array(mu_list)

    return lB_arr, mu_arr


def get_mu_conc(mu_salt_folder, lB_list, rho_salt, beads_2_M=1, num_rows_hdr=2,
               naming_structure='NA(100)NB(100)lB(*)', ext='PD'):
    """


    Parameters
    ----------
    mu_salt_folder : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    rho_salt : TYPE
        DESCRIPTION.

    Returns
    -------
    mu_conc : TYPE
        DESCRIPTION.

    """
    # computes chemical potential of salt solution at different Bjerrum lengths
    lB_arr, mu_arr = get_mu_conc_lB(mu_salt_folder, rho_salt, naming_structure=naming_structure,
                                    ext=ext, beads_2_M=beads_2_M, num_rows_hdr=num_rows_hdr)
    if len(lB_arr) == 0:
        return []

    mu_conc_list = []
    for lB in lB_list:
        mu_conc_list += [np.interp(lB, lB_arr, mu_arr)]

    mu_conc = np.array(mu_conc_list)

    return mu_conc


def get_N_lB(filepath):
    """
    Returns the degree of polymerization N and Bjerrum length lB from filepath.
    *Assumes symmetric polymers.

    Parameters:
    -----------
    filepath : string
        Full filepath to data file
    ext : string
        Extension for data file, including period
    Returns:
    --------
    N : int
        Degree of polymerization
    lB : float
        Bjerrum length (non-dimensionalized by sigma, bead diameter)
    """
    # extracts filename from filepath
    folder = filepath.split('\\')[-2]
    # extracts N
    i_N_begin = folder.find('NA')+len('NA(')
    i_N_end = folder.find(')NB')
    N = int(folder[i_N_begin:i_N_end])
    # extracts lB
    i_lB_begin = folder.find('lB(') + len('lB(')
    i_lB_end = folder[i_lB_begin:].find(')') + i_lB_begin
    lB = float(folder[i_lB_begin:i_lB_end])

    return N, lB


def interp_s(s, df, qty, chris=False):
    """
    Interpolates values of polymer volume fraction in each phase given
    total salt volume fraction. Uses linear interpolation (np.interp).
    Parameters:
    -----------
        s : float
            quantity of salt in same units as quantity extracted from dataframe
        df : Pandas dataframe
            dataframe of densities and volume fractions, should have been processed
            by calc_vol_frac()
        qty : string
            'phi' for volume fraction or 'rho' for density
        chris : bool
            If True, uses Chris Balzer's equation for volume fraction phi. No-op
            for qty == 'rho'
    Returns:
    --------
        p_I : float
            interpolated quantity of polymer in phase I (dilute phase)
        p_II : float
            interpolated quantity of polymer in phase II (polymer-rich phase)
    """
    # loads salt and polymer quantities
    if chris and qty == 'phi':
        # uses Chris's formula for the salt quantity--only works for phi
        s_data = df[qty +  'S' + 'Chris'].to_numpy(dtype=float)
    else:
        s_data = df[qty +  'S'].to_numpy(dtype=float)
    p_I_data = df[qty + 'PAI'].to_numpy(dtype=float) # assumes anion = cation
    p_II_data = df[qty + 'PAII'].to_numpy(dtype=float) # assumes anion = cation
    # organizes and orders data in ascending order of salt volume fraction
    inds_sort = np.argsort(s_data)
    s_data = s_data[inds_sort]
    p_I_data = p_I_data[inds_sort]
    p_II_data = p_II_data[inds_sort]
    # interpolates
    p_I = np.interp(s, s_data, p_I_data)
    p_II = np.interp(s, s_data, p_II_data)

    return p_I, p_II


def calc_vol_frac(df):
    """
    Calculates the total density of counterions (salt) and
    estimates volume fractions with various formulas.
    Modifies dataframe in place.
    TODO: check formula for total rho and solvent frac.
    Parameters:
    -----------
        df : Pandas dataframe
            Dataframe of raw densities from Pengfei's simulation code (non-dim by sigma^3)
        rho_max : float
            Maximum density if no void space (non-dim by sigma^3)
    Returns: void
    """
    # adds total density of salt in system--same for anions
    df['rhoS'] = df['rhoCI'] + df['rhoCII']
    # adds total density of polymer in each phase
    df['rhoP'] = df['rhoPCI'] + df['rhoPCII']
    # removes rows with nan
    df.dropna(axis='index', inplace=True)


def calc_rho_s(df, rho_p_tot):
    """
    Calculates the total density of counterions (salt).
    Modifies dataframe in place.
    Parameters:
    -----------
    df : Pandas dataframe
        Dataframe of raw densities from Pengfei's simulation code (non-dim by sigma^3)
    rho_p_tot : float
        Total density of polymer in the system [beads/sigma^3]
    Returns: void (modifies dataframe in place)
    """
    # adds total density of each species in each phase
    df['rhoPI'] = df['rhoPCI'] + df['rhoPAI']
    df['rhoPII'] = df['rhoPCII'] + df['rhoPAII']
    df['rhoSI'] = df['rhoAI'] + df['rhoCI']
    df['rhoSII'] = df['rhoAII'] + df['rhoCII']
    # calculates overall density of salt using calculations in the beginning of repl_adhikari_2019_fig4.ipynb
    df['rhoS'] = ( df['rhoSII']*(rho_p_tot - df['rhoPI']) + df['rhoSI']*(df['rhoPII'] - rho_p_tot) )/ \
                    (df['rhoPII'] + df['rhoPI'])

    # removes rows with nan
    df.dropna(axis='index', inplace=True)


def data_fig1(data, mu_salt_folder, rho_salt, T_range, sigma, z_name,
            beads_2_M, n_pts, z_rough, ext='.PD', num_rows_hdr=2, lB_lo=1.0,
            lB_hi=3.0, rho_p=None, rho_s=None, naming_structure='NA(*)NB(*)lB(*)',
            mu_naming_structure=None,
             i_cp=-1, quiet=False, T_for_outline=[], rho_s_for_outline=[]):
    """
    Plots figure 1: 3D surface binodal, critical line, 2D line binodal at
    fixed temperature, and 2D line binodal at fixed reservoir salt
    concentration.
    """
    # allows user the option to specify different naming structure for saltwater
    if mu_naming_structure is None:
        mu_naming_structure = naming_structure
    # makes dataframe of binodal for fixed salt reservoir concentration
    lB_list = list(data.keys())
    df_mu = make_df_mu(data, mu_salt_folder, rho_salt, T_range, sigma,
                        num_rows_hdr=num_rows_hdr,
                        naming_structure=mu_naming_structure, ext=ext, quiet=quiet)
    data_mu = extract_df_mu_data(df_mu, z_name)

    # extracts critical points
    data_cp = extract_cp_data(data, beads_2_M, z_name, sigma=sigma, i_cp=i_cp)

    df_slice = make_df_slice(data, n_pts, T_range, sigma)

    data_3d = extract_df_slice_data(df_slice, z_name)

    data_z = extract_fixed_z(data_3d, z_rough)

    output = [data_3d, data_mu, data_z, data_cp]

    if (rho_p is not None) and (rho_s is not None):
        data_comp = extract_fixed_comp(data, rho_p, rho_s, beads_2_M, z_name,
                                                        T_range, sigma=sigma)
        output += [data_comp]

    if len(rho_s_for_outline) > 0 or len(T_for_outline) > 0:
        data_outlines = extract_outline_data(data_3d,
                                                rho_s_for_outline,
                                                T_for_outline)
        output += [data_outlines]

    return tuple(output)


def lB_comparison(T_range, sigma, T_room_C=20, npts=100):
    """Computes Bjerrum length [A] with fixed vs. T-dependent dielectric constant."""

    # Computes Bjerrum length in desired temperature range
    T_arr = np.linspace(*T_range, npts)
    lB_arr = pe.lB_fn(T_arr, sigma=sigma) # in units of sigma
    lB_A_arr = lB_arr*sigma*m_2_A # converts to A
    # computes Bjerrum length near room temperature
    T_room = T_room_C + 273.15 # converts from Celsius to Kelvin
    i_room = np.argmin(np.abs(T_arr - T_room))
    lB_room = lB_arr[i_room]
    lB_0_arr = lB_room*T_room/T_arr # in units of sigma
    lB_0_A_arr = lB_0_arr*sigma*m_2_A # converts to A

    return T_arr, lB_A_arr, lB_0_A_arr


def lcst_vary_rho(data, rho_fix, rho_var_range, d_rho_var, ch_var, beads_2_M, T_range=[273,373], sigma=4E-10):
    """Computes LCST for different overall concentrations of a species, the other being fixed."""
    # creates dictionary of values based on which component's density is varied
    d = plot.get_plot_dict_p_s(ch_var)

    # computes list of desired values of the varied density
    rho_var_list = [rho_var_range[0] + d_rho_var*i \
                    for i in range(int((rho_var_range[1] - rho_var_range[0])/d_rho_var))]

    # initializes list of LCSTs and successful densities
    lcst_list = []
    rho_success = []

    # computes LCST under each condition
    for rho_var in rho_var_list:
        # assigns polymer and salt densities appropriately
        rho_pair = np.array([rho_var, rho_fix])
        rho_p, rho_s = rho_pair[d['order']]
        # computes binodal for fixed overall densities
        results = fixed_rho_total(data, rho_p, rho_s, beads_2_M)
        rho_PCI_list = results['rho_PCI']
        rho_PCII_list = results['rho_PCII']
        rho_CI_list = results['rho_CI']
        rho_CII_list = results['rho_CII']
        lB_arr = results['lB']
        alpha = results['alpha']
        # computes corresponding temperatures
        T_arr = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma)
        # stores estimate of LCST (minimum temperature with two phases)
        try:
            lcst_list += [np.min(T_arr)]
            rho_pair = np.array([rho_p, rho_s])
            rho_var, rho_fix = rho_pair[d['order']]
            rho_success += [rho_var]
        except:
            print('LCST not found for rho_p = {0:.3f} M, rho_s = {1:.3f} M,'.format(rho_p, rho_s) + \
                  'in salt.lcst_vs_rho().')


    return rho_success, lcst_list


def load_data(data_folder, ext='.PD', num_rows_hdr=2, lB_lo=1, lB_hi=3,
             naming_structure='NA(100)NB(100)lB(*)'):
    """
    Loads data from data folder into a dictionary of dataframes.

    To load data from Pengfei Zhang's code's output, specify the data
    folder and use the default parameters.
    To load data from Chris Balzer's code's output, set `ext` to 'output.dat'
    and `naming_structure` to 'NA(*)NB(*)lB(*)f(*)', specifying the values of
    NA, NB, lB, or f (replacing the '*' with the specified value) if desired
    to narrow the search.
    """
    # loads list of filepaths to computation results (different Bjerrum lengths)
    filepath_list = get_filepaths(data_folder, naming_structure=naming_structure,
                                    ext=ext)

    # initializes a dictionary of Bjerrum lengths to dataframes
    data = {}

    # loops through data files and stores dataframes in dictionary
    for filepath in filepath_list:
        # parses filepath
        _, lB = get_N_lB(filepath)
        if not (lB >= lB_lo and lB <= lB_hi):
            continue
        # creates dataframe out of data
        try:
            df = pd.read_csv(filepath, header=num_rows_hdr, delim_whitespace=True)
            # stores dataframe
            data[lB] = df
        except:
            print('Loading data for lB = {0:.3f} failed.'.format(lB))
            continue
     
    # alerts user if no data found
    if len(data) == 0:
        print('No data found in the folder {0:s}'.format(data_folder))

    return data


def make_df_mu(data, mu_salt_folder, rho_salt, T_range, sigma, num_rows_hdr=2,
              naming_structure='NA(100)NB(100)lB(*)', ext='PD', quiet=True):
    """
    Makes dataframe for binodal at chemical potential for constant salt
    reservoir concentration.

    Parameters
    ----------
    quiet : bool, optional
        If False, prints failure statement if cannot find composition leading to
        requested chemical potential in `fixed_conc()`. Otherwise, prints no
        statement (default True).
    """
    # calculates the chemical potential of a salt reservoir at given concentration for different temperatures
    lB_allowed, _ = get_mu_conc_lB(mu_salt_folder, rho_salt, naming_structure=naming_structure,
                        ext=ext, num_rows_hdr=num_rows_hdr) # gets allowed Bjerrum lengths
    mu_conc = get_mu_conc(mu_salt_folder, list(data.keys()), rho_salt,
                        naming_structure=naming_structure,
                        ext=ext, num_rows_hdr=num_rows_hdr) # interpolates chem pot

    assert len(mu_conc) > 0, \
            "len(mu_conc) = 0 in salt.make_df_mu(), loading from {0:s}".format(\
                                                                mu_salt_folder)
    # computes densities at different Bjerrum lengths for fixed chemical pot
    lB_arr, rhoPAI, rhoPAII = fixed_conc(mu_conc, data, 'rhoPA', quiet=quiet)
    lB_arr, rhoAI, rhoAII = fixed_conc(mu_conc, data, 'rhoA', quiet=quiet)
    # computes corresponding temperatures [K]
    T_arr = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma)

    ### DATA FORMATTING ###
    # arranges data at fixed salt concentration into dataframe
    df_mu = tidy_df(pd.DataFrame(data = {'BJ' : lB_arr, 'T [K]' : T_arr,
                                'T [C]' : T_arr - K_2_C, 'rhoPAI' : rhoPAI,
                                'rhoPAII' : rhoPAII, 'rhoAI' : rhoAI,
                                'rhoAII' : rhoAII}), sigma=sigma)

    # removes rows corresponding to Bjerrum lengths outside of allowed range (where interpolation used for mu_conc fails)
    inds_to_remove = [i for i in range(len(df_mu)) if df_mu['BJ'].iloc[i] \
                                not in list(lB_allowed) and \
                                  df_mu['phase'].iloc[i] == 'I']
    df_mu.drop(labels=inds_to_remove, inplace=True)

    return df_mu


def make_df_slice(data, n_pts, T_range, sigma):
    """
    Makes dataframe of slice of full 3D binodal (slice meaning a subset of
    points in case the dataset has a greater density of points than the user
    wishes to plot).
    """
    # arranges full data into dataframe
    df_full_molL = pd.concat([tidy_df(df, sigma=sigma) for df in data.values()])
    # slices out subset of data for plotting
    rhoPA = slice_arr(df_full_molL['rhoPA'].to_numpy(dtype=float), n_pts)
    rhoA = slice_arr(df_full_molL['rhoA'].to_numpy(dtype=float), n_pts)
    BJ = slice_arr(df_full_molL['BJ'].to_numpy(dtype=float), n_pts)
    phase = slice_arr(list(df_full_molL['phase']), n_pts)
    # arranges sliced data into dataframe
    df_slice = pd.DataFrame(data={'rhoPA' : rhoPA, 'rhoA' : rhoA, 'BJ' : BJ, 'phase' : phase})
    df_slice['T [K]'] = pe.lB_2_T_arr(df_slice['BJ'].to_numpy(dtype=float),
                                                        T_range, sigma=sigma)
    df_slice['T [C]'] = df_slice['T [K]'].to_numpy(dtype=float) - K_2_C

    return df_slice


def match_data_pt(df, rho_p, rho_s, th0=np.pi/4):
    """Identifies the entry in the dataframe closest to the data point."""
    # removes nan entries
    df.dropna(inplace=True)

    def rms_fn(th, args):
        rho_p_I, rho_p_II, rho_s_I, rho_s_II = args
        a = (np.cos(th))**2 # restricts a to range of 0 to 1

        return (rho_s_I/(1 + 1/a) + rho_s_II/(a + 1) - rho_s)**2 + \
                (rho_p_I/(1 + 1/a) + rho_p_II/(a + 1) - rho_p)**2

    def get_rho(df, i):
        rho_p_I = df['rhoPAI'].iloc[i]
        rho_p_II = df['rhoPAII'].iloc[i]
        rho_s_I = df['rhoAI'].iloc[i]
        rho_s_II = df['rhoAII'].iloc[i]

        return rho_p_I, rho_p_II, rho_s_I, rho_s_II

    rms_arr = np.zeros([len(df)])
    for i in range(len(df)):
        args = get_rho(df, i)
        res = scipy.optimize.minimize(rms_fn, th0, (args,))
        th_min = res.x
        rms_arr[i] = rms_fn(th_min, args)

    i_min = np.argmin(rms_arr)
    rho_p_I = df['rhoPAI'].iloc[i_min]
    rho_p_II = df['rhoPAII'].iloc[i_min]
    rho_s_I = df['rhoAI'].iloc[i_min]
    rho_s_II = df['rhoAII'].iloc[i_min]
    a = (np.cos(th_min))**2

    return rho_p_I, rho_p_II, rho_s_I, rho_s_II, a


def reformat_vo_data(data_folder, rho_file='density.dat',
                        mu_file='chemPotential.dat'):
    """
    Reformats the data produced in the Wetting-Polymer-Films package for
    Voorn-Overbeek theory phase behavior developed by Chris Balzer to match the
    formatting of the CCLS package based on a liquid state theory.
    """
    # extracts Bjerrum length from folder name
    N, lB = get_N_lB(data_folder[:-1], ext='')

    df = pd.DataFrame(columns=['BJ', 'rhoPCI', 'rhoPAI', 'rhoCI', 'rhoAI',
                                'rhoPCII', 'rhoPAII', 'rhoCII', 'rhoAII',
                                 'muPCI', 'muPAI', 'muCI', 'muAI',
                                 'muPCII', 'muPAII', 'muCII', 'muAII'])

    # loads density data
    col_names = ['rhoPCI', 'rhoPAI', 'rhoCI', 'rhoAI', 'rhoPCII', 'rhoPAII',
                    'rhoCII', 'rhoAII', 'unknown']
    df_rho = pd.read_csv(data_folder + rho_file, delimiter='  ', header=None,
                            engine='python', names=col_names)
    df_rho.drop(columns=['unknown'], inplace=True)

    # loads chemical potential data
    col_names = ['muPCI', 'muPAI', 'muCI', 'muAI', 'muPCII', 'muPAII',
                    'muCII', 'muAII']
    df_mu = pd.read_csv(data_folder + mu_file, delimiter='  ', header=None,
                            engine='python', names=col_names)

    # creates final dataframe
    df_lB = pd.DataFrame()
    df_lB['BJ'] = lB*np.ones([len(df_mu)])
    df = pd.concat([df_lB, df_rho, df_mu], axis=1)

    # saves result
    df.to_csv(data_folder + 'NA({0:d})NB({0:d})lB({1:.3f}).PD'.format(N, lB),
                sep='\t', index=False)

    return df


def read_df_exp(df_exp, i, read_sigma=False, conv_vals=False):
    """Reads i^th entry of the experimental dataframe."""
    if conv_vals:
        rho_p = df_exp['rho_p (conv) [M]'].iloc[i]
        rho_s = df_exp['rho_s (conv) [M]'].iloc[i]
    else:
        rho_p = df_exp['rho_p [M]'].iloc[i]
        rho_s = df_exp['rho_s [M]'].iloc[i]
    T_exp = df_exp['T [C]'].iloc[i]
    rho_p_sup = df_exp['rho_p^sup [M]'].iloc[i]
    rho_p_co = df_exp['rho_p^co [M]'].iloc[i]

    result = [rho_p, rho_s, T_exp, rho_p_sup, rho_p_co]

    if read_sigma:
        sigma_sup = df_exp['sigma_p^sup [M]'].iloc[i]
        sigma_co = df_exp['sigma_p^co [M]'].iloc[i]
        result += [sigma_sup, sigma_co]

    return result


def slice_arr(arr, n):
    return arr[::int(len(arr)/n)]


def solv_2_soln_vol(rho_p, rho_s, mL_per_mol_pe=80, mw_salt=119,
                    density_salt=2.76, density_multiplier=2):
    """
    Returns
    -------
    Volume of 1 L H2O + added polyelectrolyte and salt [L]
    """
    # computes a conversion factor
    mL_per_mol_salt = mw_salt / (density_salt*density_multiplier)
    # computes volume of solution [L]
    V_soln = (L_2_mL + rho_p*mL_per_mol_pe + rho_s*mL_per_mol_salt) / L_2_mL

    return V_soln


def tidy_df(df, phase_list=['I', 'II'], sigma=None):
    """
    Tidies the given dataframe. Basically just adds a new column for the
    phase, renames the other columns not to include the phase, and reorganizes
    accordingly.

    Use sigma to convert from beads/sigma^3 to mol/L
    """
    if sigma is not None:
        beads_2_M = pe.get_beads_2_M(sigma, SI=True) # converts from beads/sigma^3 to mol/L
    else:
        beads_2_M = 1
    # adds new column for phase
    old_columns = list(df.columns)
    # adds phase to old columns, removing duplicates for different phases
    stripped_columns = list(set([column.rstrip('I') for column in old_columns]))
    new_columns = ['phase'] + stripped_columns
    df_tidy = pd.DataFrame(columns=new_columns)
    # moves old data
    for ph in phase_list:
        df_new = pd.DataFrame(columns=new_columns)
        for column in stripped_columns:
            old_column = column
            if(old_column not in old_columns):
                old_column += ph
            # rescales densities if sigma provided for conversion factor
            if 'rho' in column:
                # note: beads_2_M is 1 if sigma not provided
                df_new[column] = df[old_column].to_numpy(dtype=float)*beads_2_M
            else:
                df_new[column] = df[old_column]
        df_new['phase'] = ph
        df_tidy = df_tidy.append(df_new)

    return df_tidy
