"""
plot.py defines functions for plotting phase diagrams of complex
coacervate liquid separation.
"""

# standard libraries
import matplotlib.pyplot as plt
from matplotlib import cm # colormap
import numpy as np
import pandas as pd

# custom libraries
import pe
import salt as nacl

#  plotting libraries
import plotly.graph_objects as go
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, Title, Range1d
from bokeh.models.tools import HoverTool

# CONSTANTS
NA = 6.022E23 # Avogadro's number, molecules / mol
m3_2_L = 1E3
K_2_C = 273.15 # conversion from Kelvin to Celsius (subtract this)
m_2_A = 1E10



def alpha_custom_rho(data, rho_p_list, rho_s_list, beads_2_M,
                    T_range=[273.15,373.35], cmap_name='plasma', sigma=None,
                       colors=None, marker='o', lw=1, T_cels=False,
                       y_lim=[0.5, 1], square_box=False, tol=0.05, ax=None,
                       show_lgnd=True):
    """
    Plots the volume fraction of supernatant phase I (alpha) vs. the overall
    density of the varied component.

    Note: currently eliminates data points with alpha = 1 because they tend to
    be the result of numerical imprecision

    T_range : 2-tuple
        Lower and upper bounds on temperature to consider in degrees Kelvin
        (even if T_cels is True)
    tol : float, opt
        Tolerance of how close volume fraction nearest single-phase region needs
        to be to 1 to round up to 1 (for plotting dashed line)
    """
    # creates list of colors for each value of the varied density
    if colors is None:
        colors = get_colors(cmap_name, len(rho_var_list))

    # creates figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # plots volume fraction of supernatant for each composition
    for i, rho_pair in enumerate(zip(rho_p_list, rho_s_list)):
        # plots binodal for low polymer concentration [M]
        rho_p, rho_s = rho_pair
        results = nacl.fixed_rho_total(data, rho_p, rho_s, beads_2_M)
        rho_PCI_list = results['rho_PCI']
        rho_PCII_list = results['rho_PCII']
        rho_CI_list = results['rho_CI']
        rho_CII_list = results['rho_CII']
        lB_arr = results['lB']
        alpha = results['alpha']
        T_arr = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma)
        liq_h2o = (T_arr >= T_range[0]) * (T_arr <= T_range[1]) * \
                    (np.asarray(alpha) != 1)
        if T_cels:
            T_arr -= K_2_C
        # plots alpha vs. T for given rho_p
        alpha_arr = np.array(alpha)[liq_h2o]
        T_arr = T_arr[liq_h2o]
        ax.plot(T_arr, alpha_arr, color=colors[i],
                marker=marker, lw=lw, label=r'$\rho_p = $' + \
                 '{0:.2f} M, '.format(rho_p) + r'$\rho_s = $' + \
                 '{0:.2f} M'.format(rho_s))

        ### Single Phase
        # plots dashed line to lowest temperature if single phase
        # *checks if lowest plotted temperature reaches y axis
        T_min = np.min(T_arr)
        if T_min > np.min(ax.get_xlim()):
            alpha_single_phase = alpha_arr[np.argmin(T_arr)]
            # rounds up to 1 if volume fraction is close (discontinuous phase sep)
            if np.abs(alpha_single_phase - 1) < tol:
                ax.plot([T_min, T_min], [alpha_single_phase, 1], '-', lw=lw,
                color=colors[i])
                alpha_single_phase = 1
            # rounds to 0.5 if volume fraction is close (passes through LCST)
            if np.abs(alpha_single_phase - 0.5) < tol:
                alpha_single_phase = 0.5

            # plots horizontal dashed line to indicate single phase at low T
            ax.plot([ax.get_xlim()[0], T_min],
                    [alpha_single_phase, alpha_single_phase], '--',
                    lw=lw, color=colors[i])

    # determines labels and limits of axes
    if T_cels:
        x_lim = [T_range[0] - K_2_C, T_range[1] - K_2_C]
        x_label = r'$T$'
        x_unit = r'$^{\circ}$C'
    else:
        x_lim = T_range
        x_lim = r'$T$ [K]'
    y_label = r'$V_{sup}/V_{tot}$'

    # formats plot
    format_binodal(ax, x_label, x_unit, T_range, x_lim=x_lim, y_lim=y_lim,
                    y_label=y_label, square_box=square_box, show_lgnd=show_lgnd)

    return ax


def alpha_vary_rho(data, rho_var_list, rho_fix, ch_var, beads_2_M,
                    T_range=[273.15,373.35], cmap_name='plasma', sigma=None,
                       colors=None, marker='o', lw=1, T_cels=False,
                       y_lim=[0.5, 1], title=None, square_box=False):
    """
    Plots the volume fraction of supernatant phase I (alpha) vs. the overall
    density of the varied component.

    Note: currently eliminates data points with alpha = 1 because they tend to
    be the result of numerical imprecision

    T_range : 2-tuple
        Lower and upper bounds on temperature to consider in degrees Kelvin
        (even if T_cels==True)
    """
    # creates dictionary of values based on which component's density is varied
    d = get_plot_dict_p_s(ch_var)

    # creates list of colors for each value of the varied density
    if colors is None:
        colors = get_colors(cmap_name, len(rho_var_list))

    # creates figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, rho_var in enumerate(rho_var_list):
        # plots binodal for low polymer concentration [M]
        rho_pair = np.array([rho_var, rho_fix])
        rho_p, rho_s = rho_pair[d['order']]
        results = nacl.fixed_rho_total(data, rho_p, rho_s, beads_2_M)
        rho_PCI_list = results['rho_PCI']
        rho_PCII_list = results['rho_PCII']
        rho_CI_list = results['rho_CI']
        rho_CII_list = results['rho_CII']
        lB_arr = results['lB']
        alpha = results['alpha']
        T_arr = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma)
        liq_h2o = (T_arr >= T_range[0]) * (T_arr <= T_range[1]) * \
                    (np.asarray(alpha) != 1)
        if T_cels:
            T_arr -= K_2_C
        # plots alpha vs. T for given rho_p
        ax.plot(T_arr[liq_h2o], np.array(alpha)[liq_h2o], color=colors[i],
                marker=marker, lw=lw, label=r'$\rho_' + d['ch_var'] + ' = $' + \
                 '{0:.2f} M'.format(rho_var))

    # determines labels and limits of axes
    if T_cels:
        x_lim = [T_range[0] - K_2_C, T_range[1] - K_2_C]
        x_label = r'$T$'
        x_unit = r'$^{\circ}$C'
    else:
        x_lim = T_range
        x_lim = r'$T$ [K]'
    y_label = r'$V^{sup}/V^{tot}$'
    if title is None:
        title = 'Effect of Total {0:s} on Supernatant Volume, {1:s} = {2:.2f} M' \
                 .format(d['name_var'], r'$\rho_' + d['ch_fix'] + '$', rho_fix)

    # formats plot
    format_binodal(ax, x_label, x_unit, T_range, title=title, x_lim=x_lim,
                    y_lim=y_lim, y_label=y_label, square_box=square_box)

    return ax


def binodal(lB_arr, left_list, right_list, left='rhoPCI', right='rhoPCII',
            x_label='polyanion density', n_tie_lines=3, plot_T=True, sigma=None,
            T_range=[273, 373], beads_2_M=None, title='', fix_eps=False,
           deg_C=False, x_lim=None, y_lim=None, marker=True, line=False,
           c1='blue', c2='red'):
    """
    Plots binodal with polyanion density as x axis and temperature or
    Bjerrum length as y axis using Bokeh interactive plotting methods.

    Parameters
    ----------
    lB_arr : (Nx1) numpy array
        Array of Bjerrum lengths non-dimensionalized by sigma defined
        in definition of "data" dictionary.
    left_list : N-element list
        List of x-axis variable in phase I (supernatant) [beads/sigma^3]
    right_list : N-element list
        List of x-axis variable in phase II (coacervate) [beads/sigma^3]
    left : string
        Name of heading in df of the variable given in left_list
    right : string
        Name of heading in df of the variable given in right_list
    x_label : string
        Variable to be plotted along the x-axis (without units)
    n_tie_lines : int
        Number of tie lines to plot
    plot_T : bool
        y axis is temperature [K] if True, Bjerrum [sigma] if False
    T_range : 2-element list
        Lower and upper bound for temperatures to plot (to limit temperatures
        to those for which water is liquid)
    beads_2_M : float
        Conversion from beads/sigma^3 to moles of monomers / L. If None, no
        conversion is made and the units on the x axis are beads/sigma^3.
    title : string
        Title of plot
    fix_eps : bool
        Fixed epsilon to constant value if True, or allows it to vary with
        temperature if False.
    deg_C : bool, opt
        If True, temperature is shown in degrees Celsius (assuming it is
        provided in Kelvin), default = False.
    x_lim : 2-element tuple of floats, optional
        Lower and upper bounds of x axis. If None provided, automatically set.
    y_lim : 2-element tuple of floats, optional
        Lower and upper bounds of y axis. If None provided, automatically set.

    Returns
    -------
    p : bokeh plot
        Plot of binodal. Use bokeh's "show(p)" to display. Use "output_notebook()" beforehand
        to show the plot in the same cell (instead of a separate browser tab).
    """
    left_arr = np.copy(left_list)
    right_arr = np.copy(right_list)
    # calculates conversion from beads / sigma^3 to mol/L
    if beads_2_M is not None:
        left_arr *= beads_2_M
        right_arr *= beads_2_M
        units_rho = '[mol/L]'
    else:
        units_rho = '[beads/sigma^3]'

    # computes temperature corresponding to Bjerrum lengths
    T_arr = pe.lB_2_T_arr(lB_arr, T_range, fix_eps=fix_eps, sigma=sigma)

    # stores results in dataframe for plotting
    df_mu = pd.DataFrame(columns=['BJ', 'T', left, right])
    liq_h2o = np.logical_and(T_arr >= T_range[0], T_arr <= T_range[1])
    df_mu['BJ'] = lB_arr[liq_h2o]
    df_mu['T'] = T_arr[liq_h2o] - deg_C*273 # converts to degrees Celsius if requested
    df_mu[left] = left_arr[liq_h2o] # monomer density
    df_mu[right] = right_arr[liq_h2o] # monomer density
    # plots binodal at fixed chemical potential
    n_plot = len(df_mu)
    if n_plot == 0:
        print('No data to plot in plot.binodal()--error likely.')
    p = no_salt(df_mu, n_plot, left=left, right=right, x_label=x_label,
                n_tie_lines=n_tie_lines, plot_T=plot_T, marker=marker, line=line,
                title=title, units_rho=units_rho, deg_C=deg_C, c1=c1, c2=c2)

    # sets axis limits if requested
    if x_lim is not None:
        p.x_range = Range1d(*x_lim)
    if y_lim is not None:
        p.y_range = Range1d(*y_lim)


    return p


def binodal_custom_rho(data, rho_p_list, rho_s_list, beads_2_M,
                     x_var='polycation', x_label=r'$\rho_{PSS}$', sigma=None,
                     T_range=[273.15,373.15], cmap_name='plasma', colors=None,
                     marker='o', fill_left='none', fill_right='full', lw_sup=1,
                     lw_co=3, lgnd_out=True, lw=1, x_lim=None, T_cels=False,
                     c_sup='#1414FF', c_co='#FF0000', ls_sup='-',
                     square_box=False, plot_fixed_rho=False, ax=None,
                     show_lgnd=True):
    """
    Like `binodal_vary_rho()` but allows user to customize both rho_p and rho_s
    (overall) of each condition, rather than fixing one for all conditions.

    """
    # creates list of colors for each value of rho_p
    if colors is None:
        if cmap_name is not None:
            colors = get_colors(cmap_name, len(rho_var_list))

    # creates figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i, rho_pair in enumerate(zip(rho_p_list, rho_s_list)):
        rho_p, rho_s = rho_pair
        # plots binodal for low polymer concentration [M]
        results = nacl.fixed_rho_total(data, rho_p, rho_s, beads_2_M)
        rho_PCI_list = results['rho_PCI']
        rho_PCII_list = results['rho_PCII']
        rho_CI_list = results['rho_CI']
        rho_CII_list = results['rho_CII']
        rho_PAI_list = results['rho_PAI']
        rho_PAII_list = results['rho_PAII']
        rho_AI_list = results['rho_AI']
        rho_AII_list = results['rho_AII']
        lB_arr = results['lB']
        alpha = results['alpha']

        # selects the x-axis data
        if x_var == 'polycation':
            left_arr = np.array(rho_PCI_list)
            right_arr = np.array(rho_PCII_list)
        elif x_var == 'polyanion':
            left_arr = np.array(rho_PAI_list)
            right_arr = np.array(rho_PAII_list)
        elif x_var == 'cation':
            left_arr = np.array(rho_CI_list)
            right_arr = np.array(rho_CII_list)
        elif x_var == 'anion':
            left_arr = np.array(rho_AI_list)
            right_arr = np.array(rho_AII_list)
        elif x_var == 'solvent':
            left_arr = pe.calc_rho_solv(rho_PCI_list,
                                          rho_CI_list,
                                          beads_2_M)
            right_arr = pe.calc_rho_solv(rho_PCII_list,
                                           rho_CII_list,
                                           beads_2_M)
        elif x_var == 'polyelectrolyte':
            left_arr = np.array(rho_PCI_list) + np.array(rho_PAI_list)
            right_arr = np.array(rho_PCII_list) + np.array(rho_PAII_list)
        elif x_var == 'salt':
            left_arr = np.array(rho_CI_list)
            right_arr = np.array(rho_CII_list)
        else:
            print('Error. Invalid x variable in plot.binodal_vary_rho().')

        # computes temperature and identifies data within range
        T_arr = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma)
        liq_h2o = np.logical_and(T_arr >= T_range[0], T_arr <= T_range[1])
        # converts temperature from Kelvin to Celsius
        if T_cels:
            T_arr -= K_2_C

        # assigns separate colors to coacervate and supernatant if not specified
        if colors is not None:
            c_sup = colors[i]
            c_co = colors[i]

        # supernatant
        ax.plot(left_arr[liq_h2o], T_arr[liq_h2o], color=c_sup,
                marker=marker, fillstyle=fill_left, ls=ls_sup,
               label=r'$\rho_p = $' + '{0:.2f} M, '.format(rho_p) + \
                        r'$\rho_s = $' + '{0:.2f} M, supernatant'.format(rho_s),
                        lw=lw_sup)
        # coacervate
        ax.plot(right_arr[liq_h2o], T_arr[liq_h2o], color=c_co,
                marker=marker, fillstyle=fill_right,
                label=r'$\rho_p = $' + '{0:.2f} M, '.format(rho_p) + \
                r'$\rho_s = $' + '{0:.2f} M, coacervate'.format(rho_s),
                lw=lw_co)

        # plots dashed line indicating fixed density if requested
        if plot_fixed_rho:
            # defines dictionary mapping x variable to corresponding fixed
            # density
            x_var_2_rho_fixed = {'polycation' : rho_p/2,
                                'cation' : rho_s,
                                'solvent' : 1 - rho_p - rho_s,
                                'polyelectrolyte' : rho_p,
                                'salt' : rho_s}
            # selects appropriate fixed density based on x variable
            rho_fixed = x_var_2_rho_fixed[x_var]
            # determines color based on which branch is closest
            if (rho_fixed - np.max(left_arr[liq_h2o])) > \
                (np.min(right_arr[liq_h2o]) - rho_fixed):
                # coacervate branch is closest to fixed density
                color = c_co
            else:
                # supernatant branch is closest to fixed density
                color = c_sup

            # plots fixed density as vertical dashed line
            ax.plot([rho_fixed, rho_fixed], ax.get_ylim(), '--', color=color,
                        lw=lw_sup)

    # determines units of density to display on plot
    if beads_2_M is not None:
        units_rho = 'mol/L'
    else:
        units_rho = 'beads/sigma^3'

    # formats plot
    format_binodal(ax, x_label, units_rho, T_range, x_lim=x_lim,
                    T_cels=T_cels, square_box=square_box, show_lgnd=show_lgnd)

    return ax


def binodal_custom_rho_rho(data, lB_list, rho_p_list, rho_s_list,
                    beads_2_M, show_tie_line=True,
                    cmap_name='plasma', colors=None, sigma=None,
                    marker='o', fill_left='none', fill_right='full',
                    lgnd_out=True, tol=1E-4, ms=10, T_cels=False, show_lB=False,
                    T_range=[273.15, 373.15], lw=2, square_box=False, ax=None,
                    colors_symbols=None, mew=1.5, x_lim=None, y_lim=None,
                    show_lgnd=True):
    """
    Plots the binodal as a function of salt density and polyelectrolyte
    density. Different Bjerrum lengths/temperatures are represented by
    different trend lines.

    Returns
    -------
    None.

    """
    # variables defining order of plotted objects
    back = 0
    front = 10
    # lists symbols for plotting overall composition
    sym_list = ['*', '^', 's', '<', '>', 'v', '+', 'x']

    # creates list of colors for each value of rho_p
    if colors is None:
        colors = get_colors(cmap_name, len(lB_list))

    # determines units
    if beads_2_M != 1:
        units_rho = 'mol/L'
    else:
        units_rho = r'beads/$\sigma^3$'

    # creates figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # loops through each temperature / Bjerrum length in data
    for i, lB in enumerate(lB_list):
        df = data[lB]

        # loads binodal data for supernatant (I) and coacervate (II)
        # doubles polycation concentration to include polyanion in polymer
        # concentration
        ion_I_list = list(beads_2_M*df['rhoCI'])
        ion_II_list = list(beads_2_M*df['rhoCII'])
        polymer_I_list = list(2*beads_2_M*df['rhoPCI'])
        polymer_II_list = list(2*beads_2_M*df['rhoPCII'])
        # critical points
        polymer_c = polymer_I_list[-1]
        ion_c = ion_I_list[-1]

        # computes temperature
        T = pe.lB_2_T(lB, sigma=sigma)
        if T_cels:
            T_unit = r'$^{\circ}$C'
            T -= K_2_C
        else:
            T_unit = ' K'

        # plots tie lines and overall composition
        for j, rho_pair in enumerate(zip(rho_p_list, rho_s_list)):
            rho_p, rho_s = rho_pair

            results = nacl.fixed_rho_total(data, rho_p, rho_s, beads_2_M)
            rho_PCI_list = results['rho_PCI']
            rho_PCII_list = results['rho_PCII']
            rho_CI_list = results['rho_CI']
            rho_CII_list = results['rho_CII']
            lB_arr = np.asarray(results['lB'])
            alpha = results['alpha']
            # converts to arrays of polymer and salt concentrations
            rho_p_I = 2*np.asarray(rho_PCI_list)
            rho_s_I = np.asarray(rho_CI_list)
            rho_p_II = 2*np.asarray(rho_PCII_list)
            rho_s_II = np.asarray(rho_CII_list)

            # continues if no T in range has 2 phases for concentration
            # finds closest match given Bjerrum length
            try:
                i_tie = np.where(np.abs(lB_arr - lB) < tol)[0][0]
            except:
                print('lB = {0:.3f} gives 1 phase for'.format(lB) + \
                      ' rho_p = {0:.3f} [{1:s}],'.format(rho_p, units_rho) + \
                          'rho_s = {0:.3f} [{1:s}].'.format(rho_s, units_rho))
                continue

            # tie line
            if show_tie_line:
                ax.plot([rho_p_I[i_tie], rho_p_II[i_tie]],
                        [rho_s_I[i_tie], rho_s_II[i_tie]], '--',
                        color='k', lw=lw, zorder=back)
            # supernatant
            ax.plot(rho_p_I[i_tie], rho_s_I[i_tie], color=colors[i],
                    marker='o', fillstyle='none', zorder=front)
            # coacervate
            ax.plot(rho_p_II[i_tie], rho_s_II[i_tie], color=colors[i],
                    marker='o', fillstyle='none', zorder=front)

            # plots overall composition last time through
            if i == len(lB_list)-1:
                short = {'mol/L' : 'M', 'beads/sigma^3' : r'$\sigma^{-3}$'}
                if sym_list[j] == '*':
                    ms_boost = 4
                else:
                    ms_boost = 0
                # if provided, can specify marker face color
                if colors_symbols is not None:
                    mfc = colors_symbols[j]
                else:
                    mfc = 'w'
                # plots symbol representing composition
                ax.plot(rho_p, rho_s, marker=sym_list[j], markerfacecolor=mfc,
                        ms=ms+ms_boost, markeredgecolor='k',
                        markeredgewidth=mew, lw=0,
                        label=r'$\rho_p = $ ' + '{0:.2f} {1:s}'.format(rho_p,
                        short[units_rho]) + r', $\rho_s = $ ' + \
                        '{0:.2f} {1:s}'.format(rho_s, short[units_rho]),
                        zorder=front)


        # plots binodal, flipping coacervate order to be in order
        label = r'$T = $' + '{0:d}{1:s}'.format(int(T), T_unit)
        if show_lB:
            label += r', $l_B = $ ' + '{0:.3f}'.format(lB)
        ax.plot(polymer_I_list + polymer_II_list[::-1],
                ion_I_list + ion_II_list[::-1],
                color=colors[i], lw=lw,
                label=label, zorder=back)

        # plots critical point
        ax.plot(polymer_c, ion_c, marker='o',
                fillstyle='full', color=colors[i], zorder=front)


    # formats plot
    x_label = r'$\rho_p$'
    y_label = r'$\rho_s$ [' + units_rho + ']'
    # determines component with varied concentration
    name_pair = ['Polymer', 'Salt']
    format_binodal(ax, x_label, units_rho, T_range, y_label=y_label,
                    x_lim=x_lim, y_lim=y_lim, lgnd_out=lgnd_out,
                    square_box=square_box, show_lgnd=show_lgnd)

    return ax


def binodal_line_3d(data, mode='lines', ms=8, op=0.1,
                    c1='black', c2='black', lw=8, fig=None):
    """Plots line binodal in 3d plot."""
    x1, y1, z1, x2, y2, z2 = data
    fig = line_3d(x1, y1, z1, mode=mode, ms=ms, op=op, c=c1, lw=lw, fig=fig)
    fig = line_3d(x2, y2, z2, mode=mode, ms=ms, op=op, c=c2, lw=lw, fig=fig)

    return fig


def binodal_proj_fixed_conc(data, mu_salt_folder, rho_salt_M_list, color_list,
                          T_range, sigma, z_name, beads_2_M, lB_list,
                          lB_color_list, T_cels=False, marker='o', show_lB=False,
                          fill_left='none', fill_right='full', lw_sup=1, lw_co=3,
                          lw_lB=2, naming_structure='NA(100)NB(100)*', ext='PD',
                          figsize=None, vertical=True):
    """
    Computes binodal projected onto three different planes (polymer-temperature,
    salt-temperature, and polymer-salt) at fixed concentration of salt in a
    saltwater reservoir.

    show_lB : bool, optional
        If True, will show Bjerrum length in legend
    """
    ### Formats Figure
    # creates figure to plot the three 2D projections in a single row
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)

    ### Creates Axes
    if vertical:
        h = 3 # 3 plots high
        w = 1 # 1 plot wide
    else:
        h = 1 # 1 plot high
        w = 3 # 3 plots wide
    # polymer-T projection
    ax_pT = fig.add_subplot(h, w, 1)
    # salt-T projection
    ax_sT = fig.add_subplot(h, w, 2)
    # polymer-salt projection
    ax_ps = fig.add_subplot(h, w, 3)

    # computes binodal at different saltwater reservoir concentrations
    # and plots on each of the three projections
    for rho_salt_M, color in zip(rho_salt_M_list, color_list):
        # converts mol/L to beads/sigma^3
        rho_salt = rho_salt_M / beads_2_M

        # makes dataframe of binodal for fixed salt reservoir concentration
        df_mu = nacl.make_df_mu(data, mu_salt_folder, rho_salt, T_range, sigma,
                                naming_structure=naming_structure, ext=ext)
        rho_p_I, rho_s_I, T_arr, rho_p_II, rho_s_II, _ = nacl.extract_df_mu_data(df_mu, z_name)

        # computes temperature and identifies data within range
        liq_h2o = np.logical_and(T_arr >= T_range[0], T_arr <= T_range[1])
        # converts temperature from Kelvin to Celsius
        if T_cels:
            T_arr -= K_2_C

        # creates labels
        label_sup = r'$\rho_s^{res} = $' + '{0:.2f} M, supernatant'.format(rho_salt_M)
        label_co = r'$\rho_s^{res} = $' + '{0:.2f} M, coacervate'.format(rho_salt_M)

        # polymer-T projection
        ax_pT.plot(rho_p_I[liq_h2o], T_arr[liq_h2o], color=color, marker=marker,
                   fillstyle=fill_left, label=label_sup, lw=lw_sup)
        ax_pT.plot(rho_p_II[liq_h2o], T_arr[liq_h2o], color=color, marker=marker,
                  fillstyle=fill_right, label=label_co, lw=lw_co)

        # salt-T projection
        ax_sT.plot(rho_s_I[liq_h2o], T_arr[liq_h2o], color=color, marker=marker,
                  fillstyle=fill_left, label=label_sup, lw=lw_sup)
        ax_sT.plot(rho_s_II[liq_h2o], T_arr[liq_h2o], color=color, marker=marker,
                  fillstyle=fill_right, label=label_co, lw=lw_co)

        # polymer-salt projection
        ax_ps.plot(rho_p_I[liq_h2o], rho_s_I[liq_h2o], color=color, label=label_sup, lw=lw_sup, zorder=10)
        ax_ps.plot(rho_p_II[liq_h2o], rho_s_II[liq_h2o], color=color, label=label_co, lw=lw_co, zorder=10)

    # plots isothermal binodal slices in polymer-salt plane
    for lB, lB_color in zip(lB_list, lB_color_list):
        df = data[lB]
        T = pe.lB_2_T(lB, sigma=sigma)

        # loads binodal data for supernatant (I) and coacervate (II)
        # doubles polycation concentration to include polyanion in polymer
        # concentration
        ion_I_list = list(beads_2_M*df['rhoCI'])
        ion_II_list = list(beads_2_M*df['rhoCII'])
        polymer_I_list = list(2*beads_2_M*df['rhoPCI'])
        polymer_II_list = list(2*beads_2_M*df['rhoPCII'])
        # critical points
        polymer_c = polymer_I_list[-1]
        ion_c = ion_I_list[-1]
        # units for temperature
        if T_cels:
            T_unit = r'$^{\circ}$C'
            T -= K_2_C
        else:
            T_unit = ' K'
        # plots binodal, flipping coacervate order to be in order
        label = r'$T = $' + '{0:d}{1:s} '.format(int(T), T_unit)
        if show_lB:
            label += r'$l_B = $ ' + '{0:.3f}'.format(lB)
        ax_ps.plot(polymer_I_list + polymer_II_list[::-1],
                ion_I_list + ion_II_list[::-1], color=lB_color, lw=lw_lB,
                label=label, zorder=0)


        # plots critical point
        ax_ps.plot(polymer_c, ion_c, marker='o',
                fillstyle='full', color=lB_color)


    return fig, ax_pT, ax_sT, ax_ps


def binodal_rho_rho(data, lB_list, rho_var_list, rho_fix,
                    ch_var, beads_2_M, show_tie_line=True,
                    cmap_name='plasma', colors=None, sigma=None, title=None,
                    marker='o', fill_left='none', fill_right='full',
                    lgnd_out=True, tol=1E-4, ms=10, T_cels=False, show_lB=False,
                    T_range=[273.15, 373.15], lw=2, square_box=False, ax=None):
    """
    Plots the binodal as a function of salt density and polyelectrolyte
    density. Different Bjerrum lengths/temperatures are represented by
    different trend lines.

    Returns
    -------
    None.

    """
    # variables defining order of plotted objects
    back = 0
    front = 10
    # lists symbols for plotting overall composition
    sym_list = ['*', '^', 's', '<', '>', 'v', '+', 'x']

    # creates dictionary to order fixed and varied densities properly
    d = get_plot_dict_p_s(ch_var)

    # creates list of colors for each value of rho_p
    if colors is None:
        colors = get_colors(cmap_name, len(lB_list))

    # determines units
    if beads_2_M != 1:
        units_rho = 'mol/L'
    else:
        units_rho = r'beads/$\sigma^3$'

    # creates figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # loops through each temperature / Bjerrum length in data
    for i, lB in enumerate(lB_list):
        df = data[lB]

        # loads binodal data for supernatant (I) and coacervate (II)
        # doubles polycation concentration to include polyanion in polymer
        # concentration
        ion_I_list = list(beads_2_M*df['rhoCI'])
        ion_II_list = list(beads_2_M*df['rhoCII'])
        polymer_I_list = list(2*beads_2_M*df['rhoPCI'])
        polymer_II_list = list(2*beads_2_M*df['rhoPCII'])
        # critical points
        polymer_c = polymer_I_list[-1]
        ion_c = ion_I_list[-1]

        # computes temperature
        T = pe.lB_2_T(lB, sigma=sigma)
        if T_cels:
            T_unit = r'$^{\circ}$C'
            T -= K_2_C
        else:
            T_unit = ' K'

        # plots tie lines and overall composition
        for j, rho_var in enumerate(rho_var_list):
            rho_pair = np.array([rho_var, rho_fix])
            rho_p, rho_s = rho_pair[d['order']]


            results = nacl.fixed_rho_total(data, rho_p, rho_s, beads_2_M)
            rho_PCI_list = results['rho_PCI']
            rho_PCII_list = results['rho_PCII']
            rho_CI_list = results['rho_CI']
            rho_CII_list = results['rho_CII']
            lB_arr = np.asarray(results['lB'])
            alpha = results['alpha']
            # converts to arrays of polymer and salt concentrations
            rho_p_I = 2*np.asarray(rho_PCI_list)
            rho_s_I = np.asarray(rho_CI_list)
            rho_p_II = 2*np.asarray(rho_PCII_list)
            rho_s_II = np.asarray(rho_CII_list)

            # continues if no T in range has 2 phases for concentration
            # finds closest match given Bjerrum length
            try:
                i_tie = np.where(np.abs(lB_arr - lB) < tol)[0][0]
            except:
                print('lB = {0:.3f} gives 1 phase for'.format(lB) + \
                      ' rho_p = {0:.3f} [{1:s}],'.format(rho_p, units_rho) + \
                          'rho_s = {0:.3f} [{1:s}].'.format(rho_s, units_rho))
                continue

            # tie line
            if show_tie_line:
                ax.plot([rho_p_I[i_tie], rho_p_II[i_tie]],
                        [rho_s_I[i_tie], rho_s_II[i_tie]], '--',
                        color='k', lw=lw, zorder=back)
            # supernatant
            ax.plot(rho_p_I[i_tie], rho_s_I[i_tie], color=colors[i],
                    marker='o', fillstyle='none', zorder=front)
            # coacervate
            ax.plot(rho_p_II[i_tie], rho_s_II[i_tie], color=colors[i],
                    marker='o', fillstyle='none', zorder=front)

            # plots overall composition last time through
            if i == len(lB_list)-1:
                short = {'mol/L' : 'M', 'beads/sigma^3' : r'$\sigma^{-3}$'}
                if sym_list[j] == '*':
                    ms_boost = 4
                else:
                    ms_boost = 0
                ax.plot(rho_p, rho_s, marker=sym_list[j], markerfacecolor='w',
                        ms=ms+ms_boost, markeredgecolor='k',
                        markeredgewidth=1.5, lw=0,
                        label=r'$\rho_p = $ ' + '{0:.2f} {1:s}'.format(rho_p,
                        short[units_rho]) + r', $\rho_s = $ ' + \
                        '{0:.2f} {1:s}'.format(rho_s, short[units_rho]),
                        zorder=front)


        # plots binodal, flipping coacervate order to be in order
        label = r'$T = $' + '{0:d}{1:s}'.format(int(T), T_unit)
        if show_lB:
            label += r', $l_B = $ ' + '{0:.3f}'.format(lB)
        ax.plot(polymer_I_list + polymer_II_list[::-1],
                ion_I_list + ion_II_list[::-1],
                color=colors[i], lw=lw,
                label=label, zorder=front)

        # plots critical point
        ax.plot(polymer_c, ion_c, marker='o',
                fillstyle='full', color=colors[i], zorder=front)


    # formats plot
    x_label = r'$\rho_p$'
    y_label = r'$\rho_s$ [' + units_rho + ']'
    # determines component with varied concentration
    name_pair = ['Polymer', 'Salt']
    name_var = name_pair[d['order'][0]]
    if title is None:
        title = 'Vary Overall {0:s} Concentration'.format(name_var)
    format_binodal(ax, x_label, units_rho, T_range, y_label=y_label, title=title,
                        lgnd_out=lgnd_out, square_box=square_box)

    return ax

def binodal_surf_3d(data, mode='markers', ms=4, op=0.01,
                    c1='blue', c2='red', lw=0, fig=None):
    """Plots surface binodal in 3d."""
    x1, y1, z1, x2, y2, z2 = data
    if fig == None:
        fig = go.Figure()
    # plots phase I (supernatant) of full binodal
    fig = fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode=mode,
        marker=dict(
            size=ms,
            opacity=op,
            color=c1
        ),
        line=dict(
            color=c1,
            width=lw,
        ),
    ))

    # plots phase II (coacervate) of full binodal
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode=mode,
        marker=dict(
            size=ms,
            opacity=op,
            color=c2
        ),
        line=dict(
            color=c2,
            width=lw,
        ),
    ))

    return fig


def binodal_surf_3d_batch(data_3d, op, ms, lw, mode, fig=None, skip=[]):
    """
    Plots batch of data for a 3d surface binodal.
    """
    # extracts data
    x1_coll, y1_coll, z1_coll, x2_coll, y2_coll, z2_coll = data_3d
    z_arr = np.unique(z1_coll)

    # plots data at each z value
    for (i, z) in enumerate(z_arr):
        # skips indices requested
        if i in skip:
            continue
        # extracts data corresponding to current z value (T or lB)
        x1 = x1_coll[z1_coll==z]
        y1 = y1_coll[z1_coll==z]
        z1 = z1_coll[z1_coll==z]
        x2 = x2_coll[z2_coll==z]
        y2 = y2_coll[z2_coll==z]
        z2 = z2_coll[z2_coll==z]

        # plots dataon 3D plot
        fig = binodal_surf_3d((x1, y1, z1, x2, y2, z2), op=op, ms=ms, lw=lw,
                                mode=mode, fig=fig)

    return fig


def binodal_vary_conc(mu_salt_folder, data, rho_salt_list, beads_2_M, qty,
                     x_var='polycation', x_label=r'$\rho_{PSS}$', sigma=None,
                     T_range=[273,373], cmap_name='plasma', colors=None,
                     marker='o', fill_left='none', fill_right='full',
                    lgnd_out=True):
    """
    LEGACY

    Plots the binodal for different average densities of polymer.
    qty : string
        The quantity from df to return. Options include 'rhoPC', 'rhoPA',
        'rhoC', and 'rhoA'.
    """
    # creates list of colors for each value of rho_p
    if colors is None:
        colors = get_colors(cmap_name, len(rho_salt_list))

    # creates figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, rho_salt in enumerate(rho_salt_list):

        # plots binodal for low polymer concentration [M]
        mu_conc = nacl.get_mu_conc(mu_salt_folder, data, rho_salt, beads_2_M=beads_2_M)
        try:
            lB_arr, rho_PCI_list, rho_PCII_list = nacl.fixed_conc(mu_conc, data, qty, beads_2_M=beads_2_M)
        except:
            continue

        # selects the x-axis data
        left_arr = np.array(rho_PCI_list)
        right_arr = np.array(rho_PCII_list)

        # computes temperature and identifies data within range
        T_arr = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma)
        liq_h2o = np.logical_and(T_arr >= T_range[0], T_arr <= T_range[1])

        # determines units
        if beads_2_M is not None:
            units_rho = '[mol/L]'
        else:
            units_rho = '[beads/sigma^3]'


        # left binodal
        ax.plot(left_arr[liq_h2o], T_arr[liq_h2o], color=colors[i],
                marker=marker, fillstyle=fill_left,
               label=r'$\rho_{salt} = $' + '{0:.2f} {1:s}, supernatant' \
                   .format(rho_salt, units_rho))
        # right binodal
        ax.plot(right_arr[liq_h2o], T_arr[liq_h2o], color=colors[i],
                marker=marker, fillstyle=fill_right,
                label=r'$\rho_{salt} = $' + \
                    '{0:.2f} {1:s}, coacervate'.format(rho_salt, units_rho))



    # formats plot
    ax.set_ylim(T_range)
    ax.set_xlabel(x_label + ' ' + units_rho, fontsize=16)
    ax.set_ylabel(r'$T$ [K]', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title('Effect of Salt Reservoir on Binodal', fontsize=16)

    # put legend outside of plot box
    if lgnd_out:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height])
        legend_x = 1
        legend_y = 0.5
        plt.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y), fontsize=12)
    else:
        plt.legend(fontsize=12)

    return ax


def binodal_vary_f(data, f_list, color_list, T_cels=True, x_label=r'$\rho_p$',
                    units_rho='M', T_range=[273.15, 373.15], lw1=1, lw2=4,
                    square_box=True, show_lgnd=False, ax=None):
    """
    Plots binodal projected onto coordinate plane for different charge fractions
    f.
    """
    # creates figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for f, color in zip(f_list, color_list):
        # creates labels
        label_sup = r'$f$ =' + ' {0:.2f}   supernatant'.format(f)
        label_co = r'$f$ =' + ' {0:.2f}   coacervate'.format(f)

        # extracts data
        T_arr, rho_p_I, rho_p_II = data[f]

        # polymer-T projection
        ax.plot(rho_p_I, T_arr, color=color, label=label_sup, lw=lw1)
        ax.plot(rho_p_II, T_arr, color=color, label=label_co, lw=lw2)

    # formats plot
    format_binodal(ax, x_label, units_rho, T_range, T_cels=T_cels,
                                    square_box=square_box, show_lgnd=show_lgnd)

    return ax


def binodal_vary_N(data, N_list, color_list, T_cels=True, x_label=r'$\rho_p$',
                    units_rho='M', T_range=[273.15, 373.15], lw1=1, lw2=4,
                    square_box=True, show_lgnd=False, ax=None):
    """
    Plots binodal projected onto coordinate plane for different degrees of
    polymerization N.
    """
    # creates figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(131)

    for N, color in zip(N_list, color_list):
        # extracts data for given N
        T_arr, rho_p_I, rho_p_II = data[N]
        # creates labels
        label_sup = r'$N$ =' + ' {0:d}   supernatant'.format(N)
        label_co = r'$N$ =' + ' {0:d}   coacervate'.format(N)

        # polymer-T projection
        ax.plot(rho_p_I, T_arr, color=color, label=label_sup, lw=lw1)
        ax.plot(rho_p_II, T_arr, color=color, label=label_co, lw=lw2)

    # formats plot
    format_binodal(ax, x_label, units_rho, T_range, T_cels=T_cels,
                                    square_box=square_box, show_lgnd=show_lgnd)

    return ax


def binodal_vary_sigma(data, sigma_list, color_list,
                    T_cels=True, x_label=r'$\rho_p$', units_rho='M',
                    T_range=[273.15, 373.15], lw1=1, lw2=4, square_box=True,
                    show_lgnd=False, x_lim=None, ax=None):
    """
    Plots binodal projected onto coordinate plane for different charge fractions
    f.
    """
    # creates figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for sigma, color in zip(sigma_list, color_list):
        # creates labels
        label_sup = r'$\sigma$ =' + ' {0:.1f}'.format(sigma*m_2_A) + r' $\AA$   supernatant'
        label_co = r'$\sigma$ =' + ' {0:.1f}'.format(sigma*m_2_A) + r' $\AA$   coacervate'

        # extracts data
        T_arr, rho_p_I, rho_p_II = data[sigma]

        # polymer-T projection
        ax.plot(rho_p_I, T_arr, color=color, label=label_sup, lw=lw1)
        ax.plot(rho_p_II, T_arr, color=color, label=label_co, lw=lw2)

    # formats plot
    format_binodal(ax, x_label, units_rho, T_range, T_cels=T_cels, x_lim=x_lim,
                                    square_box=square_box, show_lgnd=show_lgnd)

    return ax


def binodal_vary_rho(data, rho_var_list, rho_fix, ch_var, beads_2_M,
                     x_var='polycation', x_label=r'$\rho_{PSS}$', sigma=None,
                     T_range=[273.15,373.15], cmap_name='plasma', colors=None,
                     marker='o', fill_left='none', fill_right='full', lw_sup=1,
                     lw_co=3, lgnd_out=True, lw=1, x_lim=None, T_cels=False,
                     title=None, c_sup='#1414FF', c_co='#FF0000', ls_sup='-',
                     square_box=False, ax=None):
    """
    Plots the binodal for different average densities of polymer.

    If T_cels is True, converts the temperature from Kelvin to Celsius
    """
    # creates dictionary of values based on which component's density is varied
    d = get_plot_dict_p_s(ch_var)

    # creates list of colors for each value of rho_p
    if colors is None:
        if cmap_name is not None:
            colors = get_colors(cmap_name, len(rho_var_list))

    # creates figure
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ### Plots figure
    for i, rho_var in enumerate(rho_var_list):

        # plots binodal for low polymer concentration [M]
        rho_pair = np.array([rho_var, rho_fix])
        rho_p, rho_s = rho_pair[d['order']]
        results = nacl.fixed_rho_total(data, rho_p, rho_s, beads_2_M)
        rho_PCI_list = results['rho_PCI']
        rho_PCII_list = results['rho_PCII']
        rho_CI_list = results['rho_CI']
        rho_CII_list = results['rho_CII']
        rho_PAI_list = results['rho_PAI']
        rho_PAII_list = results['rho_PAII']
        rho_AI_list = results['rho_AI']
        rho_AII_list = results['rho_AII']
        lB_arr = results['lB']
        alpha = results['alpha']

        # selects the x-axis data
        if x_var == 'polycation':
            left_arr = np.array(rho_PCI_list)
            right_arr = np.array(rho_PCII_list)
        elif x_var == 'polyanion':
            left_arr = np.array(rho_PAI_list)
            right_arr = np.array(rho_PAII_list)
        elif x_var == 'cation':
            left_arr = np.array(rho_CI_list)
            right_arr = np.array(rho_CII_list)
        elif x_var == 'anion':
            left_arr = np.array(rho_AI_list)
            right_arr = np.array(rho_AII_list)
        elif x_var == 'solvent':
            left_arr = pe.calc_rho_solv(rho_PCI_list,
                                          rho_CI_list,
                                          beads_2_M)
            right_arr = pe.calc_rho_solv(rho_PCII_list,
                                           rho_CII_list,
                                           beads_2_M)
        elif x_var == 'polyelectrolyte':
            left_arr = np.array(rho_PCI_list) + np.array(rho_PAI_list)
            right_arr = np.array(rho_PCII_list) + np.array(rho_PAII_list)
        elif x_var == 'salt':
            left_arr = np.array(rho_CI_list)
            right_arr = np.array(rho_CII_list)
        else:
            print('Error. Invalid x variable in plot.binodal_vary_rho().')

        # computes temperature and identifies data within range
        T_arr = pe.lB_2_T_arr(lB_arr, T_range, sigma=sigma)
        liq_h2o = np.logical_and(T_arr >= T_range[0], T_arr <= T_range[1])
        # converts temperature from Kelvin to Celsius
        if T_cels:
            T_arr -= K_2_C

        # assigns separate colors to coacervate and supernatant if not specified
        if colors is not None:
            c_sup = colors[i]
            c_co = colors[i]

        # supernatant
        ax.plot(left_arr[liq_h2o], T_arr[liq_h2o], color=c_sup,
                marker=marker, fillstyle=fill_left, ls=ls_sup,
               label=r'$\rho_' + d['ch_var'] + ' = $' + '{0:.2f} M, supernatant' \
                   .format(rho_var), lw=lw_sup)
        # coacervate
        ax.plot(right_arr[liq_h2o], T_arr[liq_h2o], color=c_co,
                marker=marker, fillstyle=fill_right,
                label=r'$\rho_' + d['ch_var'] + ' = $' + \
                    '{0:.2f} M, coacervate'.format(rho_var), lw=lw_co)

    # determines units of density to display on plot
    if beads_2_M is not None:
        units_rho = 'mol/L'
    else:
        units_rho = 'beads/sigma^3'

    # formats plot
    if title is None:
        title = 'Effect of {0:s} on Binodal, {1:s} = {2:.2f} M' \
                 .format(d['name_var'], r'$\rho_' + d['ch_fix'] + '$', rho_fix)
    format_binodal(ax, x_label, units_rho, T_range, title=title, x_lim=x_lim,
                    T_cels=T_cels, square_box=square_box)

    return ax



def fig4(data_pred, df_exp, rho_s_raw_list, rho_p_raw, sigma, T_range,
                        lw=3, c_sup='#1414FF', c_co='#FF0000', ms=11,
                        mfc='w', mew=1.5, x_lim=None, x_label=r'$\rho_{PSS}$',
                        conv_vals=False, tol=1E-6, show_lgnd=False,
                        figsize=None, pad=3, vertical=False, plot_errorbars=False):
    """
    Validates fit of sigma to experiments.
    """
    # computes conversion from beads/sigma^3 to mol/L
    beads_2_M = pe.get_beads_2_M(sigma, SI=True)

    # creates figure
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)

    # determines arrangement of subplots
    if vertical:
        h = len(rho_s_raw_list) # many plots high
        w = 1 # 1 plot wide
    else:
        h = 1 # 1 plot high
        w = len(rho_s_raw_list) # many plots wide

    # Plots figure
    for i, rho_s_raw in enumerate(rho_s_raw_list):
        if conv_vals:
            rho_p, rho_s = nacl.conv_ali_conc(df_exp, rho_p_raw, rho_s_raw)

        # creates subplot
        ax = fig.add_subplot(h, w, i+1)
        # polymer-temperature plane
        ax = binodal_custom_rho(data_pred, [rho_p], [rho_s], beads_2_M,
                                   x_var='polycation', x_label=x_label,
                                   x_lim=x_lim, sigma=sigma, T_range=T_range,
                                   marker='', lw=lw, lw_sup=lw, lw_co=lw,
                                   colors=None, cmap_name=None, T_cels=True,
                                   c_sup=c_sup, c_co=c_co, ls_sup='--',
                                   square_box=True, show_lgnd=show_lgnd, ax=ax)

        # plots experimental results
        for i in range(len(df_exp)):
            rho_p_exp, rho_s_exp, T_exp, \
            rho_p_sup, rho_p_co, s_rho_p_sup, \
            s_rho_p_co = nacl.read_df_exp(df_exp, i, conv_vals=conv_vals, 
                                            read_sigma=plot_errorbars)
            if (rho_p_exp == rho_p) and (rho_s_exp == rho_s):
                # plots supernatant and coacervate compositions
                rho_pss_sup = rho_p_sup/2
                rho_pss_co = rho_p_co/2
                if plot_errorbars:
                    s_rho_pss_sup = s_rho_p_sup/2
                    s_rho_pss_co = s_rho_p_co/2
                    ax.errorbar(rho_pss_sup, T_exp, xerr=s_rho_pss_sup, lw=0, marker='o', ms=ms,
                            markerfacecolor=mfc, markeredgewidth=mew, elinewidth=1,
                            markeredgecolor=c_sup, label='Ali et al. (2019), supernatant')
                    ax.errorbar(rho_pss_co, T_exp, xerr=s_rho_pss_co, lw=0, marker='o', ms=ms,
                            markerfacecolor=c_co, markeredgewidth=mew, elinewidth=1,
                            markeredgecolor=c_co, label='Ali et al. (2019), coacervate')
                else:
                    ax.plot(rho_pss_sup, T_exp, lw=0, marker='o', ms=ms,
                            markerfacecolor=mfc, markeredgewidth=mew,
                            markeredgecolor=c_sup, label='Ali et al. (2019), supernatant')
                    ax.plot(rho_pss_co, T_exp, lw=0, marker='o', ms=ms,
                            markerfacecolor=c_co, markeredgewidth=mew,
                            markeredgecolor=c_co, label='Ali et al. (2019), coacervate')

    # pads subplots with whitespace
    fig.tight_layout(pad=pad)

    return fig


def figs3(data_folder_N, data_folder_f, data_folder_sigma,
        mu_salt_folder_N, mu_salt_folder_f, mu_salt_folder_sigma,
            rho_s_M_N, rho_s_M_f, rho_s_M_sigma, ext_N, ext_f, ext_sigma,
            N_list, f_list, sigma_list, color_list_N, color_list_f,
            color_list_sigma, sigma_fixed, x_lim_sigma=[0,6], figsize=None, pad=3,
            naming_structure_sigma='NA(100)NB(100)lB(*)', lB_lo=1.3, lB_hi=2.398):
    """Plots Figure S3 of SI showing effects of N, f, and sigma on
    binodal projections in polymer-temperature plane."""
    # creates figure
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)

    ### Effect of varying N
    print('loading N data')
    # adds subplot
    axN = fig.add_subplot(131)
    # extracts data
    data_vary_N = nacl.binodal_vary_N_data(data_folder_N, mu_salt_folder_N,
                        rho_s_M_N, N_list, sigma=sigma_fixed, ext=ext_N)
    # plots data
    print('plotting N data')
    _ = binodal_vary_N(data_vary_N, N_list, color_list_N, ax=axN)

    ### Effect of varying charge fraction f
    # adds subplot
    axf = fig.add_subplot(132)
    # extracts data
    print('loading f data')
    data_vary_f = nacl.binodal_vary_f_data(data_folder_f, mu_salt_folder_f,
                                            rho_s_M_f, f_list,
                                            sigma=sigma_fixed, ext=ext_f)
    # plots data
    print('plotting f data')
    _ = binodal_vary_f(data_vary_f, f_list, color_list_f, ax=axf)

    ### Effect of varying sigma
    axsigma = fig.add_subplot(133)
    # laads all data
    print('loading sigma data')
    data = nacl.load_data(data_folder_sigma, ext=ext_sigma,
            naming_structure=naming_structure_sigma, lB_lo=lB_lo, lB_hi=lB_hi)
    # extracts relevant data
    data_vary_sigma = nacl.binodal_vary_sigma_data(data, mu_salt_folder_sigma,
                                    rho_s_M_sigma, sigma_list, ext=ext_sigma)
    # plots data
    print('plotting sigma data')
    _ = binodal_vary_sigma(data_vary_sigma, sigma_list,
                                color_list_sigma, ax=axsigma, x_lim=x_lim_sigma)

    # pads subplots with whitespace
    fig.tight_layout(pad=pad)

    return fig


def compare_to_exp(data, beads_2_M, rho_p_list=[0.3], rho_s_list=[1.6, 1.85, 1.9],
                   N=100, f=1, sigma=4, t_fs=12, T_range=[273.15, 323.15]):
    """
    Compares predictions from data to the experiment in the Prabhu group.
    """
    # sets x and y axis limits
    x_lim = (-0.05, 1.3) # [mol/L]
    y_lim = (0, 60) # [C]
    # sets temperature range
    T_range = [273, 333]

    for rho_s in rho_s_list:
        for rho_p in rho_p_list:
            # computes polycation concentrations at different temperatures for fixed polymer and salt [mol/L]
            results = nacl.fixed_rho_total(data, rho_p, rho_s, beads_2_M)
            rho_PCI_list = results['rho_PCI']
            rho_PCII_list = results['rho_PCII']
            lB_arr = results['lB']

            # plots binodal
            title = '{0:.2f} M Salt, {1:.2f} M Polymer, N = {2:d}, f = {3:.2f}, sig = {4:.2f} A'.format(rho_s, rho_p, N, f, sigma)
            p = binodal(lB_arr, rho_PCI_list, rho_PCII_list, title=title,
                        beads_2_M=1, n_tie_lines=0, deg_C=True, T_range=T_range,
                        x_lim=x_lim, y_lim=y_lim, marker=False, line=True)
            p.title.text_font_size = '{0:d}pt'.format(t_fs)
            show(p)

    return


def crit_line_3d(data_cp, c_crit, lw_crit, fig):
    """
    Plots critical line in 3D, typically for 3D surface binodal plot.

    LEGACY
    """
    polymer_c_list, salt_c_list, z_arr = data_cp
    fig.add_trace(go.Scatter3d(
                    x=polymer_c_list,
                    y=salt_c_list,
                    z=z_arr,
                    mode='lines',
                    line=dict(
                        color=c_crit,
                        width=lw_crit,
                    ),
                ),
             )

    return fig


def fig1(data_3d, data_cp, data_z, data_mu, plot_params, fixed_T=True,
            fixed_salt=True, crit_line=True, fixed_comp=False,
            data_comp=None, data_outlines=None,  skip=[], plot_axes=True,
            outline_scale_factor=1.02, toc_fig=False, has_ucst=False,
            show_labels=True):
    """
    Plots Figure 1 from CCLS paper: 3d surface binodal, fixed T 2d line binodal,
    fixed salt reservoir concentration 2d line binodal, and critical line.
    """
    # if Table of Contents (TOC) figure, removes all but LCST
    if toc_fig:
        fixed_salt = True
        crit_line = True
        fixed_comp = False
        fixed_T = False
    x_range, y_range, z_range, eye_xyz, op, ms_bin, lw_bin, \
    lw_fix, lw_crit, lw_outline, c1_T, c2_T, c1_fix, c2_fix, \
    c_crit, c_outline, mode, width, height, fs, offset = plot_params
    x, y, z = eye_xyz
    # plots 3d surface binodal
    fig = binodal_surf_3d_batch(data_3d, op, ms_bin, lw_bin, mode, skip=skip)

    if crit_line:
        # plots critical line
        fig = line_3d(*data_cp, c=c_crit, lw=lw_crit, fig=fig)
    if fixed_T:
        # plots binodal at fixed z value (temperature or Bjerrum length)
        fig = binodal_line_3d(data_z, fig=fig, lw=lw_fix, c1=c1_T, c2=c2_T)
    if fixed_salt:
        ### FIXED SALT CONCENTRATION ###
        # if there is a UCST, split the binodal in two
        if has_ucst:
            # identifies threshold between UCST and LCST by largest gap in z
            z1 = data_mu[2]
            z1_diff = np.diff(z1)
            i_thresh = np.argmax(z1_diff)
            thresh_ucst = (z1[i_thresh] + z1[i_thresh+1])/2
            # splits data below UCST and above LCST
            ucst_data = list(zip(*[(x1, y1, z1, x2, y2, z2) for x1, y1, z1, x2, y2, z2 in zip(*data_mu) if z1 < thresh_ucst]))
            lcst_data = list(zip(*[(x1, y1, z1, x2, y2, z2) for x1, y1, z1, x2, y2, z2 in zip(*data_mu) if z1 > thresh_ucst]))
            # plots UCST and LCST data separately
            fig = binodal_line_3d(ucst_data, fig=fig, lw=lw_fix, c1=c1_fix, c2=c2_fix)
            fig = binodal_line_3d(lcst_data, fig=fig, lw=lw_fix, c1=c1_fix, c2=c2_fix)
        else:
            # plots data for fixed saltwater reservoir concentration
            fig = binodal_line_3d(data_mu, fig=fig, lw=lw_fix, c1=c1_fix, c2=c2_fix)
    if fixed_comp:
        # plots binodal at fixed overall salt, polymer concentration #
        fig = binodal_line_3d(data_comp, fig=fig, lw=lw_fix, c1=c1_fix, c2=c2_fix)
    # plots outlines of the surface for definition
    if data_outlines is not None:
        for data_outline in data_outlines:
            data_outline_scaled = []
            for coord in data_outline:
                coord = outline_scale_factor*np.asarray(coord)
                data_outline_scaled += [coord]
            fig = binodal_line_3d(data_outline_scaled, c1=c_outline,
                                                    c2=c_outline, fig=fig)

    if plot_axes:
        # x-axis
        fig = line_3d(x_range, [offset, offset], [z_range[0] + offset,
                        z_range[0] + offset], lw=12, c=c_outline, fig=fig)
        # y-axis
        fig = line_3d([offset, offset], y_range, [z_range[0] + offset,
                        z_range[0] + offset], lw=12, c=c_outline, fig=fig)
        # z-axis
        fig = line_3d([offset, offset], [offset, offset], z_range,
                        c=c_outline, lw=12, fig=fig)

    ### FORMATS FIGURE ###
    fig.update_layout(
        scene = dict(xaxis = dict(range=x_range,),
                    yaxis = dict(range=y_range,),
                    zaxis = dict(range=z_range,),
                ),
        width = width,
        height = height,
        # changes initial view of figure
        scene_camera = dict(
                    eye=dict(x=x, y=y, z=z),
    #                 center=dict(x=0, y=0.3, z=0.3),
    #                 up=dict(x=0, y=0, z=1)
        ),
        font = dict(
                family='Arial',
                color='black',
                size=fs)
    )

    ### Cleanup
    # removes legend (too crowded to be off use)
    fig.update_layout(showlegend=False)

    #removes tick labels and axis titles (so I can add them myself)
    if not show_labels:
        fig.update_layout(
            scene = dict(xaxis = dict(showticklabels=False, title=''),
                         yaxis = dict(showticklabels=False, title=''),
                         zaxis = dict(showticklabels=False, title='',
                                      tickmode = 'linear',
                                      tick0 = 0,
                                      dtick = 50),
                    ),
        )

    return fig


def fig2a(rho_salt_M_list_list, data, mu_salt_folder,
                color_list, T_range, sigma, z_name,
                beads_2_M, lB_list, lB_color_list, pad,
                kwargs, units_rho='mol/L', show_lgnd=False, y_lim_T=(0, 100),
                rho_p_label=r'$\rho_p$', rho_s_label=r'$\rho_s$',
                y_lim_s=[0, 2.25]):
    """Plots Figure 2a of binodal projections at different saltwater concentrations."""
    for rho_salt_M_list in rho_salt_M_list_list:
        # plots binodal projections
        fig, ax_pT, ax_sT, \
        ax_ps = binodal_proj_fixed_conc(data, mu_salt_folder, rho_salt_M_list,
                                    color_list, T_range, sigma, z_name,
                                    beads_2_M, lB_list, lB_color_list,
                                    **kwargs)

        # formats plots
        ax_pT = format_binodal(ax_pT, rho_p_label, units_rho, T_range,
                                T_cels=kwargs['T_cels'], y_lim=y_lim_T,
                                show_lgnd=show_lgnd)
        ax_sT = format_binodal(ax_sT, rho_s_label, units_rho, T_range,
                                T_cels=kwargs['T_cels'], y_lim=y_lim_T,
                                show_lgnd=show_lgnd)
        ax_ps = format_binodal(ax_ps, rho_p_label, units_rho, T_range,
                                y_label=rho_s_label + ' [' + units_rho + ']',
                                show_lgnd=show_lgnd, y_lim=y_lim_s)

    # pads plots with whitespace
    fig.tight_layout(pad=pad)

    return fig


def fig2b(data, rho_p_list, rho_s_list, beads_2_M, lB_list, color_list,
        lB_color_list, kwargs, alpha_y_lim=(0.5,1.05),
        alpha_yticks=(0.5,0.75,1), figsize=None, pad=3, mew=0.5,
        show_lgnd=False):
    """Plots Figure 2b of binodal projections at different overall compositions."""
    ### Formats Figure
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)

    ### polymer-temperature plane ###
    ax1 = fig.add_subplot(221)
    _ = binodal_custom_rho(data, rho_p_list, rho_s_list, beads_2_M,
                                x_var='polyelectrolyte', x_label=r'$\rho_p$',
                               marker='', colors=color_list,
                               plot_fixed_rho=True, ax=ax1, show_lgnd=show_lgnd,
                               **kwargs)

    ### salt-temperature plane ###
    ax2 = fig.add_subplot(222)
    _ = binodal_custom_rho(data, rho_p_list, rho_s_list, beads_2_M,
                                x_var='salt', x_label=r'$\rho_s$', marker='',
                                colors=color_list, plot_fixed_rho=True,
                                ax=ax2, show_lgnd=show_lgnd, **kwargs)

    ### polymer-salt plane ###
    ax3 = fig.add_subplot(223)
    _ = binodal_custom_rho_rho(data, lB_list, rho_p_list, rho_s_list,
                            beads_2_M, colors=lB_color_list, mew=mew, ax=ax3,
                            show_lgnd=show_lgnd, colors_symbols=color_list,
                            **kwargs)

    ### volume fraction of supernatant vs. temperature ###
    ax4 = fig.add_subplot(224)
    _ = alpha_custom_rho(data, rho_p_list, rho_s_list, beads_2_M,
                            y_lim=alpha_y_lim, marker='',
                            colors=color_list, ax=ax4, show_lgnd=show_lgnd,
                            **kwargs)
    # customizes tick mark locations
    ax4.set_yticks(alpha_yticks)

    # pads subplots with whitespace
    fig.tight_layout(pad=pad)

    return fig


def fig3(data, lB_list, rho_p_fixed, rho_s_fixed, rho_p_varied, rho_s_varied,
        beads_2_M, kwargs, figsize=None, pad=3, vertical=True):
    """Plots Figure 3 of tie lines in polymer-salt plane."""
    # formats figure
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)

    # determines arrangement of subplots
    if vertical:
        h = 2 # 2 plots high
        w = 1 # 1 plot wide
    else:
        h = 1 # 1 plot high
        w = 2 # 2 plots wide

    ################ VARIES SALT CONCENTRATION ###############
    # creates subplot
    ax1 = fig.add_subplot(h, w, 1)
    # plots binodal
    rho_p_list = rho_p_fixed*np.ones([len(rho_s_varied)])
    rho_s_list = rho_s_varied
    _ = binodal_custom_rho_rho(data, lB_list, rho_p_list, rho_s_list,
                        beads_2_M, ax=ax1, show_lgnd=False, **kwargs)

    ############ VARIES POLYMER CONCENTRATION ####################
    # creates subplot
    ax2 = fig.add_subplot(h, w, 2)
    # plots binodal
    rho_p_list = rho_p_varied
    rho_s_list = rho_s_fixed*np.ones([len(rho_p_varied)])
    ax = binodal_custom_rho_rho(data, lB_list, rho_p_list, rho_s_list,
                        beads_2_M, ax=ax2, show_lgnd=False, **kwargs)

    # pads subplots with whitespace
    fig.tight_layout(pad=pad)

    return fig


def figs1(T_range, sigma, T_room_C=20, T_cels=True, figsize=(5,5),
          gridspec=10, lw=3, y_lim=[5.5,9.5], y_ticks=[6,7,8,9], d=0.5,
          ax_fs=16, tk_fs=16):
    """Plots Figure S1 of the SI of Bjerrum length vs. T for fixed and
    T-dependent dielectric constant."""
    # computes Bjerrum lengths
    T_arr, lB_A_arr, lB_0_A_arr = nacl.lB_comparison(T_range, sigma,
                                                            T_room_C=T_room_C)

    # creates figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                    gridspec_kw={'height_ratios': [gridspec,1]}, sharex=True)

    # adjusts temperature based on requested unit
    if T_cels:
        T_arr -= K_2_C
        unit_T = r'$^{\circ}C$'
    else:
        unit_T = 'K'

    # plots Bjerrum lengths
    ax1.plot(T_arr, lB_A_arr, lw=lw, label=r'$\epsilon(T)$')
    ax1.plot(T_arr, lB_0_A_arr, lw=lw,
                    label=r'$\epsilon(T) = \epsilon($' + \
                    '{0:d}'.format(int(T_room_C)) + r'$^{\circ}C)$')

    # formats plot
    ax2.set_xlabel(r'$T$ [' + unit_T + ']', fontsize=ax_fs)
    ax1.set_ylabel(r'$l_B$ $[\AA]$', fontsize=ax_fs)
    ax1.tick_params(axis='both', labelsize=tk_fs)
    ax2.tick_params(axis='both', labelsize=tk_fs)

    ### Creates broken axis
    # see: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html

    # set limits and ticks on upper axis
    ax1.set_ylim(y_lim)
    ax1.set_yticks(y_ticks)
    # lower axis
    ax2.set_ylim([0, 0.5])
    ax2.set_yticks([0])

    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(top=False, labeltop=False)  # don't put ticks or labels at top
    ax2.xaxis.tick_bottom()

    # plots diagonal hatch marks on y-axis--"d" is ratio of height to length
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    return fig


def format_binodal(ax, x_label, units_rho, T_range, y_label=None, title=None,
                    x_lim=None, y_lim=None, T_cels=False, lgnd_out=True,
                    square_box=True, show_lgnd=True):
    """
    Formats axes of a plot of the binodal projected onto a plane with
    temperature as the vertical axis.
    """
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.set_xlabel('{0:s} [{1:s}]'.format(x_label, units_rho), fontsize=18)
    # assumes that the y axis is temperature if another label is not given
    if y_label is None:
        T_unit = 'K'
        if T_cels:
            T_unit = r'$^{\circ}$C'
            T_range = [T - K_2_C for T in T_range]
        if y_lim is None:
            ax.set_ylim(T_range)
        else:
            ax.set_ylim(y_lim)
        ax.set_ylabel(r'$T$' + ' [{0:s}]'.format(T_unit), fontsize=18)
    else:
        ax.set_ylabel(y_label, fontsize=18)
        ax.set_ylim(y_lim)

    ax.tick_params(axis='both', labelsize=16)
    if title is not None:
        ax.set_title(title, fontsize=16)

    # makes box of plot square
    if square_box:
        ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))


    # places legend outside of plot box
    if show_lgnd:
        if lgnd_out:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height])
            legend_x = 1
            legend_y = 0.5
            ax.legend(loc='center left', bbox_to_anchor=(legend_x, legend_y),
                        fontsize=14, frameon=False)
        else:
            ax.legend(fontsize=12, frameon=False)

    return ax


def get_colors(cmap_name, n):
    """Returns list of colors using given colormap."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(val) for val in np.linspace(0, 1, n)]


def get_lgnd_labels(handles, labels, key):
    """Returns zipped handles and labels for which labels contains key."""
    return [pair for pair in zip(handles, labels) if key in pair[1]]


def get_plot_dict_p_s(ch_var):
    """Returns a dictionary of key parameters for plotting based on varied component."""
    d = {}
    # polyelectrolyte density varied
    if ch_var == 'p':
        d = {'ch_var':'p', 'ch_fix':'s', 'order':[0,1], 'name_var':'Polymer'}
    # salt density varied
    elif ch_var == 's':
        d = {'ch_var':'s', 'ch_fix':'p', 'order':[1,0], 'name_var':'Salt'}
    else:
        print('invalid ch_var character: choose s or p.')

    return d


def line_3d(x, y, z, mode='lines', ms=8, op=0.1,
                    c='black', lw=8, fig=None):
    """
    Plots line in 3D plot (plotly).
    """
    if fig == None:
        fig = go.Figure()

    # plots phase I (supernatant) of fixed salt binodal
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode=mode,
        marker=dict(
            size=ms,
            opacity=op,
            color=c
        ),
        line=dict(
            color=c,
            width=lw,
        ),
    ))

    return fig


def no_salt(df, n_plot, left='rhoPCI', right='rhoPCII', x_label='polycation density',
            p=None, n_tie_lines=0, plot_T=False, title='', line=False, marker=True,
            w=500, h=500, units_rho='[beads/sigma^3]', deg_C=False,
            leg1='supernatant', c1='blue', leg2='coacervate', c2='red'):
    """
    Plots the binodal for a polyelectrolyte in solution without
    salt.
    """
    if plot_T:
        y = 'T'
        if deg_C:
            y_label = 'Temperature [' + r'$^{\circ}$' + 'C]'
        else:
            y_label = 'Temperature [K]'
    else:
        y = 'BJ'
        y_label = 'Bjerrum length'
    # samples a uniform subset of the data
    n = len(df)
    skip = int(n / n_plot)
    sample = df.iloc[::skip]

    # creates figure object if not provided
    if p is None:
        p = figure(plot_width=w, plot_height=h)

    # loads source for plot data
    source = ColumnDataSource(sample)

    if marker:
        # creates circle glyph of polycation concentration in dilute phase
        p.circle(x=left, y=y, source=source, size=10, color=c1,
                 legend_label=leg1)
        # creates circle glyph of polycation concentration in coacervate phase
        p.circle(x=right, y=y, source=source, size=10, color=c2,
                 legend_label=leg2)
    if line:
        # creates circle glyph of polycation concentration in dilute phase
        p.line(x=left, y=y, source=source, line_width=6, line_color=c1,
                 legend_label=leg1)
        # creates circle glyph of polycation concentration in coacervate phase
        p.line(x=right, y=y, source=source, line_width=6, line_color=c2,
                 legend_label=leg2)

    # adds tie lines
    if n_tie_lines > 0:
        skip_tie_lines = int(n / n_tie_lines)
        df_tie_lines = df.iloc[::skip_tie_lines]
        for t in range(len(df_tie_lines)):
            p.line([df_tie_lines[left].iloc[t], df_tie_lines[right].iloc[t]],
                   [df_tie_lines[y].iloc[t], df_tie_lines[y].iloc[t]],
                   color='black')

    # adds plot labels
    p.xaxis.axis_label = x_label + ' ' + units_rho
    p.xaxis.axis_label_text_font_size = '18pt'
    p.xaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label = y_label
    p.yaxis.axis_label_text_font_size = '18pt'
    p.yaxis.major_label_text_font_size = '14pt'

    # adds title
    p.title.text = title
    p.title.text_font_size = '16pt'

    # formats legend
    p.legend.location = "bottom_right"
    p.legend.label_text_font_size  = '14pt'
    p.legend.click_policy = 'hide'

    # creates hover feature to read data
    hover = HoverTool()
    hover.tooltips=[
        (y_label, '@' + y),
        (x_label + ' (I)', '@' + left),
        (x_label + ' (II)', '@' + right)
    ]
    p.add_tools(hover)

    return p


def pt_3d(x, y, z, mode='markers', ms=8, op=1,
                    c='black', fig=None):
    """
    Plots line in 3D plot (plotly).
    """
    if fig == None:
        fig = go.Figure()

    # plots phase I (supernatant) of fixed salt binodal
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode=mode,
        marker=dict(
            size=ms,
            opacity=op,
            color=c
        ),
    ))

    return fig


def salt(df, n_plot, p=None, n_tie_lines=0):
    """
    Plots the binodal for a polyelectrolyte in solution with salt
    at a fixed Bjerrum length on rho_p vs. rho_s axes.
    """
    # samples a uniform subset of the data
    n = len(df)
    skip = int(n / n_plot)
    sample = df.iloc[::skip]

    # creates figure object if not provided
    if p is None:
        p = figure()

    # loads source for plot data
    source = ColumnDataSource(sample)
    # creates circle glyph of polycation concentration in dilute phase
    p.circle(x='rhoPAI', y='rhoAI', source=source, size=10, color='red', legend_label='dilute phase (I)')
    # creates circle glyph of polycation concentration in coacervate phase
    p.circle(x='rhoPAII', y='rhoAII', source=source, size=10, color='blue', legend_label='coacervate phase (II)')

    # draws tie lines
    if n_tie_lines > 0:
        skip_tie_lines = int(n / n_tie_lines)
        df_tie_lines = df.iloc[::skip_tie_lines]
        for t in range(len(df_tie_lines)):
            x = [df_tie_lines['rhoPAI'].iloc[t], df_tie_lines['rhoPAII'].iloc[t]]
            y = [df_tie_lines['rhoAI'].iloc[t], df_tie_lines['rhoAII'].iloc[t]]
            p.line(x, y, color='black')

    # adds plot labels
    p.xaxis.axis_label = 'polyanion number density'
    p.xaxis.axis_label_text_font_size = '18pt'
    p.xaxis.major_label_text_font_size = '14pt'
    p.yaxis.axis_label = 'anion number density'
    p.yaxis.axis_label_text_font_size = '18pt'
    p.yaxis.major_label_text_font_size = '14pt'

    # formats legend
    p.legend.location = "top_right"
    p.legend.label_text_font_size  = '16pt'
    p.legend.click_policy = 'hide'

    # creates hover feature to read data
    hover = HoverTool()
    hover.tooltips=[
        ('Anion Density (I)', '@rhoAI'),
        ('Anion Density (II)', '@rhoAII'),
        ('Polyanion density (I)', '@rhoPAI'),
        ('Polyanion density (II)', '@rhoPAII')
    ]
    p.add_tools(hover)

    return p


def sort_lgnd_labels(ax, sorted_keys):
    """Sorts legend labels based on order of keywords."""
    # gets handles and labels from legend
    handles, labels = ax.get_legend_handles_labels()
    # sorts by keywords
    lgnd_sorted = []
    for key in sorted_keys:
        lgnd_sorted += get_lgnd_labels(handles, labels, key)
    # removes redundant entries
    lgnd_unique = [(0,0)] # primer entry
    [lgnd_unique.append(pair) for pair in lgnd_sorted if pair[1] \
                                            not in list(zip(*lgnd_unique))[1]]
    # removes primer entry
    lgnd_unique = lgnd_unique[1:]
    # unzips
    handles_sorted, labels_sorted = zip(*lgnd_unique)

    # adds legend outside plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    legend_x = 1
    legend_y = 0.5
    ax.legend(handles_sorted, labels_sorted, loc='center left',
                bbox_to_anchor=(legend_x, legend_y),
                fontsize=14, frameon=False)

    return ax


def validate_fit(data_pred, df_exp, ch_var, rho_var_list, rho_fix, colors,
                 beads_2_M_opt, T_range=[273.15, 323.15], lw=2, sigma=None,
                 conv_vals=False, x_var='polyelectrolyte'):
    """
    Validates fit of sigma to experiments.
    """
    if conv_vals:
        rho_p = df_exp['rho_p [M]'].to_numpy(dtype=float)
        rho_p_conv = df_exp['rho_p (conv) [M]'].to_numpy(dtype=float)
        rho_s = df_exp['rho_s [M]'].to_numpy(dtype=float)
        rho_s_conv = df_exp['rho_s (conv) [M]'].to_numpy(dtype=float)

        # matches polymer and salt values with fixed and varied concentrations
        rho_var_list_conv = []
        if ch_var == 'p':
            for rho_var in rho_var_list:
                i = np.where(rho_var == rho_p)[0][0]
                rho_var_list_conv += [rho_p_conv[i]]
            rho_fix_conv = rho_s_conv[np.where(rho_fix == rho_s)[0][0]]
        elif ch_var == 's':
            for rho_var in rho_var_list:
                i = np.where(rho_var == rho_s)[0][0]
                rho_var_list_conv += [rho_s_conv[i]]
            rho_fix_conv = rho_p_conv[np.where(rho_fix == rho_p)[0][0]]


    # polymer-temperature plane
    if conv_vals:
        ax = binodal_vary_rho(data_pred, rho_var_list_conv, rho_fix_conv, ch_var,
                            beads_2_M_opt,
                            x_var=x_var, x_label=r'$\rho_p$',
                            sigma=sigma, T_range=T_range, marker='', lw=lw,
                            colors=colors, T_cels=True)
    else:
        ax = binodal_vary_rho(data_pred, rho_var_list, rho_fix, ch_var,
                            beads_2_M_opt,
                            x_var=x_var, x_label=r'$\rho_p$',
                            sigma=sigma, T_range=T_range, marker='', lw=lw,
                            colors=colors, T_cels=True)

    # plots experimental results
    for i in range(len(df_exp)):
        rho_p, rho_s, T_exp, rho_p_sup, rho_p_co = nacl.read_df_exp(df_exp, i)
        if ch_var == 'p':
            rho_var_exp = rho_p
            rho_fix_exp = rho_s
        elif ch_var == 's':
            rho_var_exp = rho_s
            rho_fix_exp = rho_p
        else:
            print('Please select s or p as ch_var')
        if (rho_var_exp in rho_var_list) and (rho_fix_exp == rho_fix):
            # determines color
            color = [colors[i] for i in range(len(colors)) if rho_var_list[i] == rho_var_exp][0]

            # plots desired species concentration
            if x_var == 'polyanion' or x_var == 'polycation':
                # if just plotting polyanion, divides total polymer
                # concentration in half (assumes symmetric solution)
                rho_sup = rho_p_sup / 2
                rho_co = rho_p_co / 2
            elif x_var == 'polyelectrolyte':
                rho_sup = rho_p_sup
                rho_co = rho_p_co
            # plots supernatant and coacervate compositions
            ax.plot(rho_sup, T_exp, color=color, marker='o', label='supernatant')
            ax.plot(rho_co, T_exp, color=color, marker='^', label='coacervate')
