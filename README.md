# ccls_analysis

Function libraries and notebooks for analyzing the theoretical predictions of the
composition of the phases of polyelectrolyte solutions.
The compositions are expected to have been computed with the
liquid state theory described in the [work of Zhang, *et al*](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.6b02160)
using the FORTRAN library coded by Prof. Pengfei Zhang and deployed by
Chris Balzer in the [Complex-Coacervation-LS](https://github.com/zgw-group/Complex-Coacervation-LS)
repo.
The results from this analysis were used to produce the figures presented in
the work of [Ylitalo, *et al*](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.1c02000).


## Libraries

- `pe.py`: library of functions useful for polyelectrolyte (pe) solutions.
Functions specific to polyelectrolyte solutions with salt are contained in
`salt.py`, however.
- `plot.py`: library of functions used for plotting in notebooks
- `salt.py`: library of functions useful for analyzing complex coacervation in
polyelectrolyte solutions with salt

## Notebooks

For the notebooks to perform the desired analysis, the appropriate data must be
generated with the liquid state theory and stored appropriately in a separate
folder.
**MORE TO COME ON THIS**

### Major Analysis

- `fit_sigma_to_prabhu_data_asymN.ipynb`, `fit_sigma_to_prabhu_data_n_1000.ipynb`:
Fit the bead size $\sigma$ parameter to the experimental data from the Prabhu
group ([Ma, *et al*](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.1c02001)).
In `*asymN.ipynb`, we used asymmetric values of the bead number $N$ based on the
[correction](https://pubs.acs.org/doi/full/10.1021/acsmacrolett.1c00727)
to their previous measurements. In `*n_1000.ipynb`, we used $N = 1000$.
- `paper_figs.ipynb`: generates all plots presented in
[Ylitalo, *et al*](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.1c02000).
- `paper_figs_asymN.ipynb`, `paper_figs_n_1000.ipynb`: generates plots as in
`paper_figs.ipynb`, but using asymmetric values of the bead number $N$
(`*asymN.ipynb`) or $N = 1000$ (`*n1000.ipynb`).
- `presentation_figs.ipynb`: generates plots used for figures in presentation
by Andrew Ylitalo at the [virtual APS March Meeting 2021](https://meetings.aps.org/Meeting/MAR21/Session/Y03.10).
- `si_figs.ipynb`: generates the figures in the supporting information (SI) of
[Ylitalo, *et al*](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.1c02000).
- `si_figs_n_1000.ipynb`: generates the figures in the supporting information (SI) of
[Ylitalo, *et al*](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.1c02000)
with bead number $N = 1000$ (less up-to-date).


### Minor Analysis

- `check_for_rhopc_Tc_scaling.ipynb`, `check_for_rhopc_Tc_scaling.ipynb`:
Reviewer \#2 was interested in the effect of temperature on the critical
composition. Here, we plot the critical line on a log-log plot and show
power-law scalings (`*rhopc*.ipynb` plots the critical polymer concentration and
`*rhosc*.ipynb` plots the critical salt concentration.)
- `chi_param_T.ipynb`: Generates and saves temperatures that correspond with
each Bjerrum length assuming the dielectric constant of pure water for use in
computing the $\chi$ parameter
- `compare_exp_vary_N_sigma_f.ipynb`: varies the bead number $N$, bead diameter
$\sigma$, and charge fraction $f$ and plots the effect on the binodal
- `reformat_vo_chris.ipynb`: reformats data from VO theory model programmed in
the repo [Wetting-Polymer-Fluids](https://github.com/zgw-group/Wetting-Polymer-Fluids).
- `repl_adhikari2019_fig4.ipynb`: compares predictions of liquid state theory
with those of the mean-field model presented in [Adhikari, *et al*](https://pubs.acs.org/doi/abs/10.1021/acs.macromol.9b01201).
- `repl_ali2019_fig3.ipynb`: compares predictions of liquid state theory
with experimental measurements presented in [Ali, *et al*](https://pubs.acs.org/doi/abs/10.1021/acsmacrolett.8b00952).
- `ucst_lcst.ipynb`: first attempts to identify conditions that lead to a UCST
and an LCST when incorporating an enthalpic $\chi$ parameter.
