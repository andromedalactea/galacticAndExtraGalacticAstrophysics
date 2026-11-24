# Galactic and Extragalactic Astrophysics

## Some results

Each row is one galaxy measured with the same criterion. The spirals NGC 628 and NGC 4254 come first; the ellipticals NGC 3379 and NGC 4374 follow.

The colored patches are brightness levels. The large patch gives an ellipse from the pixel moments: the center, the axis ratio $b/a$, and the position angle of the major axis. In the spirals that ellipse is elongated because the disk is inclined. In the ellipticals it stays nearly round, and the Hubble type follows from $n = 10(1-b/a)$: both are E1.

The third column stretches the minor axis until that ellipse is a circle, so the disk is seen face-on. The flux profile shows the bulge–disk relation. The orange curve is the exponential disk fitted in the outer region; the excess in the center, to the left of the dotted line, is the bulge. In NGC 628 and NGC 4254 that excess is small, and both are classified Scd. The ellipticals have no disk to fit: the profile is the measurement alone, bright at the center and falling outward.

![Classification of NGC 628, NGC 4254, NGC 3379, and NGC 4374](Exam%201/figuras/clasificacion_galaxias.png)

### Stellar orbits in a Kuzmin disk with an oblate dark matter halo

A thin Kuzmin disk ($M_d = 5\times 10^{11}\,M_\odot$, $a_d = 8\,\mathrm{kpc}$) is embedded in an oblate Miyamoto-Nagai dark matter halo ($M_h = 5M_d$, $a_h = 8\,\mathrm{kpc}$, $b_h = 20\,\mathrm{kpc}$). On the left, the halo isodensity contours are flattened along the $z$-axis ($q_\rho \approx 0.75$) around the disk plane. On the right, the halo lifts the total circular velocity $v_{c,\mathrm{tot}} = \sqrt{v_{c,d}^2 + v_{c,h}^2}$ and sustains the rotation curve at outer radii instead of dropping off early like the isolated disk.

![Oblate Miyamoto-Nagai halo structure and rotation curve decomposition](Exam%202/figures/fig5_halo_structure_and_vc.png)

Integrating a slightly perturbed star at $R_0 = 12\,\mathrm{kpc}$ ($z_0 = 0.40\,\mathrm{kpc}$, $\delta v_R = 0.05\,v_c$) with Velocity Verlet compares the isolated disk against the combined disk + halo potential. In the $(x, y)$ plane the orbit forms a non-closed trajectory, while in the meridional section $(R, z)$ the halo confines the radial excursion ($0.489 \to 0.400\,\mathrm{kpc}$) and pulls the mean radius closer to $R_0$. In physical time, the radial oscillations speed up strongly ($\kappa$: $35.95 \to 51.52\,\mathrm{km\,s^{-1}\,kpc^{-1}}$) and the vertical curve advances progressively in phase ($\nu_{\mathrm{eff}}$: $129.6 \to 134.2\,\mathrm{km\,s^{-1}\,kpc^{-1}}$), confirming the analytical predictions.

![Simultaneous disk and halo orbit compared with the isolated Kuzmin disk](Exam%202/figures/fig7_disk_halo_orbit.png)
