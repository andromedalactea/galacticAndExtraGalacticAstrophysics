"""
part3_disk_halo.py
Numeral 3: Halo oblato de materia oscura (Miyamoto-Nagai), predicción teórica
del cambio en las frecuencias kappa y nu, y simulación simultánea disco + halo.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from potentials import (
    T_UNIT_GYR, PARAMS,
    phi_kuzmin, acc_kuzmin_xyz, vc_kuzmin, omega_kuzmin, kappa_kuzmin,
    gz_kuzmin, Fz_kuzmin, nu_eff_kuzmin, Tz_eff_kuzmin,
    rho_mn, vc_mn, omega_mn, kappa_mn, nu_mn, Fz_mn,
    phi_total, acc_total_xyz, vc_total, omega_total, kappa_total,
    Tz_eff_total, nu_eff_total,
    integrate_verlet, compute_integrals,
    measure_period_from_peaks, measure_period_from_zero_crossings,
    configure_plots,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "figures"


def halo_axial_ratio(Rm, Mh, ah, bh):
    """Calcula la razón de ejes q_rho = z_m / R_m de la superficie de isodensidad en R_m."""
    target = rho_mn(Rm, 0.0, Mh, ah, bh)
    zm = brentq(lambda z: rho_mn(0.0, z, Mh, ah, bh) - target, 1e-4, 250.0)
    return float(zm / Rm)


def run_part3(part2_orbit=None):
    configure_plots()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    Md, ad, R0 = PARAMS["Md"], PARAMS["ad"], PARAMS["R0"]
    Mh, ah, bh = PARAMS["Mh"], PARAMS["ah"], PARAMS["bh"]

    R = np.linspace(0.05, 45.0, 800)

    # Razón de ejes del halo oblato en varios radios
    q_12 = halo_axial_ratio(12.0, Mh, ah, bh)
    q_20 = halo_axial_ratio(20.0, Mh, ah, bh)
    q_30 = halo_axial_ratio(30.0, Mh, ah, bh)

    # -------------------------------------------------------------------------
    # Figura 5: Contornos de isodensidad del halo oblato y curva de rotación
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    # (a) Contornos de isodensidad en el plano (R, z)
    ax = axes[0]
    Rg = np.linspace(-46.0, 46.0, 360)
    zg = np.linspace(-38.0, 38.0, 320)
    RR, ZZ = np.meshgrid(Rg, zg)
    rho_pc3 = rho_mn(np.abs(RR), ZZ, Mh, ah, bh) * 10.0  # Msun / pc^3
    log_rho_pc3 = np.log10(rho_pc3)

    levels = [-3.0, -2.5, -2.0, -1.6]
    cs = ax.contour(RR, ZZ, log_rho_pc3, levels=levels,
                    colors="#1f5a99", linewidths=1.5, linestyles="solid")
    ax.clabel(cs, inline=True, fontsize=8.2, fmt=r"$10^{%.1f}$")
    ax.plot([-22, 22], [0, 0], color="#c0392b", lw=2.6, label="Plano del disco de Kuzmin ($z=0$)")
    ax.axvline(0.0, color="#666666", ls=":", lw=1.0, label="Eje de achatamiento (eje $z$)")
    ax.text(-43, 28.5, rf"Halo oblato: $b_h/a_h = {bh/ah:.1f}$" + "\n"
            + rf"$q_\rho = z_m/R_m \approx {q_20:.2f}$",
            fontsize=8.8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"$z\;[\mathrm{kpc}]$")
    ax.set_title(r"(a) Contornos de isodensidad $\rho_h(R, z)\;[M_\odot\,\mathrm{pc}^{-3}]$")
    ax.set_xlim(-46, 46)
    ax.set_ylim(-38, 38)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="lower left", fontsize=8.2)

    # (b) Curva de rotación descompuesta
    ax = axes[1]
    vc_d = vc_kuzmin(R, Md, ad)
    vc_h = vc_mn(R, Mh, ah, bh)
    vc_t = vc_total(R, Md, ad, Mh, ah, bh)
    ax.plot(R, vc_t, color="#1f5a99", lw=2.2, label="Total (disco + halo)")
    ax.plot(R, vc_d, color="#d95f02", ls="--", lw=1.6, label="Disco de Kuzmin")
    ax.plot(R, vc_h, color="#2e8b57", ls="-.", lw=1.6, label="Halo de Miyamoto-Nagai")
    ax.axvline(R0, color="#666666", ls=":", lw=1.2)
    vc_d0 = vc_kuzmin(R0, Md, ad)
    vc_t0 = vc_total(R0, Md, ad, Mh, ah, bh)
    ax.scatter([R0, R0], [vc_d0, vc_t0], color=["#d95f02", "#1f5a99"], zorder=6, s=40)
    ax.annotate(
        rf"$R_0 = {R0:.0f}\,\mathrm{{kpc}}$" + "\n"
        + rf"$v_c: {vc_d0:.0f} \to {vc_t0:.0f}\,\mathrm{{km/s}}$",
        xy=(R0, vc_t0), xytext=(2.0, 445.0),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0),
        fontsize=8.8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc")
    )
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"Velocidad circular $v_c(R)\;[\mathrm{km\,s^{-1}}]$")
    ax.set_title("(b) Descomposición de la curva de rotación")
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 520)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="lower right", fontsize=8.8)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(FIG_DIR / "fig5_halo_structure_and_vc.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figura 6: Predicción analítica del cambio en kappa y nu por el halo
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    R40 = np.linspace(0.05, 40.0, 600)

    kap_d = kappa_kuzmin(R40, Md, ad)
    kap_t = kappa_total(R40, Md, ad, Mh, ah, bh)
    Om_d = omega_kuzmin(R40, Md, ad)
    Om_t = omega_total(R40, Md, ad, Mh, ah, bh)

    # (a) kappa y Omega con y sin halo
    ax = axes[0, 0]
    ax.plot(R40, kap_t, color="#c0392b", lw=2.0, label=r"$\kappa_{\mathrm{tot}}(R)$ (disco + halo)")
    ax.plot(R40, kap_d, color="#c0392b", ls="--", lw=1.5, label=r"$\kappa_d(R)$ (solo disco)")
    ax.plot(R40, Om_t, color="#1f5a99", lw=1.8, label=r"$\Omega_{\mathrm{tot}}(R)$ (disco + halo)")
    ax.plot(R40, Om_d, color="#1f5a99", ls="--", lw=1.4, label=r"$\Omega_d(R)$ (solo disco)")
    ax.axvline(R0, color="#666666", ls=":", lw=1.2)
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"Frecuencia angular $[\mathrm{km\,s^{-1}\,kpc^{-1}}]$")
    ax.set_title(r"(a) Aumento de $\kappa(R)$ y $\Omega(R)$ al añadir el halo")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 165)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=8.6)

    # (b) Fuerza vertical |F_z(R_0, z)| descompuesta (tipo Binney & Tremaine Fig. 2.23)
    ax = axes[0, 1]
    z_grid = np.linspace(1e-4, 4.0, 400)
    Kz_d = -Fz_kuzmin(R0, z_grid, Md, ad) / 1e3
    Kz_h = -Fz_mn(R0, z_grid, Mh, ah, bh) / 1e3
    Kz_t = Kz_d + Kz_h
    ax.plot(z_grid, Kz_t, color="#1f5a99", lw=2.2, label=r"Total $|F_{z,\mathrm{tot}}|$")
    ax.plot(z_grid, Kz_d, color="#d95f02", ls="--", lw=1.6, label=r"Disco de Kuzmin $|F_{z,d}|$")
    ax.plot(z_grid, Kz_h, color="#2e8b57", ls="-.", lw=1.6, label=r"Halo de Miyamoto-Nagai $|F_{z,h}|$")
    ax.axvspan(0.0, 0.4, color="#f39c12", alpha=0.14, label=r"Rango simulado ($z \leq 0.4\,\mathrm{kpc}$)")
    ax.set_xlabel(r"Altura sobre el plano $z\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"$|F_z(R_0, z)|\;[10^3\,\mathrm{km^2\,s^{-2}\,kpc^{-1}}]$")
    ax.set_title(rf"(b) Fuerza restauradora vertical en $R_0 = {R0:.0f}\,\mathrm{{kpc}}$")
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 8.8)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95,
              loc="right", bbox_to_anchor=(0.98, 0.44), fontsize=8.4)

    # (c) Frecuencia vertical efectiva nu_eff con y sin halo
    ax = axes[1, 0]
    nu_d_04 = nu_eff_kuzmin(R40, 0.4, Md, ad)
    nu_t_04 = nu_eff_total(R40, 0.4, Md, ad, Mh, ah, bh)
    nu_d_15 = nu_eff_kuzmin(R40, 1.5, Md, ad)
    nu_t_15 = nu_eff_total(R40, 1.5, Md, ad, Mh, ah, bh)
    ax.plot(R40, nu_t_04, color="#1b9e77", lw=2.0, label=r"Disco + halo ($z_{\max}=0.4\,\mathrm{kpc}$)")
    ax.plot(R40, nu_d_04, color="#1b9e77", ls="--", lw=1.4, label=r"Solo disco ($z_{\max}=0.4\,\mathrm{kpc}$)")
    ax.plot(R40, nu_t_15, color="#7570b3", lw=2.0, label=r"Disco + halo ($z_{\max}=1.5\,\mathrm{kpc}$)")
    ax.plot(R40, nu_d_15, color="#7570b3", ls="--", lw=1.4, label=r"Solo disco ($z_{\max}=1.5\,\mathrm{kpc}$)")
    ax.axvline(R0, color="#666666", ls=":", lw=1.2)
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"Frecuencia vertical $\nu_{\mathrm{eff}}\;[\mathrm{km\,s^{-1}\,kpc^{-1}}]$")
    ax.set_title(r"(c) Frecuencia vertical efectiva con y sin halo")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 360)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=8.4)

    # (d) Incremento porcentual relativo de kappa y nu_eff
    ax = axes[1, 1]
    pct_kap = (kap_t / kap_d - 1.0) * 100.0
    pct_nu_04 = (nu_t_04 / nu_d_04 - 1.0) * 100.0
    pct_nu_15 = (nu_t_15 / nu_d_15 - 1.0) * 100.0
    ax.plot(R40, pct_kap, color="#c0392b", lw=2.0, label=r"$\Delta\kappa/\kappa_d$ [%]")
    ax.plot(R40, pct_nu_15, color="#7570b3", lw=1.8, label=r"$\Delta\nu_{\mathrm{eff}}/\nu_{\mathrm{eff},d}$ ($z_{\max}=1.5\,\mathrm{kpc}$)")
    ax.plot(R40, pct_nu_04, color="#1b9e77", lw=1.8, label=r"$\Delta\nu_{\mathrm{eff}}/\nu_{\mathrm{eff},d}$ ($z_{\max}=0.4\,\mathrm{kpc}$)")
    ax.axvline(R0, color="#666666", ls=":", lw=1.2)
    p_k0 = (kappa_total(R0, Md, ad, Mh, ah, bh) / kappa_kuzmin(R0, Md, ad) - 1.0) * 100.0
    p_n0 = (nu_eff_total(R0, 0.4, Md, ad, Mh, ah, bh) / nu_eff_kuzmin(R0, 0.4, Md, ad) - 1.0) * 100.0
    ax.scatter([R0, R0], [p_k0, p_n0], color=["#c0392b", "#1b9e77"], zorder=6, s=38)
    ax.annotate(rf"$R_0$: $\Delta\kappa = +{p_k0:.1f}\%$",
                xy=(R0, p_k0), xytext=(R0 - 10.5, p_k0 + 28.0),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0),
                fontsize=8.6, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"))
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel("Incremento porcentual [%]")
    ax.set_title("(d) Magnitud relativa del efecto del halo según el radio")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 155)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper left", fontsize=8.4)

    fig.tight_layout(h_pad=2.2, w_pad=2.2)
    fig.savefig(FIG_DIR / "fig6_frequencies_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Simulación numérica en el potencial combinado Disco + Halo (R0 = 12 kpc)
    # -------------------------------------------------------------------------
    vc_t0 = vc_total(R0, Md, ad, Mh, ah, bh)
    Om_t0 = omega_total(R0, Md, ad, Mh, ah, bh)
    kap_t0 = kappa_total(R0, Md, ad, Mh, ah, bh)
    T_orb_t = 2.0 * np.pi / Om_t0
    T_kap_t_th = 2.0 * np.pi / kap_t0

    z0_pert = 0.40
    dvR_t = 0.05 * vc_t0
    A_R_t_th = dvR_t / kap_t0
    Tz_t_th = Tz_eff_total(R0, z0_pert, Md, ad, Mh, ah, bh)
    nu_t_th = nu_eff_total(R0, z0_pert, Md, ad, Mh, ah, bh)

    # Integrar 5 periodos del sistema combinado (y asegurar tiempo suficiente para comparar)
    dt = 2.5e-5
    t_max = max(5.0 * T_orb_t, 1.0)  # >= 5 periodos (~0.95 u.t. -> 1.0 u.t. ~ 5.3 periodos)
    n_steps = int(np.ceil(t_max / dt))

    r0_t = [R0, 0.0, z0_pert]
    v0_t = [dvR_t, vc_t0, 0.0]
    t_dh, r_dh, v_dh = integrate_verlet(r0_t, v0_t, acc_total_xyz, dt, n_steps)

    R_dh = np.sqrt(r_dh[:, 0]**2 + r_dh[:, 1]**2)
    z_dh = r_dh[:, 2]
    E_dh, Lz_dh = compute_integrals(r_dh, v_dh, phi_total)
    dE_dh = np.abs((E_dh - E_dh[0]) / E_dh[0])
    dLz_dh = np.abs((Lz_dh - Lz_dh[0]) / Lz_dh[0])

    T_kap_dh_num, kap_dh_num = measure_period_from_peaks(t_dh, R_dh)
    Tz_dh_num, nu_dh_num = measure_period_from_zero_crossings(t_dh, z_dh)

    # Si no se pasó la órbita de solo disco desde part2, se integra rápidamente para comparar
    if part2_orbit is None:
        vc_d0 = vc_kuzmin(R0, Md, ad)
        dvR_d = 0.05 * vc_d0
        t_d, r_d, v_d = integrate_verlet([R0, 0.0, z0_pert], [dvR_d, vc_d0, 0.0],
                                         acc_kuzmin_xyz, dt, n_steps)
        R_d = np.sqrt(r_d[:, 0]**2 + r_d[:, 1]**2)
        z_d = r_d[:, 2]
    else:
        t_d = part2_orbit["t"]
        R_d = part2_orbit["R"]
        z_d = part2_orbit["z"]

    # También verificamos un radio externo R1 = 25 kpc donde el halo domina verticalmente
    R1 = 25.0
    vc_d1 = vc_kuzmin(R1, Md, ad)
    vc_t1 = vc_total(R1, Md, ad, Mh, ah, bh)
    T_orb_t1 = 2.0 * np.pi / omega_total(R1, Md, ad, Mh, ah, bh)
    n_steps_out = int(np.ceil(3.5 * T_orb_t1 / 5.0e-5))
    t_d1, r_d1, _ = integrate_verlet([R1, 0.0, z0_pert], [0.05 * vc_d1, vc_d1, 0.0],
                                     acc_kuzmin_xyz, 5.0e-5, n_steps_out)
    t_t1, r_t1, _ = integrate_verlet([R1, 0.0, z0_pert], [0.05 * vc_t1, vc_t1, 0.0],
                                     acc_total_xyz, 5.0e-5, n_steps_out)
    _, kap_d1_num = measure_period_from_peaks(t_d1, np.sqrt(r_d1[:, 0]**2 + r_d1[:, 1]**2))
    _, kap_t1_num = measure_period_from_peaks(t_t1, np.sqrt(r_t1[:, 0]**2 + r_t1[:, 1]**2))
    _, nu_d1_num = measure_period_from_zero_crossings(t_d1, r_d1[:, 2])
    _, nu_t1_num = measure_period_from_zero_crossings(t_t1, r_t1[:, 2])

    # -------------------------------------------------------------------------
    # Figura 7: Simulación comparativa Disco + Halo vs Solo Disco en R0 = 12 kpc
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    theta = np.linspace(0, 2.0 * np.pi, 400)

    # (a) Órbita en el plano (x, y)
    ax = axes[0, 0]
    ax.plot(R0 * np.cos(theta), R0 * np.sin(theta), color="#999999", ls="--", lw=1.2,
            label=rf"Círculo $R_0={R0:.0f}\,\mathrm{{kpc}}$")
    ax.plot(r_dh[:, 0], r_dh[:, 1], color="#1f5a99", lw=1.1, label="Disco + halo (5.3 periodos)")
    ax.scatter([R0], [0.0], color="#c0392b", s=38, zorder=6, label="Inicio")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"$y\;[\mathrm{kpc}]$")
    ax.set_title("(a) Órbita en el plano $(x, y)$ con disco + halo", pad=8)
    ax.set_xlim(-15.0, 15.0)
    ax.set_ylim(-15.0, 15.0)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="center", fontsize=8.3)

    # (b) Comparación de la sección meridional (R, z)
    ax = axes[0, 1]
    ax.plot(R_d, z_d, color="#d95f02", lw=0.75, alpha=0.45, label="Solo disco de Kuzmin")
    ax.plot(R_dh, z_dh, color="#1f5a99", lw=0.9, alpha=0.88, label="Disco + halo oblato")
    ax.axvline(R0, color="#666666", ls=":", lw=1.1)
    ax.scatter([R0], [z0_pert], color="#c0392b", s=40, zorder=6)
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"Altura $z\;[\mathrm{kpc}]$")
    ax.set_title("(b) Sección meridional $(R, z)$: confinamiento por el halo")
    ax.set_xlim(11.55, 12.90)
    ax.set_ylim(-0.52, 0.64)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=8.5)

    # (c) Oscilación radial R(t) en tiempo físico (Gyr)
    ax = axes[1, 0]
    t_dh_gyr = t_dh * T_UNIT_GYR
    t_d_gyr = t_d * T_UNIT_GYR
    m_dh_r = t_dh_gyr <= 0.85
    m_d_r = t_d_gyr <= 0.85
    ax.plot(t_d_gyr[m_d_r], R_d[m_d_r], color="#d95f02", ls="--", lw=1.5,
            label=r"Solo disco ($\kappa_{\mathrm{num}}=35.95$)")
    ax.plot(t_dh_gyr[m_dh_r], R_dh[m_dh_r], color="#1f5a99", lw=1.9,
            label=rf"Disco + halo ($\kappa_{{\mathrm{{num}}}}={kap_dh_num:.2f}$)")
    ax.axhline(R0, color="#777777", ls=":", lw=1.0)
    ax.set_xlabel(r"Tiempo físico $t\;[\mathrm{Gyr}]$")
    ax.set_ylabel(r"Radio cilíndrico $R(t)\;[\mathrm{kpc}]$")
    ax.set_title(r"(c) Comparación radial $R(t)$: menor periodo con el halo")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(11.48, 13.02)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=8.5)

    # (d) Oscilación vertical z(t) en tiempo físico (Gyr), zoom primeros 0.25 Gyr
    ax = axes[1, 1]
    m_dh_z = t_dh_gyr <= 0.25
    m_d_z = t_d_gyr <= 0.25
    ax.plot(t_d_gyr[m_d_z], z_d[m_d_z], color="#d95f02", ls="--", lw=1.5,
            label=r"Solo disco ($\nu_{\mathrm{num}}=129.6$)")
    ax.plot(t_dh_gyr[m_dh_z], z_dh[m_dh_z], color="#1b9e77", lw=1.9,
            label=rf"Disco + halo ($\nu_{{\mathrm{{num}}}}={nu_dh_num:.1f}$)")
    ax.axhline(0.0, color="#777777", ls=":", lw=1.0)
    ax.set_xlabel(r"Tiempo físico $t\;[\mathrm{Gyr}]$")
    ax.set_ylabel(r"Altura $z(t)\;[\mathrm{kpc}]$")
    ax.set_title(r"(d) Comparación vertical $z(t)$: adelanto de fase por el halo")
    ax.set_xlim(0, 0.25)
    ax.set_ylim(-0.52, 0.68)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=8.5)

    fig.tight_layout(h_pad=2.2, w_pad=2.2)
    fig.savefig(FIG_DIR / "fig7_disk_halo_orbit.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "q_rho_12": q_12,
        "q_rho_20": q_20,
        "q_rho_30": q_30,
        "vc_tot_R0": float(vc_t0),
        "Om_tot_R0": float(Om_t0),
        "T_orb_tot_ut": float(T_orb_t),
        "T_orb_tot_Myr": float(T_orb_t * T_UNIT_GYR * 1e3),
        "kap_tot_th": float(kap_t0),
        "kap_tot_num": float(kap_dh_num),
        "kap_tot_err_pct": float(abs(kap_dh_num - kap_t0) / kap_t0 * 100.0),
        "T_kap_tot_th_Myr": float(T_kap_t_th * T_UNIT_GYR * 1e3),
        "T_kap_tot_num_Myr": float(T_kap_dh_num * T_UNIT_GYR * 1e3),
        "nu_h_R0": float(nu_mn(R0, Mh, ah, bh)),
        "nu_tot_th": float(nu_t_th),
        "nu_tot_num": float(nu_dh_num),
        "nu_tot_err_pct": float(abs(nu_dh_num - nu_t_th) / nu_t_th * 100.0),
        "Tz_tot_th_Myr": float(Tz_t_th * T_UNIT_GYR * 1e3),
        "Tz_tot_num_Myr": float(Tz_dh_num * T_UNIT_GYR * 1e3),
        "A_R_tot_th_kpc": float(A_R_t_th),
        "A_R_tot_num_kpc": float(0.5 * (np.max(R_dh) - np.min(R_dh))),
        "R_mean_tot_num_kpc": float(np.mean(R_dh)),
        "dh_max_dE_rel": float(np.max(dE_dh)),
        "dh_max_dLz_rel": float(np.max(dLz_dh)),
        # Radio externo R1 = 25 kpc
        "R25_kap_d_th": float(kappa_kuzmin(R1, Md, ad)),
        "R25_kap_t_th": float(kappa_total(R1, Md, ad, Mh, ah, bh)),
        "R25_kap_d_num": float(kap_d1_num),
        "R25_kap_t_num": float(kap_t1_num),
        "R25_nu_d_th": float(nu_eff_kuzmin(R1, z0_pert, Md, ad)),
        "R25_nu_t_th": float(nu_eff_total(R1, z0_pert, Md, ad, Mh, ah, bh)),
        "R25_nu_d_num": float(nu_d1_num),
        "R25_nu_t_num": float(nu_t1_num),
    }
    print("=== RESUMEN NUMERAL 3 ===")
    for k, v in summary.items():
        print(f"  {k:24s}: {v:.6g}")
    return summary


if __name__ == "__main__":
    run_part3()
