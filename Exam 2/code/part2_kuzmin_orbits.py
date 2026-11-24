"""
part2_kuzmin_orbits.py
Numeral 2: Integración numérica (Velocity Verlet) de una estrella en el
potencial analítico de Kuzmin: órbita circular de equilibrio y órbita con
pequeñas perturbaciones radial y vertical.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from potentials import (
    T_UNIT_GYR, PARAMS,
    phi_kuzmin, acc_kuzmin_xyz, vc_kuzmin, omega_kuzmin, kappa_kuzmin,
    gz_kuzmin, Tz_eff_kuzmin, nu_eff_kuzmin,
    integrate_verlet, compute_integrals,
    measure_period_from_peaks, measure_period_from_zero_crossings,
    configure_plots,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "figures"


def analytic_parabolic_z(t, z0, gz):
    """Solución analítica parabólica a trozos para z'' = -g_z sgn(z), z(0)=z0, z'(0)=0."""
    q = np.sqrt(2.0 * z0 / gz)  # cuarto de periodo
    Tz = 4.0 * q
    tau = np.mod(t, Tz)
    z = np.empty_like(tau)
    m1 = tau < q
    m2 = (tau >= q) & (tau < 3.0 * q)
    m3 = tau >= 3.0 * q
    z[m1] = z0 - 0.5 * gz * tau[m1]**2
    z[m2] = -z0 + 0.5 * gz * (tau[m2] - 2.0 * q)**2
    z[m3] = z0 - 0.5 * gz * (tau[m3] - 4.0 * q)**2
    return z


def run_part2():
    configure_plots()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    Md = PARAMS["Md"]
    ad = PARAMS["ad"]
    R0 = PARAMS["R0"]

    vc0 = vc_kuzmin(R0, Md, ad)
    Om0 = omega_kuzmin(R0, Md, ad)
    kap0 = kappa_kuzmin(R0, Md, ad)
    gz0 = gz_kuzmin(R0, Md, ad)
    T_orb = 2.0 * np.pi / Om0
    T_kap_th = 2.0 * np.pi / kap0

    # -------------------------------------------------------------------------
    # 2A. Órbita en configuración de equilibrio (circular en z = 0, 4 periodos)
    # -------------------------------------------------------------------------
    dt_eq = 5.0e-5
    n_orbits_eq = 4.0
    n_steps_eq = int(np.ceil(n_orbits_eq * T_orb / dt_eq))

    r0_eq = [R0, 0.0, 0.0]
    v0_eq = [0.0, vc0, 0.0]
    t_eq, r_eq, v_eq = integrate_verlet(r0_eq, v0_eq, acc_kuzmin_xyz, dt_eq, n_steps_eq)

    R_eq = np.sqrt(r_eq[:, 0]**2 + r_eq[:, 1]**2)
    z_eq = r_eq[:, 2]
    E_eq, Lz_eq = compute_integrals(r_eq, v_eq, phi_kuzmin)
    dE_eq = np.abs((E_eq - E_eq[0]) / E_eq[0])
    dLz_eq = np.abs((Lz_eq - Lz_eq[0]) / Lz_eq[0])

    # Medición del periodo azimutal numérico
    phi_eq = np.unwrap(np.arctan2(r_eq[:, 1], r_eq[:, 0]))
    T_orb_num = float(t_eq[-1] / (phi_eq[-1] / (2.0 * np.pi)))

    # Figura 3: Diagnóstico de la órbita circular de equilibrio
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))

    # (a) Trayectoria en el plano (x, y)
    ax = axes[0]
    theta = np.linspace(0, 2.0 * np.pi, 400)
    ax.plot(R0 * np.cos(theta), R0 * np.sin(theta), color="#cccccc", lw=4.0,
            label=rf"Círculo teórico ($R_0={R0:.0f}\,\mathrm{{kpc}}$)")
    ax.plot(r_eq[:, 0], r_eq[:, 1], color="#1f5a99", lw=1.6, label="Órbita integrada (4 periodos)")
    ax.scatter([R0], [0.0], color="#c0392b", s=45, zorder=6, label="Posición inicial")
    ax.annotate("", xy=(R0 * np.cos(0.35), R0 * np.sin(0.35)),
                xytext=(R0 * np.cos(0.08), R0 * np.sin(0.08)),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"$y\;[\mathrm{kpc}]$")
    ax.set_title("(a) Plano orbital $(x, y)$")
    ax.set_xlim(-15.0, 15.0)
    ax.set_ylim(-15.0, 15.0)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="center", fontsize=8.0)

    # (b) Estabilidad de R(t) y z(t)
    ax = axes[1]
    t_norm_eq = t_eq / T_orb
    ax.plot(t_norm_eq, R_eq, color="#1f5a99", label=r"$R(t)$ integrado")
    ax.axhline(R0, color="#c0392b", ls="--", lw=1.2, label=rf"$R_0 = {R0:.0f}\,\mathrm{{kpc}}$")
    ax.plot(t_norm_eq, z_eq, color="#2e8b57", label=r"$z(t) = 0$")
    ax.set_xlabel(r"Tiempo $[t / T_{\mathrm{orb}}]$")
    ax.set_ylabel(r"Coordenadas $R(t),\;z(t)\;[\mathrm{kpc}]$")
    ax.set_title("(b) Radio y altura en equilibrio")
    ax.set_xlim(0, n_orbits_eq)
    ax.set_ylim(-1.2, 14.2)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="center right", fontsize=8.5)

    # (c) Conservación relativa de energía y momento angular
    ax = axes[2]
    floor = 1e-16
    ax.semilogy(t_norm_eq, np.maximum(dE_eq, floor), color="#d95f02", lw=1.4,
                label=r"$|\Delta E / E_0|$")
    ax.semilogy(t_norm_eq, np.maximum(dLz_eq, floor), color="#7570b3", lw=1.4,
                label=r"$|\Delta L_z / L_{z,0}|$")
    ax.set_xlabel(r"Tiempo $[t / T_{\mathrm{orb}}]$")
    ax.set_ylabel("Error relativo")
    ax.set_title("(c) Conservación de integrales")
    ax.set_xlim(0, n_orbits_eq)
    ax.set_ylim(1e-16, 1e-10)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=8.8)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG_DIR / "fig3_equilibrium_orbit.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2B. Órbita perturbada (radial + vertical, 5 periodos orbitales)
    # -------------------------------------------------------------------------
    dvR = 0.05 * vc0      # 5% de v_c -> mantiene L_z y R_g = R_0
    z0_pert = 0.40        # kpc (3.3% de R_0, 5% de a_d)
    A_R_th = dvR / kap0   # Amplitud radial teórica en aprox. epicíclica
    Tz_th = Tz_eff_kuzmin(R0, z0_pert, Md, ad)
    nu_th = nu_eff_kuzmin(R0, z0_pert, Md, ad)

    dt_p = 2.5e-5
    n_orbits_p = 5.0
    n_steps_p = int(np.ceil(n_orbits_p * T_orb / dt_p))

    r0_p = [R0, 0.0, z0_pert]
    v0_p = [dvR, vc0, 0.0]
    t_p, r_p, v_p = integrate_verlet(r0_p, v0_p, acc_kuzmin_xyz, dt_p, n_steps_p)

    R_p = np.sqrt(r_p[:, 0]**2 + r_p[:, 1]**2)
    z_p = r_p[:, 2]
    E_p, Lz_p = compute_integrals(r_p, v_p, phi_kuzmin)
    dE_p = np.abs((E_p - E_p[0]) / E_p[0])
    dLz_p = np.abs((Lz_p - Lz_p[0]) / Lz_p[0])

    # Medición numérica de frecuencias radial y vertical
    T_kap_num, kap_num = measure_period_from_peaks(t_p, R_p)
    Tz_num, nu_num = measure_period_from_zero_crossings(t_p, z_p)
    A_R_num = 0.5 * (np.max(R_p) - np.min(R_p))
    R_mean_num = float(np.mean(R_p))
    z_max_num = float(np.max(np.abs(z_p)))

    # Predicciones analíticas en función del tiempo
    R_anal = R0 + A_R_th * np.sin(kap0 * t_p)
    z_anal = analytic_parabolic_z(t_p, z0_pert, gz0)

    # Figura 4: Órbita perturbada y comparación con la teoría
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    t_norm_p = t_p / T_orb

    # (a) Órbita no cerrada en el plano (x, y)
    ax = axes[0, 0]
    ax.plot(R0 * np.cos(theta), R0 * np.sin(theta), color="#999999", ls="--", lw=1.2,
            label=rf"Órbita de equilibrio ($R_0={R0:.0f}\,\mathrm{{kpc}}$)")
    ax.plot(r_p[:, 0], r_p[:, 1], color="#1f5a99", lw=1.1, label="Órbita perturbada (5 periodos)")
    ax.scatter([R0], [0.0], color="#c0392b", s=38, zorder=6, label="Inicio")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"$y\;[\mathrm{kpc}]$")
    ax.set_title("(a) Proyección en el plano $(x, y)$ (órbita no cerrada)")
    ax.set_xlim(-15.0, 15.0)
    ax.set_ylim(-15.0, 15.0)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="center", fontsize=8.2)

    # (b) Plano meridional (R, z)
    ax = axes[0, 1]
    ax.plot(R_p, z_p, color="#1f5a99", lw=0.85, alpha=0.85, label="Trayectoria meridional")
    ax.axvline(R0, color="#999999", ls=":", lw=1.1)
    ax.axhline(0.0, color="#999999", ls=":", lw=1.1)
    ax.scatter([R0], [z0_pert], color="#c0392b", s=42, zorder=6, label=r"Inicio $(R_0, z_0)$")
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"Altura $z\;[\mathrm{kpc}]$")
    ax.set_title("(b) Sección meridional $(R, z)$")
    ax.set_xlim(11.55, 12.90)
    ax.set_ylim(-0.52, 0.62)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=8.5)

    # (c) Oscilación radial R(t) vs predicción epicíclica
    ax = axes[1, 0]
    ax.plot(t_norm_p, R_p, color="#1f5a99", lw=1.8, label="Integración numérica $R(t)$")
    ax.plot(t_norm_p, R_anal, color="#c0392b", ls="--", lw=1.4,
            label=rf"Teoría epicíclica ($\kappa_{{\mathrm{{teo}}}}={kap0:.2f}$)")
    ax.axhline(R0, color="#777777", ls=":", lw=1.0)
    ax.set_xlabel(r"Tiempo $[t / T_{\mathrm{orb}}]$")
    ax.set_ylabel(r"Radio cilíndrico $R(t)\;[\mathrm{kpc}]$")
    ax.set_title(rf"(c) Oscilación radial ($\kappa_{{\mathrm{{num}}}}={kap_num:.2f}$, dif. ${abs(kap_num-kap0)/kap0*100:.2f}\%$)")
    ax.set_xlim(0, n_orbits_p)
    ax.set_ylim(11.45, 13.05)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=8.5)

    # (d) Oscilación vertical z(t) vs predicción parabólica (primeros 1.2 periodos)
    ax = axes[1, 1]
    mask_z = t_norm_p <= 1.2
    ax.plot(t_norm_p[mask_z], z_p[mask_z], color="#1b9e77", lw=1.9,
            label="Integración numérica $z(t)$")
    ax.plot(t_norm_p[mask_z], z_anal[mask_z], color="#d95f02", ls="--", lw=1.45,
            label=rf"Modelo parabólico ($\nu_{{\mathrm{{teo}}}}={nu_th:.1f}$)")
    ax.axhline(0.0, color="#777777", ls=":", lw=1.0)
    ax.set_xlabel(r"Tiempo $[t / T_{\mathrm{orb}}]$")
    ax.set_ylabel(r"Altura $z(t)\;[\mathrm{kpc}]$")
    ax.set_title(rf"(d) Oscilación vertical ($\nu_{{\mathrm{{num}}}}={nu_num:.1f}$, dif. ${abs(nu_num-nu_th)/nu_th*100:.2f}\%$)")
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-0.52, 0.68)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=8.5)

    fig.tight_layout(h_pad=2.2, w_pad=2.2)
    fig.savefig(FIG_DIR / "fig4_perturbed_orbit.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "eq_T_orb_th_ut": float(T_orb),
        "eq_T_orb_num_ut": float(T_orb_num),
        "eq_max_dR_kpc": float(np.max(np.abs(R_eq - R0))),
        "eq_max_dz_kpc": float(np.max(np.abs(z_eq))),
        "eq_max_dE_rel": float(np.max(dE_eq)),
        "eq_max_dLz_rel": float(np.max(dLz_eq)),
        "pert_dvR_kms": float(dvR),
        "pert_z0_kpc": float(z0_pert),
        "pert_A_R_th_kpc": float(A_R_th),
        "pert_A_R_num_kpc": float(A_R_num),
        "pert_R_mean_num_kpc": float(R_mean_num),
        "pert_z_max_num_kpc": float(z_max_num),
        "pert_T_kap_th_ut": float(T_kap_th),
        "pert_T_kap_num_ut": float(T_kap_num),
        "pert_kap_th": float(kap0),
        "pert_kap_num": float(kap_num),
        "pert_kap_err_pct": float(abs(kap_num - kap0) / kap0 * 100.0),
        "pert_Tz_th_ut": float(Tz_th),
        "pert_Tz_num_ut": float(Tz_num),
        "pert_nu_th": float(nu_th),
        "pert_nu_num": float(nu_num),
        "pert_nu_err_pct": float(abs(nu_num - nu_th) / nu_th * 100.0),
        "pert_max_dE_rel": float(np.max(dE_p)),
        "pert_max_dLz_rel": float(np.max(dLz_p)),
    }
    print("=== RESUMEN NUMERAL 2 ===")
    for k, v in summary.items():
        print(f"  {k:24s}: {v:.6g}")
    return summary, {"t": t_p, "r": r_p, "v": v_p, "R": R_p, "z": z_p}


if __name__ == "__main__":
    run_part2()
