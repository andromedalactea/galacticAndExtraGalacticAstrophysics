"""
part1_analytical.py
Numeral 1: Perfil estructural del disco de Kuzmin y predicciones analíticas
para las frecuencias orbitales (Omega, kappa y nu_eff).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from potentials import (
    G, T_UNIT_GYR, PARAMS,
    sigma_kuzmin, vc_kuzmin, omega_kuzmin, kappa_kuzmin,
    gz_kuzmin, Fz_kuzmin, nu_eff_kuzmin, Tz_eff_kuzmin,
    configure_plots,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "figures"


def run_part1():
    configure_plots()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    Md = PARAMS["Md"]
    ad = PARAMS["ad"]
    R0 = PARAMS["R0"]

    R = np.linspace(0.05, 40.0, 800)

    # Valores característicos del modelo de disco
    R_peak = np.sqrt(2.0) * ad
    R_half = np.sqrt(3.0) * ad
    vc_max = vc_kuzmin(R_peak, Md, ad)
    vc_R0 = vc_kuzmin(R0, Md, ad)
    Om_R0 = omega_kuzmin(R0, Md, ad)
    kap_R0 = kappa_kuzmin(R0, Md, ad)
    gz_R0 = gz_kuzmin(R0, Md, ad)
    Sigma_0 = sigma_kuzmin(0.0, Md, ad) * 1e4  # Msun / pc^2
    Sigma_R0 = sigma_kuzmin(R0, Md, ad) * 1e4  # Msun / pc^2

    # -------------------------------------------------------------------------
    # Figura 1: Densidad superficial y curva de rotación del disco de Kuzmin
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))

    # (a) Densidad superficial en Msun / pc^2
    ax = axes[0]
    Sigma_pc2 = sigma_kuzmin(R, Md, ad) * 1e4
    ax.semilogy(R, Sigma_pc2, color="#1f5a99", label=r"$\Sigma_K(R)$")
    ax.axvline(ad, color="#7f7f7f", ls=":", lw=1.3,
               label=rf"$a_d = {ad:.0f}\,\mathrm{{kpc}}$")
    ax.axvline(R_half, color="#d95f02", ls="--", lw=1.3,
               label=rf"$R_{{1/2}} = \sqrt{{3}}\,a_d \approx {R_half:.1f}\,\mathrm{{kpc}}$")
    ax.scatter([R0], [Sigma_R0], color="#c0392b", zorder=5, s=42,
               label=rf"$R_0 = {R0:.0f}\,\mathrm{{kpc}}$")
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"Densidad superficial $\Sigma_K(R)\;[M_\odot\,\mathrm{pc}^{-2}]$")
    ax.set_title("(a) Perfil de densidad superficial")
    ax.set_xlim(0, 40)
    ax.set_ylim(5, 3e3)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    # (b) Curva de rotación circular
    ax = axes[1]
    vc = vc_kuzmin(R, Md, ad)
    R_kep = np.linspace(12.0, 40.0, 300)
    vc_kep = np.sqrt(G * Md / R_kep)
    ax.plot(R, vc, color="#1f5a99", label=r"$v_c(R)$ (disco de Kuzmin)")
    ax.plot(R_kep, vc_kep, color="#7f7f7f", ls="--", lw=1.3,
            label=r"Límite kepleriano $\sqrt{GM_d/R}$")
    ax.axvline(R_peak, color="#2e8b57", ls=":", lw=1.3,
               label=rf"$R_{{\mathrm{{pico}}}} = \sqrt{{2}}\,a_d \approx {R_peak:.1f}\,\mathrm{{kpc}}$")
    ax.scatter([R0], [vc_R0], color="#c0392b", zorder=5, s=45,
               label=rf"$R_0 = {R0:.0f}\,\mathrm{{kpc}}$ ($v_c = {vc_R0:.1f}\,\mathrm{{km/s}}$)")
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"Velocidad circular $v_c(R)\;[\mathrm{km\,s^{-1}}]$")
    ax.set_title("(b) Curva de rotación del disco")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 460)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    fig.tight_layout(w_pad=2.2)
    fig1_path = FIG_DIR / "fig1_kuzmin_structure.png"
    fig.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figura 2: Frecuencias Omega, kappa, razón kappa/Omega, salto en F_z y nu_eff
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))

    # (a) Omega(R) y kappa(R)
    ax = axes[0, 0]
    Om = omega_kuzmin(R, Md, ad)
    kap = kappa_kuzmin(R, Md, ad)
    ax.plot(R, kap, color="#c0392b", label=r"Frecuencia de epiciclo $\kappa(R)$")
    ax.plot(R, Om, color="#1f5a99", label=r"Frecuencia circular $\Omega(R)$")
    ax.axvline(R0, color="#555555", ls=":", lw=1.2)
    ax.scatter([R0, R0], [kap_R0, Om_R0], color=["#c0392b", "#1f5a99"], zorder=5, s=36)
    ax.annotate(
        rf"$R_0={R0:.0f}\,\mathrm{{kpc}}$" + "\n"
        + rf"$\kappa={kap_R0:.1f}$, $\Omega={Om_R0:.1f}$",
        xy=(R0, kap_R0), xytext=(R0 + 5.5, kap_R0 + 26.0),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0),
        fontsize=9.0, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc")
    )
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"Frecuencia angular $[\mathrm{km\,s^{-1}\,kpc^{-1}}]$")
    ax.set_title(r"(a) Frecuencias $\Omega(R)$ y $\kappa(R)$ en el plano $z=0$")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 140)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    # (b) Razón adimensional kappa / Omega
    ax = axes[0, 1]
    ratio = kap / Om
    ax.plot(R, ratio, color="#8e44ad", label=r"$\kappa(R)/\Omega(R)$")
    ax.axhline(2.0, color="#555555", ls="--", lw=1.1, label="Cuerpo rígido ($\\kappa/\\Omega = 2$)")
    ax.axhline(1.0, color="#2e8b57", ls="--", lw=1.1, label="Kepleriano ($\\kappa/\\Omega = 1$)")
    ax.scatter([R0], [kap_R0 / Om_R0], color="#c0392b", zorder=5, s=40)
    ax.annotate(
        rf"$\kappa/\Omega(R_0) = {kap_R0/Om_R0:.3f}$",
        xy=(R0, kap_R0 / Om_R0), xytext=(R0 + 6.5, kap_R0 / Om_R0 + 0.16),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.0),
        fontsize=9.2, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc")
    )
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"Razón $\kappa(R)/\Omega(R)$")
    ax.set_title(r"(b) Transición entre núcleo armónico y régimen kepleriano")
    ax.set_xlim(0, 40)
    ax.set_ylim(0.85, 2.15)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")

    # (c) Discontinuidad de la fuerza vertical F_z(R_0, z)
    ax = axes[1, 0]
    z_neg = np.linspace(-2.0, -1e-4, 300)
    z_pos = np.linspace(1e-4, 2.0, 300)
    Fz_neg = Fz_kuzmin(R0, z_neg, Md, ad) / 1e3
    Fz_pos = Fz_kuzmin(R0, z_pos, Md, ad) / 1e3
    ax.plot(z_neg, Fz_neg, color="#1f5a99", label=r"$F_z(R_0, z) = -\partial\Phi_K/\partial z$")
    ax.plot(z_pos, Fz_pos, color="#1f5a99")
    ax.plot([0.0, 0.0], [-gz_R0 / 1e3, gz_R0 / 1e3], color="#c0392b", ls=":", lw=1.6,
            label=r"Salto $\Delta F_z = 4\pi G\Sigma_K(R_0)$")
    ax.scatter([0.0, 0.0], [gz_R0 / 1e3, -gz_R0 / 1e3],
               facecolors="white", edgecolors="#1f5a99", s=40, zorder=6)
    ax.axvspan(-0.4, 0.4, color="#f39c12", alpha=0.14,
               label=r"Zona $|z| \leq 0.4\,\mathrm{kpc}$ ($|F_z| \approx g_z$)")
    ax.axhline(0.0, color="#666666", lw=0.8)
    ax.set_xlabel(r"Altura sobre el plano $z\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"$F_z(R_0, z)\;[10^3\,\mathrm{km^2\,s^{-2}\,kpc^{-1}}]$")
    ax.set_title(rf"(c) Discontinuidad de la fuerza vertical en $R_0 = {R0:.0f}\,\mathrm{{kpc}}$")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-7.5, 7.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="lower left", fontsize=8.5)

    # (d) Frecuencia vertical efectiva nu_eff(R, z_max)
    ax = axes[1, 1]
    colors_z = ["#1b9e77", "#d95f02", "#7570b3"]
    for zmax, col in zip([0.2, 0.4, 1.0], colors_z):
        nu_e = nu_eff_kuzmin(R, zmax, Md, ad)
        ax.plot(R, nu_e, color=col,
                label=rf"$\nu_{{\mathrm{{eff}}}}(R)$ con $z_{{\max}} = {zmax:.1f}\,\mathrm{{kpc}}$")
    ax.plot(R, kap, color="#c0392b", ls="--", lw=1.4, label=r"Referencia: $\kappa(R)$")
    ax.axvline(R0, color="#555555", ls=":", lw=1.2)
    ax.set_xlabel(r"Radio cilíndrico $R\;[\mathrm{kpc}]$")
    ax.set_ylabel(r"Frecuencia angular $[\mathrm{km\,s^{-1}\,kpc^{-1}}]$")
    ax.set_title(r"(d) Frecuencia vertical efectiva $\nu_{\mathrm{eff}}(R, z_{\max})$")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 485)
    ax.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=8.8)

    fig.tight_layout(h_pad=2.2, w_pad=2.2)
    fig2_path = FIG_DIR / "fig2_kuzmin_frequencies.png"
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "Md_1e10": Md,
        "ad_kpc": ad,
        "R0_kpc": R0,
        "R_peak_kpc": float(R_peak),
        "R_half_kpc": float(R_half),
        "Sigma_0_Msun_pc2": float(Sigma_0),
        "Sigma_R0_Msun_pc2": float(Sigma_R0),
        "vc_max_kms": float(vc_max),
        "vc_R0_kms": float(vc_R0),
        "Omega_R0": float(Om_R0),
        "kappa_R0": float(kap_R0),
        "kappa_over_Omega_R0": float(kap_R0 / Om_R0),
        "T_orb_ut": float(2.0 * np.pi / Om_R0),
        "T_orb_Myr": float(2.0 * np.pi / Om_R0 * T_UNIT_GYR * 1e3),
        "T_kap_ut": float(2.0 * np.pi / kap_R0),
        "T_kap_Myr": float(2.0 * np.pi / kap_R0 * T_UNIT_GYR * 1e3),
        "gz_R0": float(gz_R0),
        "Tz_0p4_ut": float(Tz_eff_kuzmin(R0, 0.4, Md, ad)),
        "Tz_0p4_Myr": float(Tz_eff_kuzmin(R0, 0.4, Md, ad) * T_UNIT_GYR * 1e3),
        "nu_eff_0p4": float(nu_eff_kuzmin(R0, 0.4, Md, ad)),
    }
    print("=== RESUMEN NUMERAL 1 ===")
    for k, v in summary.items():
        print(f"  {k:24s}: {v:.4f}")
    return summary


if __name__ == "__main__":
    run_part1()
