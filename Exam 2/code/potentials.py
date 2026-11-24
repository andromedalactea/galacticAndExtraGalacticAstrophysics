"""
potentials.py
Funciones analíticas para el disco de Kuzmin, el halo oblato de Miyamoto-Nagai,
frecuencias orbitales e integrador simpléctico Velocity Verlet.

Sistema de unidades:
    Masa     : 1 u.m. = 10^10 Msun
    Longitud : 1 u.l. = 1 kpc
    Velocidad: 1 u.v. = 1 km/s
    Tiempo   : 1 u.t. = 1 kpc / (1 km/s) = 0.977792 Gyr (~1 Gyr)
    G        : 43007.1 kpc (km/s)^2 (10^10 Msun)^-1
"""

import numpy as np

# Constantes del sistema de unidades
G = 43007.1
T_UNIT_GYR = 0.97779222168

# Parámetros estructurales del modelo (último dígito de cédula # = 5)
PARAMS = {
    "Md": 50.0,    # 5 x 10^11 Msun
    "ad": 8.0,     # kpc (escala radial del disco de Kuzmin)
    "R0": 12.0,    # kpc (radio de referencia de la estrella de prueba)
    "Mh": 250.0,   # 2.5 x 10^12 Msun (Mh / Md = 5)
    "ah": 8.0,     # kpc (parámetro radial del halo de Miyamoto-Nagai)
    "bh": 20.0,    # kpc (parámetro vertical del halo; bh/ah = 2.5 -> oblato en z)
}


# =============================================================================
# 1. Disco de Kuzmin (Binney & Tremaine, Ec. 2.68)
# =============================================================================

def phi_kuzmin(R, z, Md=PARAMS["Md"], ad=PARAMS["ad"]):
    """Potencial gravitacional del disco de Kuzmin Phi_K(R, z)."""
    return -G * Md / np.sqrt(R**2 + (ad + np.abs(z))**2)


def sigma_kuzmin(R, Md=PARAMS["Md"], ad=PARAMS["ad"]):
    """Densidad superficial del disco de Kuzmin Sigma_K(R) en 10^10 Msun / kpc^2."""
    return (ad * Md) / (2.0 * np.pi * (R**2 + ad**2)**1.5)


def acc_kuzmin_xyz(x, y, z, Md=PARAMS["Md"], ad=PARAMS["ad"]):
    """Aceleración a = -nabla Phi_K en coordenadas cartesianas (x, y, z)."""
    R2 = x**2 + y**2
    az_term = ad + np.abs(z)
    denom = (R2 + az_term**2)**1.5
    fac = -G * Md / denom
    ax = fac * x
    ay = fac * y
    # np.sign(0.0) == 0.0 garantiza a_z = 0 exactamente sobre el plano z = 0
    az = fac * az_term * np.sign(z)
    return ax, ay, az


def Fz_kuzmin(R, z, Md=PARAMS["Md"], ad=PARAMS["ad"]):
    """Fuerza vertical por unidad de masa F_z = -dPhi_K/dz."""
    az_term = ad + np.abs(z)
    return -G * Md * az_term * np.sign(z) / (R**2 + az_term**2)**1.5


def vc_kuzmin(R, Md=PARAMS["Md"], ad=PARAMS["ad"]):
    """Velocidad circular en el plano medio z = 0 del disco de Kuzmin."""
    return np.sqrt(G * Md * R**2 / (R**2 + ad**2)**1.5)


def omega_kuzmin(R, Md=PARAMS["Md"], ad=PARAMS["ad"]):
    """Frecuencia angular circular Omega(R) en z = 0."""
    return np.sqrt(G * Md / (R**2 + ad**2)**1.5)


def kappa_kuzmin(R, Md=PARAMS["Md"], ad=PARAMS["ad"]):
    """Frecuencia de epiciclo radial kappa(R) en z = 0."""
    return np.sqrt(G * Md * (R**2 + 4.0 * ad**2) / (R**2 + ad**2)**2.5)


def gz_kuzmin(R, Md=PARAMS["Md"], ad=PARAMS["ad"]):
    """Magnitud de la aceleración vertical justo sobre el plano (z -> 0+)."""
    return G * Md * ad / (R**2 + ad**2)**1.5


def Tz_eff_kuzmin(R, zmax, Md=PARAMS["Md"], ad=PARAMS["ad"]):
    """Periodo vertical analítico para pequeñas amplitudes z_max en Kuzmin."""
    gz = gz_kuzmin(R, Md, ad)
    return 4.0 * np.sqrt(2.0 * zmax / gz)


def nu_eff_kuzmin(R, zmax, Md=PARAMS["Md"], ad=PARAMS["ad"]):
    """Frecuencia angular vertical efectiva nu_eff = 2*pi / T_z."""
    return 2.0 * np.pi / Tz_eff_kuzmin(R, zmax, Md, ad)


# =============================================================================
# 2. Halo oblato de Miyamoto-Nagai (Binney & Tremaine, Ec. 2.69)
# =============================================================================

def phi_mn(R, z, Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Potencial de Miyamoto-Nagai Phi_MN(R, z)."""
    zb = np.sqrt(z**2 + bh**2)
    return -G * Mh / np.sqrt(R**2 + (ah + zb)**2)


def rho_mn(R, z, Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Densidad volumétrica de Miyamoto-Nagai rho_MN(R, z) en 10^10 Msun / kpc^3."""
    zb = np.sqrt(z**2 + bh**2)
    azb = ah + zb
    D2 = R**2 + azb**2
    num = (bh**2 * Mh / (4.0 * np.pi)) * (ah * R**2 + (ah + 3.0 * zb) * azb**2)
    return num / (D2**2.5 * zb**3)


def acc_mn_xyz(x, y, z, Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Aceleración a = -nabla Phi_MN en coordenadas cartesianas (x, y, z)."""
    R2 = x**2 + y**2
    zb = np.sqrt(z**2 + bh**2)
    azb = ah + zb
    denom = (R2 + azb**2)**1.5
    fac = -G * Mh / denom
    ax = fac * x
    ay = fac * y
    az = fac * azb * (z / zb)
    return ax, ay, az


def Fz_mn(R, z, Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Fuerza vertical por unidad de masa F_z = -dPhi_MN/dz."""
    zb = np.sqrt(z**2 + bh**2)
    azb = ah + zb
    return -G * Mh * azb * (z / zb) / (R**2 + azb**2)**1.5


def vc_mn(R, Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Velocidad circular del halo en z = 0."""
    sh = ah + bh
    return np.sqrt(G * Mh * R**2 / (R**2 + sh**2)**1.5)


def omega_mn(R, Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Frecuencia angular circular del halo en z = 0."""
    sh = ah + bh
    return np.sqrt(G * Mh / (R**2 + sh**2)**1.5)


def kappa_mn(R, Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Frecuencia de epiciclo radial del halo en z = 0."""
    sh = ah + bh
    return np.sqrt(G * Mh * (R**2 + 4.0 * sh**2) / (R**2 + sh**2)**2.5)


def nu_mn(R, Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Frecuencia vertical armónica del halo en z = 0."""
    sh = ah + bh
    return np.sqrt(G * Mh * sh / (bh * (R**2 + sh**2)**1.5))


# =============================================================================
# 3. Potencial combinado Disco de Kuzmin + Halo de Miyamoto-Nagai
# =============================================================================

def phi_total(R, z, Md=PARAMS["Md"], ad=PARAMS["ad"],
              Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Potencial total Phi_tot = Phi_K + Phi_MN."""
    return phi_kuzmin(R, z, Md, ad) + phi_mn(R, z, Mh, ah, bh)


def acc_total_xyz(x, y, z, Md=PARAMS["Md"], ad=PARAMS["ad"],
                  Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Aceleración total en coordenadas cartesianas."""
    ax_d, ay_d, az_d = acc_kuzmin_xyz(x, y, z, Md, ad)
    ax_h, ay_h, az_h = acc_mn_xyz(x, y, z, Mh, ah, bh)
    return ax_d + ax_h, ay_d + ay_h, az_d + az_h


def vc_total(R, Md=PARAMS["Md"], ad=PARAMS["ad"],
             Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Velocidad circular total (suma en cuadratura)."""
    return np.sqrt(vc_kuzmin(R, Md, ad)**2 + vc_mn(R, Mh, ah, bh)**2)


def omega_total(R, Md=PARAMS["Md"], ad=PARAMS["ad"],
                Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Frecuencia circular total."""
    return np.sqrt(omega_kuzmin(R, Md, ad)**2 + omega_mn(R, Mh, ah, bh)**2)


def kappa_total(R, Md=PARAMS["Md"], ad=PARAMS["ad"],
                Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Frecuencia de epiciclo total."""
    return np.sqrt(kappa_kuzmin(R, Md, ad)**2 + kappa_mn(R, Mh, ah, bh)**2)


def Tz_eff_total(R, zmax, Md=PARAMS["Md"], ad=PARAMS["ad"],
                 Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """
    Periodo vertical analítico para pequeñas amplitudes z_max bajo el campo
    combinado F_z ~ -g_{z,d} sgn(z) - nu_h^2 z.
    """
    gd = gz_kuzmin(R, Md, ad)
    nuh = nu_mn(R, Mh, ah, bh)
    arg = 1.0 / (1.0 + (nuh**2 * zmax) / gd)
    return (4.0 / nuh) * np.arccos(arg)


def nu_eff_total(R, zmax, Md=PARAMS["Md"], ad=PARAMS["ad"],
                 Mh=PARAMS["Mh"], ah=PARAMS["ah"], bh=PARAMS["bh"]):
    """Frecuencia vertical efectiva total nu_eff,tot = 2*pi / T_z,tot."""
    return 2.0 * np.pi / Tz_eff_total(R, zmax, Md, ad, Mh, ah, bh)


# =============================================================================
# 4. Integrador simpléctico Velocity Verlet (Leapfrog) y diagnósticos
# =============================================================================

def integrate_verlet(r0, v0, acc_func, dt, n_steps):
    """
    Integra d^2 r / dt^2 = acc_func(x, y, z) usando el esquema Velocity Verlet
    (diferencias finitas centradas de 2o orden, simpléctico y reversible en t).
    """
    t = np.arange(n_steps + 1, dtype=float) * dt
    r = np.zeros((n_steps + 1, 3), dtype=float)
    v = np.zeros((n_steps + 1, 3), dtype=float)

    r[0] = np.asarray(r0, dtype=float)
    v[0] = np.asarray(v0, dtype=float)

    ax, ay, az = acc_func(r[0, 0], r[0, 1], r[0, 2])
    a = np.array([ax, ay, az], dtype=float)

    for i in range(n_steps):
        v_half = v[i] + 0.5 * a * dt
        r_next = r[i] + v_half * dt
        ax_n, ay_n, az_n = acc_func(r_next[0], r_next[1], r_next[2])
        a_next = np.array([ax_n, ay_n, az_n], dtype=float)
        v_next = v_half + 0.5 * a_next * dt

        r[i + 1] = r_next
        v[i + 1] = v_next
        a = a_next

    return t, r, v


def compute_integrals(r, v, phi_func):
    """Calcula la energía específica E(t) y el momento angular L_z(t)."""
    R = np.sqrt(r[:, 0]**2 + r[:, 1]**2)
    z = r[:, 2]
    v2 = np.sum(v**2, axis=1)
    E = 0.5 * v2 + phi_func(R, z)
    Lz = r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]
    return E, Lz


def measure_period_from_peaks(t, y, prominence_frac=0.2):
    """
    Estima el periodo promedio y la frecuencia angular omega = 2*pi / T
    a partir de los máximos locales de una serie oscilatoria y(t), usando
    interpolación parabólica en cada pico para precisión sub-paso.
    """
    y_centered = y - np.mean(y)
    amp = 0.5 * (np.max(y_centered) - np.min(y_centered))
    if amp <= 0.0:
        return np.nan, np.nan

    peak_times = []
    for i in range(1, len(y_centered) - 1):
        if (y_centered[i] > y_centered[i - 1]
                and y_centered[i] >= y_centered[i + 1]
                and y_centered[i] > prominence_frac * amp):
            # Ajuste parabólico de 3 puntos alrededor del máximo
            y0, y1, y2 = y_centered[i - 1], y_centered[i], y_centered[i + 1]
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
            dt = t[i + 1] - t[i]
            peak_times.append(t[i] + delta * dt)

    peak_times = np.asarray(peak_times)
    if len(peak_times) < 2:
        return np.nan, np.nan
    periods = np.diff(peak_times)
    T_mean = float(np.mean(periods))
    omega_mean = float(2.0 * np.pi / T_mean)
    return T_mean, omega_mean


def measure_period_from_zero_crossings(t, z):
    """
    Estima el periodo vertical T_z y frecuencia nu = 2*pi / T_z a partir de los
    cruces ascendentes por z = 0 con interpolación lineal.
    """
    crossings = []
    for i in range(len(z) - 1):
        if z[i] <= 0.0 and z[i + 1] > 0.0:
            frac = -z[i] / (z[i + 1] - z[i])
            crossings.append(t[i] + frac * (t[i + 1] - t[i]))
    crossings = np.asarray(crossings)
    if len(crossings) < 2:
        return np.nan, np.nan
    T_mean = float(np.mean(np.diff(crossings)))
    nu_mean = float(2.0 * np.pi / T_mean)
    return T_mean, nu_mean


def configure_plots():
    """Estilo gráfico limpio y consistente para todas las figuras del informe."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.2,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
        "figure.constrained_layout.use": False,
    })

