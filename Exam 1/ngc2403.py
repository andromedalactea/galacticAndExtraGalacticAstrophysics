"""
NGC 2403: centro, inclinacion, brillo superficial, tipo de Hubble,
angulo de paso de los brazos, distancia y escala del disco.

Datos:
  data/NGC_2403_beta_g.fits   imagen SDSS banda g (NASA-Sloan Atlas)
  data/NGC2403_Rot_Curve.dat curva de rotacion, R en kpc y V en km/s

La imagen esta en nanomaggies (m_AB = 22.5 - 2.5 log10 flujo).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, label

CARPETA = Path(__file__).resolve().parent
RUTA_IMAGEN = CARPETA / "data" / "NGC_2403_beta_g.fits"
RUTA_CURVA = CARPETA / "data" / "NGC2403_Rot_Curve.dat"
CARPETA_FIGURAS = CARPETA / "figuras"

# Espesor de un disco visto de canto. Sirve para pasar de b/a a inclinacion.
Q0 = 0.2
# Punto cero AB de un nanomaggy.
CERO_AB = 22.5
# Tully-Fisher en banda g (Pizagno et al. 2007):
#   M_g = pendiente * (log10(V) - log_v0) + cero
# V es la velocidad circular. Dispersion intrinseca ~ 0.46 mag.
TF_PENDIENTE = -5.476
TF_LOG_V0 = 2.220
TF_CERO = -20.686
TF_DISPERSION = 0.46
# Extincion interna aproximada: A_g = gamma * log10(a/b).
GAMMA_G = 1.5


def cargar_curva(ruta):
    radio, velocidad = np.loadtxt(ruta, unpack=True)
    return radio, velocidad


def cargar_imagen(ruta):
    with fits.open(ruta) as hdul:
        imagen = hdul[0].data.astype(float)
        cabecera = hdul[0].header
    # CD1_1 esta en grados por pixel.
    escala_arco = abs(cabecera["CD1_1"]) * 3600.0
    return imagen, cabecera, escala_arco


def limpiar(imagen):
    """Resta el fondo y tapa estrellas puntuales con el valor suavizado."""
    borde = np.concatenate(
        [
            imagen[:50, :].ravel(),
            imagen[-50:, :].ravel(),
            imagen[:, :50].ravel(),
            imagen[:, -50:].ravel(),
        ]
    )
    fondo = np.median(borde)
    sigma = 1.4826 * np.median(np.abs(borde - fondo))
    limpia = np.clip(imagen - fondo, 0, None)
    suave = gaussian_filter(limpia, 6)
    estrellas = (limpia - suave) > 8 * sigma
    limpia[estrellas] = suave[estrellas]
    return limpia, fondo, sigma


def momentos(mascara):
    """Elipse equivalente de una region: centro, semiejes y angulo."""
    ys, xs = np.nonzero(mascara)
    if xs.size < 200:
        return None
    xc = xs.mean()
    yc = ys.mean()
    dx = xs - xc
    dy = ys - yc
    mu20 = np.mean(dx * dx)
    mu02 = np.mean(dy * dy)
    mu11 = np.mean(dx * dy)
    valores, vectores = np.linalg.eigh([[mu20, mu11], [mu11, mu02]])
    orden = np.argsort(valores)[::-1]
    valores = valores[orden]
    vectores = vectores[:, orden]
    semimayor = 2 * np.sqrt(valores[0])
    semimenor = 2 * np.sqrt(valores[1])
    angulo = np.degrees(np.arctan2(vectores[1, 0], vectores[0, 0]))
    return xc, yc, semimayor, semimenor, semimenor / semimayor, angulo


def promedio_de_eje(angulos):
    """Promedio de un angulo de eje (da lo mismo sumar 180 grados)."""
    doble = np.deg2rad(2 * np.asarray(angulos, dtype=float))
    medio = np.arctan2(np.mean(np.sin(doble)), np.mean(np.cos(doble)))
    return np.degrees(medio) / 2


def isofotas(limpia, sigma):
    """
    Cada isofota es la region por encima de un nivel de brillo.
    La forma sale de los momentos de esa region.
    El centro es el centroide de las isofotas del disco, no el pixel mas brillante.
    """
    suave = gaussian_filter(limpia, 4)
    referencia = suave[suave > 5 * sigma]
    niveles = np.percentile(referencia, [15, 25, 35, 50, 65, 80])
    filas = []
    for nivel in niveles:
        regiones, n = label(limpia > nivel)
        if n == 0:
            continue
        conteo = np.bincount(regiones.ravel())
        conteo[0] = 0
        forma = momentos(regiones == np.argmax(conteo))
        if forma is None:
            continue
        filas.append(forma)

    # Disco: isofotas grandes y achatadas. El bulbo y el ruido quedan fuera.
    disco = [f for f in filas if 250 < f[2] < 1100 and 0.35 < f[4] < 0.75]
    if len(disco) < 2:
        disco = filas
    disco = sorted(disco, key=lambda f: f[2], reverse=True)
    externas = disco[:3]
    centro_x = np.median([f[0] for f in externas])
    centro_y = np.median([f[1] for f in externas])
    razon = np.median([f[4] for f in externas])
    angulo = promedio_de_eje([f[5] for f in externas])
    return filas, centro_x, centro_y, razon, angulo


def inclinacion(razon_ejes):
    """cos^2 i = (q^2 - q0^2) / (1 - q0^2)."""
    numerador = razon_ejes**2 - Q0**2
    denominador = 1 - Q0**2
    coseno2 = np.clip(numerador / denominador, 0, 1)
    return np.degrees(np.arccos(np.sqrt(coseno2)))


def radio_eliptico(forma, centro_x, centro_y, razon, angulo):
    yy, xx = np.indices(forma)
    dx = xx - centro_x
    dy = yy - centro_y
    ca = np.cos(np.deg2rad(angulo))
    sa = np.sin(np.deg2rad(angulo))
    eje_mayor = dx * ca + dy * sa
    eje_menor = -dx * sa + dy * ca
    return np.sqrt(eje_mayor**2 + (eje_menor / razon) ** 2)


def perfil_de_brillo(limpia, radio):
    """Brillo mediano en anillos elipticos, a lo largo del semieje mayor."""
    pasos = np.arange(10, 950, 20)
    radios = []
    brillos = []
    for r1, r2 in zip(pasos[:-1], pasos[1:]):
        anillo = (radio >= r1) & (radio < r2)
        if anillo.sum() < 40:
            continue
        radios.append(0.5 * (r1 + r2))
        brillos.append(np.median(limpia[anillo]))
    return np.array(radios), np.array(brillos)


def ajustar_disco_y_bulbo(radios, brillos, sigma):
    """
    El disco es la recta de log I contra R en la zona exterior.
    El bulbo es el exceso central sobre esa recta.
    La zona de interseccion es donde el exceso se apaga y queda solo el disco.
    """
    utiles = brillos > 4 * sigma
    radio_limite = radios[utiles].max()
    # Recta en la zona lineal: fuera del centro y antes de la caida externa.
    zona_disco = (radios > 0.20 * radio_limite) & (radios < 0.65 * radio_limite) & utiles
    pendiente, ordenada = np.polyfit(radios[zona_disco], np.log(brillos[zona_disco]), 1)
    escala_px = -1.0 / pendiente
    brillo_central_disco = np.exp(ordenada)
    modelo_disco = brillo_central_disco * np.exp(-radios / escala_px)
    exceso = brillos - modelo_disco

    # La interseccion es el ultimo radio interno donde el bulbo aun se ve
    # por encima del disco (exceso mayor que el 10% del disco).
    dentro = (exceso > 0.10 * modelo_disco) & (radios < 0.35 * radio_limite)
    if dentro.any():
        radio_interseccion = radios[dentro].max()
    else:
        radio_interseccion = radios[np.argmax(exceso)]

    # Luz: el area de un anillo eliptico es 2*pi*q*R*dR. q se cancela en el cociente.
    dr = np.median(np.diff(radios))
    peso = 2 * np.pi * radios * dr
    luz_bulbo = np.sum(np.clip(exceso, 0, None) * peso)
    luz_total = np.sum(np.clip(brillos, 0, None) * peso)
    fraccion_bulbo = luz_bulbo / luz_total
    return {
        "escala_px": escala_px,
        "brillo_central_disco": brillo_central_disco,
        "modelo_disco": modelo_disco,
        "exceso": exceso,
        "radio_interseccion": radio_interseccion,
        "fraccion_bulbo": fraccion_bulbo,
        "radio_limite": radio_limite,
    }


def flujo_total(ajuste, razon_ejes):
    """
    Flujo de un disco exponencial integrado hasta infinito, en nanomaggies.
    I = I0 exp(-R/h) y el area eliptica da 2*pi*q*I0*h^2.
    Se suma el exceso del bulbo medido en el perfil.
    """
    h = ajuste["escala_px"]
    i0 = ajuste["brillo_central_disco"]
    flujo_disco = 2 * np.pi * razon_ejes * i0 * h**2
    # El peso usado en el ajuste no incluia q; aqui si, para no contar de mas.
    return flujo_disco * (1 + ajuste["fraccion_bulbo"] / max(1 - ajuste["fraccion_bulbo"], 1e-3))


def clasificar(fraccion_bulbo, angulo_paso):
    """Tipo de Hubble a partir del bulbo y de que tan abiertos son los brazos."""
    if fraccion_bulbo > 0.4:
        tipo = "Sa"
    elif fraccion_bulbo > 0.2:
        tipo = "Sb"
    elif fraccion_bulbo > 0.08 or angulo_paso < 18:
        tipo = "Sc"
    else:
        tipo = "Scd"
    return tipo


def angulo_de_paso(limpia, centro_x, centro_y, razon, angulo):
    """
    Se deproyecta el disco y se busca la espiral logaritmica que alinea los brazos.
    En una espiral logaritmica, phi = cot(psi) * ln(R) + constante.
    """
    suave = gaussian_filter(limpia, 4)
    paso = 2
    yy, xx = np.indices(suave.shape)
    ff = suave[::paso, ::paso]
    xx = xx[::paso, ::paso]
    yy = yy[::paso, ::paso]
    dx = xx - centro_x
    dy = yy - centro_y
    ca = np.cos(np.deg2rad(angulo))
    sa = np.sin(np.deg2rad(angulo))
    eje_mayor = dx * ca + dy * sa
    eje_menor = (-dx * sa + dy * ca) / razon
    radio = np.sqrt(eje_mayor**2 + eje_menor**2)
    phi = np.arctan2(eje_menor, eje_mayor)

    r_min, r_max = 120, 620
    n_radio, n_phi = 36, 72
    m = (radio > r_min) & (radio < r_max)
    ln_r = np.log(radio[m])
    ph = phi[m]
    valor = np.clip(ff[m], 0, np.percentile(ff[m], 90))

    ln_bordes = np.linspace(np.log(r_min), np.log(r_max), n_radio + 1)
    phi_bordes = np.linspace(-np.pi, np.pi, n_phi + 1)
    mapa = np.zeros((n_radio, n_phi))
    cuenta = np.zeros((n_radio, n_phi))
    ir = np.clip(np.digitize(ln_r, ln_bordes) - 1, 0, n_radio - 1)
    ip = np.clip(np.digitize(ph, phi_bordes) - 1, 0, n_phi - 1)
    np.add.at(mapa, (ir, ip), valor)
    np.add.at(cuenta, (ir, ip), 1)
    cuenta[cuenta == 0] = 1
    mapa /= cuenta
    residuo = mapa - np.median(mapa, axis=1, keepdims=True)
    ln_centro = 0.5 * (ln_bordes[:-1] + ln_bordes[1:])
    ln0 = ln_centro[len(ln_centro) // 2]
    dphi = phi_bordes[1] - phi_bordes[0]

    def puntaje(psi_grados):
        cot = 1.0 / np.tan(np.deg2rad(psi_grados))
        perfil = np.zeros(n_phi)
        for i, ln in enumerate(ln_centro):
            casillas = int(np.round(cot * (ln - ln0) / dphi))
            perfil += np.roll(residuo[i], -casillas)
        return perfil.max() - np.median(perfil)

    candidatos = np.arange(8, 46)
    puntajes = np.array([puntaje(p) for p in candidatos])
    # Los brazos son irregulares: se suaviza el contraste y se toma el maximo.
    nucleo = np.ones(3) / 3
    suaves = np.convolve(puntajes, nucleo, mode="same")
    mejor = float(candidatos[np.argmax(suaves)])
    altos = candidatos[suaves > 0.95 * suaves.max()]
    return mejor, float(altos.min()), float(altos.max()), candidatos, puntajes


def distancia_tully_fisher(velocidad, magnitud_aparente, razon_ejes):
    """Magnitud absoluta por Tully-Fisher y modulo de distancia."""
    magnitud_absoluta = TF_PENDIENTE * (np.log10(velocidad) - TF_LOG_V0) + TF_CERO
    extincion = GAMMA_G * np.log10(1.0 / razon_ejes)
    aparente_corregida = magnitud_aparente - extincion
    modulo = aparente_corregida - magnitud_absoluta
    distancia_mpc = 10 ** ((modulo + 5) / 5) / 1e6
    # 0.46 mag de dispersion pasan a un factor en distancia.
    factor = 10 ** (TF_DISPERSION / 5)
    return {
        "magnitud_absoluta": magnitud_absoluta,
        "extincion": extincion,
        "aparente_corregida": aparente_corregida,
        "modulo": modulo,
        "distancia_mpc": distancia_mpc,
        "distancia_baja": distancia_mpc / factor,
        "distancia_alta": distancia_mpc * factor,
    }


def guardar_figuras(limpia, filas, centro_x, centro_y, razon, angulo, radios, brillos, ajuste, curva_r, curva_v, pitch_angulos, pitch_puntajes, angulo_paso):
    CARPETA_FIGURAS.mkdir(exist_ok=True)
    suave = gaussian_filter(limpia, 3)
    y0, x0 = 550, 450
    recorte = suave[y0 : y0 + 1700, x0 : x0 + 1800]

    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.imshow(np.log10(recorte + 0.02), origin="lower", cmap="gray_r")
    ax.plot(centro_x - x0, centro_y - y0, "r+", ms=10)
    for forma in filas:
        if not (250 < forma[2] < 1100):
            continue
        t = np.linspace(0, 2 * np.pi, 200)
        a = forma[2]
        b = forma[3]
        ca = np.cos(np.deg2rad(forma[5]))
        sa = np.sin(np.deg2rad(forma[5]))
        xe = forma[0] + a * np.cos(t) * ca - b * np.sin(t) * sa
        ye = forma[1] + a * np.cos(t) * sa + b * np.sin(t) * ca
        ax.plot(xe - x0, ye - y0, "c-", lw=0.7)
    ax.set_title("Isofotas y centro")
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "isofotas.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(radios, brillos, "o", ms=3, label="I(R) medida")
    ax.plot(radios, ajuste["modelo_disco"], label="disco exponencial")
    ax.axvline(ajuste["radio_interseccion"], color="k", ls="--", label="fin del bulbo")
    ax.set_yscale("log")
    ax.set_xlabel("semieje mayor (pixeles)")
    ax.set_ylabel("brillo (nanomaggies / pixel)")
    ax.set_title("Brillo superficial")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "perfil.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(curva_r, curva_v)
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel("V (km/s)")
    ax.set_title("Curva de rotacion")
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "curva.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(pitch_angulos, pitch_puntajes)
    ax.axvline(angulo_paso, color="k", ls="--")
    ax.set_xlabel("angulo de paso (grados)")
    ax.set_ylabel("contraste al alinear los brazos")
    ax.set_title("Pitch angle")
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / "pitch.png", dpi=120)
    plt.close(fig)


def main():
    imagen, cabecera, escala_arco = cargar_imagen(RUTA_IMAGEN)
    curva_r, curva_v = cargar_curva(RUTA_CURVA)
    limpia, fondo, sigma = limpiar(imagen)

    filas, centro_x, centro_y, razon, angulo = isofotas(limpia, sigma)
    inc = inclinacion(razon)
    radio = radio_eliptico(limpia.shape, centro_x, centro_y, razon, angulo)
    radios, brillos = perfil_de_brillo(limpia, radio)
    ajuste = ajustar_disco_y_bulbo(radios, brillos, sigma)

    flujo = flujo_total(ajuste, razon)
    magnitud = CERO_AB - 2.5 * np.log10(flujo)
    # La parte plana de la curva, al final del archivo.
    velocidad = np.median(curva_v[-30:])
    dist = distancia_tully_fisher(velocidad, magnitud, razon)

    paso, paso_min, paso_max, pitch_angulos, pitch_puntajes = angulo_de_paso(
        limpia, centro_x, centro_y, razon, angulo
    )
    tipo = clasificar(ajuste["fraccion_bulbo"], paso)

    kpc_por_arco = dist["distancia_mpc"] / 206.265
    escala_kpc = ajuste["escala_px"] * escala_arco * kpc_por_arco
    interseccion_kpc = ajuste["radio_interseccion"] * escala_arco * kpc_por_arco

    ra = cabecera["CRVAL1"] + cabecera["CD1_1"] * (centro_x - cabecera["CRPIX1"])
    dec = cabecera["CRVAL2"] + cabecera["CD2_2"] * (centro_y - cabecera["CRPIX2"])

    print(f"fondo = {fondo:.4f}   sigma = {sigma:.4f}")
    print(f"centro (pixel) = ({centro_x:.1f}, {centro_y:.1f})")
    print(f"centro (RA, Dec) = ({ra:.4f}, {dec:.4f}) grados")
    print(f"b/a = {razon:.3f}   inclinacion = {inc:.1f} grados")
    # PA astronomico: desde el norte hacia el este. +x es el este y +y el norte.
    pa = np.degrees(np.arctan2(np.cos(np.deg2rad(angulo)), np.sin(np.deg2rad(angulo)))) % 180
    print(f"angulo del eje mayor desde +x = {angulo:.1f} grados")
    print(f"angulo de posicion (N hacia E) = {pa:.1f} grados")
    print(f"fraccion de luz del bulbo B/T = {ajuste['fraccion_bulbo']:.3f}")
    print(f"radio de interseccion = {ajuste['radio_interseccion']:.0f} px = {interseccion_kpc:.2f} kpc")
    print(f"escala del disco = {ajuste['escala_px']:.1f} px = {ajuste['escala_px'] * escala_arco:.1f} arcsec = {escala_kpc:.2f} kpc")
    print(f"tipo de Hubble = {tipo}")
    print(f"angulo de paso = {paso:.0f} grados (zona alta: {paso_min:.0f} a {paso_max:.0f})")
    print(f"V plana = {velocidad:.1f} km/s")
    print(f"magnitud g = {magnitud:.2f}   A_g interna = {dist['extincion']:.2f}")
    print(f"M_g = {dist['magnitud_absoluta']:.2f}   modulo = {dist['modulo']:.2f}")
    print(
        f"distancia = {dist['distancia_mpc']:.2f} Mpc"
        f"  (rango {dist['distancia_baja']:.2f} a {dist['distancia_alta']:.2f})"
    )
    print(f"escala de placa = {escala_arco:.3f} arcsec/pixel")

    guardar_figuras(
        limpia,
        filas,
        centro_x,
        centro_y,
        razon,
        angulo,
        radios,
        brillos,
        ajuste,
        curva_r,
        curva_v,
        pitch_angulos,
        pitch_puntajes,
        paso,
    )


if __name__ == "__main__":
    main()
