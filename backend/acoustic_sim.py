"""
SpeakerLab Pro — Simulación Acústica con Scipy
===============================================
Modelo de circuito equivalente eléctrico-mecánico-acústico
basado en Small (1973) y Thiele (1971).

Implementa:
  · Función de transferencia H(s) exacta (4º orden reflex, 2º orden sellada)
  · SPL(f) real en dB
  · Excursión del cono X(f) en mm — detecta superación de Xmax
  · Velocidad en el puerto Vp(f) en m/s — detecta turbulencia
  · Retardo de grupo GD(f) en ms — calidad transitoria
  · Potencia disipada Pe(f) en W
  · Impedancia Z(f) en Ohm

Uso rápido:
    from acoustic_sim import simulate, plot_results
    results = simulate(DAYTON_RSS315HO)
    plot_results(results, "dayton_sim.png")
"""

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogFormatter, MultipleLocator
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════
#  CONSTANTES FÍSICAS
# ══════════════════════════════════════════════════════════════
C_AIR   = 344.0          # velocidad del sonido (m/s) @ 20°C
RHO     = 1.204          # densidad del aire (kg/m³) @ 20°C
P_REF   = 2e-5           # presión de referencia auditiva (Pa)
PI      = np.pi


# ══════════════════════════════════════════════════════════════
#  TABLAS DE ALINEACIÓN (Thiele 1971 / Small 1973)
# ══════════════════════════════════════════════════════════════
_ALIGN_TABLES = {
    "QB3":  [(.20,8.80,1.87,1.55),(.25,6.13,1.67,1.45),(.30,4.42,1.50,1.36),
             (.35,3.22,1.35,1.27),(.40,2.35,1.22,1.19),(.45,1.70,1.10,1.10),(.50,1.20,1.00,1.00)],
    "SBB4": [(.20,16.32,2.09,1.00),(.25,10.04,1.85,1.00),(.30,6.57,1.64,1.00),
             (.35,4.50,1.46,1.00),(.40,3.18,1.30,1.00),(.45,2.29,1.16,1.00),(.50,1.67,1.04,1.00)],
    "B4":   [(.20,6.97,1.56,1.56),(.25,4.47,1.41,1.41),(.30,3.05,1.28,1.28),
             (.35,2.17,1.16,1.16),(.40,1.58,1.07,1.07),(.45,1.18,0.98,0.98),(.50,0.89,0.91,0.91)],
}

def _interp_align(table_name, qts):
    qts = np.clip(qts, 0.2, 0.5)
    rows = _ALIGN_TABLES[table_name]
    for i in range(len(rows)-1):
        q0,a0,h0,f0 = rows[i]; q1,a1,h1,f1 = rows[i+1]
        if q0 <= qts <= q1:
            t = (qts-q0)/(q1-q0)
            return a0+t*(a1-a0), h0+t*(h1-h0), f0+t*(f1-f0)
    return rows[-1][1:]


# ══════════════════════════════════════════════════════════════
#  DERIVACIÓN DE PARÁMETROS FÍSICOS DESDE T/S
# ══════════════════════════════════════════════════════════════

def ts_to_physical(d: dict) -> dict:
    """
    Convierte parámetros Thiele/Small a parámetros físicos del circuito
    equivalente (impedancias mecánicas y acústicas).

    Parámetros de entrada necesarios:
        fs   : frecuencia de resonancia libre (Hz)
        vas  : volumen equivalente de aire (L)
        qts  : factor de calidad total
        qes  : factor de calidad eléctrico
        qms  : factor de calidad mecánico
        sd   : área efectiva del cono (cm²)
        re   : resistencia de la bobina (Ω) — default 6Ω
        bl   : factor de fuerza (T·m) — estimado si no se provee
        mms  : masa del cono (g) — estimada si no se provee
    """
    fs   = float(d["fs"])
    vas  = float(d["vas"]) * 1e-3          # litros → m³
    qts  = float(d["qts"])
    qes  = float(d.get("qes", qts * 1.1))
    qms  = float(d.get("qms", qts / (1/qts - 1/qes + 1e-9) if qes != qts else 10))
    sd   = float(d.get("sd", 530)) * 1e-4  # cm² → m²
    re   = float(d.get("re", 6.0))         # Ω
    xmax = float(d.get("xmax", 10)) * 1e-3 # mm → m

    ws = 2 * PI * fs

    # Masa móvil Mms (kg) — desde Vas y Sd
    # Cas = Vas / (rho * c²)  →  Mms = 1 / (ws² * Cas * Sd²)  simplificado
    cas = vas / (RHO * C_AIR**2)           # m^5/N (compliance acústica)
    cms = cas / sd**2                       # m/N  (compliance mecánica)
    mms = 1.0 / (ws**2 * cms)              # kg

    # Factor de fuerza Bl (T·m)
    # Bl = Re * Mms * ws / Qes
    bl  = np.sqrt(re * mms * ws / qes)

    # Resistencia mecánica Rms (N·s/m)
    rms = mms * ws / qms

    return {
        "fs": fs, "ws": ws,
        "vas": vas, "sd": sd, "re": re, "bl": bl,
        "mms": mms, "cms": cms, "rms": rms,
        "qts": qts, "qes": qes, "qms": qms,
        "xmax": xmax,
        "cas": cas,
    }


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN DE TRANSFERENCIA — CAJA SELLADA
#
#  H(s) = s² / (s² + s·ws/Qtc + ws_c²)
#  donde ws_c = ws·sqrt(1 + Vas/Vb)  y  Qtc = Qts·sqrt(1+Vas/Vb)
# ══════════════════════════════════════════════════════════════

def tf_closed(p: dict, vb_liters: float = None, qtc_target: float = None):
    """
    Función de transferencia 2º orden para caja sellada.
    Devuelve scipy.signal.TransferFunction normalizada.
    """
    vas  = p["vas"]
    qts  = p["qts"]
    fs   = p["fs"]

    if qtc_target is not None:
        vb_liters = round(float(vas * 1e3 / ((qtc_target / qts)**2 - 1)), 1)
        vb = vb_liters * 1e-3
    elif vb_liters is not None:
        vb = vb_liters * 1e-3
    else:
        raise ValueError("Se requiere vb_liters o qtc_target")

    alpha  = vas / vb                       # ratio
    fc     = fs * np.sqrt(1 + alpha)        # frecuencia de resonancia en caja
    qtc    = qts * np.sqrt(1 + alpha)       # Qtc real
    wc     = 2 * PI * fc

    # Numerador: s²  → [1, 0, 0]
    # Denominador: s² + (wc/Qtc)·s + wc²
    num = [1.0, 0.0, 0.0]
    den = [1.0, wc / qtc, wc**2]

    return signal.TransferFunction(num, den), fc, qtc, vb*1e3


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN DE TRANSFERENCIA — CAJA REFLEX
#
#  Modelo de Small (4º orden):
#
#  H(s) = s⁴ / D(s)
#
#  D(s) = s⁴ + s³·(wb/Qb) + s²·(wb²+wa²+wa·wb/Qts)
#              + s·(wa²·wb/Qts) + wa²·wb²
#
#  donde:
#    wa = 2π·Fs  (resonancia del altavoz en campo libre)
#    wb = 2π·Fb  (resonancia de Helmholtz del recinto)
#    Qb = factor de calidad del tubo (usualmente 7–15, default 10)
# ══════════════════════════════════════════════════════════════

def get_small_coefficients(p: dict, vb_liters: float, fb_hz: float, qb: float = 10.0):
    wa = 2 * PI * p["fs"]
    wb = 2 * PI * fb_hz
    qts = p["qts"]
    alpha = p["vas"] / (vb_liters * 1e-3)
    
    a3 = wb/qb + wa/qts
    a2 = wb**2 + wa**2 * (1 + alpha) + (wa * wb)/(qts * qb)
    a1 = (wa * wb**2)/qts + (wa**2 * wb)/qb
    a0 = wa**2 * wb**2
    return wa, wb, a3, a2, a1, a0

def tf_reflex(p: dict, vb_liters: float, fb_hz: float, qb: float = 10.0):
    """
    Función de transferencia 4º orden para caja bass-reflex.
    Devuelve scipy.signal.TransferFunction normalizada.
    """
    wa, wb, a3, a2, a1, a0 = get_small_coefficients(p, vb_liters, fb_hz, qb)

    num = [1.0, 0.0, 0.0, 0.0, 0.0]    # s⁴
    den = [1.0, a3, a2, a1, a0]

    return signal.TransferFunction(num, den)


# ══════════════════════════════════════════════════════════════
#  EXCURSIÓN DEL CONO X(f)  [mm]
#
#  |X(jw)| = |Eg| · |Bl| / (Re · Sd) · |Hx(jw)|
#
#  Hx es la FT de excursión:
#  Para reflex:   Hx = (wb² - w²) / D(jw)   (el puerto descarga la membrana)
#  Para sellada:  Hx = wb² / D(jw)           (simplificado → 1/wc²)
# ══════════════════════════════════════════════════════════════

def cone_excursion(p: dict, freqs: np.ndarray, vb_liters: float,
                   box_type: str, fb_hz: float = None,
                   eg_volts: float = 2.83) -> np.ndarray:
    """
    Excursión del cono (mm) basada en la transferencia de Small (1973).
    """
    w = 2 * PI * freqs
    s = 1j * w
    wa = 2 * PI * p["fs"]
    
    # Desplazamiento máximo en baja frecuencia (límite mecánico bajo 2.83V)
    # x_dc = (Eg * Bl * Cms) / Re
    x_dc = (eg_volts * p["bl"] * p["cms"]) / p["re"]

    if box_type == "reflex" and fb_hz:
        qb = 10.0
        wa_c, wb, a3, a2, a1, a0 = get_small_coefficients(p, vb_liters, fb_hz, qb)
        D = s**4 + a3*s**3 + a2*s**2 + a1*s + a0
        # Hx para reflex es el filtro de 2º orden sobre el cajón de 4º
        Hx = wa**2 * (s**2 + s*(wb/qb) + wb**2) / D
    else:
        # Sellada (2º orden)
        alpha = p["vas"] / (vb_liters * 1e-3)
        qtc = p["qts"] * np.sqrt(1 + alpha)
        wc = wa * np.sqrt(1 + alpha)
        D = s**2 + (wc/qtc)*s + wc**2
        Hx = wa**2 / D
        
    return np.abs(x_dc * Hx) * 1e3   # m → mm


# ══════════════════════════════════════════════════════════════
#  VELOCIDAD EN EL PUERTO Vp(f)  [m/s]
# ══════════════════════════════════════════════════════════════

def port_velocity(p: dict, freqs: np.ndarray, sp_cm2: float,
                  fb_hz: float, vb_liters: float,
                  eg_volts: float = 2.83) -> np.ndarray:
    """
    Velocidad del aire en el puerto en m/s derivando el volumen radiado neto de Small.
    Límite seguro: < 12 m/s. Turbulencia audible: > 17 m/s.
    """
    w = 2 * PI * freqs
    s = 1j * w
    x_dc = (eg_volts * p["bl"] * p["cms"]) / p["re"]
    qb = 10.0
    
    wa, wb, a3, a2, a1, a0 = get_small_coefficients(p, vb_liters, fb_hz, qb)
    D = s**4 + a3*s**3 + a2*s**2 + a1*s + a0
    
    sp_m2_val = sp_cm2 * 1e-4
    sd_m2_val = p["sd"]
    
    # Vp = (Sd/Sp) * x_dc * | wa^2 * wb^2 * w / D(jw) |
    Vp = (sd_m2_val / sp_m2_val) * x_dc * (wa**2 * wb**2 * w) / np.abs(D)
    
    return Vp


# ══════════════════════════════════════════════════════════════
#  IMPEDANCIA ELÉCTRICA Z(f)  [Ω]
# ══════════════════════════════════════════════════════════════

def impedance(p: dict, freqs: np.ndarray, box_type: str,
              vb_liters: float, fb_hz: float = None) -> np.ndarray:
    """
    Impedancia de entrada del altavoz montado en la caja.
    Pico en Fs para sellada; dos picos alrededor de Fb para reflex.
    """
    re   = p["re"]
    bl   = p["bl"]
    mms  = p["mms"]
    rms  = p["rms"]
    cms  = p["cms"]
    sd   = p["sd"]
    le   = p.get("le", 0.5e-3)     # inductancia de la bobina (H) — default 0.5 mH

    z = np.zeros(len(freqs), dtype=complex)

    for i, f in enumerate(freqs):
        w = 2 * PI * f
        s = 1j * w

        # Impedancia mecánica
        zmec = rms + s * mms + 1.0 / (s * cms)

        if box_type == "reflex" and fb_hz:
            wb  = 2 * PI * fb_hz
            vb  = vb_liters * 1e-3
            cab = vb / (RHO * C_AIR**2)
            # Puerto (Helmholtz) en paralelo con la compliance de la caja
            # Simplificado: efecto como admitancia en paralelo con Zmec
            z_helmholtz = 1j * w * RHO * C_AIR**2 / (vb * sd**2) / w**2
            zmec_total  = zmec + bl**2 / (sd**2) * (1j * w)
        else:
            zmec_total = zmec

        # Impedancia vista desde los bornes: Ze = Re + jωLe + Bl²/Zmec
        ze = re + s * le + bl**2 / zmec_total
        z[i] = ze

    return np.abs(z)


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL: simulate()
# ══════════════════════════════════════════════════════════════

def simulate(driver: dict, freqs: np.ndarray = None,
             eg_volts: float = 2.83) -> dict:
    """
    Simula la respuesta completa del altavoz en la caja configurada.

    Parámetros del driver (dict):
        Obligatorios: fs, vas, qts, sd
        Opcionales:   qes, qms, re, bl, mms, xmax, spl, le
        Caja:         box_type ("reflex"/"closed")
                      alignment ("QB3"/"SBB4"/"B4") — para reflex
                      qtc_target — para sellada
                      port_diam_cm, num_ports, k_factor
                      material_mm

    Retorna dict con todas las curvas y métricas.
    """
    if freqs is None:
        freqs = np.logspace(np.log10(10), np.log10(1000), 800)

    p       = ts_to_physical(driver)
    box     = driver.get("box_type", "reflex")
    spl_ref = float(driver.get("spl", 86.0))   # sensibilidad nominal

    result = {"freqs": freqs, "driver": driver, "phys": p}

    # ── REFLEX ──────────────────────────────────────────────
    if box == "reflex":
        align = driver.get("alignment", "QB3")
        alpha, h_fb, f3_ratio = _interp_align(align, p["qts"])
        vb_liters = round(float(p["vas"] * 1e3 / alpha), 1)     # litros
        fb        = round(float(h_fb * p["fs"]), 1)
        f3        = f3_ratio * p["fs"]

        # Puerto
        port_type = driver.get("port_type", "circular")
        port_diam = float(driver.get("port_diam_cm", 7.0))
        slot_w    = float(driver.get("slot_w_cm", 10.0))
        slot_h    = float(driver.get("slot_h_cm", 5.0))
        k         = float(driver.get("k_factor", 0.732))
        N         = int(driver.get("num_ports", 1))

        if port_type == "circular":
            sp    = np.pi * (port_diam / 2)**2      # cm²
            d_eq  = port_diam
        else:
            sp    = slot_w * slot_h
            d_eq  = 2 * np.sqrt(sp / np.pi)

        sp_total  = N * sp
        L_port    = (29974.86 * N * sp) / (fb**2 * vb_liters) - k * d_eq
        L_port    = max(L_port, 1.0)

        # Función de transferencia 4º orden
        tf = tf_reflex(p, vb_liters, fb)
        _, H = signal.freqs(tf.num, tf.den, worN=2*PI*freqs)

        # La función de transferencia ya tiende a 1.0 (0 dB) en la banda de paso
        H_mag    = np.abs(H)
        H_norm   = H_mag
        spl_curve = spl_ref + 20 * np.log10(H_norm + 1e-30)

        # Retardo de grupo (ms)
        gd_s  = -np.gradient(np.unwrap(np.angle(H)), 2*PI*freqs)
        gd_ms = gd_s * 1e3

        # Excursión del cono
        xcone = cone_excursion(p, freqs, vb_liters, "reflex", fb, eg_volts)

        # Velocidad en el puerto
        vport = port_velocity(p, freqs, sp_total, fb, vb_liters, eg_volts)

        # Impedancia
        zimp = impedance(p, freqs, "reflex", vb_liters, fb)

        result.update({
            "box_type":    "reflex",
            "alignment":   align,
            "vb_liters":   vb_liters,
            "fb":          fb,
            "f3":          f3,
            "spl":         spl_curve,
            "group_delay": gd_ms,
            "excursion":   xcone,
            "port_vel":    vport,
            "impedance":   zimp,
            "sp_cm2":      sp_total,
            "L_port_cm":   L_port,
            "port_diam":   port_diam,
            "N_ports":     N,
            "tf":          tf,
        })

    # ── SELLADA ─────────────────────────────────────────────
    else:
        qtc_t  = float(driver.get("qtc_target", 0.707))
        tf_obj, fc, qtc_real, vb_liters = tf_closed(p, vb_liters=None,
                                                      qtc_target=qtc_t)
        # F3 exacta
        wc    = 2*PI*fc
        disc  = (1/qtc_real**2 - 2)**2 + 4
        val   = (-(1/qtc_real**2 - 2) + np.sqrt(disc)) / 2
        f3    = np.sqrt(val) * fc

        _, H   = signal.freqs(tf_obj.num, tf_obj.den, worN=2*PI*freqs)
        H_mag  = np.abs(H)
        H_norm = H_mag
        spl_curve = spl_ref + 20 * np.log10(H_norm + 1e-30)

        gd_s  = -np.gradient(np.unwrap(np.angle(H)), 2*PI*freqs)
        gd_ms = gd_s * 1e3

        xcone = cone_excursion(p, freqs, vb_liters, "closed", eg_volts=eg_volts)
        zimp  = impedance(p, freqs, "closed", vb_liters)

        result.update({
            "box_type":    "closed",
            "qtc_target":  qtc_t,
            "qtc_real":    qtc_real,
            "fc":          fc,
            "vb_liters":   vb_liters,
            "f3":          f3,
            "spl":         spl_curve,
            "group_delay": gd_ms,
            "excursion":   xcone,
            "impedance":   zimp,
            "tf":          tf_obj,
        })

    # ── MÉTRICAS ESCALARES ──────────────────────────────────
    spl_arr = result["spl"]
    f_arr   = freqs

    # Sensibilidad media en banda 100–1000 Hz
    mask_band = (f_arr >= 100) & (f_arr <= 1000)
    sens_band = float(np.mean(spl_arr[mask_band]))

    # F3 real desde la curva
    target_db = sens_band - 3.0
    below = np.where(spl_arr < target_db)[0]
    f3_from_curve = float(f_arr[below[-1]]) if len(below) > 0 else result["f3"]

    # F6 y F10
    for label, delta in [("f6", 6), ("f10", 10)]:
        below_d = np.where(spl_arr < sens_band - delta)[0]
        result[label] = float(f_arr[below_d[-1]]) if len(below_d) > 0 else None

    # Xmax excedido
    xmax_mm   = p["xmax"] * 1e3
    xcone_arr = result["excursion"]
    exceed    = f_arr[xcone_arr > xmax_mm]
    result["xmax_exceeded_below"] = float(exceed.max()) if len(exceed) > 0 else None

    # Turbulencia en el puerto (reflex)
    if box == "reflex":
        vp        = result["port_vel"]
        turb_freq = f_arr[vp > 17]
        result["port_turbulence_freq"] = float(turb_freq.min()) if len(turb_freq) > 0 else None

    result["sens_band"]       = sens_band
    result["f3_from_curve"]   = f3_from_curve

    return result


# ══════════════════════════════════════════════════════════════
#  PLOT: plot_results()
# ══════════════════════════════════════════════════════════════

def plot_results(r: dict, output_path: str = "simulation.png",
                 dpi: int = 150) -> str:
    """
    Genera una figura de 4 subplots:
        1. SPL (dB) — respuesta en frecuencia
        2. Excursión del cono (mm)
        3. Velocidad en el puerto / Impedancia (Ω)
        4. Retardo de grupo (ms)
    """
    # ── Paleta ────────────────────────────────────────────
    BG      = "#0a0b0e"
    SURF    = "#12151c"
    SURF2   = "#1a1e28"
    BORDER  = "#252a38"
    GREEN   = "#00e5a0"
    BLUE    = "#0099ff"
    ORANGE  = "#ff6b35"
    YELLOW  = "#ffc94d"
    RED     = "#ff4d6d"
    MUTED   = "#5a6480"
    TEXT    = "#e8ecf5"

    freqs = r["freqs"]
    d     = r["driver"]
    box   = r["box_type"]
    model = d.get("model_name", "Altavoz")

    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    gs  = gridspec.GridSpec(4, 1, hspace=0.08, figure=fig,
                             top=0.91, bottom=0.07, left=0.08, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # ── Título ────────────────────────────────────────────
    box_label = f"Bass Reflex · {r.get('alignment','')}" if box=="reflex" else "Sellada (Closed)"
    fig.suptitle(
        f"{model}   ·   {box_label}   ·   Vb = {r['vb_liters']:.1f} L   ·   F3 = {r['f3_from_curve']:.1f} Hz",
        color=TEXT, fontsize=12, fontweight="bold", y=0.97
    )

    def style_ax(ax, ylabel, ylim=None, extra_grid=True):
        ax.set_facecolor(SURF)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        ax.set_xscale("log")
        ax.set_xlim(10, 1000)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.5, which="both", alpha=0.8)
        if ylim: ax.set_ylim(ylim)
        ax.set_xticks([10,20,30,50,70,100,150,200,300,500,700,1000])
        ax.set_xticklabels([])   # solo el último tendrá etiquetas

    # ── 1. SPL ────────────────────────────────────────────
    ax = axes[0]
    style_ax(ax, "SPL  (dB)", ylim=(r["sens_band"]-30, r["sens_band"]+6))

    ax.plot(freqs, r["spl"], color=GREEN, linewidth=2.0, zorder=5, label="SPL")
    ax.axhline(r["sens_band"],     color=MUTED, linewidth=0.7, linestyle="--")
    ax.axhline(r["sens_band"]-3,   color=BLUE,  linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(r["sens_band"]-10,  color=MUTED, linewidth=0.5, linestyle=":",  alpha=0.5)

    ax.axvline(r["f3_from_curve"], color=BLUE,   linewidth=1.2, linestyle="--", alpha=0.8)
    ax.text(r["f3_from_curve"]*1.05, r["sens_band"]-1,
            f"F3 = {r['f3_from_curve']:.1f} Hz", color=BLUE, fontsize=8)

    if box == "reflex":
        ax.axvline(r["fb"], color=ORANGE, linewidth=1.0, linestyle=":", alpha=0.7)
        ax.text(r["fb"]*1.05, r["sens_band"]-5,
                f"Fb = {r['fb']:.1f} Hz", color=ORANGE, fontsize=8)

    if r.get("f6"):
        ax.axvline(r["f6"], color=MUTED, linewidth=0.7, linestyle=":", alpha=0.5)
        ax.text(r["f6"]*1.05, r["sens_band"]-4,
                f"F6={r['f6']:.0f}Hz", color=MUTED, fontsize=7)

    ax.legend(loc="lower right", fontsize=8, facecolor=SURF2,
              edgecolor=BORDER, labelcolor=TEXT)

    # ── 2. EXCURSIÓN ──────────────────────────────────────
    ax = axes[1]
    xmax_mm = r["phys"]["xmax"] * 1e3
    style_ax(ax, "Excursión  (mm)", ylim=(0, max(xmax_mm*2.2, r["excursion"].max()*1.4)))

    ax.fill_between(freqs, r["excursion"], alpha=0.2, color=GREEN)
    ax.plot(freqs, r["excursion"], color=GREEN, linewidth=1.8, label="Excursión cono")
    ax.axhline(xmax_mm,       color=YELLOW, linewidth=1.2, linestyle="--", label=f"Xmax = {xmax_mm:.1f} mm")
    ax.axhline(xmax_mm * 1.5, color=RED,    linewidth=0.8, linestyle="--", alpha=0.6, label="1.5×Xmax (daño)")

    if r.get("xmax_exceeded_below"):
        ax.axvline(r["xmax_exceeded_below"], color=RED, linewidth=1.0, linestyle="--", alpha=0.7)
        ax.text(r["xmax_exceeded_below"]*1.05,
                xmax_mm*0.5,
                f"Xmax excedido\npor debajo de\n{r['xmax_exceeded_below']:.0f} Hz",
                color=RED, fontsize=7)

    ax.legend(loc="upper right", fontsize=8, facecolor=SURF2,
              edgecolor=BORDER, labelcolor=TEXT)

    # ── 3. VELOCIDAD PUERTO / IMPEDANCIA ──────────────────
    ax    = axes[2]
    ax2   = ax.twinx()
    style_ax(ax, "Vel. puerto  (m/s)")
    ax2.set_facecolor(SURF)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.spines["right"].set_edgecolor(BORDER)
    ax2.spines["left"].set_edgecolor(BORDER)
    ax2.set_ylabel("Impedancia  (Ω)", color=MUTED, fontsize=8)

    if box == "reflex":
        vp = r["port_vel"]
        ax.fill_between(freqs, vp, alpha=0.15, color=ORANGE)
        ax.plot(freqs, vp, color=ORANGE, linewidth=1.8, label="Vel. puerto")
        ax.axhline(12, color=YELLOW, linewidth=1.0, linestyle="--", alpha=0.7, label="12 m/s (límite)")
        ax.axhline(17, color=RED,    linewidth=1.0, linestyle="--", alpha=0.7, label="17 m/s (turbulencia)")
        ax.set_ylim(0, max(vp.max()*1.4, 20))
        if r.get("port_turbulence_freq"):
            ax.axvline(r["port_turbulence_freq"], color=RED, linewidth=1.0, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8, facecolor=SURF2,
                  edgecolor=BORDER, labelcolor=TEXT)
    else:
        ax.set_visible(False)

    ax2.plot(freqs, r["impedance"], color=BLUE, linewidth=1.5,
             linestyle="-", alpha=0.9, label="Impedancia Z(f)")
    ax2.set_ylim(0, min(r["impedance"].max()*1.4, 120))
    ax2.legend(loc="upper left", fontsize=8, facecolor=SURF2,
               edgecolor=BORDER, labelcolor=TEXT)

    # ── 4. RETARDO DE GRUPO ───────────────────────────────
    ax = axes[3]
    style_ax(ax, "Ret. grupo  (ms)")
    ax.set_xticklabels(["10","20","30","50","70","100","150","200","300","500","700","1k"])
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("Frecuencia  (Hz)", color=MUTED, fontsize=9)

    gd = np.clip(r["group_delay"], -50, 100)
    ax.fill_between(freqs, gd, alpha=0.15, color=BLUE)
    ax.plot(freqs, gd, color=BLUE, linewidth=1.8, label="Retardo de grupo")
    ax.axhline(0, color=MUTED, linewidth=0.5)

    if box == "reflex":
        ax.axvline(r["fb"], color=ORANGE, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_ylim(-15, min(gd.max()*1.3 + 5, 80))
    ax.legend(loc="upper right", fontsize=8, facecolor=SURF2,
              edgecolor=BORDER, labelcolor=TEXT)

    # ── Marca de agua ─────────────────────────────────────
    fig.text(0.97, 0.02, "SpeakerLab Pro v5.1 · Modelo Small (1973)",
             ha="right", va="bottom", color=MUTED, fontsize=7, alpha=0.6)

    plt.savefig(output_path, dpi=dpi, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"✅  Gráfica guardada: {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════
#  COMPARACIÓN DE ALINEACIONES
# ══════════════════════════════════════════════════════════════

def compare_alignments(driver_base: dict, output_path: str = "compare_alignments.png") -> str:
    """
    Superpone SPL de QB3, SBB4, B4 y Sellada (0.707) en una sola gráfica.
    Ideal para elegir la alineación óptima.
    """
    BG = "#0a0b0e"; SURF = "#12151c"; BORDER = "#252a38"; MUTED = "#5a6480"; TEXT = "#e8ecf5"
    colors_map = {"QB3":"#00e5a0", "SBB4":"#0099ff", "B4":"#ff6b35", "Closed":"#ffc94d"}
    freqs = np.logspace(np.log10(10), np.log10(1000), 600)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor(SURF)
    model = driver_base.get("model_name", "Altavoz")
    ax.set_title(f"{model} — Comparación de Alineaciones", color=TEXT, fontsize=12, fontweight="bold", pad=10)

    reference_spl = 86.0

    for align in ["QB3", "SBB4", "B4"]:
        d = {**driver_base, "box_type": "reflex", "alignment": align}
        r = simulate(d, freqs)
        offset = 0
        if align == "QB3": # initialize to first execution logically
            reference_spl = float(r["sens_band"])
        spl_plot = r["spl"] - r["sens_band"] + reference_spl
        ax.plot(freqs, spl_plot, color=colors_map[align], linewidth=2.0,
                label=f"{align}  F3={r['f3_from_curve']:.0f}Hz  Vb={r['vb_liters']:.0f}L")

    # Sellada
    d_closed = {**driver_base, "box_type": "closed", "qtc_target": 0.707}
    r_c = simulate(d_closed, freqs)
    spl_c = r_c["spl"] - r_c["sens_band"] + reference_spl
    ax.plot(freqs, spl_c, color=colors_map["Closed"], linewidth=2.0, linestyle="--",
            label=f"Sellada Qtc=0.707  F3={r_c['f3_from_curve']:.0f}Hz  Vb={r_c['vb_liters']:.0f}L")

    ax.axhline(reference_spl - 3, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.7, label="−3 dB")
    ax.set_xscale("log")
    ax.set_xlim(10, 1000)
    ax.set_ylim(reference_spl - 30, reference_spl + 6)
    ax.set_xlabel("Frecuencia (Hz)", color=MUTED, fontsize=9)
    ax.set_ylabel("SPL (dB)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xticks([10,20,30,50,100,200,300,500,1000])
    ax.set_xticklabels(["10","20","30","50","100","200","300","500","1k"])
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, linewidth=0.5, which="both", alpha=0.8)
    ax.legend(loc="lower right", fontsize=9, facecolor="#1a1e28",
              edgecolor=BORDER, labelcolor=TEXT)
    fig.text(0.97, 0.01, "SpeakerLab Pro v5.1", ha="right", color=MUTED, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"✅  Comparación guardada: {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════
#  DATOS DE EJEMPLO
# ══════════════════════════════════════════════════════════════

DAYTON_RSS315HO = {
    "model_name":   "Dayton Audio RSS315HO-4",
    "fs": 18.1, "vas": 213.6, "qts": 0.269, "qes": 0.284, "qms": 4.82,
    "xmax": 24, "sd": 855, "spl": 86.3, "re": 4.0,
    "box_type": "reflex", "alignment": "SBB4",
    "port_diam_cm": 10.0, "num_ports": 1, "k_factor": 0.732,
}

TANG_BAND_W6 = {
    "model_name":   "Tang Band W6-1139SI",
    "fs": 52, "vas": 8.3, "qts": 0.41, "qes": 0.46, "qms": 3.8,
    "xmax": 5.5, "sd": 133, "spl": 88, "re": 8.0,
    "box_type": "reflex", "alignment": "B4",
    "port_diam_cm": 5.0, "num_ports": 1, "k_factor": 0.732,
}

PEERLESS_XXLS = {
    "model_name":   "Peerless XXLS 830500",
    "fs": 19, "vas": 155, "qts": 0.26, "qes": 0.27, "qms": 6.2,
    "xmax": 17, "sd": 855, "spl": 87, "re": 8.0,
    "box_type": "reflex", "alignment": "SBB4",
    "port_diam_cm": 9.0, "num_ports": 1, "k_factor": 0.732,
}


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            driver = json.load(f)
        out_png = sys.argv[2] if len(sys.argv) > 2 else "simulation.png"
    else:
        driver  = DAYTON_RSS315HO
        out_png = "sim_dayton_rss315ho.png"

    print(f"\n🔊  Simulando: {driver.get('model_name','')}")
    r = simulate(driver)

    print(f"   Vb       = {r['vb_liters']:.1f} L")
    print(f"   F3       = {r['f3_from_curve']:.1f} Hz  (tablas: {r['f3']:.1f} Hz)")
    if r['box_type'] == 'reflex':
        print(f"   Fb       = {r['fb']:.1f} Hz")
        print(f"   L puerto = {r['L_port_cm']:.1f} cm")
        if r.get('port_turbulence_freq'):
            print(f"   ⚠  Turbulencia en el puerto por encima de {r['port_turbulence_freq']:.0f} Hz")
    if r.get('xmax_exceeded_below'):
        print(f"   ⚠  Xmax excedido por debajo de {r['xmax_exceeded_below']:.0f} Hz @ {driver.get('spl',86)} dB/1W")
    print(f"   F6       = {r.get('f6',0):.1f} Hz")
    print(f"   F10      = {r.get('f10',0):.1f} Hz  (si existe)\n")

    plot_results(r, out_png)

    # Comparación de alineaciones
    compare_alignments(DAYTON_RSS315HO, "compare_dayton_alignments.png")
    compare_alignments(PEERLESS_XXLS,   "compare_peerless_alignments.png")
