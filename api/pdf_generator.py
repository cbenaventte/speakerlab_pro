"""
SpeakerLab Pro — Generador de PDF de Planos de Corte (v6.0 Optimizado)
======================================================================
Genera un PDF profesional con:
  · Portada con resumen acústico
  · Planos de corte con dimensiones exactas (butt-joint corregido)
  · Diagrama individual de cada pieza (NUEVO)
  · Diagrama isométrico de la caja
  · Gráfica de respuesta en frecuencia (modelo corregido)
  · Lista de materiales y notas de ensamblaje
  · Diagnóstico con semáforo

Uso:
    python pdf_generator.py                    # datos de ejemplo
    python pdf_generator.py datos.json         # desde archivo JSON
    python pdf_generator.py datos.json out.pdf # ruta de salida personalizada

Cambios v6.0:
  - Corregido butt-joint (piezas no se superponen)
  - Modelo de frecuencia Butterworth 4° orden (reflex) y 2° orden (sellada)
  - Eliminada página en blanco fantasma
  - Validación de entrada robusta
  - Diagrama de piezas individuales
  - Fondo oscuro consistente
  - Cálculos vectorizados (numpy)
"""

import sys
import json
import math
import io
import os
import logging
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, KeepTogether, PageBreak
)
from reportlab.pdfgen import canvas as rl_canvas

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("SpeakerLabPro")

# ─────────────────────────────────────────────
#  Paleta de colores
# ─────────────────────────────────────────────
BG       = colors.HexColor("#0a0b0e")
SURFACE  = colors.HexColor("#12151c")
SURFACE2 = colors.HexColor("#1a1e28")
BORDER   = colors.HexColor("#252a38")
ACCENT   = colors.HexColor("#00e5a0")
ACCENT2  = colors.HexColor("#0099ff")
ACCENT3  = colors.HexColor("#ff6b35")
TEXT     = colors.HexColor("#e8ecf5")
MUTED    = colors.HexColor("#5a6480")
WHITE    = colors.white
BLACK    = colors.black

# ══════════════════════════════════════════════════════════════
#  0. VALIDACIÓN DE ENTRADA
# ══════════════════════════════════════════════════════════════
VALID_BOX_TYPES = {"reflex", "sealed"}
VALID_ALIGNMENTS = {"QB3", "SBB4", "B4"}
VALID_PORT_TYPES = {"circular", "slot"}

class ValidationError(ValueError):
    pass

def validate_input(d: dict) -> dict:
    clean = dict(d)
    required = ["fs", "vas", "qts"]
    for key in required:
        if key not in clean:
            raise ValidationError(f"Falta el campo obligatorio: '{key}'")
        val = clean[key]
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValidationError(f"'{key}' debe ser un número positivo, recibido: {val}")

    optional_positive = {
        "qes": None, "qms": None, "xmax": None, "sd": None,
        "spl": None, "inches": 10, "material_mm": 18,
        "port_diam_cm": 7.0, "slot_w_cm": 10.0, "slot_h_cm": 5.0,
        "k_factor": 0.732, "num_ports": 1, "qtc_target": 0.707,
    }
    for key, default in optional_positive.items():
        if key in clean:
            val = clean[key]
            if not isinstance(val, (int, float)):
                raise ValidationError(f"'{key}' debe ser numérico, recibido: {type(val).__name__}")
            if val <= 0:
                if default is not None:
                    logger.warning(f"'{key}' = {val} inválido, usando defecto: {default}")
                    clean[key] = default
                else:
                    clean[key] = None
        else:
            clean[key] = default

    clean.setdefault("box_type", "reflex")
    if clean["box_type"] not in VALID_BOX_TYPES:
        raise ValidationError(f"box_type debe ser {VALID_BOX_TYPES}, recibido: '{clean['box_type']}'")

    if clean["box_type"] == "reflex":
        clean.setdefault("alignment", "QB3")
        if clean["alignment"] not in VALID_ALIGNMENTS:
            raise ValidationError(f"alignment debe ser {VALID_ALIGNMENTS}, recibido: '{clean['alignment']}'")

    clean.setdefault("port_type", "circular")
    if clean["port_type"] not in VALID_PORT_TYPES:
        raise ValidationError(f"port_type debe ser {VALID_PORT_TYPES}, recibido: '{clean['port_type']}'")

    if clean["box_type"] == "sealed":
        qtc_t = clean.get("qtc_target", 0.707)
        qts = clean["qts"]
        if qtc_t <= qts:
            raise ValidationError(f"Para caja sellada, qtc_target ({qtc_t}) debe ser > qts ({qts}).")

    clean.setdefault("model_name", "Altavoz Sin Nombre")
    clean.setdefault("material_name", f"MDF {clean['material_mm']}mm")
    return clean

# ══════════════════════════════════════════════════════════════
#  1. TABLAS DE ALINEACIÓN Y CÁLCULOS ACÚSTICOS
# ══════════════════════════════════════════════════════════════
ALIGNMENT_TABLES = {
    "QB3": [
        (0.20, 8.80, 1.87, 1.55), (0.25, 6.13, 1.67, 1.45),
        (0.30, 4.42, 1.50, 1.36), (0.35, 3.22, 1.35, 1.27),
        (0.40, 2.35, 1.22, 1.19), (0.45, 1.70, 1.10, 1.10),
        (0.50, 1.20, 1.00, 1.00),
    ],
    "SBB4": [
        (0.20, 16.32, 2.09, 1.00), (0.25, 10.04, 1.85, 1.00),
        (0.30, 6.57, 1.64, 1.00),  (0.35, 4.50, 1.46, 1.00),
        (0.40, 3.18, 1.30, 1.00),  (0.45, 2.29, 1.16, 1.00),
        (0.50, 1.67, 1.04, 1.00),
    ],
    "B4": [
        (0.20, 6.97, 1.56, 1.56), (0.25, 4.47, 1.41, 1.41),
        (0.30, 3.05, 1.28, 1.28), (0.35, 2.17, 1.16, 1.16),
        (0.40, 1.58, 1.07, 1.07), (0.45, 1.18, 0.98, 0.98),
        (0.50, 0.89, 0.91, 0.91),
    ],
}

def interpolate_alignment(table_name: str, qts: float) -> Tuple[float, float, float]:
    qts_clamped = max(0.20, min(0.50, qts))
    rows = ALIGNMENT_TABLES[table_name]
    for i in range(len(rows) - 1):
        q0, a0, h0, f0 = rows[i]
        q1, a1, h1, f1 = rows[i + 1]
        if q0 <= qts_clamped <= q1:
            t = (qts_clamped - q0) / (q1 - q0)
            return (a0 + t * (a1 - a0), h0 + t * (h1 - h0), f0 + t * (f1 - f0))
    return rows[-1][1], rows[-1][2], rows[-1][3]

def calc_acoustics(d: dict) -> dict:
    fs    = d["fs"]
    vas   = d["vas"]
    qts   = d["qts"]
    qes   = d.get("qes")
    xmax  = d.get("xmax")
    sd    = d.get("sd")
    inch  = d.get("inches", 10)
    T_mm  = d.get("material_mm", 18)
    T     = T_mm / 10.0
    box   = d.get("box_type", "reflex")
    align = d.get("alignment", "QB3")
    qtc_t = d.get("qtc_target", 0.707)
    port_type = d.get("port_type", "circular")
    port_diam = d.get("port_diam_cm", 7.0)
    slot_w    = d.get("slot_w_cm", 10.0)
    slot_h    = d.get("slot_h_cm", 5.0)
    k         = d.get("k_factor", 0.732)
    N         = d.get("num_ports", 1)

    r = {"fs": fs, "vas": vas, "qts": qts, "T": T, "T_mm": T_mm, "box_type": box}
    Vdriver = 0.0035 * inch ** 2.8
    Vd = sd * (xmax / 10.0) if (sd and xmax) else None
    EBP = fs / qes if qes else None

    if box == "reflex":
        alpha, h, f3_ratio = interpolate_alignment(align, qts)
        Vb = vas / alpha
        Fb = h * fs
        F3 = f3_ratio * fs

        if port_type == "circular":
            Sp   = math.pi * (port_diam / 2.0) ** 2
            d_eq = port_diam
        else:
            Sp   = slot_w * slot_h
            d_eq = 2.0 * math.sqrt(Sp / math.pi)

        SpTotal = N * Sp
        # Fórmula de Keele: L = c²·N·Sp / (4π²·Fb²·Vb) - k·d_eq
        # Con c=34400 cm/s → c²/(4π²) ≈ 29974.86
        # Vb en litros, Sp en cm², L en cm
        L = (29974.86 * N * Sp) / (Fb ** 2 * Vb) - k * d_eq

        if L < 0:
            logger.warning(f"Longitud de puerto negativa ({L:.1f} cm).")
            L = max(L, 1.0)

        # Velocidad del aire en el puerto (m/s)
        # v = Fb × 2π × Xmax_peak × Sd / SpTotal
        # Xmax_peak = xmax(mm) / 1000 × 0.4 (factor de pico)
        if sd and xmax and SpTotal > 0:
            xmax_m = xmax / 1000.0          # mm → m
            sd_cm2 = sd                      # ya en cm²
            portVel = (Fb * 2 * math.pi * (xmax_m * 0.4) * sd_cm2) / SpTotal
        else:
            portVel = None

        Vport = N * Sp * L / 1000.0
        Vb_bruto = Vb + Vport + Vdriver + 0.05 * Vb

        if Vd:
            SPLmax = 112.2 + 20 * math.log10((Vd / 1e6) * Fb ** 2)
        else:
            SPLmax = None

        Fpipe = 34400.0 / (2.0 * L) if L > 0 else None

        r.update({
            "alignment": align, "Vb": Vb, "Fb": Fb, "F3": F3,
            "Sp": Sp, "SpTotal": SpTotal, "d_eq": d_eq, "L": L,
            "portVel": portVel, "Vport": Vport, "Vb_bruto": Vb_bruto,
            "Vd": Vd, "Vdriver": Vdriver, "SPLmax": SPLmax,
            "EBP": EBP, "N": N, "Fpipe": Fpipe,
            "port_type": port_type, "port_diam": port_diam,
            "slot_w": slot_w, "slot_h": slot_h,
        })
    else:
        ratio_sq = (qtc_t / qts) ** 2 - 1
        Vb = vas / ratio_sq
        Qtc_real = qts * math.sqrt(vas / Vb + 1)
        Fc = fs * math.sqrt(vas / Vb + 1)
        a = 1.0 / (2.0 * Qtc_real ** 2) - 1.0
        F3 = Fc * math.sqrt(a + math.sqrt(a ** 2 + 1.0))
        Vb_bruto = Vb + Vdriver + 0.05 * Vb

        r.update({
            "qtc_target": qtc_t, "Qtc_real": Qtc_real,
            "Fc": Fc, "Vb": Vb, "F3": F3,
            "Vb_bruto": Vb_bruto, "Vdriver": Vdriver, "Vd": Vd, "EBP": EBP,
        })

    # Dimensiones (proporciones áureas φ y √φ)
    RATIO_H = 1.618
    RATIO_W = 1.272
    Vb_bruto_cm3 = r["Vb_bruto"] * 1000.0
    D_int = (Vb_bruto_cm3 / (RATIO_H * RATIO_W)) ** (1.0 / 3.0)
    W_int = RATIO_W * D_int
    H_int = RATIO_H * D_int
    D_ext = D_int + 2 * T
    W_ext = W_int + 2 * T
    H_ext = H_int + 2 * T
    Fbsc = 11500.0 / W_ext

    pieces = _calc_pieces(W_ext, H_ext, D_ext, T, r)
    total_area_cm2 = sum(p["w_cm"] * p["h_cm"] * p["qty"] for p in pieces)

    r.update({
        "D_int": D_int, "W_int": W_int, "H_int": H_int,
        "D_ext": D_ext, "W_ext": W_ext, "H_ext": H_ext,
        "Fbsc": Fbsc, "total_area_m2": total_area_cm2 / 10000.0, "pieces": pieces,
    })
    return r

def _calc_pieces(W_ext, H_ext, D_ext, T, r) -> List[dict]:
    return [
        {"name": "Frontal (Baffle)", "qty": 1, "w_cm": W_ext, "h_cm": H_ext,
         "note": "Orificio altavoz" + (" + puerto" if r["box_type"] == "reflex" else "")},
        {"name": "Trasera", "qty": 1, "w_cm": W_ext, "h_cm": H_ext,
         "note": "Orificio terminal de bornes"},
        {"name": "Lateral (×2)", "qty": 2, "w_cm": D_ext - 2 * T, "h_cm": H_ext,
         "note": "Encajan entre frontal y trasera"},
        {"name": "Tapa (superior)", "qty": 1, "w_cm": W_ext - 2 * T, "h_cm": D_ext - 2 * T,
         "note": "Encaja entre las 4 paredes verticales"},
        {"name": "Base (inferior)", "qty": 1, "w_cm": W_ext - 2 * T, "h_cm": D_ext - 2 * T,
         "note": "Encaja entre las 4 paredes verticales"},
    ]

# ══════════════════════════════════════════════════════════════
#  2. GRÁFICA DE RESPUESTA EN FRECUENCIA (vectorizada)
# ══════════════════════════════════════════════════════════════
def _calc_response_reflex(freqs, F3, Fb):
    ratio = F3 / freqs
    db = -10.0 * np.log10(1.0 + ratio ** 8)
    below_fb = freqs < Fb
    if np.any(below_fb):
        db[below_fb] -= 6.0 * (Fb / freqs[below_fb] - 1.0) ** 1.5
    return np.clip(db, -40, 10)

def _calc_response_sealed(freqs, Fc, Qtc):
    fn = freqs / Fc
    fn2 = fn ** 2
    denom = (1.0 - fn2) ** 2 + fn2 / (Qtc ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_sq = np.where(denom > 1e-20, fn ** 4 / denom, 1e-20)
    db = 10.0 * np.log10(np.maximum(h_sq, 1e-20))
    return np.clip(db, -40, 10)

def make_freq_chart(r, width_px=900, height_px=320):
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=120)
    fig.patch.set_facecolor("#12151c")
    ax.set_facecolor("#12151c")
    freqs = np.geomspace(10, 800, 1500)

    if r["box_type"] == "reflex":
        db_arr = _calc_response_reflex(freqs, r["F3"], r["Fb"])
    else:
        db_arr = _calc_response_sealed(freqs, r["Fc"], r["Qtc_real"])

    ax.fill_between(freqs, db_arr, -40, color="#00e5a0", alpha=0.10)
    ax.plot(freqs, db_arr, color="#00e5a0", linewidth=2.2, zorder=5)
    ax.axvline(r["F3"], color="#0099ff", linewidth=1.3, linestyle="--", alpha=0.85)
    ax.text(r["F3"] * 1.06, 3, f"F₃ = {r['F3']:.0f} Hz", color="#0099ff", fontsize=8, fontweight="bold")

    if r["box_type"] == "reflex":
        ax.axvline(r["Fb"], color="#ff6b35", linewidth=1.0, linestyle=":", alpha=0.7)
        ax.text(r["Fb"] * 1.06, -7, f"Fb = {r['Fb']:.0f} Hz", color="#ff6b35", fontsize=7)

    ax.axhline(-3, color="#5a6480", linewidth=0.7, linestyle="--")
    ax.text(600, -2.0, "−3 dB", color="#5a6480", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(15, 700)
    ax.set_ylim(-35, 8)
    ax.set_xlabel("Frecuencia (Hz)", color="#5a6480", fontsize=9)
    ax.set_ylabel("Nivel (dB)", color="#5a6480", fontsize=9)
    ax.tick_params(colors="#5a6480", labelsize=8)
    ax.set_xticks([20, 30, 50, 70, 100, 150, 200, 300, 500])
    ax.set_xticklabels(["20", "30", "50", "70", "100", "150", "200", "300", "500"])
    for spine in ax.spines.values():
        spine.set_edgecolor("#252a38")
    ax.grid(True, color="#252a38", linewidth=0.4, which="both", alpha=0.7)

    title = "Respuesta en Frecuencia Simulada"
    if r["box_type"] == "reflex":
        title += f"  —  {r.get('alignment', '')} Reflex"
    else:
        title += f"  —  Sellada (Qtc={r['Qtc_real']:.3f})"
    ax.set_title(title, color="#e8ecf5", fontsize=10, pad=8, fontweight="bold")

    buf = io.BytesIO()
    plt.tight_layout(pad=0.5)
    plt.savefig(buf, format="png", dpi=120, facecolor="#12151c")
    plt.close(fig)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════
#  3. DIAGRAMA ISOMÉTRICO DE LA CAJA
# ══════════════════════════════════════════════════════════════
def make_box_diagram(r, width_px=540, height_px=360):
    W, H, D = r["W_ext"], r["H_ext"], r["D_ext"]
    fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=120)
    fig.patch.set_facecolor("#12151c")
    ax.set_facecolor("#12151c")
    ax.set_aspect("equal"); ax.axis("off")

    sx, sy = 0.5, 0.3
    scale = min(width_px, height_px * 1.3) / max(W, H, D) / 130
    W_s, H_s, D_s = W * scale, H * scale, D * scale
    ox, oy = width_px / 100 * 0.30, height_px / 100 * 0.15
    dx, dy = D_s * sx, D_s * sy

    front = plt.Polygon([[ox,oy],[ox+W_s,oy],[ox+W_s,oy+H_s],[ox,oy+H_s]],
                         closed=True, facecolor="#1a1e28", edgecolor="#00e5a0", linewidth=1.3)
    ax.add_patch(front)
    top = plt.Polygon([[ox,oy+H_s],[ox+W_s,oy+H_s],[ox+W_s+dx,oy+H_s+dy],[ox+dx,oy+H_s+dy]],
                       closed=True, facecolor="#252a38", edgecolor="#00e5a0", linewidth=1.3)
    ax.add_patch(top)
    side = plt.Polygon([[ox+W_s,oy],[ox+W_s+dx,oy+dy],[ox+W_s+dx,oy+H_s+dy],[ox+W_s,oy+H_s]],
                        closed=True, facecolor="#12151c", edgecolor="#00e5a0", linewidth=1.3)
    ax.add_patch(side)

    spk_r = min(W_s, H_s) * 0.26
    spk_cx, spk_cy = ox + W_s * 0.5, oy + H_s * 0.55
    for r_mult, lw, ls, c in [(1.0,1.5,"--","#5a6480"),(0.7,0.8,"-","#3a4050"),(0.25,1.2,"-","#00e5a0")]:
        circle = plt.Circle((spk_cx, spk_cy), spk_r * r_mult,
                             fill=(r_mult == 0.25),
                             facecolor="#00e5a0" if r_mult == 0.25 else "none",
                             edgecolor=c, linewidth=lw, linestyle=ls)
        ax.add_patch(circle)

    if r["box_type"] == "reflex":
        port_r = min(r.get("port_diam", 7) * scale * 0.5, W_s * 0.10)
        port_cx, port_cy = ox + W_s * 0.5, oy + H_s * 0.18
        ax.add_patch(plt.Circle((port_cx, port_cy), port_r, fill=False, edgecolor="#ff6b35", linewidth=1.5))
        ax.text(port_cx, port_cy - port_r * 2.0, "Puerto", color="#ff6b35", fontsize=7, ha="center", fontweight="bold")

    # Cotas
    y_cota = oy - H_s * 0.13
    ax.annotate("", xy=(ox+W_s, y_cota), xytext=(ox, y_cota),
                arrowprops=dict(arrowstyle="<->", color="#0099ff", lw=1.0))
    ax.text(ox+W_s/2, y_cota - H_s*0.06, f"W = {W:.1f} cm", color="#0099ff", fontsize=8, ha="center", fontweight="bold")

    x_cota = ox - W_s * 0.14
    ax.annotate("", xy=(x_cota, oy+H_s), xytext=(x_cota, oy),
                arrowprops=dict(arrowstyle="<->", color="#0099ff", lw=1.0))
    ax.text(x_cota - W_s*0.05, oy+H_s/2, f"H = {H:.1f} cm", color="#0099ff", fontsize=8, ha="right", va="center", fontweight="bold")

    ax.annotate("", xy=(ox+W_s+dx, oy+dy), xytext=(ox+W_s, oy),
                arrowprops=dict(arrowstyle="<->", color="#0099ff", lw=1.0))
    ax.text(ox+W_s+dx/2+W_s*0.06, oy+dy/2, f"D = {D:.1f} cm", color="#0099ff", fontsize=8, fontweight="bold")

    ax.set_xlim(-0.5, width_px/100+0.3)
    ax.set_ylim(-0.6, height_px/100+0.1)
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", dpi=120, facecolor="#12151c", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════
#  3b. DIAGRAMA DE PIEZAS INDIVIDUALES
# ══════════════════════════════════════════════════════════════
def make_pieces_diagram(r_data, width_px=900, height_px=500):
    pieces = r_data["pieces"]
    fig, axes = plt.subplots(2, 3, figsize=(width_px / 100, height_px / 100), dpi=120)
    fig.patch.set_facecolor("#12151c")
    fig.suptitle("PIEZAS DE CORTE — Vista Individual", color="#e8ecf5", fontsize=11, fontweight="bold", y=0.97)

    expanded = []
    for p in pieces:
        if p["qty"] > 1:
            for i in range(p["qty"]):
                label = p["name"].replace("(×2)", "").strip()
                expanded.append({**p, "name": f"{label} {'Izq.' if i == 0 else 'Der.'}", "qty": 1})
        else:
            expanded.append(p)
    while len(expanded) < 6:
        expanded.append(None)

    for idx, ax in enumerate(axes.flat):
        ax.set_facecolor("#12151c"); ax.set_aspect("equal"); ax.axis("off")
        if idx >= len(expanded) or expanded[idx] is None:
            continue
        piece = expanded[idx]
        w_cm, h_cm = piece["w_cm"], piece["h_cm"]
        max_d = max(w_cm, h_cm)
        s = 2.5 / max_d if max_d > 0 else 1
        ws, hs = w_cm * s, h_cm * s

        ax.add_patch(plt.Rectangle((-ws/2, -hs/2), ws, hs, facecolor="#1a1e28", edgecolor="#00e5a0", linewidth=1.5))
        ax.text(0, hs/2+0.35, piece["name"], color="#00e5a0", fontsize=7, ha="center", fontweight="bold")
        ax.text(0, -hs/2-0.25, f"{w_cm:.1f} cm ({w_cm*10:.0f} mm)", color="#0099ff", fontsize=6, ha="center")
        ax.text(ws/2+0.15, 0, f"{h_cm:.1f} cm\n({h_cm*10:.0f} mm)", color="#0099ff", fontsize=6, ha="left", va="center")
        ax.annotate("", xy=(ws/2, -hs/2-0.12), xytext=(-ws/2, -hs/2-0.12), arrowprops=dict(arrowstyle="<->", color="#0099ff", lw=0.7))
        ax.annotate("", xy=(ws/2+0.08, hs/2), xytext=(ws/2+0.08, -hs/2), arrowprops=dict(arrowstyle="<->", color="#0099ff", lw=0.7))
        ax.text(0, 0, f"T={r_data['T_mm']}mm", color="#5a6480", fontsize=6, ha="center", va="center")
        ax.set_xlim(-2, 2.8); ax.set_ylim(-2, 2)

    buf = io.BytesIO()
    plt.tight_layout(pad=0.3, rect=[0, 0, 1, 0.94])
    plt.savefig(buf, format="png", dpi=120, facecolor="#12151c")
    plt.close(fig)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════
#  4. ESTILOS REPORTLAB
# ══════════════════════════════════════════════════════════════
def get_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("sl_title", parent=base["Normal"], fontSize=26,
                                fontName="Helvetica-Bold", textColor=TEXT, alignment=TA_LEFT, spaceAfter=4),
        "subtitle": ParagraphStyle("sl_subtitle", parent=base["Normal"], fontSize=12,
                                   fontName="Helvetica", textColor=MUTED, alignment=TA_LEFT, spaceAfter=2),
        "section": ParagraphStyle("sl_section", parent=base["Normal"], fontSize=11,
                                  fontName="Helvetica-Bold", textColor=ACCENT, spaceBefore=14, spaceAfter=8, borderPadding=4),
        "label": ParagraphStyle("sl_label", parent=base["Normal"], fontSize=8, fontName="Helvetica", textColor=MUTED),
        "value": ParagraphStyle("sl_value", parent=base["Normal"], fontSize=14, fontName="Helvetica-Bold", textColor=TEXT),
        "body": ParagraphStyle("sl_body", parent=base["Normal"], fontSize=9, fontName="Helvetica", textColor=TEXT, leading=14),
        "note": ParagraphStyle("sl_note", parent=base["Normal"], fontSize=8, fontName="Helvetica", textColor=MUTED, leading=12),
        "mono": ParagraphStyle("sl_mono", parent=base["Normal"], fontSize=9, fontName="Courier", textColor=ACCENT2),
        "footer": ParagraphStyle("sl_footer", parent=base["Normal"], fontSize=7, fontName="Helvetica", textColor=MUTED, alignment=TA_CENTER),
    }

# ══════════════════════════════════════════════════════════════
#  5. CANVAS PERSONALIZADO (header/footer)
# ══════════════════════════════════════════════════════════════
class SpeakerLabCanvas(rl_canvas.Canvas):
    def __init__(self, *args, speaker_name: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._speaker_name = speaker_name
        self._page_num = 0

    def showPage(self):
        self._page_num += 1
        self._draw_header()
        self._draw_footer()
        super().showPage()

    def _draw_header(self):
        W, H = A4
        self.setFillColor(SURFACE)
        self.rect(0, H - 24 * mm, W, 24 * mm, fill=1, stroke=0)
        self.setFillColor(ACCENT)
        self.setFont("Helvetica-Bold", 13)
        self.drawString(15 * mm, H - 14 * mm, "SpeakerLab Pro")
        self.setFillColor(TEXT)
        self.setFont("Helvetica", 10)
        self.drawString(63 * mm, H - 14 * mm, f"— Plano de Diseño: {self._speaker_name}")
        self.setStrokeColor(ACCENT)
        self.setLineWidth(1.5)
        self.line(0, H - 24 * mm, W, H - 24 * mm)

    def _draw_footer(self):
        W, _ = A4
        self.setFillColor(SURFACE)
        self.rect(0, 0, W, 12 * mm, fill=1, stroke=0)
        self.setStrokeColor(BORDER)
        self.setLineWidth(0.5)
        self.line(0, 12 * mm, W, 12 * mm)
        self.setFillColor(MUTED)
        self.setFont("Helvetica", 7)
        self.drawString(15 * mm, 4 * mm, "SpeakerLab Pro v6.0 · Thiele (1971) / Small (1973) / Keele (1975)")
        self.drawRightString(W - 15 * mm, 4 * mm, f"Página {self._page_num}")

# ══════════════════════════════════════════════════════════════
#  6. UTILIDADES
# ══════════════════════════════════════════════════════════════
def fmt(value, decimals: int = 1, unit: str = "") -> str:
    if value is None:
        return "—"
    formatted = f"{value:.{decimals}f}"
    return f"{formatted} {unit}" if unit else formatted

# ══════════════════════════════════════════════════════════════
#  7. SECCIONES DEL DOCUMENTO
# ══════════════════════════════════════════════════════════════
def section_portada(r, d, S):
    model = d.get("model_name", "Altavoz")
    box_label = "Reflex (Bass Reflex)" if r["box_type"] == "reflex" else "Sellada (Closed Box)"
    if r["box_type"] == "reflex":
        box_label += f" · Alineación {r.get('alignment', '')}"

    elems = [
        Spacer(1, 8 * mm),
        Paragraph(model, S["title"]),
        Paragraph(f"Tipo de caja: {box_label}", S["subtitle"]),
        Spacer(1, 4 * mm),
        HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=6 * mm),
    ]

    ts_left = [
        [Paragraph("PARÁMETROS THIELE/SMALL", S["label"]), ""],
        ["Fs",   fmt(d.get("fs"), 1, "Hz")],
        ["Vas",  fmt(d.get("vas"), 1, "L")],
        ["Qts",  fmt(d.get("qts"), 3)],
        ["Qes",  fmt(d.get("qes"), 3)],
        ["Xmax", fmt(d.get("xmax"), 1, "mm")],
        ["Sd",   fmt(d.get("sd"), 0, "cm²")],
    ]

    fb_qtc_label = "Fb (sintonía)" if r["box_type"] == "reflex" else "Qtc real"
    fb_qtc_value = fmt(r.get("Fb"), 1, "Hz") if r["box_type"] == "reflex" else fmt(r.get("Qtc_real"), 3)

    ts_right = [
        [Paragraph("RESULTADOS ACÚSTICOS", S["label"]), ""],
        ["Vb neto",    fmt(r["Vb"], 1, "L")],
        ["F3 (−3 dB)", fmt(r["F3"], 1, "Hz")],
        [fb_qtc_label, fb_qtc_value],
        ["Vb bruto",   fmt(r["Vb_bruto"], 1, "L")],
        ["SPL máx.",   fmt(r.get("SPLmax"), 1, "dB")],
        ["EBP",        fmt(r.get("EBP"), 0)],
    ]

    def make_half_table(data, val_color):
        tbl = Table(data, colWidths=[28*mm, 30*mm], rowHeights=[8*mm]+[7*mm]*(len(data)-1))
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0),  SURFACE2),
            ("BACKGROUND",     (0,1),(-1,-1), SURFACE),
            ("TEXTCOLOR",      (0,0),(-1,0),  MUTED),
            ("TEXTCOLOR",      (0,1),(0,-1),  MUTED),
            ("TEXTCOLOR",      (1,1),(1,-1),  val_color),
            ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,0),  8),
            ("FONTNAME",       (1,1),(1,-1),  "Courier-Bold"),
            ("FONTSIZE",       (1,1),(1,-1),  10),
            ("FONTSIZE",       (0,1),(0,-1),  9),
            ("ALIGN",          (1,0),(1,-1),  "RIGHT"),
            ("GRID",           (0,0),(-1,-1), 0.4, BORDER),
            ("ROWBACKGROUNDS", (0,1),(-1,-1), [SURFACE, SURFACE2]),
            ("SPAN",           (0,0),(-1,0)),
            ("TOPPADDING",     (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 4),
            ("LEFTPADDING",    (0,0),(-1,-1), 6),
        ]))
        return tbl

    ts_tbl = Table([[make_half_table(ts_left, ACCENT2), Spacer(6*mm,1), make_half_table(ts_right, ACCENT)]],
                   colWidths=[58*mm, 6*mm, 58*mm])
    elems.append(ts_tbl)
    elems.append(Spacer(1, 5 * mm))
    return elems


def section_freq_chart(r, S):
    elems = [Paragraph("RESPUESTA EN FRECUENCIA SIMULADA", S["section"])]
    img_buf = make_freq_chart(r, 860, 300)
    elems.append(Image(img_buf, width=175*mm, height=61*mm))
    elems.append(Paragraph(
        "Curva simplificada basada en alineación Thiele/Small. "
        "Para simulación precisa (FRD), exporta los parámetros a VituixCAD, WinISD o REW.", S["note"]))
    elems.append(Spacer(1, 4*mm))
    return elems


def section_box_diagram(r, S):
    elems = [Paragraph("DIAGRAMA DE LA CAJA (EXTERIOR)", S["section"])]
    diag_buf = make_box_diagram(r, 500, 340)
    diag_img = Image(diag_buf, width=102*mm, height=68*mm)

    dim_data = [
        [Paragraph("DIMENSIONES", S["label"]), "", ""],
        ["Ancho ext. (W)",  f"{r['W_ext']:.1f} cm", f"{r['W_ext']*10:.0f} mm"],
        ["Alto ext. (H)",   f"{r['H_ext']:.1f} cm", f"{r['H_ext']*10:.0f} mm"],
        ["Prof. ext. (D)",  f"{r['D_ext']:.1f} cm", f"{r['D_ext']*10:.0f} mm"],
        ["Ancho int.",      f"{r['W_int']:.1f} cm", f"{r['W_int']*10:.0f} mm"],
        ["Alto int.",       f"{r['H_int']:.1f} cm", f"{r['H_int']*10:.0f} mm"],
        ["Prof. int.",      f"{r['D_int']:.1f} cm", f"{r['D_int']*10:.0f} mm"],
        ["Grosor (T)",      f"{r['T_mm']} mm",       ""],
    ]
    dim_tbl = Table(dim_data, colWidths=[42*mm, 22*mm, 22*mm], rowHeights=[7*mm]+[6.5*mm]*7)
    dim_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0),  SURFACE2),
        ("BACKGROUND",  (0,1),(-1,-1), SURFACE),
        ("TEXTCOLOR",   (0,0),(-1,0),  MUTED),
        ("TEXTCOLOR",   (1,1),(1,-1),  ACCENT),
        ("TEXTCOLOR",   (2,1),(2,-1),  MUTED),
        ("TEXTCOLOR",   (0,1),(0,-1),  TEXT),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,0),  8),
        ("FONTNAME",    (1,1),(1,-1),  "Courier-Bold"),
        ("FONTSIZE",    (1,1),(-1,-1), 10),
        ("FONTSIZE",    (0,1),(0,-1),  9),
        ("GRID",        (0,0),(-1,-1), 0.4, BORDER),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[SURFACE,SURFACE2]),
        ("SPAN",        (0,0),(-1,0)),
        ("TOPPADDING",  (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING", (0,0),(-1,-1), 6),
    ]))

    combo = Table([[diag_img, Spacer(5*mm,1), dim_tbl]], colWidths=[105*mm, 5*mm, 86*mm])
    combo.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    elems.append(combo)
    elems.append(Spacer(1, 4*mm))
    return elems


def section_pieces_diagram(r, S):
    elems = [Paragraph("PIEZAS DE CORTE — VISTA INDIVIDUAL", S["section"])]
    img_buf = make_pieces_diagram(r, 860, 480)
    elems.append(Image(img_buf, width=175*mm, height=98*mm))
    elems.append(Spacer(1, 4*mm))
    return elems


def section_cut_sheet(r, d, S):
    elems = [Paragraph("PLANOS DE CORTE — MÉTODO BUTT-JOINT", S["section"])]
    mat = d.get("material_name", f"MDF {r['T_mm']}mm")
    elems.append(Paragraph(
        f"Material: <b>{mat}</b>  ·  Grosor T = {r['T_mm']} mm  ·  "
        f"Proporciones áureas H:W:D = 1.618:1.272:1.000", S["body"]))
    elems.append(Spacer(1, 3*mm))

    pieces = r["pieces"]
    header = ["PIEZA", "CANT.", "ANCHO (cm)", "ALTO (cm)", "ANCHO (mm)", "ALTO (mm)", "NOTAS"]
    rows = [header]
    for p in pieces:
        rows.append([p["name"], str(p["qty"]),
                     f"{p['w_cm']:.1f}", f"{p['h_cm']:.1f}",
                     f"{p['w_cm']*10:.0f}", f"{p['h_cm']*10:.0f}",
                     p["note"]])

    col_w = [32*mm, 12*mm, 20*mm, 20*mm, 20*mm, 20*mm, 43*mm]
    cut_tbl = Table(rows, colWidths=col_w, rowHeights=[8*mm]+[7*mm]*len(pieces))
    cut_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0),  SURFACE2),
        ("BACKGROUND",  (0,1),(-1,-1), SURFACE),
        ("TEXTCOLOR",   (0,0),(-1,0),  MUTED),
        ("TEXTCOLOR",   (0,1),(0,-1),  ACCENT),
        ("TEXTCOLOR",   (2,1),(5,-1),  TEXT),
        ("TEXTCOLOR",   (6,1),(6,-1),  MUTED),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,0),  8),
        ("FONTNAME",    (0,1),(0,-1),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,1),(-1,-1), 9),
        ("FONTNAME",    (2,1),(5,-1),  "Courier-Bold"),
        ("FONTSIZE",    (2,1),(5,-1),  10),
        ("GRID",        (0,0),(-1,-1), 0.4, BORDER),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[SURFACE,SURFACE2]),
        ("ALIGN",       (1,0),(-2,-1), "CENTER"),
        ("TOPPADDING",  (0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING", (0,0),(-1,-1), 5),
    ]))
    elems.append(cut_tbl)

    elems.append(Spacer(1, 4*mm))
    total_area_with_margin = r["total_area_m2"] * 1.15
    summary_data = [
        ["Tablero necesario (sin margen)", f"{r['total_area_m2']:.3f} m²"],
        ["Tablero recomendado (+15% merma)", f"{total_area_with_margin:.3f} m²"],
        ["Vb bruto (para dimensiones)",     f"{r['Vb_bruto']:.1f} L"],
        ["Vb neto (acústico)",              f"{r['Vb']:.1f} L"],
    ]
    sum_tbl = Table(summary_data, colWidths=[80*mm, 40*mm])
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), SURFACE2),
        ("TEXTCOLOR",   (0,0),(0,-1),  MUTED),
        ("TEXTCOLOR",   (1,0),(1,-1),  ACCENT),
        ("FONTNAME",    (1,0),(1,-1),  "Courier-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("GRID",        (0,0),(-1,-1), 0.4, BORDER),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    elems.append(sum_tbl)
    elems.append(Spacer(1, 4*mm))
    return elems


def section_port(r, S):
    if r["box_type"] != "reflex":
        return []
    elems = [Paragraph("PUERTO BASS REFLEX", S["section"])]

    port_desc = (f"Circular ∅{r.get('port_diam',0):.1f} cm"
                 if r.get("port_type")=="circular"
                 else f"Slot {r.get('slot_w',0):.1f}×{r.get('slot_h',0):.1f} cm")

    vel_val = r.get("portVel")
    vel_txt = f"{vel_val:.1f} m/s" if vel_val else "—"
    if vel_val:
        vel_txt += "  ✓ OK" if vel_val < 12 else ("  ⚠ Límite" if vel_val < 17 else "  ✗ Turbulencia")

    port_data = [
        ["Longitud del tubo/slot",   f"{r['L']:.1f} cm   ({r['L']*10:.0f} mm)"],
        ["Tipo de puerto",           port_desc],
        ["Área de cada puerto (Sp)", f"{r['Sp']:.1f} cm²"],
        [f"Área total ({r['N']} puerto/s)", f"{r['SpTotal']:.1f} cm²"],
        ["Diámetro equivalente",     f"{r['d_eq']:.1f} cm"],
        ["Velocidad en puerto",      vel_txt],
        ["Frecuencia de sintonía Fb",f"{r['Fb']:.1f} Hz"],
    ]
    if r.get("Fpipe"):
        port_data.append(["Resonancia del tubo (Pipe)", f"{r['Fpipe']:.1f} Hz  — rellenar 1/3 con espuma"])

    p_tbl = Table(port_data, colWidths=[75*mm, 80*mm])
    p_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), SURFACE),
        ("TEXTCOLOR",   (0,0),(0,-1),  MUTED),
        ("TEXTCOLOR",   (1,0),(1,-1),  ACCENT2),
        ("FONTNAME",    (1,0),(1,-1),  "Courier-Bold"),
        ("FONTSIZE",    (0,0),(-1,-1), 9),
        ("GRID",        (0,0),(-1,-1), 0.4, BORDER),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[SURFACE,SURFACE2]),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    elems.append(p_tbl)
    elems.append(Spacer(1, 4*mm))
    return elems


def section_diagnostics(r, S):
    elems = [Paragraph("DIAGNÓSTICO Y SEMÁFORO", S["section"])]
    diags = []
    if r.get("EBP"):
        c = "✓ OK" if r["EBP"]>100 else ("△ Ambos" if r["EBP"]>50 else "✗ Sellada")
        diags.append(["EBP (Fs/Qes)", f"{r['EBP']:.0f}", c])
    if r.get("portVel") is not None:
        c = "✓ Sin turbulencia" if r["portVel"]<12 else ("△ Límite aceptable" if r["portVel"]<17 else "✗ ¡Turbulencia!")
        diags.append(["Velocidad puerto", f"{r['portVel']:.1f} m/s", c])
    if r.get("SPLmax"):
        diags.append(["SPL máximo (Keele)", f"{r['SPLmax']:.1f} dB @ 1m", "ℹ Estimación Keele 1975"])
    if r.get("Vd"):
        c = "✓ Subwoofer serio" if r["Vd"]>100 else "△ Woofer medio"
        diags.append(["Vd = Sd × Xmax", f"{r['Vd']:.0f} cm³", c])
    if r.get("Qtc_real"):
        c = "✓ Plano" if r["Qtc_real"]<0.8 else ("△ Ligero pico" if r["Qtc_real"]<1.0 else "✗ Pico pronunciado")
        diags.append(["Qtc real", f"{r['Qtc_real']:.3f}", c])
    if r.get("Fbsc"):
        diags.append(["Baffle Step (F_bsc)", f"{r['Fbsc']:.1f} Hz", "ℹ Compensar con filtro shelving"])

    rows = [["PARÁMETRO", "VALOR", "DIAGNÓSTICO"]] + diags
    d_tbl = Table(rows, colWidths=[65*mm, 35*mm, 70*mm])
    d_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,0),  SURFACE2),
        ("BACKGROUND",  (0,1),(-1,-1), SURFACE),
        ("TEXTCOLOR",   (0,0),(-1,0),  MUTED),
        ("TEXTCOLOR",   (0,1),(0,-1),  TEXT),
        ("TEXTCOLOR",   (1,1),(1,-1),  ACCENT),
        ("TEXTCOLOR",   (2,1),(2,-1),  MUTED),
        ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0),(-1,0),  8),
        ("FONTNAME",    (1,1),(1,-1),  "Courier-Bold"),
        ("FONTSIZE",    (1,1),(1,-1),  10),
        ("FONTSIZE",    (0,1),(0,-1),  9),
        ("FONTSIZE",    (2,1),(2,-1),  9),
        ("GRID",        (0,0),(-1,-1), 0.4, BORDER),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[SURFACE,SURFACE2]),
        ("TOPPADDING",  (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    elems.append(d_tbl)
    elems.append(Spacer(1, 4*mm))
    return elems


def section_assembly_notes(r, d, S):
    elems = [Paragraph("NOTAS DE ENSAMBLAJE", S["section"])]
    inch = d.get("inches", 10)
    spk_hole_d = inch * 2.54 * 0.88
    sub_filter = f"{r['Fb']*0.7:.0f} Hz (0.7 × Fb)" if r["box_type"]=="reflex" else "No requerido (sellada)"

    notes = [
        ("1. Orden de corte recomendado",
         "Frontal → Trasera → Tapa → Base → Laterales. Empezar por las piezas mayores para aprovechar el tablero."),
        ("2. Orificio del altavoz",
         f"Diámetro de corte estimado: {spk_hole_d:.1f} cm (verificar con el datasheet del altavoz). "
         "Usar sierra de corona o caladora con guía circular."),
        ("3. Sellado",
         "Aplicar sellador acrílico o silicona en todas las juntas interiores antes del ensamblaje final. "
         "Especialmente crítico en las esquinas traseras."),
        ("4. Refuerzos internos",
         "Para cajas >20 L, añadir un travesaño central de 4×4 cm entre las caras laterales "
         "para reducir resonancias del panel (ya incluido en Vb_bruto como +5%)."),
        ("5. Material absorbente",
         "Rellenar el 30-50% del volumen interior con lana de vidrio o espuma de celda abierta "
         "(no bloquear el puerto en cajas reflex)."),
        ("6. Puerto bass reflex" if r["box_type"]=="reflex" else "6. Caja sellada",
         (f"Longitud exacta del tubo: {r['L']:.1f} cm. Medir desde el interior del baffle hasta "
          "el extremo del tubo. El tubo no debe tocar la pared trasera."
          if r["box_type"]=="reflex"
          else "Asegurarse de que no queden fugas de aire. Verificar con música a alto volumen "
               "pasando la mano por las juntas.")),
        ("7. Filtro subsónico", sub_filter),
        ("8. Terminal de bornes",
         "Instalar en la cara trasera. Usar cable de 2.5 mm² mínimo para conexión interior."),
    ]
    for title, body in notes:
        elems.append(Paragraph(f"<b>{title}</b>", S["body"]))
        elems.append(Paragraph(body, S["note"]))
        elems.append(Spacer(1, 2.5*mm))
    return elems


# ══════════════════════════════════════════════════════════════
#  8. FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════
def generate_pdf(input_data: dict, output_path: str = "plano_caja.pdf"):
    d = validate_input(input_data)
    r = calc_acoustics(d)
    S = get_styles()

    speaker_name = d.get("model_name", "Altavoz")

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=30*mm, bottomMargin=18*mm,
        leftMargin=15*mm, rightMargin=15*mm,
        title=f"SpeakerLab Pro — {speaker_name}",
        author="SpeakerLab Pro v6.0",
    )

    story = []
    story += section_portada(r, d, S)
    story += section_freq_chart(r, S)
    story += section_box_diagram(r, S)
    story += section_pieces_diagram(r, S)
    story += section_cut_sheet(r, d, S)
    story += section_port(r, S)
    story += section_diagnostics(r, S)
    story += section_assembly_notes(r, d, S)

    doc.build(
        story,
        canvasmaker=lambda *a, **kw: SpeakerLabCanvas(*a, speaker_name=speaker_name, **kw),
    )
    logger.info(f"✅  PDF generado: {output_path}")
    return output_path


# ══════════════════════════════════════════════════════════════
#  9. DATOS DE EJEMPLO Y ENTRY POINT
# ══════════════════════════════════════════════════════════════
EXAMPLE_DAYTON = {
    "model_name": "Dayton Audio RSS315HO-4",
    "fs": 18.1, "vas": 213.6, "qts": 0.269, "qes": 0.284, "qms": 4.82,
    "xmax": 24, "sd": 855, "inches": 12, "spl": 86.3,
    "box_type": "reflex", "alignment": "SBB4",
    "material_mm": 18, "material_name": "MDF 18mm",
    "port_type": "circular", "port_diam_cm": 10.0, "num_ports": 1, "k_factor": 0.732,
}

EXAMPLE_TANG_BAND = {
    "model_name": "Tang Band W6-1139SI",
    "fs": 52, "vas": 8.3, "qts": 0.41, "qes": 0.46,
    "xmax": 5.5, "sd": 133, "inches": 6, "spl": 88,
    "box_type": "reflex", "alignment": "B4",
    "material_mm": 18, "material_name": "MDF 18mm",
    "port_type": "circular", "port_diam_cm": 5.0, "num_ports": 1, "k_factor": 0.732,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f_in:
            data = json.load(f_in)
        out = sys.argv[2] if len(sys.argv) > 2 else "plano_caja.pdf"
    else:
        data = EXAMPLE_DAYTON
        out = "plano_dayton_rss315ho.pdf"

    generate_pdf(data, out)