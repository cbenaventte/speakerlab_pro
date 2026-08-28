"""SpeakerLab Pro — API pública de simulación y generación de planos."""

import os, sys, time, json, secrets, base64, logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

sys.path.insert(0, str(Path(__file__).parent))  # acoustic_sim y pdf_generator están aquí
from acoustic_sim import simulate
from alignments import AlignmentEngine

logger = logging.getLogger("speakerlab")
logging.basicConfig(level=logging.INFO)

# ── CONFIG ──────────────────────────────────────────────────────────────────
FRONTEND_PATH          = Path(__file__).parent.parent / "frontend" / "index.html"
FRONTEND_DIR           = FRONTEND_PATH.parent
MAX_REQUEST_BYTES      = int(os.environ.get("MAX_REQUEST_BYTES", "1048576"))
RATE_LIMIT_PER_MINUTE  = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))
_request_windows: dict[str, deque] = defaultdict(deque)

# ── APP ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SpeakerLab Pro API — acceso libre")
    yield

app = FastAPI(title="SpeakerLab Pro API", version="2.1.0", lifespan=lifespan)
allowed_origins = [origin.strip() for origin in os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins,
                   allow_methods=["GET","POST"], allow_headers=["*"])
app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")

@app.middleware("http")
async def operational_guards(request: Request, call_next):
    """Límites básicos y cabeceras seguras para una API pública."""
    if request.url.path.startswith("/api/"):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_BYTES:
                    return JSONResponse({"detail": "Solicitud demasiado grande"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Content-Length inválido"}, status_code=400)

        client_key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = _request_windows[client_key]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                {"detail": "Demasiadas solicitudes; intenta nuevamente en un minuto"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        window.append(now)

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; "
        "connect-src 'self'; script-src 'self' 'unsafe-inline'; object-src 'none'; "
        "base-uri 'self'; frame-ancestors 'none'"
    )
    return response

# ── MODELOS ──────────────────────────────────────────────────────────────────
class DriverParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fs:           float           = Field(..., gt=5,    lt=500)
    vas:          float           = Field(..., gt=0.1,  lt=2000)
    qts:          float           = Field(..., gt=0.05, lt=2.0)
    qes:          Optional[float] = Field(None, gt=0.05, lt=5)
    qms:          Optional[float] = Field(None, gt=0.1, lt=100)
    xmax:         Optional[float] = Field(None, gt=0, le=100)
    sd:           Optional[float] = Field(None, gt=1, le=5000)
    re:           Optional[float] = Field(None, gt=0.1, le=100)
    spl:          Optional[float] = Field(None, ge=50, le=130)
    mms:          Optional[float] = Field(None, gt=0, lt=1000,
                      description="Masa móvil en gramos")
    bl:           Optional[float] = Field(None, gt=0, lt=50,
                      description="Factor de fuerza en T·m")
    le:           Optional[float] = Field(None, gt=0, lt=10,
                      description="Inductancia de bobina en mH")
    inches:       Optional[float] = Field(10, ge=1, le=30)
    model_name:   Optional[str] = Field("Altavoz", max_length=120)
    box_type:     Literal["reflex", "closed"] = "reflex"
    alignment:    Optional[Literal["QB3", "SBB4", "B4"]] = "QB3"
    qtc_target:   Optional[float] = Field(0.707, gt=0.05, lt=2)
    material_mm:  Optional[int] = Field(18, ge=6, le=50)
    port_type:    Optional[Literal["circular", "slot"]] = "circular"
    port_diam_cm: Optional[float] = Field(7.0, ge=0.5, le=50)
    slot_w_cm:    Optional[float] = Field(10.0, ge=0.5, le=200)
    slot_h_cm:    Optional[float] = Field(5.0, ge=0.5, le=200)
    num_ports:    Optional[int] = Field(1, ge=1, le=8)
    k_factor:     Optional[float] = Field(0.732, ge=0.1, le=2)

    @model_validator(mode="after")
    def validate_physical_relationships(self):
        if self.qes is not None and self.qes <= self.qts:
            raise ValueError("Qes debe ser mayor que Qts")
        if self.qms is not None and self.qms <= self.qts:
            raise ValueError("Qms debe ser mayor que Qts")
        if self.box_type == "closed" and self.qtc_target is not None and self.qtc_target <= self.qts:
            raise ValueError("Qtc objetivo debe ser mayor que Qts")
        return self

class SimulateRequest(BaseModel):
    driver:            DriverParams
    freq_min:          float = Field(10.0, ge=5.0, le=20000.0)
    freq_max:          float = Field(800.0, ge=10.0, le=40000.0)
    freq_points:       int   = Field(500, ge=50, le=5000)
    eg_volts:          float = Field(2.83, gt=0.0, le=200.0)
    include_chart_png: bool  = False

class PDFRequest(BaseModel):
    driver: DriverParams

class AlignmentRequest(BaseModel):
    fs:  float = Field(..., gt=5, lt=500)
    vas: float = Field(..., gt=0.1, lt=2000)
    qts: float = Field(..., gt=0.05, lt=2.0)
    qtc_target: float = Field(0.707, gt=0.05, lt=2.0)

def _dd(d: DriverParams) -> dict:
    d_dict = d.model_dump() if hasattr(d, "model_dump") else d.dict()
    return {k: v for k, v in d_dict.items() if v is not None}

def _arr(a) -> list:
    return [round(float(x), 4) for x in a]

def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass

# ── DIAGNÓSTICO ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.1.0", "access": "free",
            "timestamp": time.time()}

@app.get("/api/config")
async def frontend_config():
    return {
        "access": "free",
        "pdf_enabled": True,
    }

@app.get("/api/speakers")
async def get_speakers():
    """Devuelve la base de datos de altavoces mapeada desde el Excel."""
    db_path = Path(__file__).parent / "speakers_db.json"
    if db_path.exists():
        return json.loads(db_path.read_text())
    return []

@app.post("/api/alignments")
async def get_alignments(req: AlignmentRequest):
    """Calcula diseños reflex y sellado desde el motor canónico."""
    try:
        closed = simulate({
            "fs": req.fs, "vas": req.vas, "qts": req.qts,
            "box_type": "closed", "qtc_target": req.qtc_target,
        })
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {
        "alignments": AlignmentEngine(req.fs, req.qts, req.vas).get_all_alignments(),
        "closed": {
            "vb": round(closed["vb_liters"], 1),
            "f3": round(closed["f3_from_curve"], 1),
            "fc": round(closed["fc"], 1),
            "qtc": round(closed["qtc_real"], 3),
        },
    }

# ── SIMULACIÓN ───────────────────────────────────────────────────────────────
@app.post("/api/simulate")
async def api_simulate(req: SimulateRequest):
    import numpy as np
    dd       = _dd(req.driver)
    warnings = []
    if req.driver.qts < 0.2 or req.driver.qts > 0.5:
        warnings.append(
            f"Qts={req.driver.qts} fuera del rango de tablas Thiele (0.20–0.50)."
        )
    if not req.driver.mms:
        warnings.append(
            "Mms no proporcionado — se estimará desde Vas/Sd. "
            "El error puede superar el 100%. Recomendamos incluir "
            "el dato del fabricante."
        )
    if not req.driver.bl:
        warnings.append(
            "Bl no proporcionado — se estimará desde Re/Mms/Qes. "
            "Incluir el dato del fabricante mejora la precisión."
        )
    if req.freq_max <= req.freq_min:
        raise HTTPException(422, "freq_max debe ser mayor que freq_min")
    try:
        freqs  = np.logspace(np.log10(req.freq_min), np.log10(req.freq_max), req.freq_points)
        # SciPy mantiene estado nativo que no es seguro mover entre el hilo
        # principal y workers tras cálculos previos. El costo queda acotado por
        # freq_points <= 5000.
        result = simulate(dd, freqs=freqs, eg_volts=req.eg_volts)
    except Exception as e:
        logger.exception("Error interno en simulación")
        raise HTTPException(422, "No se pudo simular con los parámetros enviados") from e

    resp = {
        "freqs": _arr(result["freqs"]), "spl": _arr(result["spl"]),
        "excursion": _arr(result["excursion"]), "impedance": _arr(result["impedance"]),
        "group_delay": _arr(result["group_delay"]),
        "metrics": {
            "box_type": result["box_type"], "vb_liters": round(result["vb_liters"], 2),
            "f3": round(result["f3_from_curve"], 1),
            "f6": round(result.get("f6") or 0, 1), "f10": round(result.get("f10") or 0, 1),
            "sens_band": round(result["sens_band"], 1),
            "xmax_exceeded_below": result.get("xmax_exceeded_below"),
        },
        "warnings": warnings,
    }
    if result["box_type"] == "reflex":
        resp["port_vel"] = _arr(result["port_vel"])
        resp["metrics"].update({
            "fb": round(result["fb"], 1), "alignment": result.get("alignment"),
            "L_port_cm": round(result.get("L_port_cm", 0), 1),
            "sp_cm2": round(result.get("sp_cm2", 0), 1),
            "port_turbulence_freq": result.get("port_turbulence_freq"),
        })
    else:
        resp["metrics"].update({
            "qtc_real": round(result.get("qtc_real", 0), 3),
            "fc": round(result.get("fc", 0), 1),
        })
    if req.include_chart_png:
        try:
            from acoustic_sim import plot_results
            tmp = f"/tmp/_chart_{int(time.time())}.png"
            plot_results(result, tmp)
            with open(tmp, "rb") as f:
                resp["chart_png"] = base64.b64encode(f.read()).decode()
            os.unlink(tmp)
        except Exception:
            logger.exception("No se pudo generar la gráfica PNG")
            warnings.append("PNG no disponible")
    return resp

def _compare_sync(driver: DriverParams):
    import numpy as np
    
    # Motor matemático explícito en tablas (Excel)
    engine = AlignmentEngine(driver.fs, driver.qts, driver.vas)
    targets = engine.get_all_alignments()

    freqs = np.logspace(np.log10(15), np.log10(600), 400)
    curves = {}
    
    for align in ["QB3", "SBB4", "B4"]:
        d = {**_dd(driver), "box_type": "reflex", "alignment": align}
        try:
            r = simulate(d, freqs=freqs)
            curves[align] = {
                "spl": _arr([db - r.get("sens_band", driver.spl) for db in r["spl"]]), 
                "vb": targets[align]["vb"],
                "f3": targets[align]["f3"], 
                "fb": targets[align]["fb"]
            }
        except Exception as e:
            logger.warning("No se pudo calcular alineamiento %s: %s", align, type(e).__name__)
            curves[align] = {"error": "No se pudo calcular este alineamiento"}
            
    try:
        rc = simulate({**_dd(driver),"box_type":"closed","qtc_target":0.707}, freqs=freqs)
        # Sellada recalculando F3 vía simulación ya que las tablas son solo para Reflex
        curves["Closed"] = {
            "spl": _arr([db - rc.get("sens_band", driver.spl) for db in rc["spl"]]), 
            "vb": round(rc["vb_liters"], 1),
            "f3": round(rc["f3_from_curve"], 1),
            "qtc": round(rc.get("qtc_real", 0.707), 3)
        }
    except Exception as e:
        logger.warning("No se pudo calcular caja sellada: %s", type(e).__name__)
        curves["Closed"] = {"error": "No se pudo calcular la caja sellada"}
        
    return {"freqs": [round(float(f), 2) for f in freqs], "curves": curves}

@app.post("/api/compare")
async def api_compare(driver: DriverParams):
    try:
        return _compare_sync(driver)
    except Exception as exc:
        logger.exception("Error interno en comparación")
        raise HTTPException(422, "No se pudo comparar con los parámetros enviados") from exc

# ── PDF ───────────────────────────────────────────────────────────────────────
@app.post("/api/pdf")
async def api_pdf(req: PDFRequest, bg: BackgroundTasks):
    """Genera y descarga gratuitamente el PDF para los parámetros recibidos."""
    # Matplotlib y ReportLab se cargan sólo al solicitar un documento, evitando
    # penalizar el arranque de los endpoints de consulta y simulación.
    from pdf_generator import generate_pdf
    dd       = _dd(req.driver)
    out_path = f"/tmp/speakerlab_{int(time.time())}_{secrets.token_hex(4)}.pdf"
    try:
        # El generador comparte rutinas nativas de SciPy/Matplotlib con la
        # simulación; mantenerlas en el mismo hilo evita bloqueos nativos.
        generate_pdf(dd, out_path)
    except Exception as e:
        logger.exception("Error interno generando PDF")
        _safe_unlink(out_path)
        raise HTTPException(500, "No se pudo generar el PDF") from e

    def _stream():
        with open(out_path, "rb") as f:
            yield from f

    bg.add_task(_safe_unlink, out_path)
    raw_name = dd.get("model_name") or "altavoz"
    name = "".join(ch for ch in raw_name if ch.isalnum() or ch in "-_")[:80] or "altavoz"
    return StreamingResponse(_stream(), media_type="application/pdf",
                             headers={"Content-Disposition":
                                      f'attachment; filename="speakerlab_{name}.pdf"'})

# ── FRONTEND ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return HTMLResponse(content=FRONTEND_PATH.read_text())
    return HTMLResponse("<h1>SpeakerLab Pro API v2</h1><p>Docs: <a href='/docs'>/docs</a></p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
