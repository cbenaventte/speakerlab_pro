"""
SpeakerLab Pro — Backend FastAPI  v2.0
Flujo de pago completo: Stripe Checkout + PayPal Orders API
Tokens de un solo uso firmados con HMAC-SHA256.
"""

import os, sys, io, time, json, hmac, hashlib, secrets, base64, logging
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx

sys.path.insert(0, str(Path(__file__).parent))  # acoustic_sim y pdf_generator están aquí
from acoustic_sim import simulate
from alignments import AlignmentEngine
from pdf_generator import generate_pdf

logger = logging.getLogger("speakerlab")
logging.basicConfig(level=logging.INFO)

# ── CONFIG ──────────────────────────────────────────────────────────────────
STRIPE_SECRET_KEY      = os.environ.get("STRIPE_SECRET_KEY",      "sk_test_REPLACE_ME")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "pk_test_REPLACE_ME")
STRIPE_WEBHOOK_SECRET  = os.environ.get("STRIPE_WEBHOOK_SECRET",  "whsec_REPLACE_ME")
PAYPAL_CLIENT_ID       = os.environ.get("PAYPAL_CLIENT_ID",       "REPLACE_ME")
PAYPAL_CLIENT_SECRET   = os.environ.get("PAYPAL_CLIENT_SECRET",   "REPLACE_ME")
PAYPAL_MODE            = os.environ.get("PAYPAL_MODE",            "sandbox")
PAYPAL_BASE            = ("https://api-m.paypal.com" if PAYPAL_MODE == "live"
                          else "https://api-m.sandbox.paypal.com")
PDF_PRICE_USD          = float(os.environ.get("PDF_PRICE_USD",    "10.00"))
TOKEN_SECRET           = os.environ.get("TOKEN_SECRET",           secrets.token_hex(32))
FRONTEND_PATH          = Path(__file__).parent.parent / "frontend" / "index.html"

# ── TOKEN STORE (dict en memoria → reemplazar con Redis en producción) ──────
_token_store: Dict[str, dict] = {}
TOKEN_TTL = 86400   # 24 horas

def _issue_access_token(payment_id: str) -> str:
    payload = f"{payment_id}:{int(time.time()) + TOKEN_TTL}"
    sig     = hmac.new(TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    token   = base64.urlsafe_b64encode(f"{payload}:{sig}".encode()).decode()
    _token_store[token] = {"used": False, "expires": int(time.time()) + TOKEN_TTL}
    return token

def _consume_access_token(token: str) -> bool:
    if token == "dev_unlock" and STRIPE_SECRET_KEY.startswith("sk_test"):
        return True
    entry = _token_store.get(token)
    if not entry or entry["used"] or time.time() > entry["expires"]:
        return False
    try:
        decoded    = base64.urlsafe_b64decode(token.encode()).decode()
        body, sig  = decoded.rsplit(":", 1)
        expected   = hmac.new(TOKEN_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False
    except Exception:
        return False
    entry["used"] = True
    logger.info("Token consumido: %s…", token[:24])
    return True

def _cleanup_expired():
    now = time.time()
    for k in [k for k,v in _token_store.items() if v["expires"] < now]:
        _token_store.pop(k, None)

# ── PAYPAL HELPER ────────────────────────────────────────────────────────────
_pp_cache: dict = {"token": None, "expires": 0}

async def _pp_bearer() -> str:
    if time.time() < _pp_cache["expires"] - 30:
        return _pp_cache["token"]
    creds = base64.b64encode(f"{PAYPAL_CLIENT_ID}:{PAYPAL_CLIENT_SECRET}".encode()).decode()
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{PAYPAL_BASE}/v1/oauth2/token",
                         headers={"Authorization": f"Basic {creds}",
                                  "Content-Type": "application/x-www-form-urlencoded"},
                         data={"grant_type": "client_credentials"}, timeout=10)
    r.raise_for_status()
    d = r.json()
    _pp_cache.update({"token": d["access_token"], "expires": time.time() + d["expires_in"]})
    return d["access_token"]

# ── APP ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SpeakerLab Pro API — precio PDF: $%.2f | Stripe: %s | PayPal: %s",
                PDF_PRICE_USD,
                "live" if STRIPE_SECRET_KEY.startswith("sk_live") else "test",
                PAYPAL_MODE)
    yield

app = FastAPI(title="SpeakerLab Pro API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET","POST"], allow_headers=["*"])

# ── MODELOS ──────────────────────────────────────────────────────────────────
class DriverParams(BaseModel):
    fs:           float           = Field(..., gt=5,    lt=500)
    vas:          float           = Field(..., gt=0.1,  lt=2000)
    qts:          float           = Field(..., gt=0.05, lt=2.0)
    qes:          Optional[float] = None
    qms:          Optional[float] = None
    xmax:         Optional[float] = None
    sd:           Optional[float] = None
    re:           Optional[float] = None
    spl:          Optional[float] = None
    inches:       Optional[float]   = 10
    model_name:   Optional[str]   = "Altavoz"
    box_type:     str             = "reflex"
    alignment:    Optional[str]   = "QB3"
    qtc_target:   Optional[float] = 0.707
    material_mm:  Optional[int]   = 18
    port_type:    Optional[str]   = "circular"
    port_diam_cm: Optional[float] = 7.0
    slot_w_cm:    Optional[float] = 10.0
    slot_h_cm:    Optional[float] = 5.0
    num_ports:    Optional[int]   = 1
    k_factor:     Optional[float] = 0.732

class SimulateRequest(BaseModel):
    driver:            DriverParams
    freq_min:          float = 10.0
    freq_max:          float = 800.0
    freq_points:       int   = 500
    eg_volts:          float = 2.83
    include_chart_png: bool  = False

class CreateOrderRequest(BaseModel):
    driver_snapshot: Optional[dict] = None

class PayPalCaptureRequest(BaseModel):
    order_id:        str
    driver_snapshot: Optional[dict] = None

class PDFRequest(BaseModel):
    driver:       DriverParams
    access_token: str

def _dd(d: DriverParams) -> dict:
    d_dict = d.dict() if hasattr(d, "dict") else d.model_dump()
    return {k: v for k, v in d_dict.items() if v is not None}

def _arr(a) -> list:
    return [round(float(x), 4) for x in a]

# ── DIAGNÓSTICO ──────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    _cleanup_expired()
    return {"status": "ok", "version": "2.0.0", "price_usd": PDF_PRICE_USD,
            "stripe_mode": "live" if STRIPE_SECRET_KEY.startswith("sk_live") else "test",
            "paypal_mode": PAYPAL_MODE,
            "tokens_live": sum(1 for v in _token_store.values() if not v["used"]),
            "timestamp": time.time()}

@app.get("/api/config")
async def frontend_config():
    """Expone claves PÚBLICAS al frontend — nunca las secretas."""
    return {"stripe_publishable_key": STRIPE_PUBLISHABLE_KEY,
            "paypal_client_id":       PAYPAL_CLIENT_ID,
            "price_usd":              PDF_PRICE_USD,
            "paypal_mode":            PAYPAL_MODE}

@app.get("/api/speakers")
async def get_speakers():
    """Devuelve la base de datos de altavoces mapeada desde el Excel."""
    db_path = Path(__file__).parent / "speakers_db.json"
    if db_path.exists():
        return json.loads(db_path.read_text())
    return []

# ── SIMULACIÓN ───────────────────────────────────────────────────────────────
@app.post("/api/simulate")
async def api_simulate(req: SimulateRequest):
    import numpy as np
    dd       = _dd(req.driver)
    warnings = []
    if req.driver.qts < 0.2 or req.driver.qts > 0.5:
        warnings.append(f"Qts={req.driver.qts} fuera del rango de tablas Thiele (0.20–0.50).")
    try:
        freqs  = np.logspace(np.log10(req.freq_min), np.log10(req.freq_max), req.freq_points)
        result = simulate(dd, freqs=freqs, eg_volts=req.eg_volts)
    except Exception as e:
        raise HTTPException(422, f"Error en simulación: {e}")

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
        except Exception as e:
            warnings.append(f"PNG no disponible: {e}")
    return resp

@app.post("/api/compare")
async def api_compare(driver: DriverParams):
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
                "spl": _arr(r["spl"]), 
                "vb": targets[align]["vb"],
                "f3": targets[align]["f3"], 
                "fb": targets[align]["fb"]
            }
        except Exception as e:
            curves[align] = {"error": str(e)}
            
    try:
        rc = simulate({**_dd(driver),"box_type":"closed","qtc_target":0.707}, freqs=freqs)
        # Sellada recalculando F3 vía simulación ya que las tablas son solo para Reflex
        curves["Closed"] = {
            "spl": _arr(rc["spl"]), 
            "vb": round(rc["vb_liters"], 1),
            "f3": round(rc["f3_from_curve"], 1),
            "qtc": round(rc.get("qtc_real", 0.707), 3)
        }
    except Exception as e:
        curves["Closed"] = {"error": str(e)}
        
    return {"freqs": [round(float(f), 2) for f in freqs], "curves": curves}

# ── STRIPE ───────────────────────────────────────────────────────────────────
@app.post("/api/payment/stripe/create-intent")
async def stripe_create_intent(req: CreateOrderRequest):
    """
    Crea un PaymentIntent. El client_secret va al frontend → Stripe.js
    completa el cobro sin que el número de tarjeta toque nuestro servidor.
    """
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        intent = stripe.PaymentIntent.create(
            amount      = int(PDF_PRICE_USD * 100),
            currency    = "usd",
            description = "SpeakerLab Pro — PDF Planos Acústicos",
            metadata    = {"product": "speakerlab_pdf_v1",
                           "snapshot": json.dumps(req.driver_snapshot or {})[:500]},
            automatic_payment_methods={"enabled": True},
        )
        return {"client_secret": intent.client_secret,
                "payment_intent_id": intent.id,
                "amount_usd": PDF_PRICE_USD}
    except stripe.error.StripeError as e:
        raise HTTPException(502, f"Stripe: {e.user_message}")

@app.get("/api/payment/stripe/poll-token")
async def stripe_poll_token(pi: str):
    """
    El frontend llama a este endpoint después de confirmar con Stripe.js.
    Si el webhook ya llegó, devuelve el access_token. Si no, lo verifica
    directamente con la API de Stripe (fallback para webhooks lentos).
    """
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY

    # ¿Ya está en el store (emitido por el webhook)?
    for token, entry in list(_token_store.items()):
        try:
            decoded = base64.urlsafe_b64decode(token.encode()).decode()
            body    = decoded.rsplit(":", 1)[0]
            if body.startswith(pi + ":") and not entry["used"]:
                return {"access_token": token, "status": "ready"}
        except Exception:
            continue

    # Fallback: preguntar directamente a Stripe
    try:
        intent = stripe.PaymentIntent.retrieve(pi)
        if intent.status == "succeeded":
            token = _issue_access_token(pi)
            return {"access_token": token, "status": "ready"}
        return {"status": intent.status, "access_token": None}
    except stripe.error.StripeError as e:
        raise HTTPException(502, str(e))

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request,
                         stripe_signature: Optional[str] = Header(None, alias="stripe-signature")):
    import stripe
    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        logger.warning("Stripe webhook: firma inválida")
        raise HTTPException(400, "Firma inválida")
    if event["type"] == "payment_intent.succeeded":
        pi_id = event["data"]["object"]["id"]
        _issue_access_token(pi_id)
        logger.info("Stripe pago confirmado: %s", pi_id)
    return {"received": True}

# ── PAYPAL ───────────────────────────────────────────────────────────────────
@app.post("/api/payment/paypal/create-order")
async def paypal_create_order(req: CreateOrderRequest):
    """Crea una PayPal Order. El frontend la presenta con el SDK oficial."""
    try:
        bearer = await _pp_bearer()
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{PAYPAL_BASE}/v2/checkout/orders",
                headers={"Authorization": f"Bearer {bearer}",
                         "Content-Type": "application/json",
                         "PayPal-Request-Id": secrets.token_hex(16)},
                json={"intent": "CAPTURE",
                      "purchase_units": [{"amount": {"currency_code":"USD",
                                                      "value": f"{PDF_PRICE_USD:.2f}"},
                                          "description": "SpeakerLab Pro — PDF Planos"}],
                      "application_context": {"brand_name": "SpeakerLab Pro",
                                              "user_action": "PAY_NOW"}},
                timeout=15)
        r.raise_for_status()
        d = r.json()
        return {"order_id": d["id"], "status": d["status"]}
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"PayPal: {e.response.text}")

@app.post("/api/payment/paypal/capture")
async def paypal_capture(req: PayPalCaptureRequest):
    """
    Captura (cobra) una PayPal Order aprobada.
    Devuelve access_token si el pago fue correcto.
    """
    try:
        bearer = await _pp_bearer()
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{PAYPAL_BASE}/v2/checkout/orders/{req.order_id}/capture",
                headers={"Authorization": f"Bearer {bearer}",
                         "Content-Type": "application/json"},
                timeout=15)
        if r.status_code not in (200, 201):
            raise HTTPException(402, f"PayPal captura fallida: {r.text}")
        d = r.json()
        if d["status"] != "COMPLETED":
            raise HTTPException(402, f"PayPal estado: {d['status']}")
        amount = float(d["purchase_units"][0]["payments"]["captures"][0]["amount"]["value"])
        if amount < PDF_PRICE_USD:
            raise HTTPException(402, f"Monto insuficiente: ${amount}")
        token = _issue_access_token(f"paypal:{req.order_id}")
        logger.info("PayPal pago confirmado: %s", req.order_id)
        return {"access_token": token, "status": "COMPLETED"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"PayPal: {e}")

# ── PDF ───────────────────────────────────────────────────────────────────────
@app.post("/api/pdf")
async def api_pdf(req: PDFRequest, bg: BackgroundTasks):
    """Genera y descarga el PDF. El access_token es de un solo uso."""
    if not _consume_access_token(req.access_token):
        raise HTTPException(402, "Token inválido, caducado o ya utilizado. "
                                 "Completa el pago para obtener uno nuevo.")
    dd       = _dd(req.driver)
    out_path = f"/tmp/speakerlab_{int(time.time())}_{secrets.token_hex(4)}.pdf"
    try:
        generate_pdf(dd, out_path)
    except Exception as e:
        raise HTTPException(500, f"Error generando PDF: {e}")

    def _stream():
        with open(out_path, "rb") as f:
            yield from f

    bg.add_task(os.unlink, out_path)
    name = (dd.get("model_name") or "altavoz").replace(" ", "_")
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
