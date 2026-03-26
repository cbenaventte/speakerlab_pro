# SpeakerLab Pro

Calculadora de cajas acústicas DIY con enciclopedia integrada y generación de PDF profesional.

```
speakerlab-pro/
├── frontend/
│   └── index.html          ← App completa (Calculadora + Enciclopedia + DB)
├── backend/
│   ├── main.py             ← API FastAPI (simulación + pagos + PDF)
│   ├── acoustic_sim.py     ← Motor scipy — modelo Small 1973
│   ├── pdf_generator.py    ← ReportLab — planos profesionales
│   ├── requirements.txt
│   └── .env.example        ← Variables de entorno (copiar a .env)
├── samples/                ← Ejemplos de salida
└── .gitignore
```

## Desarrollo local

```bash
# 1. Entorno
cd backend
python -m venv .venv && source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt

# 2. Variables de entorno
cp .env.example .env
# editar .env con tus claves de Stripe y PayPal

# 3. Backend
uvicorn main:app --reload --port 8000

# 4. Frontend
# Abrir frontend/index.html en el navegador
# O servir estático:
cd ../frontend && python -m http.server 3000
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/api/health` | Estado + modo pago |
| GET | `/api/config` | Claves públicas al frontend |
| POST | `/api/simulate` | Curvas scipy completas |
| POST | `/api/compare` | QB3 vs SBB4 vs B4 vs Sellada |
| POST | `/api/payment/stripe/create-intent` | Crea PaymentIntent |
| GET | `/api/payment/stripe/poll-token` | Obtiene access token post-pago |
| POST | `/api/payment/paypal/create-order` | Crea PayPal Order |
| POST | `/api/payment/paypal/capture` | Captura y emite token |
| POST | `/api/pdf` | Descarga PDF (requiere access token) |
| POST | `/webhooks/stripe` | Webhook firmado de Stripe |

Docs interactivos: `http://localhost:8000/docs`

## Producción (Railway / Render / VPS)

1. Poner las variables de entorno del `.env.example` en tu plataforma
2. Registrar el webhook en Stripe → `https://tu-dominio.com/webhooks/stripe`
3. En `frontend/index.html` cambiar:
   ```js
   const API_BASE = 'http://localhost:8000';
   // → 'https://api.tu-dominio.com'
   ```
4. Arrancar: `uvicorn main:app --host 0.0.0.0 --port 8000`
