# SpeakerLab Pro

Calculadora libre de cajas acústicas DIY con enciclopedia, simulación científica y generación gratuita de planos PDF.

## Estructura

```text
speakerlab-pro/
├── frontend/index.html     # Estructura de la aplicación
├── frontend/css/app.css    # Sistema visual y componentes
├── frontend/js/database.js # Base de datos y tablas de alineamiento
├── frontend/js/app.js      # Estado, cálculos e integración con la API
├── api/index.py            # API FastAPI
├── api/acoustic_sim.py     # Motor acústico Small/Thiele
├── api/alignments.py       # Alineamientos clásicos
├── api/pdf_generator.py    # Planos PDF
├── api/speakers_db.json    # Base utilizada por la API
├── requirements.txt
└── vercel.json
```

## Desarrollo local

Requiere Python 3.11 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000
```

Abre `http://localhost:8000`. La API sirve también el frontend en la ruta raíz.

Los orígenes adicionales permitidos se configuran mediante una lista separada por comas:

```bash
ALLOWED_ORIGINS=https://mi-dominio.example,http://localhost:3000
```

### Proyectos locales

Los diseños pueden nombrarse, guardarse, actualizarse, renombrarse, recuperarse,
duplicarse, eliminarse e importarse o exportarse como JSON. Se almacenan
únicamente en el `localStorage` del navegador: no se crean cuentas ni se envían
proyectos al servidor. El gestor advierte antes de reemplazar un formulario con
cambios sin guardar.

Límites operacionales opcionales:

```bash
MAX_REQUEST_BYTES=1048576
RATE_LIMIT_PER_MINUTE=120
```

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/health` | Estado del servicio |
| GET | `/api/config` | Capacidades públicas |
| GET | `/api/speakers` | Base de altavoces |
| POST | `/api/alignments` | Alineamientos calculados por la tabla canónica |
| POST | `/api/simulate` | Simulación acústica completa |
| POST | `/api/compare` | Comparación de alineamientos |
| POST | `/api/pdf` | Generación gratuita del PDF |

Documentación interactiva: `http://localhost:8000/docs`.

## Pruebas automatizadas

La suite protege los valores de referencia de los alineamientos, las simulaciones
sellada y bass-reflex, los límites de entrada y el contrato público de la API.

```bash
python -m unittest discover -s tests -v
```

## Despliegue

`vercel.json` publica el frontend y la función FastAPI. El frontend utiliza `/api/*` en el mismo dominio y no necesita una URL de backend escrita en el código.

El motor carga NumPy, SciPy, Matplotlib y ReportLab. Antes de publicar, confirma que el tamaño y el tiempo de inicio se ajustan a los límites del plan de Vercel utilizado.

## Acceso

La aplicación no integra pagos. La simulación y la descarga de planos PDF son libres.
