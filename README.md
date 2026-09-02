<p align="center">
  <img src="frontend/assets/brand/speakerlab-pro-logo.svg" alt="SpeakerLab Pro" width="460">
</p>

<p align="center"><strong>Diseña. Simula. Construye.</strong></p>

Calculadora libre de cajas acústicas DIY con enciclopedia, simulación científica y generación gratuita de planos PDF.

## Alcance de la simulación

SpeakerLab Pro utiliza los modelos clásicos de Thiele y Small para estimar
recintos sellados y bass-reflex. La simulación científica calcula respuesta SPL,
F3/F6/F10, excursión del cono, velocidad del puerto, impedancia y retardo de
grupo. La gráfica JS disponible como respaldo es una **estimación simplificada**
y aparece identificada como tal.

La tensión de simulación es configurable en voltios RMS. La potencia eléctrica
mostrada es una aproximación sobre la resistencia DC del driver:

```text
P ≈ V² / Re
```

Duplicar la tensión aumenta aproximadamente 6,02 dB, duplica la excursión y
cuadruplica la potencia. El valor `Qb` representa las pérdidas combinadas del
recinto y el puerto bass-reflex: un valor menor produce mayores pérdidas. El
rango admitido es 3–30 y el valor inicial es 7.

El modelo es de campo libre y no incluye la sala, difracción del baffle,
compresión térmica, directividad ni la respuesta propia del cono. Para obtener
resultados fiables conviene introducir Mms, Bl, Re, Le, Sd y Xmax del fabricante.

## Estructura

```text
speakerlab-pro/
├── frontend/index.html     # Estructura de la aplicación
├── frontend/css/app.css    # Sistema visual y componentes
├── frontend/assets/brand/  # Identidad visual y guía de marca
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
cambios sin guardar. Además, la calculadora mantiene automáticamente un borrador
del formulario y lo recupera después de cerrar o recargar la página.

### Altavoces personalizados

La base oficial permanece intacta. Desde la vista **Base de Datos** se pueden
crear, editar y eliminar altavoces propios, además de importarlos o exportarlos
como JSON. Estos registros se identifican con la etiqueta `Local` y se almacenan
únicamente en el `localStorage` del dispositivo.

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
| POST | `/api/alignments` | Alineamientos canónicos y F3 obtenido de la curva |
| POST | `/api/simulate` | Simulación acústica completa |
| POST | `/api/compare` | Comparación de alineamientos |
| POST | `/api/pdf` | Generación gratuita del PDF |

Documentación interactiva: `http://localhost:8000/docs`.

## Pruebas automatizadas

La suite protege los valores de referencia de los alineamientos, las simulaciones
sellada y bass-reflex, los límites de entrada, el contrato público de la API y el
diseño responsivo en navegadores de 320, 375, 430, 768 y 1440 píxeles. También
comprueba la escala eléctrica, Le, Qb, puertos inviables y consistencia entre la
interfaz, el motor científico y los planos PDF.

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
python -m unittest discover -v
```

Para utilizar un Chrome ya instalado en vez del navegador administrado por
Playwright, define `PLAYWRIGHT_CHROME_PATH` con la ruta de su ejecutable. GitHub
Actions ejecuta la suite completa automáticamente en cada `push` y pull request.

En este equipo, la ejecución completa es:

```bash
PLAYWRIGHT_CHROME_PATH=/usr/bin/google-chrome python -m unittest discover -v
```

### Criterios de validez

- Las tablas reflex admiten `Qts` entre 0,20 y 0,50. Fuera de ese intervalo la
  aplicación rechaza el alineamiento en vez de recortar silenciosamente el dato.
- Una longitud de puerto inferior a 1 cm se considera inviable. No se modifica
  artificialmente y el PDF no genera un plano que altere la sintonía calculada.
- F3 se obtiene del cruce real de −3 dB de la curva científica. El valor tabular
  permanece disponible internamente como referencia.
- El PDF utiliza el mismo motor canónico, tensión, Qb, Vb, Fb, F3 y puerto que la
  interfaz.

## Despliegue

`vercel.json` publica el frontend y la función FastAPI. El frontend utiliza `/api/*` en el mismo dominio y no necesita una URL de backend escrita en el código.

El motor carga NumPy, SciPy, Matplotlib y ReportLab. Antes de publicar, confirma que el tamaño y el tiempo de inicio se ajustan a los límites del plan de Vercel utilizado.

## Acceso

La aplicación no integra pagos. La simulación y la descarga de planos PDF son libres.
