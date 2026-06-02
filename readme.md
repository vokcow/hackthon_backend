# Pose Video API

API [FastAPI](https://fastapi.tiangolo.com/) que recibe un **vídeo**, ejecuta **pose estimation** con [LibreYOLO](https://www.libreyolo.com/docs) (**YOLO-NAS**, checkpoint `LibreYOLONASn-pose.pt`) solo en **CPU**, y devuelve un **MP4 anotado** con el esqueleto COCO-17 de **una sola persona** (la de mayor confianza por frame inferido).

Pensado para despliegues ligeros (p. ej. [Railway](https://railway.app/)): pocos FPS, `vid_stride` configurable, sin GPU.

## Características

- Inferencia **CPU** (`device=cpu`)
- Modelo **nano** (`LibreYOLONASn-pose.pt`) para menor uso de RAM
- **1 persona**: se elige la detección con mayor `conf` en cada frame procesado
- **Bajo FPS de inferencia**: por defecto se infiere 1 de cada 3 frames (`POSE_VID_STRIDE=3`); entre medias se reutilizan los últimos keypoints para overlay continuo
- Salida: vídeo MP4 con caja, puntos y esqueleto dibujados con OpenCV

## Requisitos

- Python **3.10+** (recomendado 3.11)
- ~2–4 GB RAM libres (PyTorch + modelo en CPU)
- Pesos **completos** en `weights/` (Git LFS); fallback a descarga LibreYOLO si no están presentes

## Instalación

> **Importante:** `libreyolo` en PyPI exige **Python ≥ 3.10**. Si tu venv usa 3.9 verás  
> `No matching distribution found for libreyolo` — no es que el paquete no exista, sino que pip no tiene wheel para 3.9.

Comprueba la versión:

```bash
python --version   # debe ser 3.10, 3.11 o 3.12
```

### Opción A — pip + venv (recomendado: Python 3.11)

```bash
cd cursor_2jun26

# Usa explícitamente 3.11+ (en Mac con conda: python3.11)
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Pesos del modelo (Git LFS)

El checkpoint completo `weights/LibreYOLONASn-pose.pt` va en **Git LFS** (no como blob gigante en Git). Tras clonar:

```bash
brew install git-lfs    # una vez en macOS
git lfs install
git lfs pull            # si ya tenías el repo sin LFS
ls -lh weights/LibreYOLONASn-pose.pt   # ~38M, no un puntero de ~130 bytes
```

La app usa automáticamente `weights/LibreYOLONASn-pose.pt` si existe. En Docker/Railway se copia el archivo **completo** del repo (el build debe hacer checkout con LFS).

Primera vez que **subes** los weights al remoto:

```bash
git lfs install
git add .gitattributes weights/LibreYOLONASn-pose.pt
git commit -m "Track pose weights with Git LFS"
git push
```

Más detalle: [weights/README.md](weights/README.md).

### Opción B — Conda (`environment.yaml`)

```bash
conda env create -f environment.yaml
conda activate pose-video-api
```

Para actualizar dependencias:

```bash
conda env update -f environment.yaml --prune
```

## Ejecutar el servidor

Desde la raíz del proyecto, con el venv activado:

```bash
# Usa el MISMO Python que tiene cv2 instalado (no el uvicorn de conda)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

O el script incluido (siempre apunta al `.venv` del proyecto):

```bash
chmod +x scripts/run_server.sh
./scripts/run_server.sh
```

### ¿`python -c "import cv2"` funciona pero `uvicorn ...` falla?

Con `(base)` + `(.venv)` activos a la vez, el comando `uvicorn` suele ser el de **Anaconda** (`/anaconda3/bin/uvicorn`), no el del venv. Ese intérprete **no** tiene `opencv` instalado.

Compruébalo:

```bash
which python    # debería ser .../cursor_2jun26/.venv/bin/python
which uvicorn   # si es .../anaconda3/bin/uvicorn → ahí está el problema
python -c "import sys; print(sys.executable)"
```

| Comando | Qué Python usa |
|---------|----------------|
| `python -c "import cv2"` | El del **venv** (tiene cv2) |
| `uvicorn app.main:app` | A menudo el de **conda** (sin cv2) |
| `python -m uvicorn app.main:app` | El mismo que `python` (correcto) |

- Documentación interactiva: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

### Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `POSE_MODEL` | `weights/LibreYOLONASn-pose.pt` si existe | Ruta al `.pt` completo; si no, nombre para auto-descarga LibreYOLO |
| `POSE_CONF` | `0.25` | Umbral de confianza de detección |
| `POSE_IOU` | `0.45` | IoU para NMS |
| `POSE_VID_STRIDE` | `3` | Inferir 1 de cada N frames |
| `POSE_KPT_MIN_CONF` | `0.3` | Mínimo para dibujar un keypoint |
| `MAX_UPLOAD_MB` | `100` | Tamaño máximo de subida |
| `OUTPUT_DIR` | `/tmp/pose_outputs` | Carpeta temporal de trabajos |
| `PORT` | `8000` | Puerto (Railway lo inyecta) |

## API

### `GET /health`

Estado del servicio y configuración activa.

### `POST /api/v1/pose/video`

Sube un vídeo y recibe el MP4 anotado.

**Form data**

- `file` — vídeo (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.gif`)

**Query**

- `vid_stride` (int, default `3`) — cada cuántos frames se ejecuta el modelo (más alto = más rápido, keypoints que se actualizan menos a menudo)
- `return_stats_header` (bool) — si `true`, añade cabecera `X-Process-Stats` con JSON de métricas

**Cabeceras de respuesta** (siempre)

- `X-Frames-Total`, `X-Frames-Inferred`, `X-Elapsed-Seconds`, `X-Effective-Fps`

**Ejemplo con curl**

```bash
curl -X POST "http://localhost:8000/api/v1/pose/video?vid_stride=5" \
  -F "file=@/ruta/a/tu_clip.mp4" \
  -o salida_pose.mp4
```

**Ejemplo con Python**

```python
import requests

with open("clip.mp4", "rb") as f:
    r = requests.post(
        "http://localhost:8000/api/v1/pose/video",
        files={"file": ("clip.mp4", f, "video/mp4")},
        params={"vid_stride": 4},
        timeout=600,
    )
r.raise_for_status()
open("salida_pose.mp4", "wb").write(r.content)
print(r.headers.get("X-Effective-Fps"))
```

## Despliegue en Railway

Railway **no descarga Git LFS** al clonar. Este repo **siempre** materializa el modelo completo:

1. **`railway.toml`** → `bash scripts/railway_lfs.sh`: `git lfs pull` + `ensure_weights.py`.
2. **`Dockerfile`** → `python scripts/ensure_weights.py`: intenta `git lfs pull` si hay `.git`; **siempre** descarga el `.pt` completo desde el CDN de Deci (borra punteros o copias previas).

Pasos:

1. Conecta el repo y usa el `Dockerfile` (configurado en `railway.toml`).
2. Asegura **≥ 2 GB RAM** en el plan.
3. El contenedor usa el **modelo completo** en `/app/weights/LibreYOLONASn-pose.pt`.

Tras deploy, comprueba:

```bash
curl https://TU-APP.railway.app/health
# model_exists: true, model_size_mb: ~38
```

```bash
# Build local (necesitas el .pt real, no el puntero LFS)
git lfs pull
bash scripts/railway_prepare.sh
docker build -t pose-video-api .
docker run -p 8000:8000 -e PORT=8000 pose-video-api
```

El build de Docker **no** depende de `.git` en la imagen; el fallback de descarga evita el fallo `COPY .git` que rompía el deploy.

**Nota:** En Railway no hay pantalla local; la visualización es el **vídeo de salida** descargado por HTTP, no una ventana en el servidor.

## Estructura del proyecto

```
app/
  config.py        # Settings y env vars
  drawing.py       # Dibujo esqueleto COCO-17 (OpenCV)
  pose_service.py  # Carga del modelo y procesado de vídeo
  main.py          # Rutas FastAPI
requirements.txt
environment.yaml   # Entorno Conda
Dockerfile
weights/           # LibreYOLONASn-pose.pt (Git LFS)
.gitattributes     # reglas LFS
```

## Rendimiento esperado (CPU)

Orden de magnitud (depende de resolución y CPU):

- `LibreYOLONASn-pose.pt` + `vid_stride=3` → procesamiento de vídeo aprox. **0.5–3 FPS efectivos** de lectura/escritura; inferencia real en **~1/3** de los frames.
- Para vídeos largos, sube `vid_stride` (p. ej. `5`–`10`).

## Licencias y avisos

- **LibreYOLO**: MIT ([docs](https://www.libreyolo.com/docs)).
- **YOLO-NAS weights**: hospedados en CDN de Deci; revisa su licencia para producción.
- YOLO-NAS en LibreYOLO v1.2 está marcado como **experimental**; valida resultados en tu dominio.

## Solución de problemas

| Problema | Qué probar |
|----------|------------|
| `No matching distribution found for libreyolo` | Tu venv es **Python 3.9**. Borra `.venv` y créalo con `python3.11 -m venv .venv` |
| pip muy antiguo | `python -m pip install --upgrade pip` |
| OOM al arrancar | Plan con más RAM; confirma modelo `n` |
| Muy lento | Aumenta `vid_stride`; reduce resolución del vídeo de entrada |
| Sin persona dibujada | Baja `POSE_CONF`; asegura que una persona sea visible y dominante |
| `libreyolo` sin pose NAS | `pip install -U "libreyolo>=1.2.0"` o instala desde GitHub: `pip install "git+https://github.com/LibreYOLO/libreyolo.git"` |
| `weights/*.pt` pesa ~130 bytes | Solo tienes el **puntero LFS** → `git lfs pull` |
| Docker build falla en COPY weights | Ejecuta `git lfs pull` antes de `docker build` |
| `RPC failed; HTTP 400` al push | Subiste el `.pt` **sin** `git-lfs` instalado (~35 MB en Git). Instala LFS, migra y vuelve a push (ver abajo) |
| `non-fast-forward` tras migrar LFS | Historial reescrito → `git push --force-with-lease origin main` (solo si nadie más depende del remoto) |
| `git-lfs was not found on your path` | `export PATH="/opt/homebrew/bin:$PATH"` o `source scripts/lfs_path.sh` (ver [weights/README.md](weights/README.md)) |
| `weights_only` / `Unsupported operand` en Railway | PyTorch 2.6 + pesos Deci; el repo fija `torch<2.6` y parchea carga en `app/torch_compat.py`. Redeploy. |
| `weights/LibreYOLONASn-pose.pt` ~130 bytes en prod | LFS no materializado en build → activa LFS en Railway o `git lfs pull` antes del Docker build |

### Arreglar push fallido (weights sin LFS)

Si creaste `.gitattributes` pero **no** tenías `git-lfs` instalado, el `.pt` entró como blob normal y el push puede fallar.

```bash
brew install git-lfs
git lfs install
git lfs migrate import --include="weights/*.pt" --everything --yes
git lfs pull
git config http.postBuffer 524288000
git push --force-with-lease origin main
```

O: `chmod +x scripts/fix_lfs_push.sh && ./scripts/fix_lfs_push.sh`
