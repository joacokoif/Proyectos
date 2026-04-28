# 🌌 Galaxy Classifier — Clasificación de Galaxias con Deep Learning

> Clasificador de morfología galáctica entrenado con imágenes de alta resolución del **Telescopio Hubble (NASA/ESA)**. Distingue entre galaxias **elípticas**, **espirales** e **irregulares** usando Transfer Learning sobre ResNet18 y EfficientNet-B0.

---

## 🔭 Motivación

La clasificación morfológica de galaxias es una tarea clave en astronomía moderna. Con los nuevos telescopios generando millones de imágenes por año, los modelos de visión por computadora permiten automatizar lo que antes requería horas de trabajo manual de astrónomos.

---

## 🧠 Arquitectura del modelo

```
Imagen (224×224)
      ↓
Pretrained Backbone (ResNet18 / EfficientNet-B0)
      ↓
Custom FC Head → Dropout(0.3) → Linear(512→256) → ReLU → Dropout(0.2) → Linear(256→3)
      ↓
[Elíptica | Espiral | Irregular]
```

**¿Por qué Transfer Learning?**
Los features de ImageNet (bordes, texturas, formas) se transfieren bien a morfología galáctica: los brazos espirales son curvas, los bulbos son patrones circulares.

---

## 📂 Estructura del proyecto

```
galaxy_classifier/
├── run_pipeline.py       # ⭐ Punto de entrada principal (todo el pipeline)
├── train.py              # Entrenamiento del modelo
├── evaluate.py           # Evaluación completa con métricas y visualizaciones
├── model.py              # Arquitecturas (ResNet18, EfficientNet-B0)
├── dataset.py            # Dataset loader + augmentations
├── prepare_dataset.py    # Organiza imágenes descargadas en train/val/test
├── download_data.py      # Descarga imágenes desde NASA/ESA
├── app.py                # Clasificador interactivo (inference en imagen suelta)
├── config.py             # Parámetros centralizados (LR, epochs, paths, etc.)
├── utils.py              # Helpers: Grad-CAM, plots, métricas
├── data/                 # Dataset (no incluido en el repo — ver abajo)
│   └── raw/              # Imágenes sin procesar
├── results/              # Métricas y gráficos generados
│   ├── confusion_matrix_resnet18.png
│   ├── confusion_matrix_efficientnet_b0.png
│   ├── roc_curves_resnet18.png
│   ├── roc_curves_efficientnet_b0.png
│   ├── predictions_resnet18.png
│   ├── predictions_efficientnet_b0.png
│   ├── metrics_resnet18.json
│   └── history_resnet18.json
├── requirements.txt
└── .gitignore
```

---

## 🚀 Uso rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Descargar el dataset

```bash
python download_data.py
```

### 3. Preparar el dataset (split train/val/test)

```bash
python prepare_dataset.py
```

### 4. Correr el pipeline completo

```bash
python run_pipeline.py
```

### 5. Clasificar una imagen suelta

```bash
python app.py --image ruta/a/galaxia.jpg
```

---

## 📊 Resultados

| Modelo | Accuracy | Params |
|---|---|---|
| ResNet18 | ~85% | 11.7M |
| EfficientNet-B0 | ~88% | 5.3M |

Las matrices de confusión y curvas ROC se guardan automáticamente en `results/`.

---

## 🗂️ Dataset

Las imágenes fueron recopiladas de:
- **NASA Hubble Site**: [hubblesite.org](https://hubblesite.org)
- **ESA Hubble Archive**: [esahubble.org](https://esahubble.org/images/)

**Clases morfológicas:**

| Clase | Descripción |
|---|---|
| Elíptica | Forma oval suave, sin estructura interna definida |
| Espiral | Brazos curvados alrededor de un núcleo central |
| Irregular | Sin forma simétrica, frecuentemente post-colisión |

> ⚠️ Las imágenes **no están incluidas en el repositorio** por su tamaño. Usar `download_data.py` para descargarlads.

---

## ⚙️ Configuración (`config.py`)

Los parámetros clave se configuran en `config.py`:

```python
DEFAULT_MODEL    = "resnet18"      # o "efficientnet_b0"
NUM_CLASSES      = 3
LEARNING_RATE    = 1e-4
EPOCHS           = 30
BATCH_SIZE       = 32
IMG_SIZE         = 224
FREEZE_BACKBONE  = False           # True = entrenar solo el head
```

---

## 🛠️ Stack técnico

| Librería | Uso |
|---|---|
| `PyTorch` + `torchvision` | Modelo, transfer learning, training loop |
| `scikit-learn` | Métricas (accuracy, F1, ROC-AUC, confusion matrix) |
| `matplotlib` | Visualizaciones y Grad-CAM |
| `Pillow` | Carga y transformación de imágenes |
| `requests` / `tqdm` | Descarga del dataset con barra de progreso |
