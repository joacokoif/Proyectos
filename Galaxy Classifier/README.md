# 🌌 Galaxy Classifier — Clasificación de Galaxias con Deep Learning

> Clasificador de morfología galáctica entrenado con imágenes de alta resolución del **Telescopio Hubble (NASA/ESA)**. Distingue entre galaxias **elípticas**, **espirales** e **irregulares** usando una red neuronal convolucional (CNN).

---

## 🔭 Motivación

La clasificación morfológica de galaxias es una tarea clave en astronomía moderna. Con los nuevos telescopios generando millones de imágenes por año, los modelos de visión por computadora permiten automatizar lo que antes requería horas de trabajo manual de astrónomos.

---

## 📂 Dataset

Las imágenes fueron recopiladas de:

- 📁 `fotos nasa/` — Imágenes de alta resolución descargadas del archivo ESA/Hubble
- 📁 `fotos baja calidad/` — Imágenes adicionales para augmentation y robustez del modelo

**Clases morfológicas:**

| Clase | Descripción |
|---|---|
| Elíptica | Forma oval suave, sin estructura interna definida |
| Espiral | Brazos curvados alrededor de un núcleo central |
| Irregular | Sin forma simétrica definida, frecuentemente post-colisión |

---

## 🧠 Arquitectura del modelo

Red neuronal convolucional (CNN) entrenada con:

- Capas convolucionales + MaxPooling
- Dropout para regularización
- Data augmentation: rotaciones, flips, zoom y ajuste de brillo
- Transfer learning desde modelos preentrenados en ImageNet

---

## 🚀 Uso

### Entrenar el modelo

```bash
python train.py
```

### Clasificar una imagen

```bash
python predict.py --image path/a/galaxia.jpg
```

---

## 📊 Clases del problema

```
galaxias/
├── fotos nasa/          # Imágenes de alta resolución (HST)
└── fotos baja calidad/  # Imágenes complementarias
```

---

## 🛠️ Stack técnico

| Herramienta | Uso |
|---|---|
| Python | Lenguaje principal |
| TensorFlow / Keras | Entrenamiento del modelo CNN |
| OpenCV / PIL | Procesamiento de imágenes |
| Matplotlib | Visualización de resultados |
| NumPy | Manipulación de arrays |

---

## 🔬 Fuentes de datos

- **NASA Hubble Site**: [hubblesite.org](https://hubblesite.org)
- **ESA Hubble Archive**: [esahubble.org](https://esahubble.org/images/)
- Etiquetas morfológicas extraídas de metadatos de imágenes ESA

---

## 📌 Notas

- Las imágenes del Telescopio Hubble son de dominio público (NASA/ESA).
- El modelo fue diseñado para imágenes de alta resolución; puede degradarse con imágenes de menor calidad.
- El dataset presenta desequilibrio de clases (pocas irregulares) — se recomienda usar `class_weight` o oversampling.
