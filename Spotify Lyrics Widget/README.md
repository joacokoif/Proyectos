# 🎵 Spotify Lyrics Widget — Desktop Overlay

> Widget de escritorio flotante que muestra las **letras sincronizadas** de la canción que estás escuchando en Spotify, con animaciones fluidas tipo karaoke y transparencia total. Funciona sobre cualquier ventana en Windows.

---

## ✨ Features

- 🎤 **Letras sincronizadas en tiempo real** — se actualiza automáticamente con cada nueva canción
- 🎬 **Animación tipo Spotify** — la línea activa se resalta en blanco y grande, las demás en gris tenue, con scroll suave tipo física
- 🪟 **Overlay transparente** — widget flotante sin bordes, siempre encima de otras ventanas
- 🖱️ **Draggable y resizable** — arrastrá para moverlo, usá el grip para cambiar el tamaño
- 🎵 **Info de canción** — muestra título, artista y álbum en tiempo real
- ⚡ **60 FPS** — animación a 60 cuadros por segundo con easing suave

---

## 🖥️ Captura

```
╔══════════════════════════════════════╗
● Spotify Subtitles                  ✕
──────────────────────────────────────
  I'm gonna make you love me
  
  **All I know is I'm lost**
  **without you**
  
  Take me back to the night we met
──────────────────────────────────────
  All I Know • Lord Huron • Strange Trails
╚══════════════════════════════════════╝
```

---

## 🗂️ Estructura del proyecto

```
spotify_lyrics_widget/
├── main.py              # Widget principal (Tkinter + física de animación)
├── media_tracker.py     # Detección de la canción actual (Windows Media API)
├── lyrics_fetcher.py    # Fetch y parseo de letras sincronizadas
├── dist/                # Ejecutable compilado (.exe)
└── build/               # Archivos de compilación PyInstaller
```

---

## 🚀 Uso

### Opción A — Python

```bash
pip install asyncio tkinter
python main.py
```

### Opción B — Ejecutable (sin instalar Python)

```bash
dist/Spotify Subtitles.exe
```

---

## ⚙️ Cómo funciona

```
Windows Media API
      ↓
media_tracker.py  →  detecta título + artista de Spotify
      ↓
lyrics_fetcher.py  →  busca letras con timestamps
      ↓
main.py (Tkinter)  →  renderiza overlay con animación 60fps
```

1. **`media_tracker.py`** consulta la API de sesión de medios de Windows (`Windows.Media.Control`) para obtener el título y artista de lo que está sonando en Spotify.
2. **`lyrics_fetcher.py`** busca las letras con timestamps (formato LRC).
3. **`main.py`** renderiza el widget sobre el escritorio con física de scroll suave — cada línea tiene una posición `visual_y` que interpola hacia `target_y` con un factor de easing `0.15` por frame.

---

## 🛠️ Stack técnico

| Librería | Uso |
|---|---|
| `tkinter` | GUI del widget (Canvas + transparencia) |
| `asyncio` / `threading` | Loop de detección asíncrono sin bloquear la UI |
| `Windows.Media.Control` | Detección de la canción actual en Spotify |
| PyInstaller | Compilación a `.exe` standalone |

---

## 📌 Notas

- **Solo compatible con Windows** — usa la API de Windows Media Session.
- Spotify debe estar abierto y reproduciendo para que el widget detecte la canción.
- El widget se posiciona automáticamente en el centro inferior de la pantalla al iniciar.
- Si las letras no están disponibles, muestra el mensaje `"Lyrics not found for this song."`.

---

## 🔧 Compilar el ejecutable

```bash
pip install pyinstaller
pyinstaller "Spotify Subtitles.spec"
```
