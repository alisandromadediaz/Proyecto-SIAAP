# 🎓 SIAAP — Sistema Inteligente de Apoyo al Aprendizaje Personalizado

Proyecto de **Machine Learning supervisado** (clasificación multiclase) desarrollado con la
metodología **CRISP-ML(Q)**, que predice el nivel de rendimiento académico de un estudiante
(`Bajo` / `Medio` / `Alto`) a partir de variables de comportamiento de estudio, para disparar
**recomendaciones de apoyo personalizadas**.

> Proyecto académico. Los datos utilizados son **sintéticos**, generados con fines demostrativos.

---

## 📁 Estructura del repositorio

```
siaap/
├── app.py                    # App de Streamlit (demo interactiva del modelo)
├── requirements.txt          # Dependencias para ejecutar la app
├── .streamlit/
│   └── config.toml           # Tema visual de la app (colores del proyecto)
├── notebook/
│   └── proyecto_crisp_ml_rendimiento_academico.ipynb   # Desarrollo completo CRISP-ML(Q)
├── landing_page.html         # Landing page explicativa del proyecto
└── README.md
```

> ⚠️ Antes de subir el repositorio, coloca el archivo `proyecto_crisp_ml_rendimiento_academico.ipynb`
> dentro de una carpeta `notebook/` y `landing_page.html` en la raíz, tal como se muestra arriba.

---

## 🧭 Metodología: CRISP-ML(Q)

| Fase | Contenido | Dónde está |
|---|---|---|
| 1. Comprensión del negocio y datos | Definición del problema y variables | `notebook/*.ipynb` |
| 2. Preparación de datos | Limpieza, codificación, escalado | `notebook/*.ipynb` |
| 3. Modelado | Regresión Logística, Árbol de Decisión, Random Forest | `notebook/*.ipynb` |
| 4. Evaluación | F1-macro, matriz de confusión, validación cruzada | `notebook/*.ipynb` |
| 5. Despliegue | Pipeline serializado + capa de recomendación | `app.py` |
| 6. Monitoreo | Estrategia de reentrenamiento y control de *drift* | Documentado en el notebook |

**Métricas del modelo (conjunto de prueba):** F1-macro ≈ 0.84 · Recall clase `Bajo` ≈ 0.87.

---

## ▶️ Ejecutar localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/siaap.git
cd siaap

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la app
streamlit run app.py
```

La app abrirá en `http://localhost:8501`.

Para explorar el desarrollo completo del modelo, abre el notebook con Jupyter o VS Code:

```bash
pip install notebook
jupyter notebook notebook/proyecto_crisp_ml_rendimiento_academico.ipynb
```

---

## ☁️ Desplegar en GitHub + Streamlit Community Cloud

1. **Crear el repositorio en GitHub** (desde github.com → "New repository", por ejemplo `siaap`).
2. **Subir los archivos** de este proyecto:
   ```bash
   git init
   git add .
   git commit -m "Primera versión del sistema SIAAP"
   git branch -M main
   git remote add origin https://github.com/<tu-usuario>/siaap.git
   git push -u origin main
   ```
3. **Desplegar en Streamlit Cloud** (gratis):
   - Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
   - Clic en **"New app"** → selecciona el repositorio `siaap`, la rama `main` y el archivo `app.py`.
   - Clic en **"Deploy"**. En unos minutos tendrás una URL pública tipo
     `https://siaap-<hash>.streamlit.app`.
4. **Enlazar la landing page**: puedes publicar `landing_page.html` gratis con **GitHub Pages**
   (Settings → Pages → seleccionar la rama `main`) y desde ahí colocar el botón/enlace hacia la URL
   de tu app de Streamlit Cloud, y viceversa.

---

## 🤖 Sobre el flujo con Google Antigravity, Colab y NotebookLM

Estas herramientas dependen de tu cuenta personal, así que no puedo operarlas por ti desde este chat.
Aquí tienes el flujo recomendado para cada una:

- **Google Colab** (no NotebookLM) es la herramienta correcta para *ejecutar* el `.ipynb` conectado a
  tu cuenta de Google: sube el archivo a tu Drive o ábrelo directamente desde Colab
  (`Archivo → Subir notebook`), y quedará vinculado a `alisandromadediaz618@gmail.com`.
- **NotebookLM** no ejecuta código; es un asistente de análisis de documentos. Puedes subirle este
  `README.md` y el notebook exportado como PDF (`Archivo → Descargar → PDF` desde Jupyter/Colab) para
  que te genere resúmenes, mapas mentales o un audio explicativo del proyecto.
- **Google Antigravity** es un IDE de escritorio (se instala en tu computadora, no vive en este chat).
  El flujo sería: descargar este repositorio → abrirlo como carpeta de proyecto en Antigravity →
  pedirle al agente, por ejemplo: *"Revisa este repo Streamlit, corrígelo si algo falla y despliégalo
  en mi GitHub"*. El agente puede ejecutar comandos de `git`/`streamlit` directamente en tu máquina,
  algo que yo no puedo hacer desde aquí.

---

## 🖼️ Imágenes / elementos visuales tipo IA para la landing

La landing page ya incluye un elemento gráfico propio (gráfico SVG animado de progreso del
estudiante) para no depender de imágenes externas. Si quieres imágenes generadas por IA para dar más
impacto visual, algunas opciones que puedes usar tú mismo y luego colocar en una carpeta `assets/`:

- **Google AI Studio / Nano Banana** (con tu cuenta de Google, gratis).
- **Gemini** o **ChatGPT** (generación de imágenes desde el chat).
- Prompts sugeridos: *"ilustración digital minimalista de un estudiante estudiando con gráficos de
  progreso flotando alrededor, paleta azul marino y verde, estilo flat design"*, o
  *"interfaz futurista de panel educativo con IA, estilo isométrico, colores #2B4C7E y #2F8F5B"*.

Una vez generadas, colócalas en `assets/` y referencia la ruta en `landing_page.html`
(por ejemplo `<img src="assets/hero.png" alt="...">`); puedo ayudarte a integrarlas en el diseño
cuando las tengas.

---

## 📊 Variables utilizadas por el modelo

| Variable | Descripción |
|---|---|
| `horas_estudio_semanal` | Horas de estudio por semana |
| `asistencia_pct` | Porcentaje de asistencia a clases |
| `promedio_anterior` | Promedio del periodo anterior (0–10) |
| `participacion_clase` | Nivel de participación en clase (0–10) |
| `uso_plataforma_horas` | Horas semanales de uso de la plataforma virtual |
| `entregas_a_tiempo_pct` | Porcentaje de tareas entregadas a tiempo |
| `horas_sueno` | Horas promedio de sueño diario |
| `apoyo_familiar` | Nivel de apoyo familiar percibido (Bajo/Medio/Alto) |

---

## 📄 Licencia

Proyecto académico de uso educativo. Ajusta esta sección con la licencia que prefieras
(por ejemplo, MIT) antes de hacer público el repositorio.
