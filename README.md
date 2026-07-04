# 🎓 SIAAP — Sistema Inteligente de Apoyo al Aprendizaje Personalizado

Proyecto de **Machine Learning supervisado** (Clasificación + Regresión) desarrollado con la metodología **CRISP-ML(Q)**, que predice el nivel de rendimiento académico de un estudiante (`Bajo` / `Medio` / `Alto`) y su nota final estimada (escala `0.0` a `10.0`) a partir de variables de comportamiento de estudio, para disparar **recomendaciones de apoyo personalizadas**.

> Proyecto académico educativo. Los datos utilizados son **sintéticos**, generados con fines demostrativos.

---

## 📁 Estructura del repositorio

```
siaap/
├── app.py                    # App de Streamlit (demo interactiva del modelo dual)
├── requirements.txt          # Dependencias para ejecutar la app
├── assets/
│   ├── ar_education_hero.jpg # Imagen de Realidad Aumentada (Hero) generada por IA
│   └── ar_dashboard_preview.jpg # Imagen del panel educativo en RA generada por IA
├── .streamlit/
│   └── config.toml           # Tema visual de la app (colores del proyecto)
├── notebook/
│   └── proyecto_crisp_ml_rendimiento_academico.ipynb   # Desarrollo completo CRISP-ML(Q)
├── landing_page.html         # Landing page explicativa del proyecto
└── README.md
```

---

## 🧭 Metodología: CRISP-ML(Q)

| Fase | Contenido | Dónde está |
|---|---|---|
| 1. Comprensión del negocio y datos | Definición del problema y variables (Regresión + Clasificación) | `notebook/*.ipynb` |
| 2. Preparación de datos | Limpieza, codificación, escalado, splits para ambos objetivos | `notebook/*.ipynb` |
| 3. Modelado | Regresión Lineal (Nota continua) y Regresión Logística (Riesgo) | `notebook/*.ipynb` |
| 4. Evaluación | Métricas R² y MAE (Regresión) / F1-macro y Recall (Clasificación) | `notebook/*.ipynb` |
| 5. Despliegue | Pipeline serializado + capa de recomendación en Streamlit | `app.py` |
| 6. Monitoreo | Estrategia de reentrenamiento y control de *drift* | Documentado en el notebook |

**Métricas del modelo (conjunto de prueba):**
- **Regresión Lineal (Nota continua):** Coeficiente $R^2 \approx 0.86$ · Error Medio Absoluto (MAE) $\approx 0.42$ puntos.
- **Regresión Logística (Clasificación):** $F1\text{-macro} \approx 0.84$ · Recall clase `Bajo` $\approx 0.87$.

---

## ▶️ Ejecutar localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/siaap.git
cd siaap

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la app
streamlit run app.py
```

La app abrirá automáticamente en tu navegador predeterminado (normalmente en `http://localhost:8501`).

Para explorar el desarrollo completo del modelo, abre el notebook con Jupyter Notebook o VS Code:

```bash
pip install notebook
jupyter notebook notebook/proyecto_crisp_ml_rendimiento_academico.ipynb
```

---

## 🖼️ Imágenes de Realidad Aumentada (RA) Generadas por IA

Para mejorar el impacto visual del proyecto en su **Landing Page** (`landing_page.html`), se integraron dos imágenes temáticas de Realidad Aumentada aplicada a la educación generadas por IA de Antigravity:

1. **`assets/ar_education_hero.jpg`**: Muestra a un estudiante utilizando lentes de RA para estudiar gráficos tridimensionales en un aula inteligente interactiva.
2. **`assets/ar_dashboard_preview.jpg`**: Muestra una vista en primer plano de un panel interactivo holográfico en 3D que proyecta la analítica predictiva del rendimiento del alumno.

> 💡 *Nota para la copia local:* Las imágenes generadas se guardan en el directorio de conversación de Antigravity. Para verlas en la Landing Page local, asegúrate de crear la carpeta `assets/` y colocarlas ahí con los nombres exactos `ar_education_hero.jpg` y `ar_dashboard_preview.jpg`.

---

## 📄 Licencia

Proyecto académico de uso educativo. Licencia MIT.
