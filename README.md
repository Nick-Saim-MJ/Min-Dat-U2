# 📚 Minería de Datos — Sistema CRISP‑DM Completo

**Dataset:** Student Performance (UCI)  
**Autor:** Milton Edward Humpiri Flores  
**Curso:** Minería de Datos – Unidad 2 – Examen Final

---

## ✨ Visión General

Esta aplicación Streamlit implementa todo el flujo **CRISP‑DM** (Cross‑Industry Standard Process for Data Mining) para predecir si un estudiante aprobará el curso final a partir de información demográfica, social y académica.

- **Ingesta y limpieza** con detección automática de *data leakage* (exclusión de variables G1 y G2).
- **Pre‑procesado** robusto con `StandardScaler` y `LabelEncoder` ajustados **solo** en el conjunto de entrenamiento.
- **Partición estratificada** (train / validation / test) que preserva la distribución de clases.
- **Modelado** con Decision Tree, Random Forest, y opciones opcionales (XGBoost, LightGBM, K‑NN, Naïve Bayes).
- **Clustering** (K‑Means y Jerárquico) con visualizaciones de perfiles de clústeres sin sub‑plots vacíos.
- **Reportes PDF** automatizados usando ReportLab.
- **Dashboard interactivo** con métricas (Accuracy, F1‑Score, AUC‑ROC, precisión, sensibilidad) y comparativas visuales.

---

## 🛠️ Características Principales

- **Detección automática de fuga de datos** (`detect_leakage`).
- **Pre‑procesado dinámico** (`auto_preprocess`) con fallback cuando el estratificado no es posible.
- **Visualizaciones premium**:
  - Perfiles de clústeres adaptativos (sin espacios en blanco).
  - Dendrogramas y proyecciones PCA.
  - Curvas ROC superpuestas.
  - Dashboard final con métricas y silhouette scores.
- **Exportación a PDF** con estilo corporativo (colores #1a3c6e, #2980b9, #27ae60).
- **Compatibilidad opcional** con XGBoost, LightGBM y Plotly (detectados en tiempo de ejecución).

---

## 📦 Instalación

```bash
# Clonar el repositorio (si está alojado en Git)
git clone <repo-url>
cd "Examen Final"

# Crear entorno virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** Los paquetes opcionales (`xgboost`, `lightgbm`, `plotly`) se instalarán automáticamente si están incluidos en `requirements.txt` y tu entorno los soporta.

---

## ▶️ Uso rápido

```bash
streamlit run app_examen_corregido.py
```

- **Barra lateral** → Configura parámetros de modelo (árbol, bosques, número de clústeres, etc.).
- **Sección "Exploración"** → Visualiza distribución del target y correlaciones.
- **Sección "Clustering"** → Selecciona K‑Means o Jerárquico y observa perfiles de clúster.
- **Sección "Modelado"** → Entrena, evalúa y compara modelos; descarga reporte PDF.

---

## 📸 Visuales

![Streamlit Dashboard Mockup](file:///C:/Users/AORUS/.gemini/antigravity/brain/d780c26c-a2c5-405f-b99c-9fcd2eb1df8a/artifacts/streamlit_dashboard_mockup.png)

---

## 🧩 Pipeline de Modelado (`auto_preprocess`)

1. **Elimina duplicados y nulos** en la variable objetivo.
2. **Detecta y elimina columnas** con >50 % de valores nulos o marcadas para `drop`.
3. **Encodea** el target con `LabelEncoder`.
4. **Particiona** usando `train_test_split` con `stratify` cuando haya al menos 2 clases y 2 muestras por clase.
5. **Imputación** (mediana para numéricos, moda para categóricos).
6. **One‑Hot o Label Encoding** de variables categóricas según `profile[col]["encoding"]`.
7. **Escalado** con `StandardScaler` ajustado únicamente en el conjunto de entrenamiento.
8. **Registro de logs** informativos para auditoría.

---

## 🤝 Contribuir

1. Fork el repositorio.
2. Crea una rama con tu feature (`git checkout -b feature/mi-mejora`).
3. Realiza cambios y asegura que los tests pasen (`pytest` – si existieran).
4. Envía un Pull Request describiendo la mejora.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulte el archivo `LICENSE` para más detalles.
