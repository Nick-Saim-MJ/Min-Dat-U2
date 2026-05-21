"""
Minería de Datos — Sistema CRISP-DM Completo
Dataset: Student Performance (UCI)
Docente: MILTON EDWARD HUMPIRI FLORES | Curso: Minería de Datos
Ejecutar: streamlit run app_examen.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import io, os, tempfile, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              roc_auc_score, roc_curve, silhouette_score,
                              silhouette_samples, precision_score, recall_score,
                              mean_absolute_error, mean_squared_error)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ReportLab para PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable, Image as RLImage)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Intentar importar extras opcionales
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Minería de Datos — CRISP-DM | Estudiantes",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded"
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DE ESTILO PDF
# ══════════════════════════════════════════════════════════════════════════════
PDF_PRIMARY    = colors.HexColor("#1a3c6e")
PDF_SECONDARY  = colors.HexColor("#2980b9")
PDF_ACCENT     = colors.HexColor("#27ae60")
PDF_WARNING    = colors.HexColor("#e74c3c")
PDF_LIGHT_BG   = colors.HexColor("#eaf2ff")
PDF_GRAY       = colors.HexColor("#7f8c8d")

# ── Textos teóricos ─────────────────────────────────────────────────────────────
EXP = {
"data_leakage": """**¿Qué es el Data Leakage?**
El data leakage ocurre cuando información del conjunto de prueba (o del futuro)
se filtra al entrenamiento, causando que el modelo parezca más preciso de lo real.

**En el dataset Student Performance:**
- `G1` y `G2` son notas de períodos intermedios con correlación ≥ 0.85 con G3.
- En un escenario real de campaña educativa, NO tendríamos G1/G2 al momento de predecir.
- **Medidas aplicadas:** StandardScaler ajustado **solo en train**, LabelEncoder aprende solo del train,
  partición **estratificada** para preservar la proporción de clases.""",

"split": """**Partición Train / Validación / Prueba**
| Conjunto | Uso | Tamaño |
|----------|-----|--------|
| **Train** (65%) | Ajusta los parámetros del modelo | ~422 registros |
| **Validación** (15%) | Selección de hiperparámetros | ~97 registros |
| **Test** (20%) | Evaluación final — **solo se usa una vez** | ~130 registros |

La partición es **estratificada** (`stratify=y`) para mantener la proporción de clases.""",

"baseline": """**Modelo Baseline — DummyClassifier (Most Frequent)**
Predice siempre la clase mayoritaria. Es el umbral mínimo que cualquier modelo útil debe superar.
Si un modelo no supera al baseline en F1 y AUC, no aporta valor real.""",

"kmeans": """**K-Means Clustering**
Algoritmo iterativo que agrupa n registros en K clústeres minimizando la *inercia*.
**Método del codo:** elegimos K donde la reducción de inercia se estabiliza.
**Limitación:** asume clústeres esféricos y requiere definir K a priori.""",

"hierarchical": """**Clustering Jerárquico Aglomerativo (Ward)**
Parte con cada registro como su propio clúster y los va fusionando.
El criterio **Ward** minimiza la varianza interna.
El **dendrograma** visualiza la jerarquía; cortando a cierta altura obtenemos K grupos.
**Ventaja:** no requiere definir K antes.""",

"silhouette": """**Índice Silhouette** ∈ [-1, 1]
- **≈ +1**: el punto está bien asignado y lejos de otros clústeres.
- **≈ 0**: punto en el borde entre dos clústeres.
- **< 0**: el punto probablemente está en el clúster equivocado.
Valores bajos son normales en datos de comportamiento (perfiles continuos).""",

"decision_tree": """**Árbol de Decisión (CART)**
Modelo interpretable que divide el espacio de features con reglas binarias.
**Ventaja:** altamente interpretable. **Limitación:** propenso a overfitting sin poda.""",

"random_forest": """**Random Forest**
Ensemble de N árboles entrenados con muestras aleatorias (**bagging**) y subconjuntos de features.
**Ventaja:** robusto, maneja no-linealidad, provee importancia de variables.""",

"roc": """**Curva ROC (Receiver Operating Characteristic)**
Muestra la tasa de VP vs FP para todos los umbrales posibles.
**AUC:** 1.0 = perfecto, 0.5 = aleatorio, <0.5 = peor que azar.""",

"confusion_matrix": """**Matriz de Confusión**
| Celda | Nombre | Descripción |
|-------|--------|-------------|
| VP ✅ | Verdaderos Positivos | Predijo positivo y era positivo |
| VN ✅ | Verdaderos Negativos | Predijo negativo y era negativo |
| FP ⚠️ | Falso Positivo — Error Tipo I | Predijo positivo pero era negativo |
| FN ⚠️ | Falso Negativo — Error Tipo II | Predijo negativo pero era positivo |

**Para este problema educativo:** FN es más costoso (estudiante en riesgo no detectado).""",
}

# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE ARCHIVOS
# ══════════════════════════════════════════════════════════════════════════════
def load_any_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    raw  = uploaded.read()
    buf  = io.BytesIO(raw)
    if name.endswith(".csv"):
        sample = raw[:4096].decode("utf-8", errors="ignore")
        sep = ";" if sample.count(";") > sample.count(",") else ","
        buf.seek(0)
        return pd.read_csv(buf, sep=sep, encoding="utf-8", on_bad_lines="skip")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    elif name.endswith(".json"):
        return pd.read_json(buf)
    else:
        buf.seek(0)
        return pd.read_csv(buf, sep=";", on_bad_lines="skip")


# ══════════════════════════════════════════════════════════════════════════════
#  PERFILADO & PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
def profile_dataframe(df: pd.DataFrame) -> dict:
    n = len(df)
    profile = {}
    for col in df.columns:
        s = df[col]
        n_null   = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        is_num   = pd.api.types.is_numeric_dtype(s)

        if n_unique == 2:
            col_type = "binary"
        elif is_num and n_unique > 20 and n_unique / n > 0.05:
            col_type = "continuous"
        elif is_num:
            col_type = "discrete"
        elif s.dtype == object and n_unique > max(100, 0.5 * n):
            col_type = "high_cardinality"
        else:
            col_type = "categorical"

        impute   = ("none" if n_null == 0 else
                    "median" if is_num and abs(s.dropna().skew()) > 1 else
                    "mean" if is_num else "mode")
        encoding = ("none"  if col_type in ("continuous", "discrete") else
                    "drop"  if col_type == "high_cardinality" else
                    "label" if col_type == "binary" else
                    "onehot" if n_unique <= 8 else "label")

        profile[col] = dict(col_type=col_type, is_num=is_num,
                            n_null=n_null, null_pct=round(n_null / n * 100, 2),
                            n_unique=n_unique, impute=impute, encoding=encoding)
    return profile


def detect_leakage(df, target, profile):
    leakage_cols, logs = [], []
    y = df[target]
    try:
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        is_binary = len(le.classes_) == 2
        for col in df.columns:
            if col == target: continue
            p = profile[col]
            if p["null_pct"] > 50 or p["encoding"] == "drop": continue
            dt = DecisionTreeClassifier(max_depth=1, random_state=42)
            X_col = df[[col]].copy()
            try:
                if p["is_num"]:
                    X_col = X_col.fillna(X_col.median())
                else:
                    X_col = pd.get_dummies(X_col[col])
                dt.fit(X_col, y_enc)
                preds = dt.predict_proba(X_col)
                auc = roc_auc_score(y_enc, preds[:, 1] if is_binary else preds,
                                    multi_class="raise" if is_binary else "ovr")
                if auc > 0.95:
                    leakage_cols.append(col)
                    logs.append(f"🚨 **Leakage:** `{col}` AUC={auc:.3f} → excluida")
            except Exception:
                pass
    except Exception:
        pass
    return leakage_cols, logs


def auto_preprocess(df, target, profile, val_pct=0.15, test_pct=0.20, manual_drop=None):
    log  = []
    work = df.copy()

    dup_before = len(work)
    work = work.dropna(subset=[target]).drop_duplicates()
    removed = dup_before - len(work)
    if removed > 0:
        log.append(f"🧹 Eliminados {removed:,} duplicados/nulos en target")

    y = work.pop(target).reset_index(drop=True)
    work.reset_index(drop=True, inplace=True)

    leak_cols, leak_logs = detect_leakage(df, target, profile)
    log.extend(leak_logs)

    drop = [c for c, p in profile.items()
            if c != target and (p["null_pct"] > 50 or p["encoding"] == "drop")]
    drop = list(set(drop + leak_cols + (manual_drop or [])))
    if drop:
        work.drop(columns=[c for c in drop if c in work.columns], inplace=True)
        log.append(f"🗑️ Eliminadas: `{'`, `'.join(drop)}`")

    le_target = LabelEncoder()
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(le_target.fit_transform(y.astype(str)), name=target)
        log.append(f"🎯 Target `{target}` codificado: {list(le_target.classes_)}")

    # Ensure stratification when possible (both classes present and at least 2 samples per class)
    stratify = y if y.nunique() >= 2 and y.value_counts().min() >= 2 else None
    try:
        X_work, X_test, y_work, y_test = train_test_split(
            work, y, test_size=test_pct, random_state=42, stratify=stratify)
    except ValueError:
        X_work, X_test, y_work, y_test = train_test_split(
            work, y, test_size=test_pct, random_state=42)

    if val_pct > 0:
        val_ratio = val_pct / (1 - test_pct)
        # Use same stratify logic for validation split
        strat2 = y_work if y_work.nunique() >= 2 and y_work.value_counts().min() >= 2 else None
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_work, y_work, test_size=val_ratio, random_state=42, stratify=strat2)
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X_work, y_work, test_size=val_ratio, random_state=42)
    else:
        X_train, X_val = X_work.copy(), pd.DataFrame(columns=X_work.columns)
        y_train, y_val = y_work.copy(), pd.Series([], dtype=y_work.dtype)

    # Warn if test set has only one class (can cause ambiguous metrics)
    if y_test.nunique() < 2:
        log.append("⚠️ Test set contiene una sola clase; métricas pueden ser ambiguas.")

    log.append(f"✂️ Split → train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    splits_all = [s for s in [X_train, X_val, X_test] if len(s) > 0]

    if num_cols and X_train[num_cols].isna().any().any():
        imp_n = SimpleImputer(strategy="median").fit(X_train[num_cols])
        for s in splits_all:
            s[num_cols] = imp_n.transform(s[num_cols])

    if cat_cols and X_train[cat_cols].isna().any().any():
        imp_c = SimpleImputer(strategy="most_frequent").fit(X_train[cat_cols])
        for s in splits_all:
            s[cat_cols] = imp_c.transform(s[cat_cols])

    ohe_done, le_done = [], []
    for col in cat_cols:
        enc = profile.get(col, {}).get("encoding", "label")
        if enc == "onehot":
            dummies_tr = pd.get_dummies(X_train[col], prefix=col, drop_first=True, dtype=int)
            X_train = pd.concat([X_train.drop(columns=[col]), dummies_tr], axis=1)
            for s in [X_val, X_test]:
                if len(s) > 0:
                    d = pd.get_dummies(s[col], prefix=col, drop_first=True, dtype=int)
                    for c in dummies_tr.columns:
                        if c not in d.columns:
                            d[c] = 0
                    s.drop(columns=[col], inplace=True)
                    for c in dummies_tr.columns:
                        s[c] = d[c].values
            ohe_done.append(col)
        else:
            le = LabelEncoder().fit(X_train[col].astype(str))
            X_train[col] = X_train[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1)
            
            for s in [X_val, X_test]:
                if len(s) > 0:
                    s[col] = s[col].astype(str).map(
                        lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1)
                    
            le_done.append(col)

    if ohe_done:
        log.append(f"🔠 OneHot → `{'`, `'.join(ohe_done)}`")
    if le_done:
        log.append(f"🔢 Label  → `{'`, `'.join(le_done)}`")

    feats  = X_train.columns.tolist()
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=feats)
    X_test  = pd.DataFrame(scaler.transform(X_test),  columns=feats)
    if len(X_val) > 0:
        X_val = pd.DataFrame(scaler.transform(X_val), columns=feats)
    else:
        X_val = pd.DataFrame(columns=feats)
    log.append("📏 StandardScaler ajustado solo con train → sin data leakage")

    return X_train, X_val, X_test, y_train, y_val, y_test, feats, log
    log.append(f"✂️ Split → train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    splits_all = [s for s in [X_train, X_val, X_test] if len(s) > 0]

    if num_cols and X_train[num_cols].isna().any().any():
        imp_n = SimpleImputer(strategy="median").fit(X_train[num_cols])
        for s in splits_all:
            s[num_cols] = imp_n.transform(s[num_cols])

    if cat_cols and X_train[cat_cols].isna().any().any():
        imp_c = SimpleImputer(strategy="most_frequent").fit(X_train[cat_cols])
        for s in splits_all:
            s[cat_cols] = imp_c.transform(s[cat_cols])

    ohe_done, le_done = [], []
    for col in cat_cols:
        enc = profile.get(col, {}).get("encoding", "label")
        if enc == "onehot":
            dummies_tr = pd.get_dummies(X_train[col], prefix=col, drop_first=True, dtype=int)
            X_train = pd.concat([X_train.drop(columns=[col]), dummies_tr], axis=1)
            for s in [X_val, X_test]:
                if len(s) > 0:
                    d = pd.get_dummies(s[col], prefix=col, drop_first=True, dtype=int)
                    for c in dummies_tr.columns:
                        if c not in d.columns: d[c] = 0
                    idx = s.index
                    s.drop(columns=[col], inplace=True)
                    for c in dummies_tr.columns:
                        s[c] = d[c].values
            ohe_done.append(col)
        else:
            le = LabelEncoder().fit(X_train[col].astype(str))
            X_train[col] = X_train[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1)
            for s in [X_val, X_test]:
                if len(s) > 0:
                    s[col] = s[col].astype(str).map(
                        lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1)
            le_done.append(col)

    if ohe_done: log.append(f"🔠 OneHot → `{'`, `'.join(ohe_done)}`")
    if le_done:  log.append(f"🔢 Label  → `{'`, `'.join(le_done)}`")

    feats  = X_train.columns.tolist()
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=feats)
    X_test  = pd.DataFrame(scaler.transform(X_test),  columns=feats)
    if len(X_val) > 0:
        X_val = pd.DataFrame(scaler.transform(X_val), columns=feats)
    else:
        X_val = pd.DataFrame(columns=feats)
    log.append("📏 StandardScaler ajustado solo con train → sin data leakage")

    return X_train, X_val, X_test, y_train, y_val, y_test, feats, log


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
def compute_all_metrics(y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred)
    binary = cm.shape == (2, 2)

    if binary:
        VN, FP, FN, VP = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    else:
        VP = int(np.diag(cm).sum())
        FP = int((cm.sum(axis=0) - np.diag(cm)).sum())
        FN = int((cm.sum(axis=1) - np.diag(cm)).sum())
        VN = int(cm.sum()) - VP - FP - FN

    total = VP + FP + FN + VN
    safe = lambda n, d: round(n/d, 4) if d else 0.0

    metrics = dict(
        VP=VP, FP=FP, FN=FN, VN=VN,
        Precisión      = safe(VP, VP+FP),
        Exactitud      = safe(VP+VN, total),
        Especificidad  = safe(VN, VN+FP),
        Sensibilidad   = safe(VP, VP+FN),
        Tasa_FN        = safe(FN, FN+VP),
        F1_Score       = safe(2*VP, 2*VP+FP+FN),
        AUC_ROC        = None,
        cm_matrix      = cm,
        binary         = binary,
    )
    if y_proba is not None:
        try:
            if binary:
                metrics["AUC_ROC"] = round(roc_auc_score(y_true, y_proba[:, 1]), 4)
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                metrics["roc"] = (fpr, tpr)
                metrics["roc_per_class"] = {"clase 1": (fpr, tpr, metrics["AUC_ROC"])}
            else:
                metrics["AUC_ROC"] = round(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"), 4)
        except Exception:
            pass
    return metrics


def calcular_metricas_cm(VN, FP, FN, VP):
    total = VN + FP + FN + VP
    safe = lambda n, d: round(n/d, 4) if d > 0 else 0.0
    return {
        "Exactitud (Accuracy)":   safe(VP+VN, total),
        "Precisión":              safe(VP, VP+FP),
        "Sensibilidad (Recall)":  safe(VP, VP+FN),
        "Especificidad":          safe(VN, VN+FP),
        "F1-Score":               safe(2*VP, 2*VP+FP+FN),
        "Tasa FN (Miss Rate)":    safe(FN, FN+VP),
        "Tasa FP (Fall-out)":     safe(FP, FP+VN),
        "VPP (Valor Pred. +)":    safe(VP, VP+FP),
        "VPN (Valor Pred. -)":    safe(VN, VN+FN),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZACIONES (matplotlib — para PDF y Streamlit)
# ══════════════════════════════════════════════════════════════════════════════
def fig_target_distribution(df, target_col):
    vc = df[target_col].value_counts()
    pct = df[target_col].value_counts(normalize=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors_t = ["#e74c3c", "#2ecc71"][:len(vc)]
    labels_t = [str(v) for v in vc.index]
    axes[0].bar(labels_t, vc.values, color=colors_t, edgecolor="white", width=0.5)
    for i, (v, p) in enumerate(zip(vc.values, pct.values)):
        axes[0].text(i, v + 5, f"{v}\n({p:.1f}%)", ha="center", fontweight="bold", fontsize=9)
    axes[0].set_title(f"Distribución del Target — {target_col}", fontweight="bold")
    axes[0].set_ylabel("Nº de registros")
    axes[0].set_ylim(0, max(vc.values) * 1.25)
    axes[1].pie(vc.values, labels=labels_t, autopct="%1.1f%%",
                colors=colors_t, startangle=90,
                wedgeprops={"edgecolor": "white"})
    axes[1].set_title("Proporción de Clases", fontweight="bold")
    plt.tight_layout()
    return fig


def fig_numeric_distributions(df, num_cols, target_col, n_cols=3):
    n = min(len(num_cols), 9)
    cols_show = num_cols[:n]
    rows_n = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(rows_n, n_cols, figsize=(14, rows_n * 3.5))
    axes = np.array(axes).flatten()
    for i, c in enumerate(cols_show):
        ax = axes[i]
        ax.hist(df[c].dropna(), bins=25, color="#3498db", edgecolor="white", alpha=0.8)
        series = pd.to_numeric(df[c], errors="coerce").dropna()
        ax.axvline(series.mean(), color="red", linestyle="--", linewidth=1.5, label="Media")
        ax.axvline(series.median(), color="orange", linestyle=":", linewidth=1.5, label="Mediana")
        ax.set_title(f"{c}  (sesgo={series.skew():.2f})", fontweight="bold", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    for j in range(len(cols_show), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribuciones Numéricas", fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig


def fig_boxplots_by_target(df, num_cols, target_col, n_cols=3):
    n = min(len(num_cols), 6)
    cols_show = num_cols[:n]
    rows_n = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(rows_n, n_cols, figsize=(14, rows_n * 3.5))
    axes = np.array(axes).flatten()
    df_p = df.copy()
    df_p[target_col] = df_p[target_col].astype(str)
    classes = sorted(df_p[target_col].unique())
    colors_b = ["#e74c3c", "#2ecc71", "#3498db"]
    for i, c in enumerate(cols_show):
        ax = axes[i]
        data_by_class = [df_p.loc[df_p[target_col] == cl, c].dropna().values for cl in classes]
        bp = ax.boxplot(data_by_class, labels=classes, patch_artist=True, notch=False)
        for patch, color in zip(bp["boxes"], colors_b):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"{c} vs {target_col}", fontweight="bold", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    for j in range(len(cols_show), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"Variables Numéricas vs Target — {target_col}", fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig


def fig_correlation_heatmap(df, num_cols):
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 0.7), max(6, len(num_cols) * 0.6)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax,
                mask=mask, annot_kws={"size": 8})
    ax.set_title("Matriz de Correlación de Pearson", fontweight="bold")
    plt.tight_layout()
    return fig


def fig_elbow_silhouette(X_clust, K_range=range(2, 11)):
    inertias, silhouettes = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_clust)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_clust, km.labels_,
                               sample_size=min(500, len(X_clust)),
                               random_state=RANDOM_STATE)
        silhouettes.append(sil)

    best_k_sil = list(K_range)[silhouettes.index(max(silhouettes))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(list(K_range), inertias, "bo-", linewidth=2, markersize=8)
    axes[0].axvline(x=4, color="red", linestyle="--", alpha=0.7, label="K=4 (codo aprox.)")
    axes[0].set_title("Método del Codo — Inercia", fontweight="bold")
    axes[0].set_xlabel("Número de Clústeres (K)")
    axes[0].set_ylabel("Inercia")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(list(K_range), silhouettes, "rs-", linewidth=2, markersize=8, color="darkgreen")
    axes[1].axvline(x=best_k_sil, color="orange", linestyle="--",
                    label=f"Mejor K silhouette = {best_k_sil}")
    axes[1].set_title("Silhouette Score por K", fontweight="bold")
    axes[1].set_xlabel("Número de Clústeres (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Selección del número óptimo de clústeres", fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig, inertias, silhouettes, best_k_sil


def fig_silhouette_detail(X_arr, labels, title="Silhouette por Clúster"):
    sil_vals = silhouette_samples(X_arr, labels)
    k = len(np.unique(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    fig, ax = plt.subplots(figsize=(9, max(4, k * 1.3)))
    y_lower = 10
    for i in range(k):
        vals_i = np.sort(sil_vals[labels == i])
        size_i = vals_i.shape[0]
        y_upper = y_lower + size_i
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_i,
                         color=colors[i], alpha=0.75, label=f"Clúster {i}")
        ax.text(-0.06, y_lower + size_i / 2, f"C{i}", ha="right",
                fontsize=10, fontweight="bold", color=colors[i])
        y_lower = y_upper + 8

    sil_mean = sil_vals.mean()
    ax.axvline(x=sil_mean, color="red", linestyle="--", linewidth=1.5,
               label=f"Media = {sil_mean:.4f}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Coeficiente Silhouette")
    ax.set_ylabel("Índice de muestra")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_pca_scatter(X_arr, labels, title="Proyección PCA 2D", cluster_centers=None):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_arr)
    var_exp = pca.explained_variance_ratio_
    k = len(np.unique(labels))
    colors_c = plt.cm.tab10(np.linspace(0, 1, k))

    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(k):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors_c[i]], label=f"Clúster {i}",
                   alpha=0.5, s=40, edgecolors="white", linewidth=0.3)

    if cluster_centers is not None:
        centroids_pca = pca.transform(cluster_centers)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   c="black", marker="X", s=200, zorder=5, label="Centroides")

    ax.set_title(f"{title}\nVarianza explicada: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}%",
                 fontweight="bold")
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig


def fig_cluster_profiles(profile_df, key_vars, k, title="Perfiles de Clústeres"):
    """Generate cluster profile bar charts.

    Parameters
    ----------
    profile_df : pd.DataFrame
        DataFrame containing aggregated cluster statistics. Rows correspond to clusters.
    key_vars : list of str
        Variables/columns to plot for each cluster.
    k : int
        Expected number of clusters (e.g., from the clustering algorithm).
    title : str, optional
        Plot title.
    """
    # Determine actual number of clusters present to avoid indexing errors
    actual_k = min(k, len(profile_df))
    if actual_k == 0:
        raise ValueError("profile_df is empty; cannot plot cluster profiles.")

    n_vars = len(key_vars)
    cols_n = 3
    rows_n = (n_vars + cols_n - 1) // cols_n
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(14, rows_n * 3.5))
    axes = np.array(axes).flatten()

    # Use a colormap that matches the actual number of clusters
    colors_c = plt.cm.tab10(np.linspace(0, 1, actual_k))
    cluster_names = [f"C{i}" for i in range(actual_k)]

    for idx, var in enumerate(key_vars):
        ax = axes[idx]
        # Extract values safely; missing clusters get NaN which we treat as 0
        vals = []
        for i in range(actual_k):
            try:
                vals.append(profile_df.iloc[i][var])
            except Exception:
                vals.append(0)
        # Ensure vals is numeric for proper scaling
        vals = np.nan_to_num(vals, nan=0.0)
        bars = ax.bar(cluster_names, vals, color=colors_c, edgecolor="white", alpha=0.85)
        ax.set_title(var, fontweight="bold", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # Hide any unused subplots
    for j in range(len(key_vars), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig
    n_vars = len(key_vars)
    cols_n = 3
    rows_n = (n_vars + cols_n - 1) // cols_n
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(14, rows_n * 3.5))
    axes = np.array(axes).flatten()
    colors_c = plt.cm.tab10(np.linspace(0, 1, k))
    cluster_names = [f"C{i}" for i in range(k)]

    for idx, var in enumerate(key_vars):
        ax = axes[idx]
        vals = [profile_df.iloc[i][var] if i < len(profile_df) else 0 for i in range(k)]
        bars = ax.bar(cluster_names, vals, color=colors_c[:k], edgecolor="white", alpha=0.85)
        ax.set_title(var, fontweight="bold", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    for j in range(len(key_vars), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig


def fig_dendrogram(X_clust, k_hc, sample_n=80):
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_clust), min(sample_n, len(X_clust)), replace=False)
    X_samp = X_clust[idx]
    Z = linkage(X_samp, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    ct = Z[-(k_hc - 1), 2] if k_hc > 1 and len(Z) >= k_hc else 0
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=7,
               color_threshold=ct, above_threshold_color="gray")
    if k_hc > 1 and len(Z) >= k_hc:
        cut_h = (Z[-(k_hc), 2] + Z[-(k_hc-1), 2]) / 2 if k_hc < len(Z) else Z[-1, 2] / 2
        ax.axhline(y=cut_h, color="red", linestyle="--", linewidth=1.5,
                   label=f"Corte → K={k_hc} (h≈{cut_h:.1f})")
        ax.legend()
    ax.set_title(f"Dendrograma — Clustering Jerárquico Ward (muestra {len(X_samp)} registros)",
                 fontweight="bold")
    ax.set_xlabel("Índice de muestra")
    ax.set_ylabel("Distancia (Ward)")
    plt.tight_layout()
    return fig


def fig_confusion_matrix_detail(y_true, y_pred, model_name, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    VN, FP, FN, VP = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    total = VN + FP + FN + VP

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0)

    labels = [
        [f"VN\n{VN}\n({VN/total*100:.1f}%)", f"FP (Error I)\n{FP}\n({FP/total*100:.1f}%)"],
        [f"FN (Error II)\n{FN}\n({FN/total*100:.1f}%)", f"VP\n{VP}\n({VP/total*100:.1f}%)"],
    ]
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_title(f"Matriz de Confusión\n{model_name}", fontweight="bold", fontsize=12)
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No aprobó (0)", "Aprobó (1)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No aprobó (0)", "Aprobó (1)"])
    plt.tight_layout()
    return fig, VN, FP, FN, VP


def fig_roc_comparison(models_roc_data):
    """models_roc_data: list of (name, fpr, tpr, auc)"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Azar — AUC = 0.50")
    colors_r = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, (name, fpr, tpr, auc) in enumerate(models_roc_data):
        ax.plot(fpr, tpr, color=colors_r[i % len(colors_r)], linewidth=2.5,
                label=f"{name} — AUC = {auc:.4f}")
    ax.set_title("Curvas ROC — Comparativa de Modelos", fontweight="bold", fontsize=13)
    ax.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def fig_metrics_bar_comparison(history):
    models = list(history.keys())
    accs  = [history[m].get("Accuracy", 0) for m in models]
    f1s   = [history[m].get("F1", 0)       for m in models]
    aucs  = [history[m].get("AUC-ROC", 0)  for m in models]

    x = np.arange(len(models))
    w = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    data_list = [("Accuracy", accs, "#3498db"), ("F1-Score", f1s, "#e74c3c"), ("AUC-ROC", aucs, "#2ecc71")]
    for ax, (metric, vals, color) in zip(axes, data_list):
        bars = ax.bar(x, vals, color=color, edgecolor="white", width=0.55, alpha=0.85)
        if metric == "AUC-ROC":
            ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Azar")
            ax.legend(fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Comparativa de Modelos — Métricas Principales", fontweight="bold", fontsize=13)
    plt.tight_layout()
    return fig


def fig_feature_importance_compare(model_a, model_b, name_a, name_b, features, n=12):
    if not (hasattr(model_a, "feature_importances_") and hasattr(model_b, "feature_importances_")):
        return None
    imp_a = pd.Series(model_a.feature_importances_, index=features)
    imp_b = pd.Series(model_b.feature_importances_, index=features)
    top_f = (imp_a + imp_b).sort_values(ascending=False).head(n).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, imp, name, color in [(axes[0], imp_a[top_f], name_a, "#3498db"),
                                  (axes[1], imp_b[top_f], name_b, "#e74c3c")]:
        imp.sort_values().plot(kind="barh", ax=ax, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"Importancia — {name}", fontweight="bold")
        ax.set_xlabel("Importancia (Gini)")
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Importancia de Variables — Comparativa", fontweight="bold", fontsize=12)
    plt.tight_layout()
    return fig


def fig_dashboard_final(history, roc_data, sil_km, sil_hc):
    models = list(history.keys())
    accs  = [history[m].get("Accuracy", 0) for m in models]
    f1s   = [history[m].get("F1", 0)       for m in models]
    aucs  = [history[m].get("AUC-ROC", 0)  for m in models]

    fig = plt.figure(figsize=(16, 10))
    colors_b = ["#95a5a6", "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    # 1 — Accuracy
    ax1 = fig.add_subplot(2, 3, 1)
    bars = ax1.bar(models, accs, color=colors_b[:len(models)], edgecolor="white", width=0.5)
    ax1.set_ylim(0, 1.15); ax1.set_title("Accuracy", fontweight="bold")
    ax1.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    for b, v in zip(bars, accs):
        ax1.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
                 ha="center", fontsize=8, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # 2 — F1
    ax2 = fig.add_subplot(2, 3, 2)
    bars2 = ax2.bar(models, f1s, color=colors_b[:len(models)], edgecolor="white", width=0.5)
    ax2.set_ylim(0, 1.15); ax2.set_title("F1-Score", fontweight="bold")
    ax2.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    for b, v in zip(bars2, f1s):
        ax2.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
                 ha="center", fontsize=8, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # 3 — AUC
    ax3 = fig.add_subplot(2, 3, 3)
    bars3 = ax3.bar(models, aucs, color=colors_b[:len(models)], edgecolor="white", width=0.5)
    ax3.axhline(0.5, color="gray", linestyle="--", lw=1, label="Azar")
    ax3.set_ylim(0, 1.15); ax3.set_title("AUC-ROC", fontweight="bold")
    ax3.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
    for b, v in zip(bars3, aucs):
        ax3.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
                 ha="center", fontsize=8, fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    # 4 — Curvas ROC
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot([0, 1], [0, 1], "k--", lw=1, label="Azar")
    pal = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for ci, (mnm, class_dict) in enumerate(roc_data.items()):
        if isinstance(class_dict, dict):
            for cls, (fpr_i, tpr_i, auc_i) in class_dict.items():
                ax4.plot(fpr_i, tpr_i, color=pal[ci % len(pal)], lw=2,
                         label=f"{mnm} (AUC={auc_i:.3f})")
                break
    ax4.set_xlim([0, 1]); ax4.set_ylim([0, 1.02])
    ax4.set_xlabel("Tasa FP", fontsize=9); ax4.set_ylabel("Tasa VP", fontsize=9)
    ax4.set_title("Curvas ROC", fontweight="bold")
    ax4.legend(fontsize=7, loc="lower right"); ax4.grid(alpha=0.3)

    # 5 — Tabla comparativa
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis("off")
    col_labels = ["Accuracy", "F1", "AUC"]
    row_data = [[f"{history[m].get('Accuracy',0):.3f}",
                 f"{history[m].get('F1',0):.3f}",
                 f"{history[m].get('AUC-ROC',0):.3f}"]
                for m in models]
    tbl = ax5.table(cellText=row_data, rowLabels=models,
                    colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    if aucs:
        best_idx = aucs.index(max(aucs))
        for j in range(3):
            tbl[(best_idx+1, j)].set_facecolor("#d5f5e3")
    ax5.set_title("Tabla Comparativa", fontweight="bold", pad=15)

    # 6 — Silhouette Clustering
    ax6 = fig.add_subplot(2, 3, 6)
    alg_names_s, sil_vals_s = [], []
    if sil_km: alg_names_s.append("K-Means"); sil_vals_s.append(sil_km)
    if sil_hc: alg_names_s.append("Jerárquico"); sil_vals_s.append(sil_hc)
    if alg_names_s:
        bars6 = ax6.bar(alg_names_s, sil_vals_s,
                        color=["#3498db", "#e74c3c"][:len(alg_names_s)],
                        edgecolor="white", width=0.4)
        ax6.set_ylim(0, max(sil_vals_s) * 1.4)
        for bar, v in zip(bars6, sil_vals_s):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                     f"{v:.4f}", ha="center", fontweight="bold")
    ax6.set_title("Silhouette — Clustering", fontweight="bold")
    ax6.grid(axis="y", alpha=0.3)

    plt.suptitle("Dashboard Final — Resumen del Análisis CRISP-DM\nMinería de Datos | MILTON EDWARD HUMPIRI FLORES",
                 fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DE GUARDADO DE FIGURAS
# ══════════════════════════════════════════════════════════════════════════════
def save_fig_to_bytes(fig, dpi=120) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def save_fig_to_tmp(fig, dpi=120) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADOR DE PDF (ReportLab)
# ══════════════════════════════════════════════════════════════════════════════
def build_pdf_report(session) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            title="Informe Minería de Datos",
                            author="Sistema CRISP-DM")
    styles = getSampleStyleSheet()

    # Estilos personalizados
    S_TITLE = ParagraphStyle("TitlePage", fontSize=22, leading=28, alignment=TA_CENTER,
                              textColor=PDF_PRIMARY, fontName="Helvetica-Bold")
    S_H1    = ParagraphStyle("H1", fontSize=14, leading=18, spaceBefore=16, spaceAfter=6,
                              textColor=PDF_PRIMARY, fontName="Helvetica-Bold")
    S_H2    = ParagraphStyle("H2", fontSize=12, leading=15, spaceBefore=10, spaceAfter=4,
                              textColor=PDF_SECONDARY, fontName="Helvetica-Bold")
    S_BODY  = ParagraphStyle("Body", fontSize=9.5, leading=13, spaceAfter=4,
                              fontName="Helvetica", alignment=TA_JUSTIFY)
    S_SMALL = ParagraphStyle("Small", fontSize=8.5, leading=11, textColor=PDF_GRAY,
                              fontName="Helvetica", spaceAfter=3)
    S_META  = ParagraphStyle("Meta", fontSize=11, leading=14, alignment=TA_CENTER,
                              textColor=PDF_GRAY, fontName="Helvetica")
    S_NOTE  = ParagraphStyle("Note", fontSize=8.5, leading=11, textColor=PDF_GRAY,
                              fontName="Helvetica-Oblique", leftIndent=12)

    story = []

    def hr(): return HRFlowable(width="100%", thickness=1, color=PDF_SECONDARY, spaceAfter=8)

    def add_image(fig, width=16*cm, caption=None):
        path = save_fig_to_tmp(fig)
        story.append(Spacer(1, 6))
        story.append(RLImage(path, width=width, height=width * 0.6))
        if caption:
            story.append(Paragraph(f"<i>{caption}</i>", S_NOTE))
        story.append(Spacer(1, 6))

    def section_table(data, col_widths=None, header_bg=PDF_PRIMARY):
        if not data or len(data) < 2:
            return
        t = Table(data, colWidths=col_widths, repeatRows=1)
        style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 9),
            ("ALIGN",      (0, 0), (-1, 0), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PDF_LIGHT_BG]),
            ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 1), (-1, -1), 8.5),
            ("ALIGN",      (1, 1), (-1, -1), "CENTER"),
            ("ALIGN",      (0, 1), (0, -1),  "LEFT"),
            ("GRID",       (0, 0), (-1, -1), 0.4, PDF_GRAY),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
        t.setStyle(style)
        story.append(t)
        story.append(Spacer(1, 8))

    # ─── PORTADA ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("UNIVERSIDAD", S_META))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("INFORME FINAL DE EXAMEN", S_META))
    story.append(Spacer(1, 0.8*cm))
    story.append(hr())
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Minería de Datos", S_TITLE))
    story.append(Paragraph("Sistema CRISP-DM — Análisis Completo", S_TITLE))
    story.append(Spacer(1, 0.5*cm))
    story.append(hr())
    story.append(Spacer(1, 1*cm))

    target_col = session.get("target", "passed")
    df_orig    = session.get("df")
    n_rows     = len(df_orig) if df_orig is not None else "N/A"
    n_feats    = len(session.get("feature_names", []))

    meta_data = [
        ["Dataset", "Student Performance (UCI) — Portugués"],
        ["Objetivo", f"Predecir si un estudiante aprobará ({target_col})"],
        ["Registros", str(n_rows)],
        ["Features", str(n_feats)],
        ["Docente", "MILTON EDWARD HUMPIRI FLORES"],
        ["Curso", "Minería de Datos"],
        ["Metodología", "CRISP-DM — Cross-Industry Standard Process for Data Mining"],
    ]
    for row in meta_data:
        story.append(Paragraph(f"<b>{row[0]}:</b>  {row[1]}", S_BODY))
    story.append(PageBreak())

    # ─── ÍNDICE ───────────────────────────────────────────────────────────────
    story.append(Paragraph("ÍNDICE", S_H1))
    story.append(hr())
    index_items = [
        "1.  Carga y Exploración de Datos (EDA)",
        "2.  Partición y Baseline — Data Leakage",
        "3.  Segmentación — K-Means y Clustering Jerárquico",
        "4.  Clasificación — Árbol de Decisión y Random Forest",
        "5.  Evaluación Comparativa de Modelos — Curva ROC",
        "6.  Matriz de Confusión con Métricas Completas",
        "7.  Conclusiones y Recomendaciones Gerenciales",
    ]
    for item in index_items:
        story.append(Paragraph(item, S_BODY))
    story.append(PageBreak())

    # ─── 1. EDA ───────────────────────────────────────────────────────────────
    story.append(Paragraph("1. Carga y Exploración de Datos (EDA)", S_H1))
    story.append(hr())
    story.append(Paragraph(
        "El dataset <b>Student Performance</b> (UCI) contiene información demográfica, social y "
        "académica de estudiantes de secundaria en Portugal. Se usa el archivo de Portugués "
        "(649 registros, 33 variables). El objetivo es predecir si un estudiante aprobará "
        "el curso (<i>passed = 1</i> si G3 ≥ 10).",
        S_BODY))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Variables clave del dataset:", S_H2))
    var_data = [["Variable", "Descripción"],
                ["school", "Escuela del estudiante (GP / MS)"],
                ["age", "Edad del estudiante"],
                ["failures", "Nº de materias reprobadas anteriormente"],
                ["studytime", "Horas de estudio semanales"],
                ["absences", "Nº de ausencias"],
                ["Dalc / Walc", "Consumo de alcohol (laboral / fin de semana)"],
                ["G1, G2", "Notas del 1er y 2do período (EXCLUIDAS — leakage)"],
                ["G3", "Nota final (origen del target passed)"],
                ["passed", "Target: 1 si G3 >= 10, 0 en caso contrario"]]
    section_table(var_data, col_widths=[4*cm, 12*cm])

    story.append(Paragraph("Calidad de Datos:", S_H2))
    if df_orig is not None:
        eda_data = [["Métrica", "Valor"],
                    ["Filas totales", str(len(df_orig))],
                    ["Columnas", str(len(df_orig.columns))],
                    ["Valores nulos", str(df_orig.isnull().sum().sum())],
                    ["Filas duplicadas", str(df_orig.duplicated().sum())],
                    ["Clases en target", str(df_orig[target_col].value_counts().to_dict()) if target_col in df_orig.columns else "N/A"]]
        section_table(eda_data, col_widths=[8*cm, 8*cm])

    # Gráfico EDA — distribución target
    fig_key = "pdf_fig_target"
    if df_orig is not None and target_col in df_orig.columns:
        fig_t = fig_target_distribution(df_orig, target_col)
        add_image(fig_t, caption=f"Figura 1.1 — Distribución del target '{target_col}'")

    # Numéricas
    if df_orig is not None:
        num_c = session.get("feature_names", [])
        if num_c:
            num_c_orig = [c for c in num_c if c in df_orig.columns][:9]
            if num_c_orig:
                fig_n = fig_numeric_distributions(df_orig, num_c_orig, target_col)
                add_image(fig_n, width=16*cm,
                          caption="Figura 1.2 — Distribuciones numéricas (histograma + media/mediana)")

    # Correlación
    if df_orig is not None:
        num_only = [c for c in df_orig.columns if pd.api.types.is_numeric_dtype(df_orig[c])]
        if len(num_only) >= 3:
            fig_corr = fig_correlation_heatmap(df_orig, num_only[:12])
            add_image(fig_corr, width=15*cm,
                      caption="Figura 1.3 — Matriz de correlación de Pearson (triángulo inferior)")

    story.append(PageBreak())

    # ─── 2. PARTICIÓN & BASELINE ──────────────────────────────────────────────
    story.append(Paragraph("2. Partición y Baseline — Data Leakage", S_H1))
    story.append(hr())

    story.append(Paragraph("2.1 ¿Qué es el Data Leakage?", S_H2))
    story.append(Paragraph(
        "El <b>data leakage</b> ocurre cuando información del conjunto de prueba (o del futuro) "
        "se filtra al entrenamiento, causando que el modelo parezca más preciso de lo real. "
        "En este dataset, <b>G1 y G2</b> son calificaciones de períodos intermedios con "
        "correlación ≥ 0.85 con G3 — son excluidas obligatoriamente.",
        S_BODY))

    story.append(Paragraph("Medidas aplicadas para prevenir leakage:", S_H2))
    measures = [
        ["Medida", "Implementación"],
        ["StandardScaler", "Ajustado solo con X_train — transform aplicado a val/test"],
        ["LabelEncoder", "Aprende categorías solo del conjunto de entrenamiento"],
        ["Partición estratificada", "stratify=y preserva la proporción de clases en los 3 conjuntos"],
        ["Estadísticas de imputación", "Mediana/moda calculada solo en train"],
        ["Detección automática", "Variables con AUC > 0.95 son excluidas automáticamente"],
    ]
    section_table(measures, col_widths=[6*cm, 10*cm])

    story.append(Paragraph("2.2 Esquema de Partición Train / Validación / Test", S_H2))
    split_data = [["Conjunto", "Tamaño", "Uso"],
                  ["Train (65%)", f"{len(session.get('X_train', [])):,}", "Ajusta los parámetros del modelo"],
                  ["Validación (15%)", f"{len(session.get('X_val', [])):,}", "Selección de hiperparámetros"],
                  ["Test (20%)", f"{len(session.get('X_test', [])):,}", "Evaluación final — solo se usa una vez"]]
    section_table(split_data, col_widths=[5*cm, 3.5*cm, 7.5*cm])

    story.append(Paragraph("2.3 Modelo Baseline", S_H2))
    story.append(Paragraph(
        "El <b>DummyClassifier(strategy='most_frequent')</b> predice siempre la clase mayoritaria. "
        "Representa el rendimiento mínimo que cualquier modelo útil debe superar. "
        "Dado que el dataset está desbalanceado (~84.6% aprobados), el baseline alcanza una "
        "Accuracy engañosamente alta — por eso se prioriza el <b>F1-Score</b> y <b>AUC-ROC</b>.",
        S_BODY))
    hist = session.get("results_history", {})
    if "Baseline (Dummy)" in hist:
        b = hist["Baseline (Dummy)"]
        baseline_data = [["Métrica", "Valor"],
                         ["Accuracy", f"{b.get('Accuracy',0):.4f}"],
                         ["F1-Score", f"{b.get('F1',0):.4f}"],
                         ["AUC-ROC", "0.5000 (aleatorio)"],
                         ["Estrategia", "Most Frequent"]]
        section_table(baseline_data, col_widths=[8*cm, 8*cm])

    story.append(PageBreak())

    # ─── 3. SEGMENTACIÓN ─────────────────────────────────────────────────────
    story.append(Paragraph("3. Segmentación — K-Means y Clustering Jerárquico", S_H1))
    story.append(hr())

    story.append(Paragraph(
        "El clustering es un aprendizaje <b>no supervisado</b> que agrupa registros similares sin "
        "usar la etiqueta. Permite descubrir <b>perfiles naturales de estudiantes</b>. "
        "Se usan variables numéricas de comportamiento: edad, educación parental, tiempo de estudio, "
        "ausencias, consumo de alcohol, etc.",
        S_BODY))

    story.append(Paragraph("3.1 K-Means Clustering", S_H2))
    story.append(Paragraph(
        "Algoritmo iterativo que agrupa n registros en K clústeres minimizando la <i>inercia</i> "
        "(suma de distancias cuadradas al centroide). El <b>método del codo</b> indica K≈4.",
        S_BODY))

    # Figuras de clustering guardadas en sesión
    for fig_key_s in ["pdf_km_elbow", "pdf_km_sil", "pdf_km_pca", "pdf_km_profiles"]:
        if session.get(fig_key_s):
            captions = {
                "pdf_km_elbow": "Figura 3.1 — Método del Codo y Silhouette Score para selección de K",
                "pdf_km_sil": "Figura 3.2 — Diagrama Silhouette detallado por clúster (K-Means)",
                "pdf_km_pca": "Figura 3.3 — Proyección PCA 2D de los clústeres K-Means",
                "pdf_km_profiles": "Figura 3.4 — Perfiles de clústeres: variables clave por grupo",
            }
            path = session[fig_key_s]
            story.append(Spacer(1, 6))
            story.append(RLImage(path, width=15*cm, height=9*cm))
            story.append(Paragraph(f"<i>{captions[fig_key_s]}</i>", S_NOTE))
            story.append(Spacer(1, 6))

    sil_km = session.get("kmeans_sil")
    if sil_km:
        grade = ("Excelente (>0.70)" if sil_km > 0.7 else
                 "Razonable (>0.50)" if sil_km > 0.5 else
                 "Débil — datos continuos" if sil_km > 0.25 else "Sin estructura fuerte")
        km_results = [["Métrica", "Valor"],
                      ["Silhouette Score", f"{sil_km:.4f}"],
                      ["Calificación", grade],
                      ["K elegido", str(session.get("km_k_final", 4))]]
        section_table(km_results, col_widths=[8*cm, 8*cm])
        story.append(Paragraph(
            "<i>Nota: valores bajos de Silhouette son comunes en datos de comportamiento estudiantil, "
            "ya que los perfiles son continuos y no forman islas bien separadas.</i>",
            S_NOTE))

    story.append(Paragraph("3.2 Clustering Jerárquico (Ward)", S_H2))
    story.append(Paragraph(
        "Parte con cada registro como su propio clúster y los fusiona progresivamente. "
        "El criterio <b>Ward</b> minimiza la varianza interna. "
        "El <b>dendrograma</b> visualiza la jerarquía completa; "
        "cortando horizontalmente se obtiene cualquier número de grupos.",
        S_BODY))

    for fig_key_s in ["pdf_hc_dend", "pdf_hc_pca"]:
        if session.get(fig_key_s):
            captions = {
                "pdf_hc_dend": "Figura 3.5 — Dendrograma de Clustering Jerárquico (Ward)",
                "pdf_hc_pca": "Figura 3.6 — Proyección PCA 2D del Clustering Jerárquico",
            }
            path = session[fig_key_s]
            story.append(Spacer(1, 6))
            story.append(RLImage(path, width=15*cm, height=9*cm))
            story.append(Paragraph(f"<i>{captions[fig_key_s]}</i>", S_NOTE))
            story.append(Spacer(1, 6))

    sil_hc = session.get("hc_sil")
    if sil_hc:
        hc_results = [["Métrica", "Valor"],
                      ["Silhouette Score", f"{sil_hc:.4f}"],
                      ["K elegido", str(session.get("hc_k_final", 4))]]
        section_table(hc_results, col_widths=[8*cm, 8*cm])

    story.append(Paragraph("3.3 Comparativa K-Means vs Jerárquico", S_H2))
    story.append(Paragraph(
        "El índice Silhouette evalúa qué tan bien asignado está cada punto: "
        "valores cercanos a +1 indican clústeres compactos y separados. "
        "K-Means es más escalable para grandes datasets; el jerárquico es más flexible y "
        "su dendrograma permite explorar la estructura sin definir K a priori.",
        S_BODY))

    if sil_km or sil_hc:
        comp_rows = [["Algoritmo", "Silhouette", "Escalabilidad", "Requiere K a priori"]]
        if sil_km: comp_rows.append(["K-Means", f"{sil_km:.4f}", "Alta", "Sí"])
        if sil_hc: comp_rows.append(["Jerárquico (Ward)", f"{sil_hc:.4f}", "Media", "No"])
        section_table(comp_rows, col_widths=[5*cm, 4*cm, 4*cm, 3*cm])

    story.append(PageBreak())

    # ─── 4. CLASIFICACIÓN ────────────────────────────────────────────────────
    story.append(Paragraph("4. Clasificación — Árbol de Decisión y Random Forest", S_H1))
    story.append(hr())

    story.append(Paragraph(
        "La clasificación es un aprendizaje <b>supervisado</b> que aprende a predecir si un "
        "estudiante aprobará el curso (<i>passed = 1</i>) basándose en sus características "
        "demográficas, sociales y de comportamiento.",
        S_BODY))

    story.append(Paragraph("4.1 Árbol de Decisión (CART)", S_H2))
    story.append(Paragraph(
        "Divide el espacio de features con reglas binarias interpretables. "
        "Cada nodo hace una pregunta sobre una variable; las hojas son las predicciones. "
        "<b>Ventaja:</b> totalmente interpretable. "
        "<b>Limitación:</b> propenso a overfitting sin poda (max_depth, min_samples_leaf).",
        S_BODY))

    story.append(Paragraph("4.2 Random Forest", S_H2))
    story.append(Paragraph(
        "Ensemble de N árboles entrenados con muestras aleatorias del dataset (<b>bagging</b>) "
        "y subconjuntos aleatorios de features en cada nodo. "
        "La predicción final es la mayoría de votos. "
        "<b>Ventaja:</b> robusto, maneja no-linealidad, provee importancia de variables. "
        "<b>Limitación:</b> menos interpretable que un único árbol.",
        S_BODY))

    # Resultados de modelos
    hist = session.get("results_history", {})
    model_figs = session.get("pdf_model_figs", {})

    for model_name in ["Árbol de Decisión", "Random Forest"]:
        if model_name in hist:
            m = hist[model_name]
            story.append(Paragraph(f"Resultados — {model_name}:", S_H2))
            res_data = [["Métrica", "Valor", "Interpretación"],
                        ["Accuracy", f"{m.get('Accuracy',0):.4f}", "Proporción global de aciertos"],
                        ["F1-Score", f"{m.get('F1',0):.4f}", "Balance Precisión/Sensibilidad"],
                        ["AUC-ROC", f"{m.get('AUC-ROC',0):.4f}", "Capacidad global de discriminación"],
                        ["Precisión", f"{m.get('Precisión',0):.4f}", "VP / (VP+FP)"],
                        ["Sensibilidad", f"{m.get('Sensibilidad',0):.4f}", "VP / (VP+FN)"]]
            section_table(res_data, col_widths=[4*cm, 4*cm, 8*cm])

    # Importancia de variables comparativa
    if session.get("pdf_feat_importance"):
        story.append(Spacer(1, 6))
        story.append(RLImage(session["pdf_feat_importance"], width=15*cm, height=7*cm))
        story.append(Paragraph(
            "<i>Figura 4.1 — Importancia de variables: Árbol de Decisión vs Random Forest</i>", S_NOTE))

    story.append(PageBreak())

    # ─── 5. EVALUACIÓN COMPARATIVA ────────────────────────────────────────────
    story.append(Paragraph("5. Evaluación Comparativa de Modelos — Curva ROC", S_H1))
    story.append(hr())

    story.append(Paragraph("5.1 Tabla Comparativa de Modelos", S_H2))
    hist = session.get("results_history", {})
    if hist:
        rows_t = [["Modelo", "Accuracy", "F1-Score", "Precisión", "Sensibilidad", "AUC-ROC"]]
        sorted_models = sorted(hist.items(), key=lambda x: x[1].get("AUC-ROC", 0), reverse=True)
        for name, v in sorted_models:
            rows_t.append([name,
                           f"{v.get('Accuracy',0):.4f}",
                           f"{v.get('F1',0):.4f}",
                           f"{v.get('Precisión',0):.4f}",
                           f"{v.get('Sensibilidad',0):.4f}",
                           f"{v.get('AUC-ROC',0):.4f}"])
        section_table(rows_t, col_widths=[4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        best_name = sorted_models[0][0]
        best_auc  = sorted_models[0][1].get("AUC-ROC", 0)
        story.append(Paragraph(
            f"<b>Mejor modelo:</b> {best_name} con AUC-ROC = {best_auc:.4f}", S_BODY))

    story.append(Paragraph("5.2 Curvas ROC Superpuestas", S_H2))
    story.append(Paragraph(
        "La <b>curva ROC</b> muestra la tasa de Verdaderos Positivos (Sensibilidad) frente "
        "a la tasa de Falsos Positivos para todos los umbrales posibles. "
        "Un <b>AUC = 1.0</b> indica clasificador perfecto; "
        "<b>AUC = 0.5</b> equivale a adivinar al azar.",
        S_BODY))

    if session.get("pdf_roc_fig"):
        story.append(Spacer(1, 6))
        story.append(RLImage(session["pdf_roc_fig"], width=13*cm, height=9*cm))
        story.append(Paragraph(
            "<i>Figura 5.1 — Curvas ROC superpuestas — el modelo con mayor área es el mejor discriminador</i>",
            S_NOTE))

    story.append(Paragraph("5.3 Dashboard Comparativo", S_H2))
    if session.get("pdf_metrics_bar"):
        story.append(Spacer(1, 6))
        story.append(RLImage(session["pdf_metrics_bar"], width=15*cm, height=6*cm))
        story.append(Paragraph(
            "<i>Figura 5.2 — Comparativa de Accuracy, F1-Score y AUC-ROC por modelo</i>", S_NOTE))

    story.append(Paragraph("5.4 Resumen para Equipo Gerencial", S_H2))
    if hist and sorted_models:
        b_name = sorted_models[0][0]
        b_auc  = sorted_models[0][1].get("AUC-ROC", 0)
        b_acc  = sorted_models[0][1].get("Accuracy", 0)
        b_f1   = sorted_models[0][1].get("F1", 0)
        grade_g = ("excelente" if b_auc >= 0.90 else "bueno" if b_auc >= 0.80
                   else "aceptable" if b_auc >= 0.70 else "necesita mejoras")
        story.append(Paragraph(
            f"El modelo recomendado es <b>{b_name}</b>. "
            f"En términos prácticos: si se ordenan 100 estudiantes por probabilidad de aprobar, "
            f"el sistema coloca a un estudiante que SÍ aprobará antes que uno que NO aprobará "
            f"el <b>{b_auc:.0%} de las veces</b>. "
            f"El rendimiento global (AUC = {b_auc:.4f}) es calificado como <b>{grade_g}</b>. "
            f"{'El modelo está listo para una prueba piloto.' if b_auc >= 0.75 else 'Se recomienda mejorar el modelo antes de implementarlo.'}",
            S_BODY))

    story.append(PageBreak())

    # ─── 6. MATRIZ DE CONFUSIÓN ────────────────────────────────────────────────
    story.append(Paragraph("6. Matriz de Confusión con Métricas Completas", S_H1))
    story.append(hr())

    story.append(Paragraph("6.1 Estructura de la Matriz de Confusión", S_H2))
    cm_theory = [["", "Predicción: NO (0)", "Predicción: SÍ (1)"],
                 ["Real: NO (0)", "VN — Verdaderos Negativos", "FP — Falso Positivo (Error Tipo I)"],
                 ["Real: SÍ (1)", "FN — Falso Negativo (Error Tipo II)", "VP — Verdaderos Positivos"]]
    section_table(cm_theory, col_widths=[4*cm, 6*cm, 6*cm], header_bg=PDF_SECONDARY)

    story.append(Paragraph("Interpretación de cada celda:", S_H2))
    interp_data = [["Celda", "Nombre", "Significado práctico"],
                   ["VP", "Verdaderos Positivos", "Predijo 'aprobará' y sí aprobó → acierto correcto"],
                   ["VN", "Verdaderos Negativos", "Predijo 'no aprobará' y no aprobó → acierto correcto"],
                   ["FP", "Falso Positivo — Error Tipo I", "Predijo 'aprobará' pero NO aprobó → recurso mal dirigido"],
                   ["FN", "Falso Negativo — Error Tipo II", "Predijo 'no aprobará' pero SÍ aprobó → caso no detectado (más costoso)"]]
    section_table(interp_data, col_widths=[2*cm, 5*cm, 9*cm])

    story.append(Paragraph("6.2 Matrices de Confusión — Árbol y Random Forest", S_H2))
    for fig_key_s in ["pdf_cm_dt", "pdf_cm_rf"]:
        if session.get(fig_key_s):
            labels_cm = {"pdf_cm_dt": "Figura 6.1 — Matriz de Confusión: Árbol de Decisión",
                         "pdf_cm_rf": "Figura 6.2 — Matriz de Confusión: Random Forest"}
            story.append(Spacer(1, 6))
            story.append(RLImage(session[fig_key_s], width=10*cm, height=7.5*cm))
            story.append(Paragraph(f"<i>{labels_cm[fig_key_s]}</i>", S_NOTE))
            story.append(Spacer(1, 6))

    story.append(Paragraph("6.3 Métricas Derivadas de la Matriz de Confusión", S_H2))
    metrics_def = [["Métrica", "Fórmula", "Cuándo priorizar"],
                   ["Precisión", "VP / (VP+FP)", "Cuando los FP son costosos (recursos mal dirigidos)"],
                   ["Sensibilidad", "VP / (VP+FN)", "Cuando los FN son costosos (caso no detectado)"],
                   ["Especificidad", "VN / (VN+FP)", "Para identificar correctamente los negativos"],
                   ["F1-Score", "2·VP / (2·VP+FP+FN)", "Balance entre Precisión y Sensibilidad"],
                   ["AUC-ROC", "Área bajo curva ROC", "Evaluación global independiente del umbral"],
                   ["VPP", "VP / (VP+FP)", "Valor Predictivo Positivo"],
                   ["VPN", "VN / (VN+FN)", "Valor Predictivo Negativo"]]
    section_table(metrics_def, col_widths=[3.5*cm, 4*cm, 8.5*cm])

    if hist:
        best_name_cm = max(hist, key=lambda m: hist[m].get("AUC-ROC", 0)
                           if m != "Baseline (Dummy)" else 0)
        bm = hist.get(best_name_cm, {})
        if bm.get("Accuracy"):
            story.append(Paragraph(
                f"<b>Interpretación del mejor modelo ({best_name_cm}):</b>",
                S_H2))
            story.append(Paragraph(
                f"Con una Sensibilidad de {bm.get('Sensibilidad', 0):.4f}, el modelo detecta "
                f"el {bm.get('Sensibilidad', 0)*100:.1f}% de los estudiantes que realmente aprueban. "
                f"La Especificidad de {bm.get('Especificidad', 0):.4f} indica que identifica correctamente "
                f"el {bm.get('Especificidad', 0)*100:.1f}% de los que no aprueban. "
                f"<b>Recomendación de umbral:</b> para este problema educativo, priorizar la Sensibilidad "
                f"(detectar a quienes van a reprobar) reduce el FN, aunque aumente ligeramente los FP.",
                S_BODY))

    story.append(PageBreak())

    # ─── 7. CONCLUSIONES ──────────────────────────────────────────────────────
    story.append(Paragraph("7. Conclusiones y Recomendaciones Gerenciales", S_H1))
    story.append(hr())

    story.append(Paragraph("7.1 Resumen Ejecutivo", S_H2))
    conc_data = [["Aspecto", "Resultado"],
                 ["Dataset", "Student Performance (UCI) — 649 registros, 33 variables"],
                 ["Objetivo", "Predecir si un estudiante aprobará (G3 >= 10)"],
                 ["Leakage detectado", "G1 y G2 excluidas (r ≥ 0.85 con G3)"],
                 ["Desbalanceo", "~84.6% aprobados → class_weight='balanced'"],
                 ["Segmentación", "4 perfiles de estudiantes identificados"],
                 ["Variable más predictora", "failures, absences, studytime, Walc"],
                 ["Metodología anti-leakage", "Scaler y Encoder ajustados solo en train"]]
    if hist:
        best_m  = max(hist, key=lambda m: hist[m].get("AUC-ROC", 0) if m != "Baseline (Dummy)" else 0)
        best_v  = hist[best_m]
        conc_data.append(["Mejor modelo", f"{best_m} — AUC={best_v.get('AUC-ROC',0):.4f}, F1={best_v.get('F1',0):.4f}"])
    section_table(conc_data, col_widths=[6*cm, 10*cm])

    story.append(Paragraph("7.2 Hallazgos Principales", S_H2))
    hallazgos = [
        "Leakage crítico: G1 y G2 se correlacionan con G3 en más de 0.85. Su exclusión es obligatoria para un modelo real.",
        "Desbalanceo: el baseline (predecir siempre 'aprobó') alcanza ~84.6% de Accuracy, demostrando que esta métrica sola es engañosa.",
        "Segmentación: los 4 clústeres revelan perfiles diferenciados por dedicación al estudio vs. tiempo social/consumo de alcohol.",
        "Random Forest supera al Árbol de Decisión en todas las métricas al combinar múltiples árboles que se corrigen mutuamente.",
        "Para el negocio educativo: priorizar la Sensibilidad (detectar quienes van a reprobar) reduce el FN y el costo de intervención tardía.",
    ]
    for h in hallazgos:
        story.append(Paragraph(f"• {h}", S_BODY))

    story.append(Paragraph("7.3 Recomendaciones Prácticas", S_H2))
    recs = [
        "Usar el modelo para priorizar tutorías en los estudiantes con probabilidad de aprobar < 40%.",
        "Monitorear resultados trimestralmente y reentrenar con nuevos datos de cada cohorte.",
        "Considerar técnicas de resampling (SMOTE) o ajuste de class_weight para mejorar la detección de la clase minoritaria.",
        "Incluir variables de seguimiento en tiempo real (asistencia, participación) para mejorar la predicción.",
        "Presentar el modelo como una herramienta de apoyo, no como decisión definitiva, respetando el juicio docente.",
    ]
    for r in recs:
        story.append(Paragraph(f"• {r}", S_BODY))

    # Dashboard final
    if session.get("pdf_dashboard"):
        story.append(PageBreak())
        story.append(Paragraph("Anexo: Dashboard Final de Resultados", S_H1))
        story.append(hr())
        story.append(Spacer(1, 6))
        story.append(RLImage(session["pdf_dashboard"], width=16*cm, height=10*cm))
        story.append(Paragraph(
            "<i>Figura A.1 — Dashboard resumen completo: métricas, curvas ROC, "
            "importancia de variables y silhouette de clustering</i>", S_NOTE))

    # Pie de página info
    story.append(Spacer(1, 1*cm))
    story.append(hr())
    story.append(Paragraph(
        "Informe generado automáticamente por el Sistema CRISP-DM — "
        "Minería de Datos | Docente: MILTON EDWARD HUMPIRI FLORES",
        S_SMALL))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
#  ALMACENAMIENTO DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
def store_clf_result(model_name, m):
    if "results_history" not in st.session_state:
        st.session_state.results_history = {}
    if "roc_data" not in st.session_state:
        st.session_state.roc_data = {}

    st.session_state.results_history[model_name] = {
        "Accuracy":     m["Exactitud"],
        "Precisión":    m["Precisión"],
        "Sensibilidad": m["Sensibilidad"],
        "Especificidad": m["Especificidad"],
        "F1":           m["F1_Score"],
        "AUC-ROC":      m["AUC_ROC"] or 0.0,
    }
    roc_pc = m.get("roc_per_class", {})
    if roc_pc:
        st.session_state.roc_data[model_name] = roc_pc


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN EDA
# ══════════════════════════════════════════════════════════════════════════════
def render_eda():
    df      = st.session_state.df
    target  = st.session_state.target
    profile = st.session_state.profile

    st.header("📊 Comprensión de los Datos — CRISP-DM Fase 2")

    with st.expander("📖 ¿Qué es la Comprensión de Datos en CRISP-DM?"):
        st.markdown("""
La **fase 2 (Data Understanding)** busca:
- Describir el dataset: filas, columnas, tipos de datos.
- Identificar problemas de calidad: nulos, duplicados, outliers.
- Descubrir relaciones entre variables (análisis bivariado).
        """)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filas",     f"{len(df):,}")
    c2.metric("Columnas",  len(df.columns))
    c3.metric("Nulos",     df.isna().sum().sum())
    c4.metric("Duplicados", df.duplicated().sum())
    c5.metric("Tarea",     "Clasificación")
    st.divider()

    with st.expander("📋 Perfil automático de columnas", expanded=True):
        rows = [{"Columna": c, "Tipo": p["col_type"], "Únicos": p["n_unique"],
                 "Nulos %": p["null_pct"], "Imputación": p["impute"],
                 "Encoding": p["encoding"]}
                for c, p in profile.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader(f"🎯 Variable Objetivo: `{target}`")
    fig_t = fig_target_distribution(df, target)
    st.pyplot(fig_t)
    plt.close(fig_t)

    min_r = df[target].value_counts(normalize=True).min() if df[target].nunique() > 1 else 1
    if min_r < 0.15:
        st.warning(f"⚠️ **Dataset desbalanceado**: clase minoritaria = {min_r:.2%}. "
                   "Se usará `class_weight='balanced'` en los clasificadores.")
    st.divider()

    num_cols = [c for c, p in profile.items() if p["is_num"] and c != target]
    if num_cols:
        st.subheader("📈 Distribuciones Numéricas")
        fig_n = fig_numeric_distributions(df, num_cols, target)
        st.pyplot(fig_n)
        plt.close(fig_n)

        st.subheader("📦 Variables Numéricas vs Target")
        fig_b = fig_boxplots_by_target(df, num_cols[:6], target)
        st.pyplot(fig_b)
        plt.close(fig_b)

    if len(num_cols) >= 3:
        st.divider()
        st.subheader("🔥 Correlación de Pearson")
        fig_corr = fig_correlation_heatmap(df, num_cols[:14])
        st.pyplot(fig_corr)
        plt.close(fig_corr)
        st.caption("Alta correlación entre features puede indicar redundancia. "
                   "Valores cercanos a ±1 con el target señalan buenos predictores.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN PREPARACIÓN
# ══════════════════════════════════════════════════════════════════════════════
def render_preparacion():
    st.header("⚙️ Preparación de Datos — CRISP-DM Fase 3")

    with st.expander("📖 Data Leakage — definición y cómo evitarlo", expanded=True):
        st.markdown(EXP["data_leakage"])

    with st.expander("📖 Partición Train / Validación / Test", expanded=True):
        st.markdown(EXP["split"])

    with st.expander("📖 Modelo Baseline"):
        st.markdown(EXP["baseline"])

    st.divider()

    log     = st.session_state.prep_log
    X_train = st.session_state.X_train
    X_val   = st.session_state.X_val
    X_test  = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test  = st.session_state.y_test
    feats   = st.session_state.feature_names

    st.subheader("✅ Pasos de limpieza y codificación aplicados")
    for d in log:
        st.markdown(f"- {d}")

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features resultantes", len(feats))
    c2.metric("Train",      f"{len(X_train):,}")
    c3.metric("Validación", f"{len(X_val):,}" if len(X_val) > 0 else "No usado")
    c4.metric("Test",       f"{len(X_test):,}")

    # Gráfico de partición
    splits_df = pd.DataFrame({
        "Conjunto": ["Train", "Validación", "Test"],
        "Registros": [len(X_train), len(X_val), len(X_test)]
    }).query("Registros > 0")

    fig, ax = plt.subplots(figsize=(5, 4))
    colors_s = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax.pie(splits_df["Registros"], labels=splits_df["Conjunto"],
           autopct="%1.1f%%", colors=colors_s[:len(splits_df)],
           startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Distribución de la Partición", fontweight="bold")
    st.pyplot(fig); plt.close(fig)

    # Balance de clases por conjunto
    st.subheader("⚖️ Balance de clases por conjunto (verificación del stratify)")
    cols = st.columns(3)
    for col_ui, nm, yy in [(cols[0], "Train", y_train),
                            (cols[1], "Validación", st.session_state.y_val),
                            (cols[2], "Test", y_test)]:
        if len(yy) > 0:
            with col_ui:
                u, cnt = np.unique(yy, return_counts=True)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar([str(x) for x in u], cnt/cnt.sum(),
                       color=["#e74c3c", "#2ecc71"][:len(u)], edgecolor="white")
                ax.set_title(nm, fontweight="bold")
                ax.set_ylim(0, 1.15)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
                ax.grid(axis="y", alpha=0.3)
                st.pyplot(fig); plt.close(fig)

    st.divider()
    st.subheader("📏 Modelo Baseline — DummyClassifier (most_frequent)")
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    acc_d = accuracy_score(y_test, dummy.predict(X_test))
    f1_d  = f1_score(y_test, dummy.predict(X_test), average="weighted", zero_division=0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy Baseline", f"{acc_d:.4f}")
    c2.metric("F1 Baseline",       f"{f1_d:.4f}")
    c3.metric("Estrategia",        "Most Frequent")
    st.info(f"🎯 Cualquier modelo útil debe superar Accuracy > {acc_d:.2%} y F1 > {f1_d:.4f}")
    st.session_state.results_history["Baseline (Dummy)"] = {
        "Accuracy": round(acc_d, 4), "F1": round(f1_d, 4), "AUC-ROC": 0.5,
        "Precisión": 0.0, "Sensibilidad": 0.0, "Especificidad": 0.0
    }

    with st.expander("🔎 Vista previa — datos procesados (train, primeras 5 filas)"):
        st.dataframe(X_train.head(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN SEGMENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
def render_segmentacion():
    st.header("📉 Segmentación — CRISP-DM Fase 4 (Modelos Descriptivos)")

    X_train = st.session_state.X_train
    X_test  = st.session_state.X_test
    feats   = st.session_state.feature_names

    with st.expander("📖 K-Means"):             st.markdown(EXP["kmeans"])
    with st.expander("📖 Clustering Jerárquico"): st.markdown(EXP["hierarchical"])
    with st.expander("📖 Índice Silhouette"):    st.markdown(EXP["silhouette"])

    tab_km, tab_hc, tab_comp = st.tabs(["🔵 K-Means", "🌳 Jerárquico", "📊 Comparativa"])

    # ── K-MEANS ───────────────────────────────────────────────────────────────
    with tab_km:
        st.subheader("K-Means Clustering")
        k_val = st.slider("Número de clústeres K", 2, 12, 4, key="km_k")
        run_km = st.button("▶ Ejecutar K-Means", type="primary", key="btn_km")

        if run_km:
            with st.spinner("Calculando inercias y silhouette..."):
                fig_es, inertias, silhouettes, best_k = fig_elbow_silhouette(
                    X_train.values, range(2, 11))
            st.pyplot(fig_es)
            st.caption(f"📌 K con mayor Silhouette: K={best_k}. Usamos K={k_val} (balance codo/interpretabilidad).")

            # Modelo final
            kmeans = KMeans(n_clusters=k_val, random_state=RANDOM_STATE, n_init=10)
            kmeans.fit(X_train)

            X_all  = np.vstack([X_train.values, X_test.values])
            lbl_all = np.concatenate([kmeans.labels_, kmeans.predict(X_test)])
            sil_f   = silhouette_score(X_all, lbl_all,
                                       sample_size=min(5000, len(X_all)), random_state=RANDOM_STATE)

            c1, c2 = st.columns(2)
            c1.metric("Silhouette Score final", f"{sil_f:.4f}")
            c2.metric("Inercia final",           f"{kmeans.inertia_:,.0f}")

            grade = ("🟢 Excelente" if sil_f > 0.7 else "🟡 Razonable" if sil_f > 0.5
                     else "🟠 Débil (datos continuos)" if sil_f > 0.25 else "🔴 Sin estructura")
            st.info(f"**Silhouette = {sil_f:.4f} → {grade}**")

            # Diagrama silhouette detallado
            st.subheader("📊 Diagrama Silhouette Detallado")
            fig_sil = fig_silhouette_detail(X_all, lbl_all, f"Silhouette — K-Means (K={k_val})")
            st.pyplot(fig_sil)

            # PCA
            st.subheader("🔵 Proyección PCA 2D")
            fig_pca = fig_pca_scatter(X_all, lbl_all,
                                      f"K-Means (K={k_val}) — PCA 2D",
                                      cluster_centers=kmeans.cluster_centers_)
            st.pyplot(fig_pca)
            st.caption("📌 Clústeres bien separados en 2D sugieren buena separabilidad real.")

            # Perfiles
            st.subheader("👥 Perfiles de Clústeres")
            df_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=feats)
            df_centroids.index = [f"Clúster {i}" for i in range(k_val)]
            st.dataframe(df_centroids.round(3), use_container_width=True)

            key_vars = feats[:min(6, len(feats))]
            fig_prof = fig_cluster_profiles(df_centroids, key_vars, k_val,
                                            f"Perfiles de Clústeres — K-Means (K={k_val})")
            st.pyplot(fig_prof)

            # Distribución del target
            y_test_arr = np.array(st.session_state.y_test)
            lbl_test = kmeans.predict(X_test)
            df_cl = pd.DataFrame({"Clúster": lbl_test.astype(str), "Target": y_test_arr})
            df_agg = df_cl.groupby("Clúster")["Target"].mean().reset_index()
            df_agg.columns = ["Clúster", "% Aprobados (media)"]

            fig_ct, ax_ct = plt.subplots(figsize=(8, 4))
            bars = ax_ct.bar(df_agg["Clúster"], df_agg["% Aprobados (media)"],
                             color=plt.cm.tab10(np.linspace(0, 1, k_val)),
                             edgecolor="white", width=0.5)
            ax_ct.set_title("Proporción de Aprobados por Clúster", fontweight="bold")
            ax_ct.set_ylabel("Media Target (Aprobado)")
            ax_ct.set_ylim(0, 1.2)
            for bar, v in zip(bars, df_agg["% Aprobados (media)"]):
                ax_ct.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                           f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
            ax_ct.grid(axis="y", alpha=0.3)
            st.pyplot(fig_ct)
            st.caption("📌 Clústeres con alta/baja proporción de aprobados son los **segmentos más homogéneos**.")

            # Guardar figuras para PDF
            st.session_state["kmeans_sil"] = sil_f
            st.session_state["km_k_final"] = k_val
            st.session_state["pdf_km_elbow"]   = save_fig_to_tmp(fig_es)
            st.session_state["pdf_km_sil"]     = save_fig_to_tmp(fig_sil)
            st.session_state["pdf_km_pca"]     = save_fig_to_tmp(fig_pca)
            st.session_state["pdf_km_profiles"] = save_fig_to_tmp(fig_prof)

    # ── CLUSTERING JERÁRQUICO ──────────────────────────────────────────────────
    with tab_hc:
        st.subheader("Clustering Jerárquico Aglomerativo (Ward)")
        k_hc = st.slider("Número de clústeres (corte del dendrograma)", 2, 12, 4, key="hc_k")
        run_hc = st.button("▶ Ejecutar Clustering Jerárquico", type="primary", key="btn_hc")

        if run_hc:
            sample_n = min(150, len(X_train))
            with st.spinner(f"Construyendo dendrograma sobre {sample_n} registros..."):
                fig_dend = fig_dendrogram(X_train.values, k_hc, sample_n)
            st.pyplot(fig_dend)
            st.caption("📌 La línea roja marca el corte que genera los K clústeres solicitados.")

            agg = AgglomerativeClustering(n_clusters=k_hc, linkage="ward")
            labels_train = agg.fit_predict(X_train)
            sil_hc = silhouette_score(X_train, labels_train,
                                      sample_size=min(3000, len(X_train)), random_state=RANDOM_STATE)

            c1, c2 = st.columns(2)
            c1.metric("Silhouette Score", f"{sil_hc:.4f}")
            c2.metric("Clústeres",        k_hc)

            grade_h = ("🟢 Excelente" if sil_hc > 0.7 else "🟡 Razonable" if sil_hc > 0.5
                       else "🟠 Débil" if sil_hc > 0.25 else "🔴 Sin estructura")
            st.info(f"**Silhouette = {sil_hc:.4f} → {grade_h}**")

            # PCA
            fig_pca_h = fig_pca_scatter(X_train.values, labels_train,
                                        f"Clustering Jerárquico (K={k_hc}) — PCA 2D")
            st.pyplot(fig_pca_h)

            # Perfiles
            df_prof_h = X_train.copy()
            df_prof_h["Clúster"] = labels_train
            profile_means = df_prof_h.groupby("Clúster").mean().round(3)
            st.subheader("👥 Perfiles de Clústeres — Medias")
            st.dataframe(profile_means, use_container_width=True)

            key_vars_h = feats[:min(6, len(feats))]
            fig_prof_h = fig_cluster_profiles(profile_means, key_vars_h, k_hc,
                                              f"Perfiles Jerárquico (K={k_hc})")
            st.pyplot(fig_prof_h)

            st.session_state["hc_sil"]     = sil_hc
            st.session_state["hc_k_final"] = k_hc
            st.session_state["pdf_hc_dend"] = save_fig_to_tmp(fig_dend)
            st.session_state["pdf_hc_pca"]  = save_fig_to_tmp(fig_pca_h)

    # ── COMPARATIVA ────────────────────────────────────────────────────────────
    with tab_comp:
        st.subheader("📊 Comparativa: K-Means vs Clustering Jerárquico")
        sil_km = st.session_state.get("kmeans_sil")
        sil_hc = st.session_state.get("hc_sil")

        if not sil_km and not sil_hc:
            st.info("Ejecuta ambos algoritmos para ver la comparativa.")
        else:
            alg_names_c, sil_vals_c = [], []
            if sil_km: alg_names_c.append("K-Means"); sil_vals_c.append(sil_km)
            if sil_hc: alg_names_c.append("Jerárquico"); sil_vals_c.append(sil_hc)

            fig_comp, ax_comp = plt.subplots(figsize=(6, 4))
            bars = ax_comp.bar(alg_names_c, sil_vals_c,
                               color=["#3498db", "#e74c3c"][:len(alg_names_c)],
                               edgecolor="white", width=0.4, alpha=0.85)
            for bar, v in zip(bars, sil_vals_c):
                ax_comp.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                             f"{v:.4f}", ha="center", fontweight="bold")
            ax_comp.set_ylim(0, max(sil_vals_c) * 1.4 if sil_vals_c else 1)
            ax_comp.set_title("Silhouette Score — Comparativa", fontweight="bold")
            ax_comp.grid(axis="y", alpha=0.3)
            st.pyplot(fig_comp); plt.close(fig_comp)

            st.markdown("""
**¿Cómo elegir el mejor algoritmo?**
- **Silhouette más alto** → clústeres más compactos y separados.
- **K-Means** es más escalable para grandes datasets.
- **Clustering Jerárquico** no requiere K a priori — el dendrograma guía la selección.
- Para **perfiles estudiantiles**: K-Means es el estándar; jerárquico se usa para exploración inicial.
            """)


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════
def render_clasificacion():
    st.header("🌳 Clasificación — CRISP-DM Fase 4 (Modelos Predictivos)")

    X_train = st.session_state.X_train
    X_val   = st.session_state.X_val
    X_test  = st.session_state.X_test
    y_train = st.session_state.y_train
    y_val   = st.session_state.y_val
    y_test  = st.session_state.y_test
    feats   = st.session_state.feature_names

    with st.expander("📖 Árbol de Decisión"):   st.markdown(EXP["decision_tree"])
    with st.expander("📖 Random Forest"):        st.markdown(EXP["random_forest"])
    with st.expander("📖 Matriz de Confusión"):  st.markdown(EXP["confusion_matrix"])

    # Todos los clasificadores disponibles
    tab_names = ["🌿 Árbol", "🌲 Random Forest", "📈 Logística", "🎯 KNN", "🧠 Naive Bayes"]
    clf_keys  = [("Árbol de Decisión", "dt"),
                 ("Random Forest", "rf"),
                 ("Regresión Logística", "lr"),
                 ("KNN", "knn"),
                 ("Naive Bayes", "nb")]
    if HAS_XGB:
        tab_names.append("🚀 XGBoost"); clf_keys.append(("XGBoost", "xgb"))
    if HAS_LGB:
        tab_names.append("⚡ LightGBM"); clf_keys.append(("LightGBM", "lgb"))

    tabs = st.tabs(tab_names)

    for (tab, (name, key)) in zip(tabs, clf_keys):
        with tab:
            st.subheader(name)
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if key == "dt":
                    max_d  = st.slider("max_depth", 2, 20, 5, key=f"md_{key}")
                    min_sl = st.slider("min_samples_leaf", 1, 50, 5, key=f"msl_{key}")
                elif key == "rf":
                    n_est  = st.slider("n_estimators", 50, 500, 150, step=50, key=f"ne_{key}")
                    max_d  = st.slider("max_depth", 2, 30, 10, key=f"md_{key}")
                    min_sl = st.slider("min_samples_leaf", 1, 20, 4, key=f"msl_{key}")
                elif key == "lr":
                    C_reg  = st.select_slider("C (regularización)",
                                              options=[0.001, 0.01, 0.1, 1, 10, 100],
                                              value=1.0, key=f"C_{key}")
                    max_iter = st.slider("max_iter", 100, 2000, 500, step=100, key=f"mi_{key}")
                elif key == "knn":
                    n_neighbors = st.slider("n_neighbors", 1, 50, 5, key=f"nn_{key}")
                elif key == "nb":
                    st.info("Naive Bayes: no requiere hiperparámetros adicionales.")
                elif key in ("xgb", "lgb"):
                    n_est  = st.slider("n_estimators", 50, 500, 150, step=50, key=f"ne_{key}")
                    max_d  = st.slider("max_depth", 2, 20, 6, key=f"md_{key}")
                    lr_val = st.select_slider("learning_rate",
                                             options=[0.01, 0.05, 0.1, 0.2, 0.3],
                                             value=0.1, key=f"lr_{key}")
            with col_p2:
                run_cv = st.checkbox("Cross-Validation (k=5)", value=True, key=f"cv_{key}")
                use_val = len(X_val) > 0

            run_btn = st.button(f"🚀 Entrenar {name}", type="primary", key=f"btn_{key}")

            if run_btn:
                with st.spinner(f"Entrenando {name}..."):
                    if key == "dt":
                        model = DecisionTreeClassifier(max_depth=max_d, min_samples_leaf=min_sl,
                                                       class_weight="balanced", random_state=42)
                    elif key == "rf":
                        model = RandomForestClassifier(class_weight="balanced", n_estimators=n_est, max_depth=max_d,
                               min_samples_leaf=min_sl, random_state=42, n_jobs=-1)
                    elif key == "lr":
                        model = LogisticRegression(C=C_reg, solver="lbfgs", max_iter=max_iter,
                                                   class_weight="balanced", random_state=42)
                    elif key == "knn":
                        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
                    elif key == "nb":
                        model = GaussianNB()
                    elif key == "xgb":
                        model = xgb.XGBClassifier(n_estimators=n_est, max_depth=max_d,
                                                   learning_rate=lr_val, random_state=42, n_jobs=-1)
                    elif key == "lgb":
                        model = lgb.LGBMClassifier(n_estimators=n_est, max_depth=max_d,
                                                   learning_rate=lr_val, class_weight="balanced",
                                                   random_state=42, n_jobs=-1, verbose=-1)
                    model.fit(X_train, y_train)

                if run_cv:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                                scoring="accuracy", n_jobs=-1)
                    st.info(f"🔁 **CV (k=5):** Accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                y_proba = model.predict_proba(X_test)
                y_pred  = model.predict(X_test)
                acc_train = accuracy_score(y_train, model.predict(X_train))
                m_test  = compute_all_metrics(y_test, y_pred, y_proba)
                gap = acc_train - m_test["Exactitud"]

                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Accuracy (Test)",  f"{m_test['Exactitud']:.4f}")
                c2.metric("🔧 Accuracy (Train)", f"{acc_train:.4f}")
                c3.metric("⚠️ Gap Train-Test",   f"{gap:.4f}",
                          delta="-Posible overfitting" if gap > 0.15 else "OK")

                if gap > 0.15:
                    st.warning(f"⚠️ Gap = {gap:.4f} > 0.15 → **sobreajuste**. "
                               "Prueba mayor min_samples_leaf o menor max_depth.")

                if use_val:
                    m_val = compute_all_metrics(y_val, model.predict(X_val),
                                                model.predict_proba(X_val))
                    cv1, cv2, cv3 = st.columns(3)
                    cv1.metric("Accuracy (Val)",  f"{m_val['Exactitud']:.4f}")
                    cv2.metric("F1-Score (Val)",  f"{m_val['F1_Score']:.4f}")
                    cv3.metric("AUC-ROC (Val)",   f"{m_val['AUC_ROC'] or 'N/A'}")

                st.divider()
                st.subheader("📐 Matriz de Confusión — Test Set")

                col_cm, col_roc = st.columns(2)
                with col_cm:
                    fig_cm, VN, FP, FN, VP = fig_confusion_matrix_detail(y_test, y_pred, name)
                    st.pyplot(fig_cm)
                    total = VN + FP + FN + VP
                    st.caption(f"VP={VP}, VN={VN}, FP={FP} (Error I), FN={FN} (Error II) | Total={total}")

                    # Métricas derivadas
                    m_cm = calcular_metricas_cm(VN, FP, FN, VP)
                    st.markdown("**Métricas derivadas:**")
                    cols_m = st.columns(3)
                    for idx, (k_m, v_m) in enumerate(m_cm.items()):
                        cols_m[idx % 3].metric(k_m, f"{v_m:.4f}")

                with col_roc:
                    roc_pc = m_test.get("roc_per_class", {})
                    if roc_pc:
                        roc_list = [(name, fpr_i, tpr_i, auc_i)
                                    for cls, (fpr_i, tpr_i, auc_i) in roc_pc.items()]
                        fig_roc = fig_roc_comparison(roc_list)
                        st.pyplot(fig_roc)
                        st.caption(f"📌 AUC-ROC = {m_test['AUC_ROC']:.4f}")

                # Semáforo AUC
                auc = m_test["AUC_ROC"] or 0
                color_s = "🟢" if auc >= 0.90 else "🟡" if auc >= 0.75 else "🟠" if auc >= 0.60 else "🔴"
                label_s = ("Excelente" if auc >= 0.90 else "Buena" if auc >= 0.75
                           else "Aceptable" if auc >= 0.60 else "Débil — revisar modelo")
                st.success(f"{color_s} **AUC-ROC = {auc:.4f} → {label_s}** | "
                           f"F1 = {m_test['F1_Score']:.4f} | "
                           f"Precisión = {m_test['Precisión']:.4f} | "
                           f"Sensibilidad = {m_test['Sensibilidad']:.4f}")

                # Importancia de variables
                if hasattr(model, "feature_importances_"):
                    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=True).tail(12)
                    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                    imp.plot(kind="barh", ax=ax_imp, color="#2ecc71", edgecolor="white", alpha=0.85)
                    ax_imp.set_title(f"Importancia de Variables — {name}", fontweight="bold")
                    ax_imp.set_xlabel("Importancia (Gini)")
                    ax_imp.grid(axis="x", alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_imp)
                    st.caption("📌 Variables con mayor importancia son las más influyentes en las predicciones.")

                # Reglas del árbol
                if key == "dt":
                    with st.expander("📋 Reglas del Árbol (texto)"):
                        rules = export_text(model, feature_names=feats, max_depth=5)
                        st.code(rules, language="text")

                # Classification report
                with st.expander("📋 Classification Report completo"):
                    rep = classification_report(y_test, y_pred, output_dict=True)
                    rows_r = [{"Clase": k, **{m: round(v[m], 4)
                               for m in ["precision", "recall", "f1-score", "support"]}}
                              for k, v in rep.items() if isinstance(v, dict)]
                    st.dataframe(pd.DataFrame(rows_r), use_container_width=True, hide_index=True)

                # Guardar para comparativa y PDF
                store_clf_result(name, m_test)
                st.session_state["pdf_cm_" + key] = save_fig_to_tmp(fig_cm)

                # Guardar importancia comparativa DT vs RF
                if key in ("dt", "rf"):
                    st.session_state[f"model_{key}"]   = model
                    st.session_state[f"m_test_{key}"]  = m_test
                    if "model_dt" in st.session_state and "model_rf" in st.session_state:
                        fig_fi = fig_feature_importance_compare(
                            st.session_state["model_dt"],
                            st.session_state["model_rf"],
                            "Árbol de Decisión", "Random Forest", feats)
                        if fig_fi:
                            st.session_state["pdf_feat_importance"] = save_fig_to_tmp(fig_fi)

                st.info(f"✅ Resultado de **{name}** guardado para la comparativa.")
                plt.close("all")


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════
def render_comparativa():
    st.header("📈 Comparativa de Modelos — CRISP-DM Fase 5 (Evaluación)")

    with st.expander("📖 Curva ROC y AUC"): st.markdown(EXP["roc"])

    history  = st.session_state.get("results_history", {})
    roc_data = st.session_state.get("roc_data", {})

    if len(history) < 2:
        st.info("Entrena al menos **2 modelos** para ver la comparativa.")
        return

    # Tabla comparativa
    st.subheader("📋 Tabla Comparativa de Modelos")
    df_comp = pd.DataFrame([{"Modelo": m, **v} for m, v in history.items()])
    if "AUC-ROC" in df_comp.columns:
        df_comp = df_comp.sort_values("AUC-ROC", ascending=False)
    st.dataframe(df_comp.set_index("Modelo"), use_container_width=True)

    # Gráfico comparativo
    fig_bar = fig_metrics_bar_comparison(history)
    st.pyplot(fig_bar)
    tmp_bar = save_fig_to_tmp(fig_bar)
    st.session_state["pdf_metrics_bar"] = tmp_bar

    # Curvas ROC superpuestas
    if roc_data:
        st.subheader("📉 Curvas ROC Superpuestas")
        roc_list = []
        for model_nm, class_dict in roc_data.items():
            if isinstance(class_dict, dict):
                for cls, (fpr_i, tpr_i, auc_i) in class_dict.items():
                    roc_list.append((model_nm, fpr_i, tpr_i, auc_i))
                    break
        if roc_list:
            fig_roc = fig_roc_comparison(roc_list)
            st.pyplot(fig_roc)
            tmp_roc = save_fig_to_tmp(fig_roc)
            st.session_state["pdf_roc_fig"] = tmp_roc
            st.caption("📌 El modelo con la curva más alejada de la diagonal (mayor área) es el mejor. "
                       "AUC = 0.5 equivale a adivinar al azar.")

    # Dashboard final
    st.subheader("📊 Dashboard Resumen — Evaluación Completa")
    sil_km = st.session_state.get("kmeans_sil")
    sil_hc = st.session_state.get("hc_sil")
    try:
        fig_dash = fig_dashboard_final(history, roc_data, sil_km, sil_hc)
        st.pyplot(fig_dash)
        st.session_state["pdf_dashboard"] = save_fig_to_tmp(fig_dash)
    except Exception as e:
        st.warning(f"Dashboard: {e}")

    # Resumen gerencial
    st.divider()
    st.subheader("📣 Resumen para Equipo Gerencial (Lenguaje no técnico)")
    with st.expander("Ver resumen ejecutivo", expanded=True):
        best_row = df_comp.iloc[0]
        best_model = best_row["Modelo"]
        best_auc   = best_row.get("AUC-ROC", 0)
        best_acc   = best_row.get("Accuracy", 0)
        best_f1    = best_row.get("F1", 0)
        grade_g = ("excelente" if best_auc >= 0.90 else "bueno" if best_auc >= 0.80
                   else "aceptable" if best_auc >= 0.70 else "mejorable")
        st.markdown(f"""
### 🏆 Mejor modelo: **{best_model}**

Se analizaron **{len(df_comp)} modelos** de inteligencia artificial sobre el dataset de estudiantes.
El modelo recomendado es **{best_model}**, que obtuvo:

- **Exactitud (Accuracy):** {best_acc:.2%} — de cada 100 predicciones, acierta ~{int(best_acc*100)}.
- **F1-Score:** {best_f1:.4f} — equilibrio entre detectar correctamente los que van a reprobar y no generar falsas alarmas.
- **AUC-ROC:** {best_auc:.4f} — capacidad global de discriminación, calificada como **{grade_g}**.

**¿Por qué importa el AUC-ROC?**
Si se ordenan 100 estudiantes por probabilidad de aprobar, el modelo asignará mayor
probabilidad al que realmente aprueba el **{best_auc:.0%} de las veces**.

**Recomendación:**
{'✅ El modelo está listo para una prueba piloto — priorizar tutorías para estudiantes con prob. < 40%.' if best_auc >= 0.75
else '⚠️ El modelo necesita mejoras antes de implementarse (más datos, tuning de hiperparámetros).'}

**Comparativa con el modelo de referencia (Baseline):**
Predecir siempre la clase mayoritaria da AUC = 0.50.
Nuestro mejor modelo supera eso en **{(best_auc - 0.5)*100:.1f} puntos porcentuales**.
        """)

    if st.button("🗑️ Limpiar historial de modelos"):
        st.session_state.results_history = {}
        st.session_state.roc_data = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN REPORTE PDF
# ══════════════════════════════════════════════════════════════════════════════
def render_pdf_report():
    st.header("📄 Generar Reporte PDF — Examen Minería de Datos")

    st.info("""
**¿Qué incluye el reporte?**
- Portada con datos del curso (Docente: MILTON EDWARD HUMPIRI FLORES)
- EDA completo con tablas y gráficos
- Partición y análisis de Data Leakage
- Segmentación K-Means y Jerárquico con dendrograma y silhouette
- Clasificación con matrices de confusión y métricas completas
- Evaluación comparativa con curvas ROC superpuestas
- Conclusiones y recomendaciones gerenciales
- Dashboard final anexo

⚠️ **Para incluir todos los gráficos**, primero ejecuta los algoritmos en las secciones anteriores.
    """)

    history = st.session_state.get("results_history", {})
    sil_km  = st.session_state.get("kmeans_sil")
    sil_hc  = st.session_state.get("hc_sil")

    col1, col2, col3 = st.columns(3)
    col1.metric("Modelos entrenados", len(history))
    col2.metric("K-Means ejecutado", "✅" if sil_km else "❌")
    col3.metric("Jerárquico ejecutado", "✅" if sil_hc else "❌")

    if st.button("🖨️ Generar Reporte PDF", type="primary", use_container_width=True):
        with st.spinner("Generando reporte PDF... esto puede tomar unos segundos."):
            try:
                # Preparar un diccionario con todo el estado necesario
                session_data = {
                    "df":               st.session_state.get("df"),
                    "target":           st.session_state.get("target", "passed"),
                    "feature_names":    st.session_state.get("feature_names", []),
                    "X_train":          st.session_state.get("X_train"),
                    "X_val":            st.session_state.get("X_val"),
                    "X_test":           st.session_state.get("X_test"),
                    "y_train":          st.session_state.get("y_train"),
                    "y_test":           st.session_state.get("y_test"),
                    "results_history":  st.session_state.get("results_history", {}),
                    "roc_data":         st.session_state.get("roc_data", {}),
                    "kmeans_sil":       sil_km,
                    "hc_sil":           sil_hc,
                    "km_k_final":       st.session_state.get("km_k_final", 4),
                    "hc_k_final":       st.session_state.get("hc_k_final", 4),
                    # Figuras guardadas como paths temporales
                    "pdf_km_elbow":     st.session_state.get("pdf_km_elbow"),
                    "pdf_km_sil":       st.session_state.get("pdf_km_sil"),
                    "pdf_km_pca":       st.session_state.get("pdf_km_pca"),
                    "pdf_km_profiles":  st.session_state.get("pdf_km_profiles"),
                    "pdf_hc_dend":      st.session_state.get("pdf_hc_dend"),
                    "pdf_hc_pca":       st.session_state.get("pdf_hc_pca"),
                    "pdf_cm_dt":        st.session_state.get("pdf_cm_dt"),
                    "pdf_cm_rf":        st.session_state.get("pdf_cm_rf"),
                    "pdf_feat_importance": st.session_state.get("pdf_feat_importance"),
                    "pdf_roc_fig":      st.session_state.get("pdf_roc_fig"),
                    "pdf_metrics_bar":  st.session_state.get("pdf_metrics_bar"),
                    "pdf_dashboard":    st.session_state.get("pdf_dashboard"),
                }

                # Si no hay figuras de comparativa, generarlas ahora
                if not session_data["pdf_roc_fig"]:
                    roc_data = st.session_state.get("roc_data", {})
                    if roc_data:
                        roc_list = [(nm, fpr_i, tpr_i, auc_i)
                                    for nm, cd in roc_data.items()
                                    for cls, (fpr_i, tpr_i, auc_i) in cd.items()
                                    if isinstance(cd, dict)][:4]
                        if roc_list:
                            session_data["pdf_roc_fig"] = save_fig_to_tmp(fig_roc_comparison(roc_list))

                if not session_data["pdf_metrics_bar"] and history:
                    session_data["pdf_metrics_bar"] = save_fig_to_tmp(fig_metrics_bar_comparison(history))

                if not session_data["pdf_dashboard"] and history:
                    session_data["pdf_dashboard"] = save_fig_to_tmp(
                        fig_dashboard_final(history, st.session_state.get("roc_data", {}), sil_km, sil_hc))

                pdf_bytes = build_pdf_report(session_data)
                st.success("✅ Reporte generado exitosamente.")
                st.download_button(
                    label="⬇️ Descargar Reporte PDF",
                    data=pdf_bytes,
                    file_name="Examen_Mineria_Datos_CRISP_DM.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error al generar el PDF: {e}")
                import traceback
                st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.title("🎓 Minería de Datos — Sistema CRISP-DM")
    st.caption("Dataset: Student Performance (UCI) | Docente: MILTON EDWARD HUMPIRI FLORES")

    # Sidebar
    with st.sidebar:
        st.header("📁 Cargar Dataset")
        st.caption("Formatos: CSV (sep=;), Excel, JSON")
        uploaded = st.file_uploader("Subir archivo",
                                    type=["csv", "tsv", "xlsx", "xls", "json"])

        if uploaded:
            try:
                df = load_any_file(uploaded)
                if df is None or len(df) == 0:
                    st.error("Archivo vacío o no reconocido.")
                    st.stop()
                st.session_state.df = df
                st.success(f"✅ {len(df):,} filas × {len(df.columns)} cols")
            except Exception as e:
                st.error(f"Error al leer: {e}")
                st.stop()

        if "df" not in st.session_state:
            st.info("Sube el archivo **student-por.csv** o **student-mat.csv** para comenzar.")
            st.stop()

        df = st.session_state.df

        st.header("⚙️ Configuración")
        target = st.selectbox("Variable objetivo (target)",
                              df.columns.tolist(),
                              index=len(df.columns)-1)
        cols_to_drop = st.multiselect(
            "Columnas a excluir (opcional)",
            [c for c in df.columns if c != target],
            help="Selecciona variables que causarían leakage (ej: G1, G2, G3).")

        st.markdown("**Partición:**")
        split_mode = st.radio("Esquema de partición",
                              ["80/20 (Train/Test)",
                               "65/15/20 (Train/Val/Test)",
                               "70/10/20 (Train/Val/Test)",
                               "Personalizado"],
                              index=1)
        if split_mode == "80/20 (Train/Test)":
            val_pct, test_pct = 0.0, 0.20
        elif split_mode == "65/15/20 (Train/Val/Test)":
            val_pct, test_pct = 0.15, 0.20
        elif split_mode == "70/10/20 (Train/Val/Test)":
            val_pct, test_pct = 0.10, 0.20
        else:
            val_pct  = st.slider("% Validación", 5, 30, 15) / 100
            test_pct = st.slider("% Test",       5, 30, 20) / 100

        if st.button("🔍 Analizar y Preparar", type="primary", use_container_width=True):
            with st.spinner("Analizando y preparando datos..."):
                profile = profile_dataframe(df)
                out = auto_preprocess(df, target, profile, val_pct, test_pct,
                                      manual_drop=cols_to_drop)
                X_tr, X_v, X_te, y_tr, y_v, y_te, feats, log = out
                st.session_state.update(dict(
                    target=target, profile=profile, task="classification",
                    X_train=X_tr, X_val=X_v, X_test=X_te,
                    y_train=y_tr, y_val=y_v, y_test=y_te,
                    feature_names=feats, prep_log=log,
                    results_history={}, roc_data={},
                    kmeans_sil=None, hc_sil=None,
                ))
            st.success(f"✅ Features: **{len(feats)}** | Target: **{target}**")

        if "task" in st.session_state:
            st.divider()
            tr = len(st.session_state.X_train)
            va = len(st.session_state.X_val)
            te = len(st.session_state.X_test)
            tot = tr + va + te
            st.markdown(f"**Target:** `{st.session_state.target}`")
            st.markdown(f"**Features:** `{len(st.session_state.feature_names)}`")
            st.markdown(f"**Split:** {tr/tot:.0%} / {va/tot:.0%} / {te/tot:.0%}")

    # Pantalla de bienvenida
    if "profile" not in st.session_state:
        st.markdown("""
## 👋 Bienvenido — Sistema CRISP-DM para Examen

**Dataset recomendado:** `student-por.csv` (separador `;`)

| Fase | Pestaña | Contenido |
|------|---------|-----------|
| **2. Data Understanding** | 📊 EDA | Distribuciones, correlación, perfilado |
| **3. Data Preparation**   | ⚙️ Preparación | Limpieza, encoding, partición 3-way, Data Leakage, Baseline |
| **4. Modeling**           | 📉 Segmentación | K-Means + Clustering Jerárquico + Silhouette |
| **4. Modeling**           | 🌳 Clasificación | Árbol · Random Forest · KNN · NB + Matriz de Confusión completa |
| **5. Evaluation**         | 📈 Comparativa | ROC superpuestas · tabla comparativa · resumen gerencial |
| **Exportar**              | 📄 Reporte PDF | Informe listo para presentar — sin código |

---
⬅️ Sube tu dataset y haz clic en **Analizar y Preparar**.

> **Sugerencia de target:** crea la columna `passed = (G3 >= 10).astype(int)` antes de subir,
> o sube el archivo tal cual y elige `G3` como target (el sistema lo detectará como clasificación binaria si usas un threshold).
        """)
        return

    # Tabs principales
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 EDA",
        "⚙️ Preparación",
        "📉 Segmentación",
        "🌳 Clasificación",
        "📈 Comparativa",
        "📄 Reporte PDF",
    ])
    with t1: render_eda()
    with t2: render_preparacion()
    with t3: render_segmentacion()
    with t4: render_clasificacion()
    with t5: render_comparativa()
    with t6: render_pdf_report()


if __name__ == "__main__":
    main()