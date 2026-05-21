"""
Minería de Datos — Sistema CRISP-DM Completo
Python 3.12.9  |  Ejecutar: streamlit run app.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import io, numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              roc_auc_score, roc_curve, silhouette_score,
                              mean_absolute_error, mean_squared_error, mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Minería de Datos CRISP-DM",
                   page_icon="⛏️", layout="wide",
                   initial_sidebar_state="expanded")

# ── Explicaciones teóricas (reutilizadas en la UI) ─────────────────────────────
EXP = {
"data_leakage":
"""**¿Qué es el Data Leakage?**
El data leakage ocurre cuando información del conjunto de prueba (o del futuro)
se filtra al entrenamiento, causando que el modelo parezca más preciso de lo que
realmente es. **Cómo evitarlo aquí:**
- **Detección automática**: el sistema excluye variables ("post-evento") que predicen el target con precisión irreal (AUC > 0.95).
- El `StandardScaler` se ajusta **solo** con `X_train` y luego se aplica a val/test.
- El `LabelEncoder` aprende categorías solo del train.
- Las estadísticas de imputación (mediana/moda) se calculan solo en train.
Normalizar **antes** de partir sería data leakage.""",

"split":
"""**Partición Train / Validación / Prueba**
| Conjunto | Uso | Tamaño típico |
|----------|-----|---------------|
| **Train** | Ajusta los parámetros del modelo | 60–70 % |
| **Validación** | Ajusta hiperparámetros, selección de modelo | 15–20 % |
| **Test** | Evaluación final, **solo se usa una vez** | 15–20 % |
La partición es **estratificada** para mantener la proporción de clases.""",

"baseline":
"""**Modelo Baseline (línea base)**
Un baseline es el resultado mínimo que cualquier modelo útil debe superar.
Usamos `DummyClassifier(strategy='most_frequent')` que siempre predice la clase
más frecuente. Si un modelo no supera al baseline, no aporta valor real.""",

"kmeans":
"""**K-Means Clustering**
Algoritmo iterativo que agrupa n registros en K clústeres minimizando la
*inercia* (suma de distancias cuadradas al centroide).
**Método del codo:** elegimos K donde la reducción de inercia se estabiliza.
**Limitación:** asume clústeres esféricos y requiere definir K a priori.""",

"hierarchical":
"""**Clustering Jerárquico Aglomerativo**
Parte con cada registro como su propio clúster y los va fusionando según
el criterio de enlace (**Ward** minimiza la varianza interna).
El **dendrograma** visualiza la jerarquía; cortando a cierta altura obtenemos K grupos.
**Ventaja:** no requiere definir K antes; **desventaja:** costoso en datasets grandes.""",

"silhouette":
"""**Índice Silhouette** ∈ [-1, 1]
Mide qué tan bien separado está cada punto respecto a su clúster:
- **≈ +1**: el punto está bien asignado y lejos de otros clústeres.
- **≈ 0**: el punto está en el borde entre dos clústeres.
- **< 0**: el punto probablemente está en el clúster equivocado.
Promedio alto → clústeres bien definidos.""",

"decision_tree":
"""**Árbol de Decisión (CART)**
Modelo interpretable que divide el espacio de features en regiones mediante
reglas binarias. Cada nodo pregunta sobre una feature; las hojas son las
predicciones. **Ventaja:** altamente interpretable. **Limitación:** propenso a
overfitting si no se poda (`max_depth`).""",

"random_forest":
"""**Random Forest**
Ensemble de N árboles entrenados con muestras aleatorias del dataset
(**bagging**) y subconjuntos aleatorios de features en cada nodo.
La predicción final es la mayoría de votos.
**Ventajas:** robusto, maneja no-linealidad, provee importancia de variables.
**Limitación:** menos interpretable que un único árbol.""",

"knn":
"""**K-Nearest Neighbors (KNN)**
Clasifica una muestra según la clase mayoritaria de sus K vecinos más cercanos 
en el espacio de características. Es simple pero sensible a la escala de los datos.""",

"naive_bayes":
"""**Naive Bayes**
Basado en el Teorema de Bayes asumiendo independencia entre las features. 
Es extremadamente rápido y funciona muy bien con datos de alta dimensionalidad o texto.""",

"xgboost":
"""**XGBoost (eXtreme Gradient Boosting)**
Algoritmo de ensamble que construye árboles secuencialmente para corregir errores 
de los anteriores. Altamente optimizado, maneja valores nulos y es el estándar 
de oro en competencias de Machine Learning.""",

"lightgbm":
"""**LightGBM**
Framework de Microsoft similar a XGBoost pero optimizado para velocidad y eficiencia 
de memoria. Divide los árboles por hojas (leaf-wise) en lugar de niveles, logrando alta precisión en tiempos récord.""",

"confusion_matrix":
"""**Matriz de Confusión** — VP=Verdaderos Positivos, VN=Verdaderos Negativos,
FP=Falsos Positivos (Error Tipo I), FN=Falsos Negativos (Error Tipo II).
Cada métrica responde una pregunta diferente sobre el desempeño del modelo
dependiendo del costo de cada tipo de error en el problema de negocio.""",

"roc":
"""**Curva ROC (Receiver Operating Characteristic)**
Muestra la tasa de verdaderos positivos vs falsos positivos para todos los
umbrales de clasificación posibles. **AUC** (Área Bajo la Curva) resume la
curva en un solo número: 1.0 = perfecto, 0.5 = aleatorio, <0.5 = peor que azar.""",
}


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE ARCHIVOS (multi-formato)
# ══════════════════════════════════════════════════════════════════════════════
def load_any_file(uploaded) -> pd.DataFrame:
    """Carga CSV, Excel, JSON, Parquet, TSV detectando el formato por extensión."""
    name = uploaded.name.lower()
    raw  = uploaded.read()
    buf  = io.BytesIO(raw)

    if name.endswith(".csv"):
        # Detectar separador automáticamente
        sample = raw[:4096].decode("utf-8", errors="ignore")
        sep = ";" if sample.count(";") > sample.count(",") else ","
        buf.seek(0)
        return pd.read_csv(buf, sep=sep, encoding="utf-8", on_bad_lines="skip")
    elif name.endswith(".tsv") or name.endswith(".txt"):
        buf.seek(0)
        return pd.read_csv(buf, sep="\t", encoding="utf-8", on_bad_lines="skip")
    elif name.endswith((".xlsx", ".xls")):
        buf.seek(0)
        return pd.read_excel(buf)
    elif name.endswith(".json"):
        buf.seek(0)
        return pd.read_json(buf)
    elif name.endswith(".parquet"):
        buf.seek(0)
        return pd.read_parquet(buf)
    else:
        # Intentar CSV por defecto
        buf.seek(0)
        return pd.read_csv(buf, on_bad_lines="skip")


# ══════════════════════════════════════════════════════════════════════════════
#  PERFILADO AUTOMÁTICO
# ══════════════════════════════════════════════════════════════════════════════
def profile_dataframe(df: pd.DataFrame) -> dict:
    n = len(df)
    profile = {}
    for col in df.columns:
        s        = df[col]
        n_null   = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        is_num   = pd.api.types.is_numeric_dtype(s)
        is_dt    = pd.api.types.is_datetime64_any_dtype(s)

        if not is_dt and s.dtype == object:
            try:
                pd.to_datetime(s.dropna().head(20), infer_datetime_format=True)
                is_dt = True
            except Exception:
                pass

        if is_dt:
            col_type = "datetime"
        elif n_unique == 2:
            col_type = "binary"
        elif is_num and n_unique > 20 and n_unique / n > 0.05:
            col_type = "continuous"
        elif is_num:
            col_type = "discrete"
        elif s.dtype == object and n_unique > max(100, 0.5 * n):
            col_type = "high_cardinality"
        else:
            col_type = "categorical"

        impute   = ("none" if n_null == 0
                    else ("median" if is_num and abs(s.dropna().skew()) > 1
                          else ("mean" if is_num else "mode")))
        encoding = ("none"  if col_type in ("continuous","discrete","datetime")
                    else "drop"  if col_type == "high_cardinality"
                    else "label" if col_type == "binary"
                    else "onehot" if n_unique <= 8
                    else "label")

        profile[col] = dict(col_type=col_type, is_num=is_num,
                            n_null=n_null, null_pct=round(n_null/n*100,2),
                            n_unique=n_unique, impute=impute, encoding=encoding)
    return profile


def detect_task(df, target, profile):
    p = profile[target]
    if p["col_type"] in ("binary","categorical","discrete") and p["n_unique"] <= 20:
        return "classification"
    return "regression"


def detect_leakage(df, target, task, profile):
    """Detecta variables que causan data leakage (ej. AUC > 0.95 con una sola regla)."""
    leakage_cols = []
    logs = []
    y = df[target]
    
    if task == "classification":
        le = LabelEncoder()
        try:
            y_enc = le.fit_transform(y.astype(str))
            is_binary = len(le.classes_) == 2
            
            for col in df.columns:
                if col == target: continue
                p = profile[col]
                if p["null_pct"] > 50 or p["encoding"] == "drop" or p["col_type"] == "datetime": continue
                
                # Probar predicción con un árbol de decisión simple (1 regla)
                dt = DecisionTreeClassifier(max_depth=1, random_state=42)
                X_col = df[[col]].copy()
                
                try:
                    if p["is_num"]:
                        X_col = X_col.fillna(X_col.median())
                    else:
                        X_col = pd.get_dummies(X_col[col])
                        
                    dt.fit(X_col, y_enc)
                    
                    if is_binary:
                        preds = dt.predict_proba(X_col)[:, 1]
                        auc = roc_auc_score(y_enc, preds)
                    else:
                        preds = dt.predict_proba(X_col)
                        auc = roc_auc_score(y_enc, preds, multi_class="ovr")
                        
                    if auc > 0.95:
                        leakage_cols.append(col)
                        logs.append(f"🚨 **Leakage detectado:** `{col}` predice el target casi perfectamente (AUC = {auc:.3f}). Excluida (posible variable post-evento).")
                except Exception:
                    pass
        except Exception:
            pass
            
    else: # regression
        for col in df.columns:
            if col == target: continue
            if profile[col]["is_num"]:
                corr = abs(df[col].corr(pd.to_numeric(y, errors='coerce')))
                if corr > 0.95:
                    leakage_cols.append(col)
                    logs.append(f"🚨 **Leakage detectado:** `{col}` altamente correlacionada con el target (|r| = {corr:.3f}). Excluida (posible variable post-evento).")
                    
    return leakage_cols, logs


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESAMIENTO 3-WAY (Train / Val / Test) — sin data leakage
# ══════════════════════════════════════════════════════════════════════════════
def auto_preprocess(df, target, profile, val_pct=0.15, test_pct=0.20, manual_drop=None):
    """
    Pipeline libre de data leakage:
    0. Eliminar duplicados y target nulos.
    1. Eliminar columnas no útiles.
    2. Imputar nulos (fit solo en train).
    3. Encoding (fit solo en train).
    4. Escalar (fit solo en train).
    5. Split estratificado Train/Val/Test.
    """
    log  = []
    work = df.copy()

    # 0. Eliminar duplicados y filas con target nulo
    dup_before = len(work)
    work = work.dropna(subset=[target])
    work = work.drop_duplicates()
    dups_removed = dup_before - len(work)
    if dups_removed > 0:
        log.append(f"🧹 Duplicados o target nulo eliminados: {dups_removed:,} filas")

    y = work.pop(target).reset_index(drop=True)
    work.reset_index(drop=True, inplace=True)

    task = detect_task(df, target, profile)
    
    # 1. Detección automática de Data Leakage
    leak_cols, leak_logs = detect_leakage(df, target, task, profile)
    log.extend(leak_logs)

    # 2. Eliminar columnas problemáticas o con leakage
    drop = [c for c, p in profile.items()
            if c != target and (p["null_pct"] > 50 or
                                p["encoding"] == "drop" or
                                p["col_type"] == "datetime")]
    drop = list(set(drop + leak_cols))
    if manual_drop:
        drop = list(set(drop + manual_drop))
        
    if drop:
        work.drop(columns=[c for c in drop if c in work.columns], inplace=True)
        log.append(f"🗑️ Eliminadas ({len(drop)} cols): `{'`, `'.join(drop)}`")

    # 3. Codificar target si categórico
    le_target = LabelEncoder()
    if not pd.api.types.is_numeric_dtype(y) or y.dtype == object or str(y.dtype) == "category" or str(y.dtype) == "string":
        y = pd.Series(le_target.fit_transform(y.astype(str)), name=target)
        log.append(f"🎯 Target `{target}` codificado: {list(le_target.classes_)}")

    # 4. Partición temporal (soporta val_pct = 0.0)
    is_ts = (task == "timeseries")
    stratify = y if task == "classification" and y.nunique() <= 20 and not is_ts else None
    
    X_work, X_test, y_work, y_test = train_test_split(
        work, y, test_size=test_pct, random_state=42, 
        stratify=stratify, shuffle=not is_ts)

    use_val = val_pct > 0.0
    if use_val:
        val_ratio = val_pct / (1 - test_pct)
        strat2    = y_work if stratify is not None and not is_ts else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_work, y_work, test_size=val_ratio, random_state=42, 
            stratify=strat2, shuffle=not is_ts)
    else:
        X_train, X_val = X_work.copy(), pd.DataFrame(columns=X_work.columns)
        y_train, y_val = y_work.copy(), pd.Series([], dtype=y_work.dtype)
        
    if is_ts:
        log.append("⏳ Partición **cronológica** (sin shuffle).")

    log.append(f"✂️ Split → train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")

    # 5. Imputar — fit SOLO en train
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    splits_all = [s for s in (X_train, X_val, X_test) if len(s) > 0]
    
    if num_cols and X_train[num_cols].isna().any().any():
        n_nulls = X_train[num_cols].isna().sum().sum()
        imp_n = SimpleImputer(strategy="median").fit(X_train[num_cols])
        for split in splits_all:
            split[num_cols] = imp_n.transform(split[num_cols])
        log.append(f"🔧 Nulos numéricos ({n_nulls}) → mediana (fit en train)")

    if cat_cols and X_train[cat_cols].isna().any().any():
        n_nulls_c = X_train[cat_cols].isna().sum().sum()
        imp_c = SimpleImputer(strategy="most_frequent").fit(X_train[cat_cols])
        for split in splits_all:
            split[cat_cols] = imp_c.transform(split[cat_cols])
        log.append(f"🔧 Nulos categóricos ({n_nulls_c}) → moda (fit en train)")

    # 6. Encoding — fit SOLO en train
    val_splits = [s for s in (X_val, X_test) if len(s) > 0]
    ohe_done, le_done = [], []
    for col in cat_cols:
        enc = profile.get(col, {}).get("encoding", "label")
        if enc == "onehot":
            dummies_tr = pd.get_dummies(X_train[col], prefix=col, drop_first=True, dtype=int)
            X_train = pd.concat([X_train.drop(columns=[col]), dummies_tr], axis=1)
            for split in val_splits:
                d = pd.get_dummies(split[col], prefix=col, drop_first=True, dtype=int)
                split.drop(columns=[col], inplace=True)
                for c in dummies_tr.columns:
                    if c not in d.columns:
                        d[c] = 0
                split[list(dummies_tr.columns)] = d[list(dummies_tr.columns)].values
            ohe_done.append(col)
        else:
            le = LabelEncoder().fit(X_train[col].astype(str))
            for split in splits_all:
                split[col] = split[col].astype(str).map(
                    lambda v, le=le: le.transform([v])[0]
                    if v in le.classes_ else -1)
            le_done.append(col)

    if ohe_done: log.append(f"🔠 OneHotEncoding → `{'`, `'.join(ohe_done)}`")
    if le_done:  log.append(f"🔢 LabelEncoding  → `{'`, `'.join(le_done)}`  (fit en train)")

    # 7. Escalar — fit SOLO en train
    feats  = X_train.columns.tolist()
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=feats)
    X_test  = pd.DataFrame(scaler.transform(X_test),  columns=feats)
    if len(X_val) > 0:
        X_val = pd.DataFrame(scaler.transform(X_val), columns=feats)
    else:
        X_val = pd.DataFrame(columns=feats)
    log.append("📏 StandardScaler (fit en train, transform en val+test → sin data leakage)")

    return X_train, X_val, X_test, y_train, y_val, y_test, feats, log


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS COMPLETAS DE MATRIZ DE CONFUSIÓN
# ══════════════════════════════════════════════════════════════════════════════
def compute_all_metrics(y_true, y_pred, y_proba=None):
    """Calcula todas las métricas del diagrama de la Matriz de Confusión."""
    cm = confusion_matrix(y_true, y_pred)
    binary = cm.shape == (2, 2)

    if binary:
        VN, FP, FN, VP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    else:
        VP = int(np.diag(cm).sum())
        FP = int((cm.sum(axis=0) - np.diag(cm)).sum())
        FN = int((cm.sum(axis=1) - np.diag(cm)).sum())
        VN = int(cm.sum()) - VP - FP - FN
    VP, FP, FN, VN = int(VP), int(FP), int(FN), int(VN)
    total = VP + FP + FN + VN

    def safe(num, den): return round(num/den, 4) if den else 0.0

    metrics = dict(
        VP=VP, FP=FP, FN=FN, VN=VN,
        Precisión          = safe(VP, VP+FP),
        Exactitud          = safe(VP+VN, total),
        Especificidad      = safe(VN, VN+FP),
        Sensibilidad       = safe(VP, VP+FN),
        Tasa_FN            = safe(FN, FN+VP),
        VPP                = safe(VP, FP+VP),
        VPN                = safe(VN, VN+FN),
        F1_Score           = safe(2*VP, 2*VP+FP+FN),
        AUC_ROC            = None,
        cm_matrix          = cm,
        binary             = binary,
    )
    if y_proba is not None:
        try:
            classes = np.unique(y_true)
            if binary:
                metrics["AUC_ROC"] = round(roc_auc_score(y_true, y_proba[:,1]), 4)
                fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
                metrics["roc"] = (fpr, tpr)
                metrics["roc_per_class"] = {"clase 1": (fpr, tpr, metrics["AUC_ROC"])}
            else:
                metrics["AUC_ROC"] = round(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"), 4)
                # ROC por clase (micro-average para graficar)
                y_bin = label_binarize(y_true, classes=classes)
                roc_per = {}
                for i, cls in enumerate(classes):
                    if y_bin[:, i].sum() > 0:
                        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                        auc_i = round(roc_auc_score(y_bin[:, i], y_proba[:, i]), 4)
                        roc_per[str(cls)] = (fpr_i, tpr_i, auc_i)
                metrics["roc_per_class"] = roc_per
        except Exception:
            pass
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════
def fig_confusion_heatmap(m, name):
    cm = m["cm_matrix"]
    labels = [str(i) for i in range(cm.shape[0])]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=labels, y=labels,
                    labels=dict(x="Predicho", y="Real"),
                    title=f"Matriz de Confusión — {name}")
    fig.update_layout(height=350, coloraxis_showscale=False)
    return fig


def show_metrics_cards(m):
    """Muestra tarjetas de métricas estilo el diagrama del PDF."""
    if m["binary"]:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("VP (Verdaderos +)", m["VP"])
        c2.metric("FP (Falsos +) Error I", m["FP"])
        c3.metric("FN (Falsos -) Error II", m["FN"])
        c4.metric("VN (Verdaderos -)", m["VN"])

    st.markdown("#### 📐 Métricas derivadas")
    row1 = st.columns(4)
    row1[0].metric("Precisión",   f"{m['Precisión']:.4f}",
                   help="VP / (VP+FP)")
    row1[1].metric("Exactitud",   f"{m['Exactitud']:.4f}",
                   help="(VP+VN) / Total")
    row1[2].metric("Especificidad",f"{m['Especificidad']:.4f}",
                   help="VN / (VN+FP)")
    row1[3].metric("Sensibilidad", f"{m['Sensibilidad']:.4f}",
                   help="VP / (VP+FN) — Recall")

    row2 = st.columns(4)
    row2[0].metric("F1-Score",    f"{m['F1_Score']:.4f}",
                   help="2·VP / (2·VP+FP+FN)")
    row2[1].metric("Tasa FN",     f"{m['Tasa_FN']:.4f}",
                   help="FN / (FN+VP)")
    row2[2].metric("VPP",         f"{m['VPP']:.4f}",
                   help="Valor Predictivo Positivo: VP/(FP+VP)")
    row2[3].metric("AUC-ROC",     f"{m['AUC_ROC'] or 'N/A'}",
                   help="Área bajo la curva ROC")


def fig_roc_single(fpr, tpr, auc, name, color="darkorange"):
    fig = go.Figure([
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.4f})",
                   line=dict(color=color, width=2)),
        go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar",
                   line=dict(color="gray", dash="dash"))
    ])
    fig.update_layout(title=f"Curva ROC — {name}", height=360,
                      xaxis_title="Tasa FP", yaxis_title="Tasa VP",
                      legend=dict(x=0.55, y=0.1))
    return fig


def fig_feature_importance(model, features, name, n=15):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True).tail(n)
    fig = px.bar(imp, orientation="h", title=f"Importancia de Variables — {name}",
                 labels={"value":"Importancia","index":"Variable"})
    fig.update_layout(height=400, showlegend=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN EDA
# ══════════════════════════════════════════════════════════════════════════════
def render_eda():
    df      = st.session_state.df
    target  = st.session_state.target
    profile = st.session_state.profile
    task    = st.session_state.task

    st.header("📊 Comprensión de los Datos — CRISP-DM Fase 2")

    with st.expander("📖 ¿Qué es la Comprensión de Datos en CRISP-DM?", expanded=False):
        st.markdown("""
En la metodología CRISP-DM la **fase 2 (Data Understanding)** busca:
- Describir el dataset: filas, columnas, tipos de datos.
- Identificar problemas de calidad: nulos, duplicados, outliers.
- Descubrir relaciones entre variables (análisis bivariado).
Esta fase informa las decisiones de la **fase 3 (Data Preparation)**.
        """)

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Filas",     f"{len(df):,}")
    c2.metric("Columnas",  len(df.columns))
    c3.metric("Nulos",     df.isna().sum().sum())
    c4.metric("Duplicados",df.duplicated().sum())
    c5.metric("Tarea",     task.capitalize())

    st.divider()

    # Perfil
    with st.expander("📋 Perfil automático de columnas — decisiones de limpieza", expanded=True):
        rows = [{"Columna":c, "Tipo":p["col_type"], "Dtype":str(df[c].dtype),
                 "Únicos":p["n_unique"], "Nulos %":p["null_pct"],
                 "Imputación":p["impute"], "Encoding sugerido":p["encoding"]}
                for c, p in profile.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.info("**Encoding:** `onehot` para categóricas de baja cardinalidad (≤8 valores) — "
                "evita imponer orden artificial. `label` para alta cardinalidad o binarias. "
                "`drop` para columnas de texto libre o >50 % nulos — no aportan señal.")

    # Target
    st.subheader(f"🎯 Variable Objetivo: `{target}` — Tarea: {task}")
    vc = df[target].value_counts().reset_index()
    vc.columns = [target, "count"]
    vc[target] = vc[target].astype(str)
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(vc, x=target, y="count", color=target,
                     text="count", title="Distribución del Target")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.pie(vc, names=target, values="count",
                      title="Proporción de Clases", hole=0.35)
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)

    if task == "classification":
        min_r = df[target].value_counts(normalize=True).min()
        if min_r < 0.10:
            st.warning(f"⚠️ **Dataset desbalanceado**: clase minoritaria = {min_r:.2%}. "
                       "Los modelos tenderán a ignorar la clase rara. "
                       "Considera `class_weight='balanced'` o técnicas de resampling.")

    st.divider()

    # Numéricas
    num_cols = [c for c, p in profile.items() if p["is_num"] and c != target]
    if num_cols:
        st.subheader("📈 Distribuciones Numéricas y Outliers")
        st.caption("Se muestra histograma + boxplot marginal. Una distribución muy sesgada "
                   "sugiere usar **mediana** para imputar y puede beneficiarse de transformación log.")
        for i in range(0, min(len(num_cols), 9), 3):
            batch = num_cols[i:i+3]
            cols  = st.columns(len(batch))
            for ui, c in zip(cols, batch):
                with ui:
                    skew = df[c].skew()
                    fig = px.histogram(df, x=c, nbins=30, marginal="box",
                                       title=f"{c}  (sesgo={skew:.2f})", height=270)
                    fig.update_layout(showlegend=False, margin=dict(t=40,b=10),
                                      title_font_size=11)
                    st.plotly_chart(fig, use_container_width=True)

    # Categóricas
    cat_cols = [c for c, p in profile.items()
                if p["col_type"] in ("categorical","binary","discrete")
                and c != target and p["n_unique"] <= 20]
    if cat_cols:
        st.subheader("📊 Distribuciones Categóricas")
        for i in range(0, min(len(cat_cols),6), 3):
            batch = cat_cols[i:i+3]
            cols  = st.columns(len(batch))
            for ui, c in zip(cols, batch):
                with ui:
                    vc_c = df[c].value_counts().reset_index()
                    vc_c.columns = [c,"n"]
                    fig = px.bar(vc_c, x=c, y="n", title=c, height=260)
                    fig.update_layout(showlegend=False, margin=dict(t=40,b=10),
                                      title_font_size=11)
                    st.plotly_chart(fig, use_container_width=True)

    # Correlación
    if len(num_cols) >= 3:
        st.divider()
        st.subheader("🔥 Correlación de Pearson")
        st.caption("Valores cercanos a ±1 indican relación lineal fuerte. "
                   "Alta correlación entre features sugiere redundancia (posible eliminación o PCA).")
        cols_c = (num_cols+[target]) if profile[target]["is_num"] else num_cols
        corr   = df[cols_c].corr()
        fig    = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                           aspect="auto", height=max(350, len(cols_c)*35))
        st.plotly_chart(fig, use_container_width=True)

    # Bivariado
    if task == "classification" and num_cols:
        st.divider()
        st.subheader("🔍 Features Numéricas vs Target")
        st.caption("Boxplots por clase: solapamiento alto → la variable discrimina poco. "
                   "Medias separadas → buena variable predictora.")
        df_p = df.copy(); df_p[target] = df_p[target].astype(str)
        for i in range(0, min(len(num_cols),6), 3):
            batch = num_cols[i:i+3]
            cols  = st.columns(len(batch))
            for ui, c in zip(cols, batch):
                with ui:
                    fig = px.box(df_p, x=target, y=c, color=target,
                                 title=f"{c} vs {target}", height=270)
                    fig.update_layout(showlegend=False, margin=dict(t=40,b=10),
                                      title_font_size=11)
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN PREPARACIÓN & PARTICIÓN
# ══════════════════════════════════════════════════════════════════════════════
def render_preparacion():
    st.header("⚙️ Preparación de Datos — CRISP-DM Fase 3")

    with st.expander("📖 Data Leakage — definición y cómo evitarlo", expanded=True):
        st.markdown(EXP["data_leakage"])

    with st.expander("📖 Partición Train / Validación / Test", expanded=True):
        st.markdown(EXP["split"])

    with st.expander("📖 Modelo Baseline", expanded=False):
        st.markdown(EXP["baseline"])

    st.divider()

    log          = st.session_state.prep_log
    X_train      = st.session_state.X_train
    X_val        = st.session_state.X_val
    X_test       = st.session_state.X_test
    y_train      = st.session_state.y_train
    y_val        = st.session_state.y_val
    y_test       = st.session_state.y_test
    feats        = st.session_state.feature_names
    task         = st.session_state.task

    # Pasos aplicados
    st.subheader("✅ Pasos de limpieza y codificación aplicados")
    for d in log:
        st.markdown(f"- {d}")

    st.divider()

    # Tamaños del split
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Features resultantes", len(feats))
    c2.metric("Train",      f"{len(X_train):,}")
    c3.metric("Validación", f"{len(X_val):,}" if len(X_val)>0 else "No usado")
    c4.metric("Test",       f"{len(X_test):,}")

    # Visualizar distribución de la partición
    splits_df = pd.DataFrame({
        "Conjunto": ["Train","Validación","Test"],
        "Registros": [len(X_train), len(X_val), len(X_test)]
    })
    splits_df = splits_df[splits_df["Registros"] > 0]
    fig = px.pie(splits_df, names="Conjunto", values="Registros",
                 title="Distribución de la Partición", hole=0.35,
                 color_discrete_sequence=["#2ecc71","#f39c12","#e74c3c"])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Balance de clases en cada conjunto
    if task == "classification":
        st.subheader("⚖️ Balance de clases por conjunto")
        st.caption("Un buen split estratificado mantiene la misma proporción de clases en "
                   "los tres conjuntos, garantizando que las métricas sean comparables.")
        col_l, col_c, col_r = st.columns(3)
        for col_ui, nm, yy in [(col_l,"Train",y_train),
                                (col_c,"Validación",y_val),
                                (col_r,"Test",y_test)]:
            if len(yy) > 0:
                with col_ui:
                    u, cnt = np.unique(yy, return_counts=True)
                    df_b = pd.DataFrame({"Clase":u.astype(str), "Proporción":cnt/cnt.sum()})
                    fig = px.bar(df_b, x="Clase", y="Proporción", title=nm,
                                 text_auto=".2%", height=280)
                    fig.update_layout(yaxis_tickformat=".0%", title_font_size=12)
                    st.plotly_chart(fig, use_container_width=True)

    # Baseline
    st.divider()
    st.subheader("📏 Modelo Baseline — DummyClassifier (most_frequent)")
    if task == "classification":
        dummy = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy.fit(X_train, y_train)
        acc_d  = accuracy_score(y_test, dummy.predict(X_test))
        f1_d   = f1_score(y_test, dummy.predict(X_test), average="weighted", zero_division=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy Baseline", f"{acc_d:.4f}")
        c2.metric("F1 Baseline",       f"{f1_d:.4f}")
        c3.metric("Estrategia",        "Most Frequent")

        st.info(f"🎯 **Interpretación**: Cualquier modelo con Accuracy > {acc_d:.2%} y "
                f"F1 > {f1_d:.4f} supera al baseline y aporta valor real.")

        # Guardar baseline para comparativa
        st.session_state.results_history["Baseline (Dummy)"] = {
            "Accuracy": round(acc_d, 4), "F1": round(f1_d, 4), "AUC-ROC": 0.5
        }
    else:
        st.info("El baseline de regresión es predecir siempre la media del target. "
                "Cambia a una tarea de clasificación para ver el baseline completo.")

    # Preview data procesada
    with st.expander("🔎 Vista previa — datos procesados (train, primeras 5 filas)"):
        st.dataframe(X_train.head(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN SEGMENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════
def render_segmentacion():
    st.header("📉 Segmentación — CRISP-DM Fase 4 (Modelos Descriptivos)")

    X_train = st.session_state.X_train
    X_test  = st.session_state.X_test
    y_test  = st.session_state.y_test
    feats   = st.session_state.feature_names

    with st.expander("📖 K-Means",             expanded=False): st.markdown(EXP["kmeans"])
    with st.expander("📖 Clustering Jerárquico",expanded=False): st.markdown(EXP["hierarchical"])
    with st.expander("📖 Índice Silhouette",    expanded=False): st.markdown(EXP["silhouette"])

    tab_km, tab_hc, tab_comp = st.tabs(["🔵 K-Means","🌳 Jerárquico","📊 Comparativa Clústeres"])

    # ── K-Means ──────────────────────────────────────────────────────────────
    with tab_km:
        st.subheader("K-Means Clustering")
        k_val = st.slider("Número de clústeres K", 2, 12, 5, key="km_k")
        run_km = st.button("▶ Ejecutar K-Means", type="primary", key="btn_km")

        if run_km:
            with st.spinner("Calculando inercias para el método del codo..."):
                inertias, sils = [], []
                K_range = range(2, 13)
                for ki in K_range:
                    km_i = KMeans(n_clusters=ki, random_state=42, n_init=10)
                    km_i.fit(X_train)
                    inertias.append(km_i.inertia_)
                    sample_n = min(3000, len(X_train))
                    sils.append(silhouette_score(X_train, km_i.labels_,
                                                  sample_size=sample_n, random_state=42))

            col_e, col_s = st.columns(2)
            with col_e:
                fig_el = px.line(x=list(K_range), y=inertias, markers=True,
                                 title="Método del Codo — Inercia",
                                 labels={"x":"K","y":"Inercia"})
                fig_el.add_vline(x=k_val, line_dash="dash", line_color="red",
                                 annotation_text=f"K={k_val}")
                fig_el.update_layout(height=320)
                st.plotly_chart(fig_el, use_container_width=True)
                st.caption("📌 El 'codo' es donde la reducción de inercia se estabiliza. "
                           "Añadir más clústeres más allá del codo da pocos beneficios.")

            with col_s:
                fig_sl = px.line(x=list(K_range), y=sils, markers=True,
                                 title="Silhouette Score por K",
                                 labels={"x":"K","y":"Silhouette"})
                fig_sl.add_vline(x=k_val, line_dash="dash", line_color="red")
                fig_sl.update_layout(height=320)
                st.plotly_chart(fig_sl, use_container_width=True)
                st.caption("📌 Mayor silhouette → clústeres más compactos y separados.")

            # Modelo final
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            kmeans.fit(X_train)

            X_all  = pd.concat([X_train, pd.DataFrame(X_test, columns=feats)],
                                ignore_index=True)
            lbl_all= np.concatenate([kmeans.labels_, kmeans.predict(X_test)])
            sil_f  = silhouette_score(X_all, lbl_all, sample_size=min(5000, len(X_all)),
                                       random_state=42)

            c1, c2 = st.columns(2)
            c1.metric("Silhouette Score final", f"{sil_f:.4f}")
            c2.metric("Inercia final",           f"{kmeans.inertia_:,.0f}")

            # Interpretación silhouette
            grade = ("🟢 Excelente" if sil_f > 0.7 else "🟡 Razonable" if sil_f > 0.5
                     else "🟠 Débil" if sil_f > 0.25 else "🔴 Sin estructura")
            st.info(f"**Silhouette = {sil_f:.4f} → {grade}**\n\n"
                    "Un valor cercano a 1 indica clústeres bien separados. "
                    "Si es bajo, considera cambiar K o revisar el preprocesamiento. "
                    "*(Nota: En datos de marketing de comportamiento es normal obtener scores bajos, ya que los segmentos suelen ser continuos en lugar de agruparse en islas separadas).*")

            # Visualización PCA
            pca    = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_all)
            df_vis = pd.DataFrame({"PC1":coords[:,0], "PC2":coords[:,1],
                                   "Clúster":lbl_all.astype(str)})
            fig_pca = px.scatter(df_vis, x="PC1", y="PC2", color="Clúster",
                                  title=f"K-Means (K={k_val}) — Proyección PCA 2D",
                                  opacity=0.4, height=420)
            st.plotly_chart(fig_pca, use_container_width=True)
            st.caption("📌 PCA proyecta todas las features en 2 dimensiones para visualizar. "
                       "Clústeres bien separados en 2D sugieren buena separabilidad real.")

            # Perfiles de clústeres
            st.subheader("👥 Perfiles de Clústeres")
            st.caption("Centroide de cada clúster en escala original (invertiendo StandardScaler).")
            df_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=feats)
            df_centroids.index = [f"Clúster {i}" for i in range(k_val)]
            st.dataframe(df_centroids.round(3), use_container_width=True)

            # Distribución del target por clúster
            lbl_test = kmeans.predict(X_test)
            y_test_arr = np.array(y_test)
            target_numeric = pd.to_numeric(pd.Series(y_test_arr), errors="coerce")
            is_numeric_target = target_numeric.notna().all()

            df_cl = pd.DataFrame({"Clúster": lbl_test.astype(str),
                                  "Target": y_test_arr})
            if is_numeric_target:
                df_cl["Target"] = target_numeric.values
                df_agg = df_cl.groupby("Clúster")["Target"].agg(["mean","count"]).reset_index()
                df_agg.columns = ["Clúster","Media Target","N registros"]
                fig_ct = px.bar(df_agg, x="Clúster", y="Media Target",
                                text=df_agg["N registros"].astype(str)+" registros",
                                title="Media del Target por Clúster (numérico)", height=320)
            else:
                df_agg = df_cl.groupby("Clúster")["Target"].agg(
                    lambda x: x.value_counts().index[0]).reset_index()
                cnt   = df_cl.groupby("Clúster").size().reset_index(name="N registros")
                df_agg = df_agg.merge(cnt, on="Clúster")
                df_agg.columns = ["Clúster","Clase Dominante","N registros"]
                fig_ct = px.bar(df_agg, x="Clúster", y="N registros", color="Clase Dominante",
                                text="Clase Dominante",
                                title="Clase dominante del Target por Clúster", height=320)
            fig_ct.update_traces(textposition="outside")
            st.plotly_chart(fig_ct, use_container_width=True)
            st.caption("📌 Clústeres con alta concentración en una clase son los **segmentos más homogéneos**.")

            # Guardar para comparativa
            st.session_state["kmeans_sil"] = sil_f

    # ── Clustering Jerárquico ─────────────────────────────────────────────────
    with tab_hc:
        st.subheader("Clustering Jerárquico Aglomerativo (Ward)")
        k_hc  = st.slider("Número de clústeres (corte del dendrograma)", 2, 12, 4, key="hc_k")
        run_hc = st.button("▶ Ejecutar Clustering Jerárquico", type="primary", key="btn_hc")

        if run_hc:
            # Muestra para el dendrograma (costoso computacionalmente)
            sample_n = min(150, len(X_train))
            sample_df = X_train.sample(sample_n, random_state=42)
            with st.spinner(f"Construyendo dendrograma sobre {sample_n} registros..."):
                Z = linkage(sample_df.values, method="ward")
                try:
                    fig_dend = ff.create_dendrogram(
                        sample_df.values,
                        linkagefun=lambda x: linkage(x, method="ward"),
                        color_threshold=0)
                    fig_dend.update_layout(
                        title=f"Dendrograma Jerárquico (muestra {sample_n} registros)",
                        height=420, xaxis_showticklabels=False,
                        yaxis_title="Distancia (Ward)")
                    st.plotly_chart(fig_dend, use_container_width=True)
                    st.caption("📌 La altura del corte determina el número de clústeres. "
                               "Se observan las fusiones de grupos de menor a mayor distancia.")
                except Exception as e:
                    st.warning(f"No se pudo renderizar el dendrograma: {e}")

            # Modelo completo
            agg = AgglomerativeClustering(n_clusters=k_hc, linkage="ward")
            labels_train = agg.fit_predict(X_train)
            sil_hc = silhouette_score(X_train, labels_train,
                                       sample_size=min(3000, len(X_train)), random_state=42)
            labels_test = AgglomerativeClustering(n_clusters=k_hc, linkage="ward").fit_predict(X_test)

            c1, c2 = st.columns(2)
            c1.metric("Silhouette Score", f"{sil_hc:.4f}")
            c2.metric("Clústeres",        k_hc)

            grade = ("🟢 Excelente" if sil_hc > 0.7 else "🟡 Razonable" if sil_hc > 0.5
                     else "🟠 Débil" if sil_hc > 0.25 else "🔴 Sin estructura")
            st.info(f"**Silhouette = {sil_hc:.4f} → {grade}**\n\n"
                    "*(Nota: En datos de comportamiento/marketing es muy común obtener un score bajo debido a la falta de separación natural extrema).*")

            # PCA
            pca    = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_train)
            df_vis = pd.DataFrame({"PC1":coords[:,0], "PC2":coords[:,1],
                                   "Clúster":labels_train.astype(str)})
            fig_pca = px.scatter(df_vis, x="PC1", y="PC2", color="Clúster",
                                  title=f"Clustering Jerárquico (K={k_hc}) — PCA 2D",
                                  opacity=0.5, height=400)
            st.plotly_chart(fig_pca, use_container_width=True)

            # Perfiles
            df_prof = X_train.copy()
            df_prof["Clúster"] = labels_train
            profile_means = df_prof.groupby("Clúster").mean().round(3)
            st.subheader("👥 Perfiles de Clústeres — Medias por grupo")
            st.dataframe(profile_means, use_container_width=True)

            st.session_state["hc_sil"] = sil_hc

    # ── Comparativa clústeres ─────────────────────────────────────────────────
    with tab_comp:
        st.subheader("📊 Comparativa: K-Means vs Clustering Jerárquico")
        sil_km = st.session_state.get("kmeans_sil")
        sil_hc = st.session_state.get("hc_sil")
        if sil_km is None and sil_hc is None:
            st.info("Ejecuta ambos algoritmos (K-Means y Jerárquico) para ver la comparativa.")
        else:
            rows = []
            if sil_km: rows.append({"Algoritmo":"K-Means",    "Silhouette":sil_km})
            if sil_hc: rows.append({"Algoritmo":"Jerárquico", "Silhouette":sil_hc})
            df_comp = pd.DataFrame(rows)
            fig = px.bar(df_comp, x="Algoritmo", y="Silhouette", text_auto=".4f",
                         color="Algoritmo", title="Silhouette Score — Comparativa", height=320)
            fig.update_traces(textposition="outside")
            fig.update_yaxes(range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
**¿Cómo elegir el mejor algoritmo de clustering?**
- **Silhouette más alto** → clústeres más compactos y bien separados.
- **K-Means** es más escalable y rápido para grandes datasets.
- **Clustering Jerárquico** es más flexible (no requiere definir K a priori) y
  el dendrograma ayuda a entender la estructura del dato.
- Para **segmentación de clientes**: K-Means es el estándar industrial;
  el jerárquico se usa para exploración inicial.
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
    task    = st.session_state.task

    if task != "classification":
        st.warning("La variable objetivo seleccionada es continua (regresión). "
                   "Para clasificación, selecciona un target categórico o binario.")
        return

    with st.expander("📖 Árbol de Decisión",  expanded=False): st.markdown(EXP["decision_tree"])
    with st.expander("📖 Random Forest",       expanded=False): st.markdown(EXP["random_forest"])
    with st.expander("📖 Otros Modelos (KNN, NB, XGBoost, LightGBM)", expanded=False):
        st.markdown(EXP["knn"])
        st.markdown(EXP["naive_bayes"])
        st.markdown(EXP["xgboost"])
        st.markdown(EXP["lightgbm"])
    with st.expander("📖 Matriz de Confusión y sus Métricas", expanded=False):
        st.markdown(EXP["confusion_matrix"])

    tab_dt, tab_rf, tab_lr, tab_knn, tab_nb, tab_xgb, tab_lgb = st.tabs(
        ["🌿 Árbol", "🌲 Random Forest", "📈 Logística", "🎯 KNN", "🧠 Naive Bayes", "🚀 XGBoost", "⚡ LightGBM"])

    CLF_TABS = [
        (tab_dt, "Árbol de Decisión",    "dt"),
        (tab_rf, "Random Forest",        "rf"),
        (tab_lr, "Regresión Logística",  "lr"),
        (tab_knn, "K-Nearest Neighbors", "knn"),
        (tab_nb, "Naive Bayes", "nb"),
        (tab_xgb, "XGBoost", "xgb"),
        (tab_lgb, "LightGBM", "lgb"),
    ]

    for tab, name, key in CLF_TABS:
        with tab:
            st.subheader(f"{name}")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if key == "dt":
                    max_d = st.slider("max_depth", 2, 20, 5, key=f"md_{key}")
                    min_sl = st.slider("min_samples_leaf", 1, 50, 5, key=f"msl_{key}",
                                       help="Mínimo muestras en hoja — reduce overfitting")
                elif key == "rf":
                    n_est = st.slider("n_estimators", 50, 500, 150, step=50, key=f"ne_{key}")
                    max_d = st.slider("max_depth", 2, 30, 10, key=f"md_{key}")
                    min_sl = st.slider("min_samples_leaf", 1, 20, 4, key=f"msl_{key}",
                                       help="Controla overfitting del bosque")
                elif key == "lr":
                    C_reg = st.select_slider("C (regularización)", options=[0.001,0.01,0.1,1,10,100],
                                             value=1.0, key=f"C_{key}",
                                             help="Menor C = más regularización = menos overfitting")
                    max_iter = st.slider("max_iter", 100, 2000, 500, step=100, key=f"mi_{key}")
                elif key == "knn":
                    n_neighbors = st.slider("n_neighbors", 1, 50, 5, key=f"nn_{key}", help="Número de vecinos")
                elif key == "nb":
                    st.info("Naive Bayes no requiere hiperparámetros complejos para esta configuración.")
                elif key in ["xgb", "lgb"]:
                    n_est = st.slider("n_estimators", 50, 500, 150, step=50, key=f"ne_{key}")
                    max_d = st.slider("max_depth", 2, 20, 6, key=f"md_{key}")
                    lr_val = st.select_slider("learning_rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1, key=f"lr_{key}")
            with col_p2:
                use_val = st.checkbox("Evaluar en Validación", value=len(X_val)>0, key=f"uv_{key}",
                                      disabled=len(X_val)==0, help="Deshabilitado si Validation=0%")
                run_cv  = st.checkbox("Cross-Validation (k=5)", value=True, key=f"cv_{key}",
                                      help="Estimación robusta del rendimiento en Train")

            run_btn = st.button(f"🚀 Entrenar {name}", type="primary", key=f"btn_{key}")

            if run_btn:
                with st.spinner(f"Entrenando {name}..."):
                    if key == "dt":
                        model = DecisionTreeClassifier(max_depth=max_d,
                                                       min_samples_leaf=min_sl,
                                                       class_weight="balanced",
                                                       random_state=42)
                    elif key == "rf":
                        model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                                       min_samples_leaf=min_sl,
                                                       class_weight="balanced",
                                                       random_state=42, n_jobs=-1)
                    elif key == "lr":
                        model = LogisticRegression(C=C_reg, solver="lbfgs", max_iter=max_iter,
                                                   class_weight="balanced", random_state=42)
                    elif key == "knn":
                        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
                    elif key == "nb":
                        model = GaussianNB()
                    elif key == "xgb":
                        scale_pos_weight = 1
                        if len(np.unique(y_train)) == 2:
                            c_counts = np.bincount(y_train)
                            scale_pos_weight = c_counts[0] / c_counts[1] if len(c_counts)>1 else 1
                        obj = "binary:logistic" if len(np.unique(y_train)) == 2 else "multi:softprob"
                        model = xgb.XGBClassifier(n_estimators=n_est, max_depth=max_d, learning_rate=lr_val, 
                                                  scale_pos_weight=scale_pos_weight if obj=="binary:logistic" else None,
                                                  objective=obj, random_state=42, n_jobs=-1)
                    elif key == "lgb":
                        model = lgb.LGBMClassifier(n_estimators=n_est, max_depth=max_d, learning_rate=lr_val, 
                                                   class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
                    model.fit(X_train, y_train)

                # Cross-Validation sobre train
                if run_cv:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                                scoring="accuracy", n_jobs=-1)
                    st.info(f"🔁 **Cross-Validation (k=5):** Accuracy = "
                            f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}  "
                            f"(min={cv_scores.min():.4f}, max={cv_scores.max():.4f})")

                # Métricas en test
                y_proba = model.predict_proba(X_test)
                y_pred  = model.predict(X_test)
                acc_train = accuracy_score(y_train, model.predict(X_train))
                m_test  = compute_all_metrics(y_test, y_pred, y_proba)
                gap = acc_train - m_test["Exactitud"]

                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("✅ Accuracy (Test)", f"{m_test['Exactitud']:.4f}")
                col_s2.metric("🔧 Accuracy (Train)", f"{acc_train:.4f}")
                col_s3.metric("⚠️ Gap Train-Test", f"{gap:.4f}",
                              delta=f"{'-Posible overfitting' if gap > 0.15 else 'OK'}")
                if gap > 0.15:
                    st.warning(f"⚠️ Gap Train-Test = {gap:.4f} > 0.15. El modelo presenta **sobreajuste significativo**. "
                               "Prueba aumentar `min_samples_leaf`, reducir `max_depth` o usar más datos.")
                st.success(f"✅ AUC-ROC en Test: {m_test['AUC_ROC'] or 'N/A'}")

                # Validación (opcional)
                if use_val and len(X_val) > 0:
                    y_pred_v  = model.predict(X_val)
                    y_proba_v = model.predict_proba(X_val)
                    m_val     = compute_all_metrics(y_val, y_pred_v, y_proba_v)
                    cv1, cv2, cv3 = st.columns(3)
                    cv1.metric("Accuracy (Val)",  f"{m_val['Exactitud']:.4f}")
                    cv2.metric("F1-Score (Val)",  f"{m_val['F1_Score']:.4f}")
                    cv3.metric("AUC-ROC (Val)",   f"{m_val['AUC_ROC'] or 'N/A'}")
                    st.caption("Validación detecta overfitting antes de tocar el test.")

                st.divider()
                st.subheader("📐 Matriz de Confusión completa — Test Set")
                with st.expander("ℹ️ ¿Cómo leer estas métricas?"):
                    st.markdown("""
| Métrica | Fórmula | Cuándo priorizar |
|---------|---------|-----------------|
| **Precisión** | VP/(VP+FP) | Cuando los FP son costosos (ej: spam) |
| **Sensibilidad** | VP/(VP+FN) | Cuando los FN son costosos (ej: cáncer) |
| **Especificidad** | VN/(VN+FP) | Cuando correctamente identificar negativos es clave |
| **F1-Score** | 2·VP/(2·VP+FP+FN) | Balance entre precisión y sensibilidad |
| **AUC-ROC** | Área bajo ROC | Evaluación global independiente del umbral |
                    """)

                col_cm, col_roc = st.columns(2)
                with col_cm:
                    st.plotly_chart(fig_confusion_heatmap(m_test, name), use_container_width=True)
                with col_roc:
                    roc_pc = m_test.get("roc_per_class", {})
                    if roc_pc:
                        colors_roc = px.colors.qualitative.Set2
                        fig_r = go.Figure()
                        fig_r.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                                    name="Azar",line=dict(color="gray",dash="dash")))
                        for ci,(cls,(fpr_i,tpr_i,auc_i)) in enumerate(roc_pc.items()):
                            fig_r.add_trace(go.Scatter(
                                x=fpr_i, y=tpr_i, mode="lines",
                                name=f"Clase {cls} (AUC={auc_i:.3f})",
                                line=dict(color=colors_roc[ci % len(colors_roc)], width=2)))
                        fig_r.update_layout(title=f"Curva ROC — {name}",
                                            xaxis_title="Tasa FP", yaxis_title="Tasa VP",
                                            height=360, legend=dict(x=0.55, y=0.05))
                        st.plotly_chart(fig_r, use_container_width=True)

                show_metrics_cards(m_test)

                # Importancia de variables
                fi = fig_feature_importance(model, feats, name)
                if fi:
                    st.plotly_chart(fi, use_container_width=True)
                    st.caption("📌 Las variables con mayor importancia son las más influyentes "
                               "en las decisiones del modelo. Útil para explicar resultados al negocio.")

                # Reglas del árbol (solo DT)
                if key == "dt":
                    with st.expander("📋 Reglas del Árbol de Decisión (texto)"):
                        rules = export_text(model, feature_names=feats, max_depth=5)
                        st.code(rules, language="text")
                        st.caption("Cada `|---` es un nivel del árbol. "
                                   "`class:` indica la predicción de esa hoja.")

                # Classification report completo
                with st.expander("📋 Classification Report completo"):
                    rep = classification_report(y_test, y_pred, output_dict=True)
                    rows = [{"Clase": k, **{m: round(v[m],4) for m in
                              ["precision","recall","f1-score","support"]}}
                             for k, v in rep.items() if isinstance(v, dict)]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Guardar para comparativa
                _store_clf_result(name, m_test)
                st.info(f"✅ Resultado de **{name}** guardado para la comparativa.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════
def render_comparativa():
    st.header("📈 Comparativa de Modelos — CRISP-DM Fase 5 (Evaluación)")

    with st.expander("📖 Curva ROC y AUC", expanded=False):
        st.markdown(EXP["roc"])

    history = st.session_state.get("results_history", {})
    roc_data = st.session_state.get("roc_data", {})

    if len(history) < 2:
        st.info("Entrena al menos **2 modelos** (incluyendo Baseline) "
                "en las secciones de Preparación y Clasificación para ver la comparativa.")
        return

    # ── Tabla comparativa ────────────────────────────────────────────────────
    st.subheader("📋 Tabla Comparativa de Modelos")
    rows = [{"Modelo": m, **v} for m, v in history.items()]
    df_comp = pd.DataFrame(rows)

    # Ordenar por AUC-ROC
    if "AUC-ROC" in df_comp.columns:
        df_comp = df_comp.sort_values("AUC-ROC", ascending=False)
    st.dataframe(df_comp.set_index("Modelo"), use_container_width=True)

    # Ranking visual
    st.subheader("🏆 Ranking por AUC-ROC")
    if "AUC-ROC" in df_comp.columns:
        fig_rank = px.bar(df_comp, x="Modelo", y="AUC-ROC", color="AUC-ROC",
                          color_continuous_scale="RdYlGn", text="AUC-ROC",
                          title="Comparativa AUC-ROC por Modelo")
        fig_rank.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_rank.add_hline(y=0.5, line_dash="dash", line_color="red",
                           annotation_text="Azar (0.50)")
        fig_rank.update_yaxes(range=[0, 1.1])
        fig_rank.update_layout(height=400, coloraxis_showscale=False, xaxis_tickangle=-15)
        st.plotly_chart(fig_rank, use_container_width=True)

    # ── Curvas ROC superpuestas ───────────────────────────────────────────────
    if roc_data:
        st.subheader("📉 Curvas ROC Superpuestas")
        # roc_data[model_name] = dict de {clase: (fpr, tpr, auc)}
        colors = px.colors.qualitative.Set1
        fig_roc_all = go.Figure()
        fig_roc_all.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                          name="Azar", line=dict(color="gray", dash="dash")))
        color_i = 0
        for model_nm, class_dict in roc_data.items():
            # Para cada modelo graficar la curva promedio (macro) si multiclass
            if isinstance(class_dict, dict):
                fprs, tprs = [], []
                aucs = []
                for cls, (fpr_i, tpr_i, auc_i) in class_dict.items():
                    fprs.append(fpr_i); tprs.append(tpr_i); aucs.append(auc_i)
                # Usar la primera clase como representativa (o graficar todas)
                if len(class_dict) == 1:  # binario
                    fpr_p, tpr_p = fprs[0], tprs[0]
                    auc_p = aucs[0]
                else:  # multiclase: graficar la curva de cada clase separada
                    for ci, (cls, (fpr_i, tpr_i, auc_i)) in enumerate(class_dict.items()):
                        fig_roc_all.add_trace(go.Scatter(
                            x=fpr_i, y=tpr_i, mode="lines",
                            name=f"{model_nm} | Clase {cls} (AUC={auc_i:.3f})",
                            line=dict(color=colors[color_i % len(colors)], width=1.5),
                            opacity=0.8))
                        color_i += 1
                    continue
                fig_roc_all.add_trace(go.Scatter(
                    x=fpr_p, y=tpr_p, mode="lines",
                    name=f"{model_nm} (AUC={auc_p:.4f})",
                    line=dict(color=colors[color_i % len(colors)], width=2)))
                color_i += 1

        fig_roc_all.update_layout(
            title="Curvas ROC — Todos los Modelos",
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos",
            height=500, legend=dict(x=0.55, y=0.05))
        st.plotly_chart(fig_roc_all, use_container_width=True)
        st.caption("📌 El modelo con la curva más alejada de la diagonal (mayor área) es el mejor. "
                   "Un AUC = 1.0 sería perfecto; AUC = 0.5 equivale a adivinar al azar.")

    # ── Matriz de correlación en Comparativa ─────────────────────────────
    df_orig = st.session_state.get("df")
    if df_orig is not None:
        st.divider()
        st.subheader("🔥 Matriz de Correlación de Features")
        target_col = st.session_state.get("target")
        profile_s  = st.session_state.get("profile", {})
        num_c = [c for c, p in profile_s.items() if p["is_num"]]
        if len(num_c) >= 2:
            corr = df_orig[num_c].corr()
            fig_corr = px.imshow(corr, text_auto=".2f",
                                 color_continuous_scale="RdBu_r",
                                 aspect="auto",
                                 height=max(400, len(num_c)*32),
                                 title="Correlación de Pearson entre variables numéricas")
            st.plotly_chart(fig_corr, use_container_width=True)
            if target_col and target_col in num_c:
                corr_target = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
                st.markdown(f"**Variables más correlacionadas con `{target_col}`:**")
                st.dataframe(corr_target.rename("|r con target|").round(4).head(10),
                             use_container_width=True)

    # ── Resumen para equipo gerencial ─────────────────────────────────────────
    st.divider()
    st.subheader("📣 Resumen para Equipo Gerencial (Lenguaje no técnico)")
    with st.expander("Ver resumen ejecutivo", expanded=True):
        best_model = df_comp.iloc[0]["Modelo"] if len(df_comp) else "N/A"
        best_auc   = df_comp.iloc[0].get("AUC-ROC", 0) if len(df_comp) else 0
        best_acc   = df_comp.iloc[0].get("Accuracy", 0) if "Accuracy" in df_comp.columns else 0
        best_f1    = df_comp.iloc[0].get("F1", 0) if "F1" in df_comp.columns else 0

        auc_grade = ("excelente" if best_auc >= 0.90 else "bueno" if best_auc >= 0.80
                     else "aceptable" if best_auc >= 0.70 else "mejorable")

        st.markdown(f"""
### 🏆 Mejor modelo: **{best_model}**

**¿Qué significa esto en términos de negocio?**

Se analizaron **{len(df_comp)} modelos** de inteligencia artificial sobre este dataset.
El modelo recomendado es **{best_model}**, que obtuvo:

- **Exactitud (Accuracy):** {best_acc:.2%} — de cada 100 predicciones, acierta ~{int(best_acc*100)}.
- **F1-Score:** {best_f1:.4f} — equilibrio entre detectar correctamente los casos positivos
  y no generar falsas alarmas.
- **AUC-ROC:** {best_auc:.4f} — capacidad global de discriminación, calificada como **{auc_grade}**.

**¿Por qué importa el AUC-ROC?**
Imagine que desea identificar clientes que van a comprar. Un AUC de {best_auc:.2f} significa que,
si toma a un cliente que **sí compra** y uno que **no compra**, el modelo le asignará
mayor probabilidad de compra al primero el **{best_auc:.0%} de las veces**.

**Recomendación:**
{'✅ El modelo está listo para una prueba piloto en producción.' if best_auc >= 0.75
 else '⚠️ El modelo necesita mejoras antes de implementarse (más datos, tuning de hiperparámetros).'}

**Comparativa con el modelo de referencia (Baseline):**
Predecir siempre la clase más frecuente (sin IA) da un AUC de 0.50.
Nuestro mejor modelo supera eso en **{(best_auc - 0.5)*100:.1f} puntos porcentuales**.
        """)

    if st.button("🗑️ Limpiar historial de modelos"):
        st.session_state.results_history = {}
        st.session_state.roc_data = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _store_clf_result(model_name: str, m: dict):
    if "results_history" not in st.session_state:
        st.session_state.results_history = {}
    if "roc_data" not in st.session_state:
        st.session_state.roc_data = {}

    st.session_state.results_history[model_name] = {
        "Accuracy": m["Exactitud"],
        "Precisión": m["Precisión"],
        "Sensibilidad": m["Sensibilidad"],
        "Especificidad": m["Especificidad"],
        "F1": m["F1_Score"],
        "AUC-ROC": m["AUC_ROC"] or 0.0,
    }
    # Guardar curvas ROC para binario y multiclase
    roc_pc = m.get("roc_per_class", {})
    if roc_pc:
        st.session_state.roc_data[model_name] = roc_pc


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN SERIES DE TIEMPO
# ══════════════════════════════════════════════════════════════════════════════
def render_timeseries():
    st.header("📈 Series de Tiempo — CRISP-DM Fase 4 (Pronóstico)")

    y_train = st.session_state.y_train
    y_test  = st.session_state.y_test
    
    if len(y_train) == 0:
        st.warning("No hay datos de entrenamiento. Revisa la partición.")
        return

    st.info("📌 En Series de Tiempo, el Target se pronostica basándose en su propio comportamiento histórico. La validación se hace comparando con el conjunto de Test (datos reales más recientes).")

    tab_hw, tab_arima = st.tabs(["📉 Suavizado Exponencial", "📊 AutoARIMA"])

    # Suavizado Exponencial
    with tab_hw:
        st.subheader("Suavizado Exponencial (Holt-Winters)")
        c1, c2 = st.columns(2)
        with c1:
            trend = st.selectbox("Tendencia", ["add", "mul", "Ninguna"], index=0, key="hw_trend")
            trend_val = None if trend == "Ninguna" else trend
        with c2:
            seasonal = st.selectbox("Estacionalidad", ["add", "mul", "Ninguna"], index=2, key="hw_seasonal")
            seasonal_val = None if seasonal == "Ninguna" else seasonal
            periods = st.number_input("Periodos por ciclo estacional", min_value=2, value=12, key="hw_periods")

        if st.button("🚀 Entrenar Holt-Winters", type="primary"):
            with st.spinner("Entrenando modelo de Suavizado Exponencial..."):
                try:
                    model = ExponentialSmoothing(y_train, trend=trend_val, seasonal=seasonal_val, seasonal_periods=periods if seasonal_val else None)
                    fit_model = model.fit()
                    
                    preds = fit_model.forecast(len(y_test))
                    future_preds = fit_model.forecast(len(y_test) + 30) # 30 pasos extra
                    
                    mae = mean_absolute_error(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("MAE (Test)", f"{mae:.4f}", help="Error Absoluto Medio (menor es mejor)")
                    col_m2.metric("RMSE (Test)", f"{rmse:.4f}", help="Raíz del Error Cuadrático Medio (penaliza errores grandes)")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_train, mode='lines', name='Train'))
                    x_test = range(len(y_train), len(y_train)+len(y_test))
                    fig.add_trace(go.Scatter(x=list(x_test), y=y_test, mode='lines', name='Test (Real)', line=dict(color='green')))
                    x_future = range(len(y_train), len(y_train)+len(future_preds))
                    fig.add_trace(go.Scatter(x=list(x_future), y=future_preds, mode='lines', name='Predicción (+30 pasos)', line=dict(dash='dot', color='red')))
                    
                    fig.update_layout(title="Predicción Holt-Winters vs Realidad", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store as clf result to use the same comparativa code
                    _store_clf_result("Holt-Winters", {"Exactitud": mae, "Precisión": rmse, "Sensibilidad": 0, "Especificidad": 0, "F1_Score": 0, "AUC_ROC": None})
                    st.caption("Nota: Para Series de Tiempo, en la comparativa el valor 'Exactitud' mostrará el MAE y 'Precisión' el RMSE (Menor es mejor).")
                except Exception as e:
                    st.error(f"Error al entrenar el modelo (revisa la configuración de estacionalidad): {e}")

    # ARIMA
    with tab_arima:
        st.subheader("AutoARIMA")
        st.caption("Busca automáticamente los mejores parámetros p, d, q.")
        seasonality = st.checkbox("Incluir estacionalidad (SARIMA)", value=False)
        m_periods = st.number_input("Periodos (m) para estacionalidad", min_value=2, value=12, key="arima_m") if seasonality else 1
        
        if st.button("🚀 Entrenar AutoARIMA", type="primary"):
            with st.spinner("Buscando los mejores parámetros y entrenando... (Puede tardar varios segundos)"):
                try:
                    model = pm.auto_arima(y_train, seasonal=seasonality, m=m_periods, trace=True, error_action='ignore', suppress_warnings=True)
                    
                    preds = model.predict(n_periods=len(y_test))
                    future_preds = model.predict(n_periods=len(y_test) + 30)
                    
                    mae = mean_absolute_error(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("MAE (Test)", f"{mae:.4f}")
                    col_m2.metric("RMSE (Test)", f"{rmse:.4f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=y_train, mode='lines', name='Train'))
                    x_test = range(len(y_train), len(y_train)+len(y_test))
                    fig.add_trace(go.Scatter(x=list(x_test), y=y_test, mode='lines', name='Test (Real)', line=dict(color='green')))
                    x_future = range(len(y_train), len(y_train)+len(future_preds))
                    fig.add_trace(go.Scatter(x=list(x_future), y=future_preds, mode='lines', name='Predicción (+30 pasos)', line=dict(dash='dot', color='red')))
                    
                    fig.update_layout(title="Predicción AutoARIMA vs Realidad", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Ver resumen estadístico del modelo"):
                        st.text(str(model.summary()))
                    
                    _store_clf_result("AutoARIMA", {"Exactitud": mae, "Precisión": rmse, "Sensibilidad": 0, "Especificidad": 0, "F1_Score": 0, "AUC_ROC": None})
                except Exception as e:
                    st.error(f"Error al entrenar el modelo: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.title("⛏️ Minería de Datos — Sistema CRISP-DM Completo")
    st.caption("Carga · EDA · Preparación · Segmentación · Clasificación · Comparativa")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📁 Cargar Dataset")
        st.caption("Formatos soportados: CSV, TSV, Excel, JSON, Parquet")
        uploaded = st.file_uploader(
            "Subir archivo",
            type=["csv","tsv","txt","xlsx","xls","json","parquet"])

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
            st.info("Sube un archivo de datos para comenzar.")
            st.stop()

        df = st.session_state.df

        st.header("⚙️ Configuración")
        target = st.selectbox("Variable objetivo (target)",
                              df.columns.tolist(), index=len(df.columns)-1)
                              
        cols_to_drop = st.multiselect("Columnas a excluir (opcional)", 
                                      [c for c in df.columns if c != target],
                                      help="Selecciona variables que sepas que causan leakage combinado o no aportan valor.")
                                      
        time_col = st.selectbox("Columna de Tiempo (Opcional - para Series de Tiempo)",
                                ["Ninguna"] + df.columns.tolist(), index=0,
                                help="Si seleccionas una, el pipeline cambiará a Series de Tiempo (partición cronológica).")

        st.markdown("**Partición del dataset:**")
        split_mode = st.radio(
            "Esquema de partición",
            ["80 / 20  (Train / Test)",
             "70 / 15 / 15  (Train / Val / Test)",
             "60 / 20 / 20  (Train / Val / Test)",
             "Personalizado"],
            index=1, key="split_mode")
        if split_mode == "80 / 20  (Train / Test)":
            val_pct, test_pct = 0.0, 0.20
        elif split_mode == "70 / 15 / 15  (Train / Val / Test)":
            val_pct, test_pct = 0.15, 0.15
        elif split_mode == "60 / 20 / 20  (Train / Val / Test)":
            val_pct, test_pct = 0.20, 0.20
        else:
            val_pct  = st.slider("% Validación", 5, 30, 15, key="val_pct") / 100
            test_pct = st.slider("% Test",       5, 30, 20, key="test_pct") / 100

        if st.button("🔍 Analizar y Preparar", type="primary", use_container_width=True):
            if val_pct + test_pct >= 0.9:
                st.error("La suma de validación + test no puede superar el 90 %.")
                st.stop()
            with st.spinner("Analizando y preparando datos..."):
                if time_col != "Ninguna":
                    df = df.sort_values(by=time_col).reset_index(drop=True)
                    # Forzar task a timeseries independientemente de la cardinalidad del target
                    task = "timeseries"
                    profile = profile_dataframe(df)
                else:
                    profile = profile_dataframe(df)
                    task    = detect_task(df, target, profile)
                    
                out     = auto_preprocess(df, target, profile, val_pct, test_pct, manual_drop=cols_to_drop)
                X_tr, X_v, X_te, y_tr, y_v, y_te, feats, log = out

                st.session_state.update(dict(
                    target=target, profile=profile, task=task,
                    X_train=X_tr, X_val=X_v,   X_test=X_te,
                    y_train=y_tr, y_val=y_v,   y_test=y_te,
                    feature_names=feats, prep_log=log,
                    results_history={}, roc_data={},
                    kmeans_sil=None, hc_sil=None,
                ))
            st.success(f"✅ Tarea: **{task}** | Features: **{len(feats)}**")

        if "task" in st.session_state:
            st.divider()
            st.markdown(f"**Tarea:** `{st.session_state.task}`")
            st.markdown(f"**Target:** `{st.session_state.target}`")
            st.markdown(f"**Features:** `{len(st.session_state.feature_names)}`")
            tr = len(st.session_state.X_train)
            va = len(st.session_state.X_val)
            te = len(st.session_state.X_test)
            tot = tr + va + te
            st.markdown(f"**Split:** {tr/tot:.0%} / {va/tot:.0%} / {te/tot:.0%}")

    # ── Bienvenida ─────────────────────────────────────────────────────────────
    if "profile" not in st.session_state:
        st.markdown("""
## 👋 Bienvenido al Sistema CRISP-DM

Este sistema implementa el flujo completo de minería de datos según la metodología **CRISP-DM**:

| Fase | Pestaña | Contenido |
|------|---------|-----------|
| **2. Data Understanding** | 📊 EDA | Perfil, distribuciones, correlaciones, análisis bivariado |
| **3. Data Preparation** | ⚙️ Preparación | Limpieza, encoding, partición 3-way, data leakage, baseline |
| **4. Modeling** | 📉 Segmentación | K-Means + Clustering Jerárquico + Silhouette |
| **4. Modeling** | 🌳 Clasificación | Árbol · Random Forest · Log. Reg. + Matriz de confusión completa |
| **5. Evaluation** | 📈 Comparativa | ROC superpuestas · tabla comparativa · resumen gerencial |

**Archivos soportados:** CSV · TSV · Excel (.xlsx/.xls) · JSON · Parquet

---
⬅️ Sube tu dataset y haz clic en **Analizar y Preparar**.
        """)
        return

    # ── Tabs ───────────────────────────────────────────────────────────────────
    task_name = st.session_state.get("task", "classification")
    tab_model_name = "📈 Series de Tiempo" if task_name == "timeseries" else "🌳 Clasificación"
    
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 EDA",
        "⚙️ Preparación",
        "📉 Segmentación",
        tab_model_name,
        "📈 Comparativa",
    ])
    with t1: render_eda()
    with t2: render_preparacion()
    with t3: render_segmentacion()
    with t4: 
        if task_name == "timeseries":
            render_timeseries()
        else:
            render_clasificacion()
    with t5: render_comparativa()


if __name__ == "__main__":
    main()
