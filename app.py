import pandas as pd
import streamlit as st
from pathlib import Path
from PIL import Image

from utils.inference import (
    ACCURACY_REFERENCIA,
    ARQUITECTURA_POR_CONDICION,
    CHECKPOINTS,
    FAMILIA_POR_CONDICION,
    cargar_modelo,
    predecir,
)


# ==================================================================
# Rutas base del dataset
# ==================================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "dataset"

ORIGINAL_DIR    = DATASET_ROOT / "ORL_limpio"
DEGRADACION_DIR = DATASET_ROOT / "Degradacion"
TECNICAS_DIR    = DATASET_ROOT / "Tecnica De mejora de imagen"

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".pgm", ".bmp"}
UMBRAL_CONFIANZA   = 50.0
MAX_SUJETOS_UI     = 10   # solo s1…s10

# ==================================================================
# Catálogo de datasets disponibles para la UI
# ==================================================================
DATASETS = {
    "Original (ORL limpio)": {
        "path": ORIGINAL_DIR,
        "tec_esc": None,
    },
    "Degradada — escala x2": {
        "path": DEGRADACION_DIR / "ORL_escala2",
        "tec_esc": None,
    },
    "Degradada — escala x4": {
        "path": DEGRADACION_DIR / "ORL_escala4",   # 
        "tec_esc": None,
    },
    "CLAHE x2":            {"path": TECNICAS_DIR / "ORL_CLAHE_x2",           "tec_esc": ("CLAHE", 2)},
    "CLAHE x4":            {"path": TECNICAS_DIR / "ORL_CLAHE_x4",           "tec_esc": ("CLAHE", 4)},
    "Super Resolution x2": {"path": TECNICAS_DIR / "ORL_SuperResolution_x2", "tec_esc": ("SR", 2)},
    "Super Resolution x4": {"path": TECNICAS_DIR / "ORL_SuperResolution_x4", "tec_esc": ("SR", 4)},
    "Lite-ESRGAN x2":      {"path": TECNICAS_DIR / "ORL_LiteESRGAN_x2",      "tec_esc": ("LE", 2)},
    "Lite-ESRGAN x4":      {"path": TECNICAS_DIR / "ORL_LiteESRGAN_x4",      "tec_esc": ("LE", 4)},
}

# ==================================================================
# Experimentos por modelo
# ------------------------------------------------------------------
# "condicion": clave base sin sufijo de familia, p.ej. "CLAHE x2" u "ORL"
# "variante":  "CE" | "ArcFace" | "noDFC" | "ORL"
# ==================================================================
EXPERIMENTO_BASE    = "Experimento 1 — Modelo base ORL (sin reentrenar)"
EXPERIMENTO_CE      = "Experimento 2 — Reentrenado (Cross-Entropy)"
EXPERIMENTO_ARCFACE = "Experimento 3 — PReLU + ArcFace"
EXPERIMENTO_NODFC   = "Experimento 4 — Sin DFC (Cross-Entropy)"

EXPERIMENTOS_POR_MODELO = {
    "MobileFaceNet": {
        EXPERIMENTO_BASE:    {"condicion": "ORL",      "variante": "ORL"},
        EXPERIMENTO_CE:      {"condicion": "CLAHE x2", "variante": "CE"},
        EXPERIMENTO_ARCFACE: {"condicion": "CLAHE x2", "variante": "ArcFace"},
    },
    "ShuffleFaceNet": {
        EXPERIMENTO_BASE:    {"condicion": "ORL",      "variante": "ORL"},
        EXPERIMENTO_CE:      {"condicion": "CLAHE x2", "variante": "CE"},
        EXPERIMENTO_ARCFACE: {"condicion": "CLAHE x2", "variante": "ArcFace"},
    },
    "GhostFaceNetV2": {
        EXPERIMENTO_BASE:    {"condicion": "ORL",      "variante": "ORL"},
        EXPERIMENTO_CE:      {"condicion": "CLAHE x2", "variante": "CE"},
        EXPERIMENTO_ARCFACE: {"condicion": "CLAHE x2", "variante": "ArcFace"},
        EXPERIMENTO_NODFC:   {"condicion": "CLAHE x2", "variante": "noDFC"},
    },
}


# ==================================================================
# Helpers: construir clave de CHECKPOINTS
# ==================================================================
def _clave_checkpoint(modelo: str, condicion: str, variante: str) -> str:
    """
    ("MobileFaceNet", "ORL", "ORL")           -> "ORL original"
    ("ShuffleFaceNet", "ORL", "ORL")          -> "ORL original (Shuffle)"
    ("GhostFaceNetV2", "ORL", "ORL")          -> "ORL original (Ghost)"
    ("MobileFaceNet", "CLAHE x2", "CE")       -> "CLAHE x2 (CE)"
    ("ShuffleFaceNet", "CLAHE x2", "ArcFace") -> "CLAHE x2 (Shuffle ArcFace)"
    ("GhostFaceNetV2", "CLAHE x2", "noDFC")   -> "CLAHE x2 (Ghost noDFC)"
    """
    if condicion == "ORL":
        if modelo == "MobileFaceNet":
            return "ORL original"
        if modelo == "ShuffleFaceNet":
            return "ORL original (Shuffle)"
        if modelo == "GhostFaceNetV2":
            return "ORL original (Ghost)"
        raise ValueError(f"Modelo desconocido: {modelo}")

    if modelo == "MobileFaceNet":
        return f"{condicion} ({variante})"
    if modelo == "ShuffleFaceNet":
        return f"{condicion} (Shuffle {variante})"
    if modelo == "GhostFaceNetV2":
        return f"{condicion} (Ghost {variante})"
    raise ValueError(f"Modelo desconocido: {modelo}")


def _condicion_dataset_a_condicion_modelo(tec_esc):
    """('CLAHE', 2) -> 'CLAHE x2'; ('LE', 4) -> 'LE x4'; None -> None."""
    if tec_esc is None:
        return None
    tec, esc = tec_esc
    return f"{tec} x{esc}"


# ==================================================================
# Helpers de filesystem
# ==================================================================
def _testing_dir(dataset_path: Path) -> Path:
    """Devuelve <dataset>/Testing (o testing/test). Fallback: dataset_path."""
    for nombre in ("Testing", "testing", "Test", "test"):
        candidato = dataset_path / nombre
        if candidato.is_dir():
            return candidato
    return dataset_path


def listar_sujetos(dataset_path: Path) -> list[Path]:
    testing = _testing_dir(dataset_path)
    if not testing.is_dir():
        return []

    sujetos = [
        p for p in testing.iterdir()
        if p.is_dir() and p.name.startswith("s") and p.name[1:].isdigit()
    ]
    sujetos = sorted(sujetos, key=lambda p: int(p.name[1:]))
    return sujetos[:MAX_SUJETOS_UI]


def listar_imagenes(directorio: Path) -> list[Path]:
    return sorted(
        (
            p for p in directorio.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONES_IMAGEN
        ),
        key=lambda p: p.name.lower(),
    )


# ==================================================================
# Carga de modelo cacheada
# ==================================================================
@st.cache_resource
def obtener_modelo(checkpoint_path: str, familia: str, tipo: str):
    return cargar_modelo(checkpoint_path, familia, tipo)


# ==================================================================
# UI
# ==================================================================
def main():
    st.set_page_config(
        page_title="Reconocimiento Facial — Abel Mora",
        page_icon="🧑",
        layout="centered",
    )

    st.title("Reconocimiento Facial")
    st.caption(
        "Selecciona un dataset de prueba, un modelo entrenado y una imagen "
        "para evaluar el reconocimiento."
    )

    # ---------- ① Conjunto de prueba ----------
    st.subheader("① Conjunto de prueba")
    tipo = st.radio(
        "Tipo de imagen:",
        ["Original", "Degradada", "Técnica de mejora"],
        horizontal=True,
    )

    if tipo == "Original":
        opciones = ["Original (ORL limpio)"]
    elif tipo == "Degradada":
        opciones = [k for k in DATASETS if k.startswith("Degradada")]
    else:
        opciones = [k for k, v in DATASETS.items() if v["tec_esc"] is not None]

    dataset_label = st.selectbox("Variante:", opciones)
    dataset_cfg = DATASETS[dataset_label]
    dataset_path = dataset_cfg["path"]

    if not dataset_path.is_dir():
        st.error(
            f"No se encontró la carpeta `{dataset_path}`. "
            "Revisa la estructura descrita en `DATASETS`."
        )
        return

    condicion_dataset = _condicion_dataset_a_condicion_modelo(dataset_cfg["tec_esc"])

    # ---------- ② Sujeto ----------
    st.subheader("② Sujeto de prueba")
    sujetos = listar_sujetos(dataset_path)
    if not sujetos:
        st.error(
            f"No hay sujetos (`s1`, `s2`, …) en "
            f"`{_testing_dir(dataset_path)}`."
        )
        return

    sujeto_sel = st.selectbox(
        "Selecciona un sujeto:",
        sujetos,
        format_func=lambda p: p.name,
    )

    # ---------- ③ Modelo y experimento ----------
    st.subheader("③ Modelo y experimento")
    modelo_sel = st.radio(
        "Modelo:",
        list(EXPERIMENTOS_POR_MODELO),
        horizontal=True,
    )
    experimentos = EXPERIMENTOS_POR_MODELO[modelo_sel]
    exp_sel = st.radio("Experimento:", list(experimentos))

    exp_cfg = experimentos[exp_sel]
    condicion_modelo = exp_cfg["condicion"]
    variante_modelo  = exp_cfg["variante"]

    clave_ckpt = _clave_checkpoint(modelo_sel, condicion_modelo, variante_modelo)

    if clave_ckpt not in CHECKPOINTS:
        st.error(
            f"El checkpoint `{clave_ckpt}` no está registrado en "
            "`utils/inference.py` (CHECKPOINTS)."
        )
        return

    checkpoint_path = CHECKPOINTS[clave_ckpt]
    familia_arq    = FAMILIA_POR_CONDICION[clave_ckpt]
    tipo_arq       = ARQUITECTURA_POR_CONDICION[clave_ckpt]
    accuracy_ref   = ACCURACY_REFERENCIA.get(clave_ckpt, "—")

    # Aviso de generalización: dataset ≠ condición del modelo
    if condicion_dataset is not None and condicion_dataset != condicion_modelo:
        st.warning(
            f"⚠️ Este modelo fue entrenado para **{condicion_modelo}**, pero "
            f"estás evaluando sobre **{dataset_label}**. "
            "El resultado es una prueba de generalización, no el accuracy oficial."
        )

    # ---------- ④ Imagen ----------
    st.subheader("④ Imagen")
    imagenes = listar_imagenes(sujeto_sel)
    if not imagenes:
        st.error(f"No hay imágenes compatibles en `{sujeto_sel}`.")
        return

    imagen_sel = st.selectbox(
        "Selecciona una imagen:",
        imagenes,
        format_func=lambda p: p.name,
    )

    try:
        with Image.open(imagen_sel) as imagen:
            imagen_pil = imagen.convert("L")
    except OSError as e:
        st.error(f"No se pudo leer la imagen: {e}")
        return

    st.image(
        imagen_pil,
        caption=f"{sujeto_sel.name}/{imagen_sel.name} — {dataset_label}",
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy referencia", accuracy_ref)
    col2.metric("Arquitectura", f"{familia_arq} ({tipo_arq})")
    col3.metric("Dataset", dataset_label)

    # ---------- ⑤ Evaluación ----------
    if st.button("Evaluar imagen", type="primary", use_container_width=True):
        try:
            with st.spinner("Cargando modelo..."):
                modelo, arcface = obtener_modelo(
                    str(checkpoint_path), familia_arq, tipo_arq,
                )
        except FileNotFoundError:
            st.error(f"No se encontró el checkpoint `{checkpoint_path}`.")
            return
        except Exception as e:
            st.error(f"Error cargando el modelo: {e}")
            return

        with st.spinner("Procesando imagen..."):
            _, confianza, top_k, sujeto_pred = predecir(
                modelo, arcface, imagen_pil, top_k=5,
            )

        sujeto_real  = sujeto_sel.name
        acierto_top1 = sujeto_pred == sujeto_real
        acierto_top5 = any(s == sujeto_real for _, _, s in top_k)

        st.subheader("⑤ Resultado")
        c1, c2, c3 = st.columns(3)
        c1.metric("Top-1", "Correcta" if acierto_top1 else "Incorrecta")
        c2.metric("Confianza top-1", f"{confianza:.2f}%")
        c3.metric("Top-5", "Sí" if acierto_top5 else "No")

        if acierto_top1:
            st.success(f"✅ **{sujeto_pred}** — {confianza:.2f}%")
        elif confianza < UMBRAL_CONFIANZA:
            st.warning(
                f"⚠️ Predicción poco fiable: **{sujeto_pred}** "
                f"con {confianza:.2f}%."
            )
        else:
            st.warning(
                f"❌ Real: **{sujeto_real}** · Predicho: **{sujeto_pred}** "
                f"({confianza:.2f}%)."
            )

        st.markdown("**Top 5 predicciones**")
        df_top = pd.DataFrame([
            {
                "Sujeto": s,
                "Confianza (%)": round(p, 2),
                "Es el sujeto real": s == sujeto_real,
            }
            for _, p, s in top_k
        ])
        st.bar_chart(df_top.set_index("Sujeto")["Confianza (%)"])
        st.dataframe(df_top, hide_index=True, use_container_width=True)

        with st.expander("📋 Reporte"):
            st.markdown(f"""
| Campo | Valor |
|---|---|
| Dataset | {dataset_label} |
| Imagen | `{sujeto_real}/{imagen_sel.name}` |
| Sujeto real | {sujeto_real} |
| Sujeto predicho | {sujeto_pred} |
| Top-1 correcto | {"Sí" if acierto_top1 else "No"} |
| Top-5 correcto | {"Sí" if acierto_top5 else "No"} |
| Confianza | {confianza:.2f}% |
| Modelo | {modelo_sel} |
| Experimento | {exp_sel} |
| Arquitectura | {familia_arq} ({tipo_arq}) |
| Checkpoint | `{checkpoint_path.name}` |
| Accuracy ref. | {accuracy_ref} |
""")

    st.divider()
    st.caption(
        "MobileFaceNet · ShuffleFaceNet · GhostFaceNetV2 · ORL (AT&T) · CPU"
    )


if __name__ == "__main__":
    main()