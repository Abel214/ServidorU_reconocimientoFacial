from pathlib import Path
import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from models.architecture import (
    MobileFaceNet,
    MobileFaceNet2,
    ShuffleFaceNet,
    ShuffleFaceNet2_Arface,
    GhostFaceNetV2,
    GhostFaceNetV2_ArcFace,
    ArcFaceHead,
    IMG_HEIGHT,
    IMG_WIDTH,
    IMG_SIZE_SHUFFLE,
    IMG_SIZE_GHOST,          # añadir en architecture.py (p.ej. 112)
    NUM_CLASSES,
)

DEVICE = torch.device("cpu")

# ------------------------------------------------------------------
# Directorios base
# ------------------------------------------------------------------
# BASE_DIR: carpeta que contiene las subcarpetas de pesos.
# Por defecto: <proyecto>/checkpoints
# Se puede sobreescribir con la variable de entorno CHECKPOINTS_BASE
BASE_DIR = Path(
    os.environ.get(
        "CHECKPOINTS_BASE",
        Path(__file__).resolve().parent.parent / "models" / "checkpoints",
    )
)

DIR_MOBILE  = BASE_DIR / "Pesos MobileFacenet"
DIR_SHUFFLE = BASE_DIR / "Pesos ShuffleFaceNet"
DIR_GHOST   = BASE_DIR / "Pesos GhostFacenetV2"

# ------------------------------------------------------------------
# Clases (igual que antes)
# ------------------------------------------------------------------
def _obtener_clases_alfabeticas():
    # ImageFolder ordena las carpetas alfabéticamente. 
    # Usamos esto para replicar exactamente lo que vio el modelo al entrenar.
    clases = [f"s{i}" for i in range(1, NUM_CLASSES + 1)]
    return sorted(clases)

CLASES_MOBILEFACENET  = _obtener_clases_alfabeticas()
CLASES_SHUFFLEFACENET = _obtener_clases_alfabeticas()
CLASES_GHOSTFACENET   = _obtener_clases_alfabeticas()


def indice_a_sujeto(indice, familia):
    if familia == "MobileFaceNet":
        return CLASES_MOBILEFACENET[indice]
    elif familia == "ShuffleFaceNet":
        return CLASES_SHUFFLEFACENET[indice]
    elif familia == "GhostFaceNetV2":
        return CLASES_GHOSTFACENET[indice]
    else:
        raise ValueError(f"Familia no soportada: {familia}")


# ------------------------------------------------------------------
# Checkpoints (reorganizados por carpeta de familia)
# ------------------------------------------------------------------
CHECKPOINTS = {
    # ================= MobileFaceNet =================
    "ORL original":           DIR_MOBILE / "best_mobilefacenet_orl.pth",

    # --- CLAHE x2 ---
    "CLAHE x2 (CE)":          DIR_MOBILE / "best_mobilefacenet_experimento2_clahe_x2.pth",
    "CLAHE x2 (ArcFace)":     DIR_MOBILE / "best_mobilefacenet_arcface_experimento3_clahe_x2.pth",

    # --- CLAHE x4 ---
    "CLAHE x4 (CE)":          DIR_MOBILE / "best_mobilefacenet_clahe_x4_experimento2.pth",
    "CLAHE x4 (ArcFace)":     DIR_MOBILE / "best_mobilefacenet_arcface_clahe_x4.pth",

    # --- SR x2 ---
    "SR x2 (CE)":             DIR_MOBILE / "best_mobilefacenet_experimento2_sr_x2_v2.pth",
    "SR x2 (ArcFace)":        DIR_MOBILE / "best_mobilefacenet_arcface_experimento3_sr_x2.pth",

    # --- SR x4 ---
    "SR x4 (CE)":             DIR_MOBILE / "best_mobilefacenet_experimento2_sr_x4.pth",
    "SR x4 (ArcFace)":        DIR_MOBILE / "best_mobilefacenet_arcface_experimento3_sr_x4.pth",

    # --- Lite-ESRGAN x2 ---
    "LE x2 (CE)":             DIR_MOBILE / "best_mobilefacenet_experimento2_le_x2.pth",
    "LE x2 (ArcFace)":        DIR_MOBILE / "best_mobilefacenet_arcface_experimento3_le_x2.pth",

    # --- Lite-ESRGAN x4 ---
    "LE x4 (CE)":             DIR_MOBILE / "best_mobilefacenet_experimento2_le_x4.pth",
    "LE x4 (ArcFace)":        DIR_MOBILE / "best_mobilefacenet_arcface_experimento3_le_x4.pth",

    # ================= ShuffleFaceNet =================
    "ORL original (Shuffle)": DIR_SHUFFLE / "best_shufflefacenet_orl.pth",

    # --- CLAHE x2 ---
    "CLAHE x2 (Shuffle CE)":        DIR_SHUFFLE / "best_shufflefacenet_clahe_x2_experimento2.pth",
    "CLAHE x2 (Shuffle ArcFace)":   DIR_SHUFFLE / "best_shufflefacenet2_clahe_x2_experimento3.pth",

    # --- CLAHE x4 ---
    "CLAHE x4 (Shuffle CE)":        DIR_SHUFFLE / "best_shufflefacenet_clahe_x4_experimento2.pth",
    "CLAHE x4 (Shuffle ArcFace)":   DIR_SHUFFLE / "best_shufflefacenet2_clahe_x4_experimento3.pth",

    # --- SR x2 ---
    "SR x2 (Shuffle CE)":           DIR_SHUFFLE / "best_shufflefacenet_sr_x2_experimento2.pth",
    "SR x2 (Shuffle ArcFace)":      DIR_SHUFFLE / "best_shufflefacenet2_sr_x2_experimento3.pth",

    # --- SR x4 ---
    "SR x4 (Shuffle CE)":           DIR_SHUFFLE / "best_shufflefacenet_sr_x4_experimento2.pth",
    "SR x4 (Shuffle ArcFace)":      DIR_SHUFFLE / "best_shufflefacenet2_sr_x4_experimento3.pth",

    # --- Lite-ESRGAN x2 ---
    "LE x2 (Shuffle CE)":           DIR_SHUFFLE / "best_shufflefacenet_le_x2_experimento2.pth",
    "LE x2 (Shuffle ArcFace)":      DIR_SHUFFLE / "best_shufflefacenet2_le_x2_experimento3.pth",

    # --- Lite-ESRGAN x4 ---
    "LE x4 (Shuffle CE)":           DIR_SHUFFLE / "best_shufflefacenet_le_x4_experimento2.pth",
    "LE x4 (Shuffle ArcFace)":      DIR_SHUFFLE / "best_shufflefacenet2_le_x4_experimento3.pth",

    # ================= GhostFaceNetV2 =================
    "ORL original (Ghost)":         DIR_GHOST / "best_ghostfacenet_orl.pth",

    # --- CLAHE x2 ---
    "CLAHE x2 (Ghost CE)":          DIR_GHOST / "best_ghostfacenet_clahe_x2_experimento2.pth",
    "CLAHE x2 (Ghost ArcFace)":     DIR_GHOST / "best_ghostfacenet_arcface_clahe_x2_experimento3.pth",
    "CLAHE x2 (Ghost noDFC)":       DIR_GHOST / "best_ghostfacenet_noDFC_clahe_x2_experimento4.pth",

    # --- CLAHE x4 ---
    "CLAHE x4 (Ghost CE)":          DIR_GHOST / "best_ghostfacenet_clahe_x4_experimento2.pth",
    "CLAHE x4 (Ghost ArcFace)":     DIR_GHOST / "best_ghostfacenet_arcface_clahe_x4_experimento3.pth",
    "CLAHE x4 (Ghost noDFC)":       DIR_GHOST / "best_ghostfacenet_noDFC_clahe_x4_experimento4.pth",

    # --- SR x2 ---
    "SR x2 (Ghost CE)":             DIR_GHOST / "best_ghostfacenet_SR_x2_experimento2.pth",
    "SR x2 (Ghost ArcFace)":        DIR_GHOST / "best_ghostfacenet_arcface_SR_x2_experimento3.pth",
    "SR x2 (Ghost noDFC)":          DIR_GHOST / "best_ghostfacenet_noDFC_SR_x2_experimento4.pth",

    # --- SR x4 ---
    "SR x4 (Ghost CE)":             DIR_GHOST / "best_ghostfacenet_SR_x4_experimento2.pth",
    "SR x4 (Ghost ArcFace)":        DIR_GHOST / "best_ghostfacenet_arcface_SR_x4_experimento3.pth",
    "SR x4 (Ghost noDFC)":          DIR_GHOST / "best_ghostfacenet_noDFC_SR_x4_experimento4.pth",

    # --- Lite-ESRGAN x2 ---
    "LE x2 (Ghost CE)":             DIR_GHOST / "best_ghostfacenet_LE_x2_experimento2.pth",
    "LE x2 (Ghost ArcFace)":        DIR_GHOST / "best_ghostfacenet_arcface_LE_x2_experimento3.pth",
    "LE x2 (Ghost noDFC)":          DIR_GHOST / "best_ghostfacenet_noDFC_LE_x2_experimento4.pth",

    # --- Lite-ESRGAN x4 ---
    "LE x4 (Ghost CE)":             DIR_GHOST / "best_ghostfacenet_LE_x4_experimento2.pth",
    "LE x4 (Ghost ArcFace)":        DIR_GHOST / "best_ghostfacenet_arcface_LE_x4_experimento3.pth",
    "LE x4 (Ghost noDFC)":          DIR_GHOST / "best_ghostfacenet_noDFC_LE_x4_experimento4.pth",
}

# ------------------------------------------------------------------
# Familia por condición
# ------------------------------------------------------------------
FAMILIA_POR_CONDICION = {}

# MobileFaceNet
for _k in ["ORL original", "CLAHE x2 (CE)", "CLAHE x4 (CE)", "SR x2 (CE)", "SR x4 (CE)",
           "LE x2 (CE)", "LE x4 (CE)",
           "CLAHE x2 (ArcFace)", "CLAHE x4 (ArcFace)", "SR x2 (ArcFace)", "SR x4 (ArcFace)",
           "LE x2 (ArcFace)", "LE x4 (ArcFace)"]:
    FAMILIA_POR_CONDICION[_k] = "MobileFaceNet"

# ShuffleFaceNet
for _k in ["ORL original (Shuffle)",
           "CLAHE x2 (Shuffle CE)", "CLAHE x4 (Shuffle CE)", "SR x2 (Shuffle CE)", "SR x4 (Shuffle CE)",
           "LE x2 (Shuffle CE)", "LE x4 (Shuffle CE)",
           "CLAHE x2 (Shuffle ArcFace)", "CLAHE x4 (Shuffle ArcFace)",
           "SR x2 (Shuffle ArcFace)", "SR x4 (Shuffle ArcFace)",
           "LE x2 (Shuffle ArcFace)", "LE x4 (Shuffle ArcFace)"]:
    FAMILIA_POR_CONDICION[_k] = "ShuffleFaceNet"

# GhostFaceNetV2
for _k in ["ORL original (Ghost)",
           "CLAHE x2 (Ghost CE)", "CLAHE x4 (Ghost CE)", "SR x2 (Ghost CE)", "SR x4 (Ghost CE)",
           "LE x2 (Ghost CE)", "LE x4 (Ghost CE)",
           "CLAHE x2 (Ghost ArcFace)", "CLAHE x4 (Ghost ArcFace)",
           "SR x2 (Ghost ArcFace)", "SR x4 (Ghost ArcFace)",
           "LE x2 (Ghost ArcFace)", "LE x4 (Ghost ArcFace)",
           "CLAHE x2 (Ghost noDFC)", "CLAHE x4 (Ghost noDFC)",
           "SR x2 (Ghost noDFC)", "SR x4 (Ghost noDFC)",
           "LE x2 (Ghost noDFC)", "LE x4 (Ghost noDFC)"]:
    FAMILIA_POR_CONDICION[_k] = "GhostFaceNetV2"

# ------------------------------------------------------------------
# Tipo de cabeza / loss
# ------------------------------------------------------------------
ARQUITECTURA_POR_CONDICION = {}

for _k in ["ORL original", "CLAHE x2 (CE)", "CLAHE x4 (CE)", "SR x2 (CE)", "SR x4 (CE)", "LE x2 (CE)", "LE x4 (CE)"]:
    ARQUITECTURA_POR_CONDICION[_k] = "CE"
for _k in ["CLAHE x2 (ArcFace)", "CLAHE x4 (ArcFace)", "SR x2 (ArcFace)", "SR x4 (ArcFace)",
           "LE x2 (ArcFace)", "LE x4 (ArcFace)"]:
    ARQUITECTURA_POR_CONDICION[_k] = "ArcFace"

for _k in ["ORL original (Shuffle)",
           "CLAHE x2 (Shuffle CE)", "CLAHE x4 (Shuffle CE)", "SR x2 (Shuffle CE)", "SR x4 (Shuffle CE)",
           "LE x2 (Shuffle CE)", "LE x4 (Shuffle CE)"]:
    ARQUITECTURA_POR_CONDICION[_k] = "CE"
for _k in ["CLAHE x2 (Shuffle ArcFace)", "CLAHE x4 (Shuffle ArcFace)",
           "SR x2 (Shuffle ArcFace)", "SR x4 (Shuffle ArcFace)",
           "LE x2 (Shuffle ArcFace)", "LE x4 (Shuffle ArcFace)"]:
    ARQUITECTURA_POR_CONDICION[_k] = "ArcFace"

for _k in ["ORL original (Ghost)",
           "CLAHE x2 (Ghost CE)", "CLAHE x4 (Ghost CE)", "SR x2 (Ghost CE)", "SR x4 (Ghost CE)",
           "LE x2 (Ghost CE)", "LE x4 (Ghost CE)"]:
    ARQUITECTURA_POR_CONDICION[_k] = "CE"
for _k in [
    "CLAHE x2 (Ghost ArcFace)", "CLAHE x4 (Ghost ArcFace)",
    "SR x2 (Ghost ArcFace)",    "SR x4 (Ghost ArcFace)",
    "LE x2 (Ghost ArcFace)",    "LE x4 (Ghost ArcFace)",
]:
    ARQUITECTURA_POR_CONDICION[_k] = "ArcFace"
for _k in ["CLAHE x2 (Ghost noDFC)", "CLAHE x4 (Ghost noDFC)",
           "SR x2 (Ghost noDFC)", "SR x4 (Ghost noDFC)",
           "LE x2 (Ghost noDFC)", "LE x4 (Ghost noDFC)"]:
    ARQUITECTURA_POR_CONDICION[_k] = "noDFC"   

# ------------------------------------------------------------------
# Degradación (técnica, escala)
# ------------------------------------------------------------------
DEGRADACION_POR_CONDICION = {
    "ORL original": None,
    "ORL original (Shuffle)": None,
    "ORL original (Ghost)": None,
}

for _tec in ["CLAHE", "SR", "LE"]:
    for _esc in [2, 4]:
        # MobileFaceNet
        if _tec == "LE":
            DEGRADACION_POR_CONDICION[f"LE x{_esc} (CE)"]      = (_tec, _esc)
            DEGRADACION_POR_CONDICION[f"LE x{_esc} (ArcFace)"] = (_tec, _esc)
        else:
            DEGRADACION_POR_CONDICION[f"{_tec} x{_esc} (CE)"]      = (_tec, _esc)
            DEGRADACION_POR_CONDICION[f"{_tec} x{_esc} (ArcFace)"] = (_tec, _esc)

        # ShuffleFaceNet
        if _tec == "LE":
            DEGRADACION_POR_CONDICION[f"LE x{_esc} (Shuffle CE)"]      = (_tec, _esc)
            DEGRADACION_POR_CONDICION[f"LE x{_esc} (Shuffle ArcFace)"] = (_tec, _esc)
        else:
            DEGRADACION_POR_CONDICION[f"{_tec} x{_esc} (Shuffle CE)"]      = (_tec, _esc)
            DEGRADACION_POR_CONDICION[f"{_tec} x{_esc} (Shuffle ArcFace)"] = (_tec, _esc)

        # GhostFaceNetV2
        for _tipo in ["CE", "ArcFace", "noDFC"]:
            DEGRADACION_POR_CONDICION[f"{_tec} x{_esc} (Ghost {_tipo})"] = (_tec, _esc)

# ------------------------------------------------------------------
# Accuracy de referencia (rellena con los valores reales que tengas)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Accuracy de referencia (Fase 2) — según experimentos reportados
# ------------------------------------------------------------------

ACCURACY_REFERENCIA = {
    # ================= MobileFaceNet =================
    "ORL original":        "100.00%",

    "CLAHE x2 (CE)":       "97.50%",
    "CLAHE x2 (ArcFace)":  "92.50%",

    "CLAHE x4 (CE)":       "62.50%",
    "CLAHE x4 (ArcFace)":  "85.00%",

    "SR x2 (CE)":          "92.50%",
    "SR x2 (ArcFace)":     "95.00%",

    "SR x4 (CE)":          "82.50%",
    "SR x4 (ArcFace)":     "95.00%",

    "LE x2 (CE)":          "92.50%",
    "LE x2 (ArcFace)":     "92.50%",

    "LE x4 (CE)":          "90.00%",
    "LE x4 (ArcFace)":     "92.50%",

    # ================= ShuffleFaceNet =================
    "ORL original (Shuffle)":       "100.00%",

    "CLAHE x2 (Shuffle CE)":        "97.50%",
    "CLAHE x2 (Shuffle ArcFace)":   "92.50%",

    "CLAHE x4 (Shuffle CE)":        "62.50%",
    "CLAHE x4 (Shuffle ArcFace)":   "85.00%",

    "SR x2 (Shuffle CE)":           "92.50%",
    "SR x2 (Shuffle ArcFace)":      "95.00%",

    "SR x4 (Shuffle CE)":           "82.50%",
    "SR x4 (Shuffle ArcFace)":      "95.00%",

    "LE x2 (Shuffle CE)":           "92.50%",
    "LE x2 (Shuffle ArcFace)":      "97.50%",

    "LE x4 (Shuffle CE)":           "92.50%",
    "LE x4 (Shuffle ArcFace)":      "95.00%",

    # ================= GhostFaceNetV2 =================
    "ORL original (Ghost)":         "95.00%",

    "CLAHE x2 (Ghost CE)":          "95.00%",
    "CLAHE x2 (Ghost ArcFace)":     "95.00%",
    "CLAHE x2 (Ghost noDFC)":       "92.50%",

    "CLAHE x4 (Ghost CE)":          "85.00%",
    "CLAHE x4 (Ghost ArcFace)":     "87.50%",
    "CLAHE x4 (Ghost noDFC)":       "82.50%",

    "SR x2 (Ghost CE)":             "95.00%",
    "SR x2 (Ghost ArcFace)":        "95.00%",
    "SR x2 (Ghost noDFC)":          "95.00%",

    "SR x4 (Ghost CE)":             "87.50%",
    "SR x4 (Ghost ArcFace)":        "92.50%",
    "SR x4 (Ghost noDFC)":          "87.50%",

    "LE x2 (Ghost CE)":             "95.00%",
    "LE x2 (Ghost ArcFace)":        "97.50%",
    "LE x2 (Ghost noDFC)":          "95.00%",

    "LE x4 (Ghost CE)":             "95.00%",
    "LE x4 (Ghost ArcFace)":        "97.50%",
    "LE x4 (Ghost noDFC)":          "95.00%",
}
# ------------------------------------------------------------------
# Transformaciones
# ------------------------------------------------------------------
_transform_mobilefacenet = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

_transform_shufflefacenet = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE_SHUFFLE, IMG_SIZE_SHUFFLE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

_transform_ghostfacenet = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE_GHOST, IMG_SIZE_GHOST)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ------------------------------------------------------------------
# Cargar modelo
# ------------------------------------------------------------------
def cargar_modelo(checkpoint_path, familia, tipo):
    """
    familia : "MobileFaceNet" | "ShuffleFaceNet" | "GhostFaceNetV2"
    tipo    : "CE" | "ArcFace" | "noDFC"
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint:\n{checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    sd_model = state_dict.get("model", state_dict) if isinstance(state_dict, dict) else state_dict

    # ---------------- CE ----------------
    if tipo == "CE":
        if familia == "MobileFaceNet":
            modelo = MobileFaceNet(num_classes=NUM_CLASSES).to(DEVICE)
        elif familia == "ShuffleFaceNet":
            modelo = ShuffleFaceNet(num_classes=NUM_CLASSES).to(DEVICE)
        elif familia == "GhostFaceNetV2":
            modelo = GhostFaceNetV2(num_classes=NUM_CLASSES).to(DEVICE)
        else:
            raise ValueError(f"Familia no soportada: {familia}")

        modelo.load_state_dict(sd_model, strict=False)
        modelo.eval()
        return modelo, None

    # ---------------- ArcFace / noDFC ----------------
    if familia == "MobileFaceNet":
        modelo = MobileFaceNet2(embedding_dim=128).to(DEVICE)
    elif familia == "ShuffleFaceNet":
        modelo = ShuffleFaceNet2_Arface(embedding_dim=128).to(DEVICE)
    elif familia == "GhostFaceNetV2":
        # noDFC usa la MISMA clase pero SIN atenciones DFC
        if tipo == "noDFC":
            modelo = GhostFaceNetV2_ArcFace(embedding_dim=128, use_dfc=False).to(DEVICE)
        else:
            modelo = GhostFaceNetV2_ArcFace(embedding_dim=128, use_dfc=True).to(DEVICE)
    else:
        raise ValueError(f"Familia no soportada: {familia}")

    arcface = ArcFaceHead(embedding_dim=128, num_classes=NUM_CLASSES, s=30.0).to(DEVICE)

    sd_arcface = (
        state_dict.get("arcface") or state_dict.get("criterion")
        if isinstance(state_dict, dict) else None
    )

    modelo.load_state_dict(sd_model, strict=False)
    if sd_arcface is not None:
        arcface.load_state_dict(sd_arcface, strict=False)

    modelo.eval()
    arcface.eval()
    return modelo, arcface

# ------------------------------------------------------------------
# Predicción
# ------------------------------------------------------------------
def predecir(modelo, arcface, imagen_pil, top_k=5):
    if isinstance(modelo, (ShuffleFaceNet, ShuffleFaceNet2_Arface)):
        transform = _transform_shufflefacenet
        familia = "ShuffleFaceNet"
    elif isinstance(modelo, (GhostFaceNetV2, GhostFaceNetV2_ArcFace)):
        transform = _transform_ghostfacenet
        familia = "GhostFaceNetV2"
    else:
        transform = _transform_mobilefacenet
        familia = "MobileFaceNet"

    tensor = transform(imagen_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        salida = modelo(tensor)
        if arcface is None:
            probs = F.softmax(salida, dim=1).squeeze(0)
        else:
            logits = arcface(salida)
            probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.shape[0]))
    top_k_resultados = [(idx.item(), prob.item() * 100) for idx, prob in zip(top_indices, top_probs)]

    clase = top_k_resultados[0][0]
    confianza = top_k_resultados[0][1]
    sujeto = indice_a_sujeto(clase, familia)

    top_k_resultados = [
        (idx, prob, indice_a_sujeto(idx, familia)) for idx, prob, in top_k_resultados
    ]

    return clase, confianza, top_k_resultados, sujeto

# ------------------------------------------------------------------
# Utilidad
# ------------------------------------------------------------------
def numpy_a_pil(imagen_array):
    return Image.fromarray(imagen_array).convert("L")