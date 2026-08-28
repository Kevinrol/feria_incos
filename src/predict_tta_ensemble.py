"""
predict_tta_ensemble.py
=======================
Prediccion de alta precision usando:
  1. TTA (Test-Time Augmentation): 8 variaciones de la imagen + promedio para Kallpa
  2. Ensemble de 3 modelos: MVP6 (MobileNetV2) + Kallpa (EfficientNetB0) con TTA
  3. Fusion ponderada optimizada para condiciones de campo
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT_DIR = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de clase
# ─────────────────────────────────────────────────────────────────────────────
def get_class_info(raw_class_name, mapping_dict=None):
    if mapping_dict and raw_class_name in mapping_dict:
        m = mapping_dict[raw_class_name]
        return m.get("crop_es", "Cultivo"), m.get("disease_es", "Estado")
    parts = raw_class_name.split("___")
    crop = parts[0].replace("_", " ").title()
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
    return crop, disease


# ─────────────────────────────────────────────────────────────────────────────
# TTA: generar N aumentaciones de la misma imagen
# ─────────────────────────────────────────────────────────────────────────────
def generate_tta_batch(img_array_raw, n_augmentations=8):
    """
    Genera N variaciones aumentadas de la imagen para TTA.
    img_array_raw: numpy array [H, W, 3] en [0, 255]
    Retorna: numpy array [n_augmentations, 224, 224, 3]
    """
    imgs = []
    # 1. Original (sin aumentacion)
    imgs.append(img_array_raw.copy())

    # 2. Flip horizontal
    imgs.append(np.fliplr(img_array_raw))

    # 3. Flip vertical
    imgs.append(np.flipud(img_array_raw))

    # 4. Flip horizontal + vertical
    imgs.append(np.flipud(np.fliplr(img_array_raw)))

    # 5. Rotacion 90
    imgs.append(np.rot90(img_array_raw, k=1))

    # 6. Rotacion 270
    imgs.append(np.rot90(img_array_raw, k=3))

    # 7. Brillo +15%
    bright = np.clip(img_array_raw * 1.15, 0, 255)
    imgs.append(bright)

    # 8. Brillo -15%
    dark = np.clip(img_array_raw * 0.85, 0, 255)
    imgs.append(dark)

    # Si se piden mas de 8
    while len(imgs) < n_augmentations:
        # Contraste aleatorio suave
        factor = np.random.uniform(0.85, 1.15)
        imgs.append(np.clip(img_array_raw * factor, 0, 255))

    return np.stack(imgs[:n_augmentations], axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Prediccion con TTA para un modelo
# ─────────────────────────────────────────────────────────────────────────────
def predict_with_tta(model, img_raw, preprocess_fn=None, n_aug=8):
    """
    Predice con TTA: genera N augmentaciones, predice cada una y promedia.
    img_raw: numpy [224, 224, 3] en [0, 255]
    Retorna: numpy [num_classes] promedio de probabilidades
    """
    batch = generate_tta_batch(img_raw, n_augmentations=n_aug)  # [N, 224, 224, 3]
    if preprocess_fn is not None:
        batch = preprocess_fn(batch)
    preds = model.predict(batch, verbose=0)  # [N, num_classes]
    return preds.mean(axis=0)  # [num_classes]


# ─────────────────────────────────────────────────────────────────────────────
# Prediccion principal con TTA + Ensemble
# ─────────────────────────────────────────────────────────────────────────────
def predict_tta_ensemble(
    image_path,
    kallpa_model_path="models/LeafNet_kallpa.keras",
    mvp6_model_path="models/agrovision_mobilenetv2_mvp6.keras",
    classes_path="models/class_names_mvp2.json",
    mapping_path="models/class_mapping_mvp2.json",
    n_tta=8,
    weight_kallpa=0.60,
    weight_mvp6=0.40,
    top_k=5,
):
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"No se encontro la imagen: {img_file.resolve()}")

    # Cargar etiquetas y mapeo
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    mapping_dict = {}
    if Path(mapping_path).exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_dict = json.load(f).get("metadata", {})

    # Cargar imagen
    img_pil = Image.open(img_file).convert("RGB")
    img_resized = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
    img_raw = np.array(img_resized, dtype=np.float32)  # [0, 255]

    all_preds = []
    model_names = []
    model_weights = []

    # ── Kallpa (EfficientNetB0) con TTA ─────────────────────────────────────
    kallpa_file = Path(kallpa_model_path)
    if kallpa_file.exists():
        print(f"  [+] Cargando LeafNet-kallpa ({kallpa_model_path})...")
        model_kallpa = tf.keras.models.load_model(str(kallpa_file), compile=False)
        print(f"      TTA x{n_tta} augmentaciones...")
        # EfficientNet espera [0, 255] directamente
        preds_kallpa = predict_with_tta(model_kallpa, img_raw, preprocess_fn=None, n_aug=n_tta)

        # Padding si tiene menos clases que el JSON (no deberia pasar)
        if len(preds_kallpa) < num_classes:
            pad = np.zeros(num_classes - len(preds_kallpa))
            preds_kallpa = np.concatenate([preds_kallpa, pad])
        elif len(preds_kallpa) > num_classes:
            preds_kallpa = preds_kallpa[:num_classes]

        all_preds.append(preds_kallpa)
        model_names.append("LeafNet-kallpa (EfficientNetB0)")
        model_weights.append(weight_kallpa)
        del model_kallpa

    # ── MVP6 (MobileNetV2) ──────────────────────────────────────────────────
    mvp6_file = Path(mvp6_model_path)
    if mvp6_file.exists():
        print(f"  [+] Cargando MVP 6 ({mvp6_model_path})...")
        model_mvp6 = tf.keras.models.load_model(str(mvp6_file), compile=False)
        # MobileNetV2 usa preprocess_input de ImageNet
        preds_mvp6 = predict_with_tta(
            model_mvp6, img_raw,
            preprocess_fn=lambda b: preprocess_input(b.copy()),
            n_aug=n_tta
        )
        if len(preds_mvp6) < num_classes:
            pad = np.zeros(num_classes - len(preds_mvp6))
            preds_mvp6 = np.concatenate([preds_mvp6, pad])
        all_preds.append(preds_mvp6)
        model_names.append("MVP 6 (MobileNetV2)")
        model_weights.append(weight_mvp6)
        del model_mvp6

    if not all_preds:
        raise RuntimeError("No se encontro ningun modelo valido.")

    # Fusion ponderada
    weights_arr = np.array(model_weights, dtype=np.float32)
    weights_arr /= weights_arr.sum()
    ensemble_preds = sum(w * p for w, p in zip(weights_arr, all_preds))

    top_idx_ens = np.argsort(ensemble_preds)[::-1]
    best_idx = top_idx_ens[0]
    best_raw = class_names[best_idx]
    best_conf = ensemble_preds[best_idx] * 100
    crop_det, cond_det = get_class_info(best_raw, mapping_dict)

    # Mostrar resultados
    print("\n" + "=" * 70)
    print("    AGROVISION — DIAGNOSTICO TTA + ENSEMBLE (MVP6 + LeafNet-kallpa)")
    print("=" * 70)
    print(f"  Archivo:   {img_file.name}")
    print(f"  Ruta:      {img_file.resolve()}\n")

    print(f"  Modelos usados:")
    for name, w, p in zip(model_names, weights_arr, all_preds):
        top_i = int(np.argmax(p))
        top_c, top_d = get_class_info(class_names[top_i], mapping_dict)
        print(f"     • {name:<38} peso: {w*100:.0f}%")
        print(f"       Prediccion individual: {top_c} - {top_d} ({p[top_i]*100:.1f}%)")
    print(f"\n  TTA augmentaciones por modelo: {n_tta}x")
    print("  " + "─" * 66)
    print(f"\n  🌱 Planta detectada:     {crop_det}")
    print(f"  🔬 Estado / Patologia:   {cond_det}")
    print(f"  🎯 Nivel de Confianza:   {best_conf:.2f}%\n")

    print("  Distribucion de probabilidades (Top):")
    print("  " + "-" * 66)
    for i in range(min(top_k, num_classes)):
        idx = top_idx_ens[i]
        raw_c = class_names[idx]
        crop, disease = get_class_info(raw_c, mapping_dict)
        prob = ensemble_preds[idx] * 100
        label_str = f"{crop} - {disease}"
        if len(label_str) > 44:
            label_str = label_str[:41] + "..."
        per_model = "  |  ".join(
            f"{mn.split('(')[0].strip()[:8]}: {p[idx]*100:5.1f}%"
            for mn, p in zip(model_names, all_preds)
        )
        print(f"   [{i+1}] {label_str:<44} {prob:6.2f}%")
    print("=" * 70 + "\n")

    return {
        "image": str(img_file),
        "crop": crop_det,
        "condition": cond_det,
        "confidence": float(best_conf),
        "raw_class": best_raw,
        "n_tta": n_tta,
        "models": [
            {"name": n, "weight": float(w), "top1": class_names[int(np.argmax(p))], "top1_conf": float(np.max(p)*100)}
            for n, w, p in zip(model_names, weights_arr, all_preds)
        ],
        "top_predictions": [
            {
                "class": class_names[idx],
                "crop": get_class_info(class_names[idx], mapping_dict)[0],
                "condition": get_class_info(class_names[idx], mapping_dict)[1],
                "probability": float(ensemble_preds[idx] * 100)
            }
            for idx in top_idx_ens[:top_k]
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroVision: Prediccion TTA + Ensemble MVP6 + Kallpa")
    parser.add_argument("--image",   type=str, required=True,  help="Ruta de la imagen")
    parser.add_argument("--kallpa",  type=str, default="models/LeafNet_kallpa.keras")
    parser.add_argument("--mvp6",    type=str, default="models/agrovision_mobilenetv2_mvp6.keras")
    parser.add_argument("--classes", type=str, default="models/class_names_mvp2.json")
    parser.add_argument("--mapping", type=str, default="models/class_mapping_mvp2.json")
    parser.add_argument("--tta",     type=int, default=8,  help="Numero de augmentaciones TTA")
    parser.add_argument("--wk",      type=float, default=0.60, help="Peso para LeafNet-kallpa")
    parser.add_argument("--wm",      type=float, default=0.40, help="Peso para MVP6")
    parser.add_argument("--top",     type=int, default=5,  help="Top K resultados")
    args = parser.parse_args()

    predict_tta_ensemble(
        image_path=args.image,
        kallpa_model_path=args.kallpa,
        mvp6_model_path=args.mvp6,
        classes_path=args.classes,
        mapping_path=args.mapping,
        n_tta=args.tta,
        weight_kallpa=args.wk,
        weight_mvp6=args.wm,
        top_k=args.top,
    )
