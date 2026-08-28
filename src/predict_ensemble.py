import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def get_class_info(raw_class_name, mapping_dict=None):
    if mapping_dict and raw_class_name in mapping_dict:
        m = mapping_dict[raw_class_name]
        return m.get("crop_es", "Cultivo"), m.get("disease_es", "Estado")
    
    parts = raw_class_name.split("___")
    crop = parts[0].replace("_", " ").title()
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
    return crop, disease

def predict_ensemble(
    image_path="test-plant.jpg",
    model4_path="models/agrovision_mobilenetv2_mvp4.keras",
    model6_path="models/agrovision_mobilenetv2_mvp6.keras",
    classes_path="models/class_names_mvp2.json",
    mapping_path="models/class_mapping_mvp2.json",
    weight4=0.5,
    weight6=0.5,
    top_k=5
):
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"No se encontró la imagen en: {img_file.resolve()}")

    p4_file = Path(model4_path)
    p6_file = Path(model6_path)

    if not p4_file.exists():
        raise FileNotFoundError(f"No se encontró el modelo MVP 4 en: {p4_file.resolve()}")
    if not p6_file.exists():
        raise FileNotFoundError(f"No se encontró el modelo MVP 6 en: {p6_file.resolve()}")

    # 1. Cargar clases y mapeo
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    mapping_dict = {}
    if Path(mapping_path).exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_dict = json.load(f).get("metadata", {})

    # 2. Cargar ambos modelos
    model4 = tf.keras.models.load_model(p4_file, compile=False)
    model6 = tf.keras.models.load_model(p6_file, compile=False)

    # 3. Preprocesar imagen
    img = Image.open(img_file).convert("RGB")
    img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # 4. Inferencia individual
    preds4 = model4.predict(img_preprocessed, verbose=0)[0]
    preds6 = model6.predict(img_preprocessed, verbose=0)[0]

    # 5. Fusión Probabilística Ponderada (Ensemble)
    total_weight = weight4 + weight6
    w4_norm = weight4 / total_weight
    w6_norm = weight6 / total_weight

    ensemble_preds = (w4_norm * preds4) + (w6_norm * preds6)

    # Ordenar índices
    top_idx_ens = np.argsort(ensemble_preds)[::-1]
    top_idx_m4 = np.argsort(preds4)[::-1]
    top_idx_m6 = np.argsort(preds6)[::-1]

    best_idx = top_idx_ens[0]
    best_raw = class_names[best_idx]
    best_conf = ensemble_preds[best_idx] * 100
    crop_det, cond_det = get_class_info(best_raw, mapping_dict)

    # Diagnósticos individuales
    crop_m4, cond_m4 = get_class_info(class_names[top_idx_m4[0]], mapping_dict)
    conf_m4 = preds4[top_idx_m4[0]] * 100

    crop_m6, cond_m6 = get_class_info(class_names[top_idx_m6[0]], mapping_dict)
    conf_m6 = preds6[top_idx_m6[0]] * 100

    # 6. Despliegue de Resultados
    print("\n" + "=" * 70)
    print("       AGROVISION — DIAGNOSTICO POR ENSEMBLE MULTIDOMINIO (MVP 4 + 6)")
    print("=" * 70)
    print(f"  Archivo:   {img_file.name}")
    print(f"  Ruta:      {img_file.resolve()}\n")

    print(f"  📊 Modelos individuales:")
    print(f"     • MVP 4 (Laboratorio):  {crop_m4} - {cond_m4:<28} (Conf: {conf_m4:5.2f}%)")
    print(f"     • MVP 6 (In-the-Wild):  {crop_m6} - {cond_m6:<28} (Conf: {conf_m6:5.2f}%)")
    print(f"     • Ponderación aplicada: MVP 4 ({w4_norm*100:.0f}%) | MVP 6 ({w6_norm*100:.0f}%)\n")

    print("  " + "─" * 66)
    print(f"  🌱 Diagnóstico Consolidado: {crop_det} — {cond_det}")
    print(f"  🎯 Nivel de Confianza:     {best_conf:.2f}%")
    print("  " + "─" * 66 + "\n")

    print("  Distribución de probabilidades (Top Consolidado):")
    print("  " + "-" * 66)
    for i in range(min(top_k, len(class_names))):
        idx = top_idx_ens[i]
        raw_c = class_names[idx]
        crop, disease = get_class_info(raw_c, mapping_dict)
        prob = ensemble_preds[idx] * 100
        p4_i = preds4[idx] * 100
        p6_i = preds6[idx] * 100
        label_str = f"{crop} - {disease}"
        if len(label_str) > 34:
            label_str = label_str[:31] + "..."
        print(f"   [{i+1}] {label_str:<34} {prob:6.2f}%  (M4: {p4_i:5.1f}% | M6: {p6_i:5.1f}%)")

    print("=" * 70 + "\n")

    return {
        "image": str(img_file),
        "crop": crop_det,
        "condition": cond_det,
        "confidence": float(best_conf),
        "raw_class": best_raw,
        "model4_top": {"class": class_names[top_idx_m4[0]], "conf": float(conf_m4)},
        "model6_top": {"class": class_names[top_idx_m6[0]], "conf": float(conf_m6)},
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
    parser = argparse.ArgumentParser(description="AgroVision: Predecir con Ensemble MVP 4 + MVP 6")
    parser.add_argument("--image", type=str, default="test-plant.jpg", help="Ruta de la imagen")
    parser.add_argument("--model4", type=str, default="models/agrovision_mobilenetv2_mvp4.keras", help="Ruta de MVP 4")
    parser.add_argument("--model6", type=str, default="models/agrovision_mobilenetv2_mvp6.keras", help="Ruta de MVP 6")
    parser.add_argument("--classes", type=str, default="models/class_names_mvp2.json", help="Ruta de class_names")
    parser.add_argument("--mapping", type=str, default="models/class_mapping_mvp2.json", help="Ruta de class_mapping")
    parser.add_argument("--weight4", type=float, default=0.5, help="Peso relativo para MVP 4")
    parser.add_argument("--weight6", type=float, default=0.5, help="Peso relativo para MVP 6")
    parser.add_argument("--top", type=int, default=5, help="Número de probabilidades")
    args = parser.parse_args()

    predict_ensemble(
        image_path=args.image,
        model4_path=args.model4,
        model6_path=args.model6,
        classes_path=args.classes,
        mapping_path=args.mapping,
        weight4=args.weight4,
        weight6=args.weight6,
        top_k=args.top
    )
