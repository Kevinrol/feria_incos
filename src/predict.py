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

def predict_single_image(
    image_path="test-plant.jpg",
    model_path="models/agrovision_mobilenetv2_mvp2.keras",
    classes_path="models/class_names_mvp2.json",
    mapping_path="models/class_mapping_mvp2.json",
    top_k=5
):
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"No se encontro la imagen en: {img_file.resolve()}")

    # Fallback si el modelo MVP 2 aún no existe
    if not Path(model_path).exists():
        fallback_model = "models/plant_disease_mobilenetv2.keras"
        fallback_classes = "models/class_names.json"
        if Path(fallback_model).exists():
            print(f"[!] Aviso: Modelo MVP 2 no encontrado, utilizando modelo MVP 1 ({fallback_model})")
            model_path = fallback_model
            classes_path = fallback_classes
        else:
            raise FileNotFoundError(f"No se encontro el modelo en: {model_path}. Ejecute src/train.py primero.")

    # Auto-detectar archivo de clases según el modelo
    # LeafNet-kallpa (MVP7) y MVP2 usan 27 clases (class_names_mvp2.json)
    if classes_path == "models/class_names_mvp2.json":
        is_kallpa = "kallpa" in str(model_path).lower() or "mvp7" in str(model_path).lower()
        is_mvp2 = "mvp2" in str(model_path).lower()
        if is_kallpa and not Path(classes_path).exists():
            classes_path = "models/class_names_mvp2.json"
        # Verificar que el archivo de clases tiene el número correcto de clases
        # Si se usa el modelo kallpa con class_names.json (13 clases), auto-corregir
        model_name_lower = str(model_path).lower()
        if "kallpa" in model_name_lower or "mvp2" in model_name_lower:
            preferred = "models/class_names_mvp2.json"
            if classes_path != preferred and Path(preferred).exists():
                import json as _json
                with open(classes_path) as _f:
                    _n = len(_json.load(_f))
                if _n < 20:  # class_names.json solo tiene 13 clases
                    print(f"[!] Auto-corrigiendo clases: '{classes_path}' ({_n} clases) → '{preferred}' (27 clases)")
                    classes_path = preferred

    # 1. Cargar clases y metadatos
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    mapping_dict = {}
    if Path(mapping_path).exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_dict = json.load(f).get("metadata", {})

    # 2. Cargar modelo
    model = tf.keras.models.load_model(model_path, compile=False)

    # 3. Cargar y preprocesar imagen
    img = Image.open(img_file).convert("RGB")
    img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Preprocesamiento adaptable según arquitectura
    is_efficientnet = "kallpa" in str(model_path).lower() or any("efficientnet" in str(getattr(l, "name", "")).lower() for l in model.layers)
    if is_efficientnet:
        img_preprocessed = img_batch  # EfficientNet tiene capa interna de normalización [0, 255]
    else:
        img_preprocessed = preprocess_input(img_batch)

    # 4. Inferencia
    preds = model.predict(img_preprocessed, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1]

    best_idx = top_indices[0]
    best_raw_class = class_names[best_idx]
    best_conf = preds[best_idx] * 100
    crop_detected, condition_detected = get_class_info(best_raw_class, mapping_dict)

    # 5. Formatear y mostrar resultados
    print("\n" + "=" * 65)
    print("                AGROVISION — DIAGNOSTICO DE LA PLANTA")
    print("=" * 65)
    print(f"  Archivo:   {img_file.name}")
    print(f"  Ruta:      {img_file.resolve()}\n")
    print(f"  🌱 Planta detectada:     {crop_detected}")
    print(f"  🔬 Estado / Patología:   {condition_detected}")
    print(f"  🎯 Nivel de Confianza:   {best_conf:.2f}%\n")
    print("  Distribucion de probabilidades (Top):")
    print("  " + "-" * 61)

    for i in range(min(top_k, len(class_names))):
        idx = top_indices[i]
        raw_c = class_names[idx]
        crop, disease = get_class_info(raw_c, mapping_dict)
        prob = preds[idx] * 100
        label_str = f"{crop} - {disease}"
        if len(label_str) > 46:
            label_str = label_str[:43] + "..."
        print(f"   [{i+1}] {label_str:<46} {prob:6.2f}%")

    print("=" * 65 + "\n")

    return {
        "image": str(img_file),
        "crop": crop_detected,
        "condition": condition_detected,
        "confidence": float(best_conf),
        "raw_class": best_raw_class,
        "top_predictions": [
            {
                "class": class_names[idx],
                "crop": get_class_info(class_names[idx], mapping_dict)[0],
                "condition": get_class_info(class_names[idx], mapping_dict)[1],
                "probability": float(preds[idx] * 100)
            }
            for idx in top_indices[:top_k]
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgroVision: Predecir estado/enfermedad")
    parser.add_argument("--image", type=str, default="test-plant.jpg", help="Ruta de la imagen")
    parser.add_argument("--model", type=str, default="models/agrovision_mobilenetv2_mvp2.keras", help="Ruta del modelo")
    parser.add_argument("--classes", type=str, default="models/class_names_mvp2.json", help="Ruta de class_names")
    parser.add_argument("--mapping", type=str, default="models/class_mapping_mvp2.json", help="Ruta de class_mapping")
    parser.add_argument("--top", type=int, default=5, help="Numero de probabilidades")
    args = parser.parse_args()

    predict_single_image(
        image_path=args.image,
        model_path=args.model,
        classes_path=args.classes,
        mapping_path=args.mapping,
        top_k=args.top
    )
