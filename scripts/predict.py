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

def parse_class_info(raw_class_name):
    """
    Convierte clases del dataset como 'Tomato___Late_blight' a nombres 100% en español:
    ('Tomate', 'Tizón tardío')
    """
    crop_translations = {
        "tomato": "Tomate",
        "potato": "Papa",
        "pepper,_bell": "Pimiento morrón",
        "pepper": "Pimiento",
        "apple": "Manzana",
        "blueberry": "Arándano",
        "cherry_(including_sour)": "Cereza",
        "cherry": "Cereza",
        "corn_(maize)": "Maíz",
        "corn": "Maíz",
        "grape": "Vid (Uva)",
        "orange": "Naranja",
        "peach": "Melocotón (Durazno)",
        "raspberry": "Frambuesa",
        "soybean": "Soja",
        "squash": "Calabaza",
        "strawberry": "Fresa"
    }

    disease_translations = {
        "bacterial_spot": "Mancha bacteriana",
        "bacterial spot": "Mancha bacteriana",
        "early_blight": "Tizón temprano",
        "early blight": "Tizón temprano",
        "late_blight": "Tizón tardío",
        "late blight": "Tizón tardío",
        "leaf_mold": "Moho foliar",
        "leaf mold": "Moho foliar",
        "septoria_leaf_spot": "Mancha foliar por Septoria",
        "septoria leaf spot": "Mancha foliar por Septoria",
        "spider_mites two-spotted_spider_mite": "Ácaros / Arañuela roja",
        "spider mites two-spotted spider mite": "Ácaros / Arañuela roja",
        "target_spot": "Mancha diana",
        "target spot": "Mancha diana",
        "tomato_yellow_leaf_curl_virus": "Virus del enrollamiento amarillo de la hoja (TYLCV)",
        "tomato yellow leaf curl virus": "Virus del enrollamiento amarillo de la hoja (TYLCV)",
        "tomato_mosaic_virus": "Virus del mosaico del tomate (TMV)",
        "tomato mosaic virus": "Virus del mosaico del tomate (TMV)",
        "healthy": "Saludable",
        "apple_scab": "Sarna del manzano",
        "black_rot": "Podredumbre negra",
        "cedar_apple_rust": "Roya del cedro y manzano",
        "powdery_mildew": "Oídio (Cenicilla)",
        "cercospora_leaf_spot gray_leaf_spot": "Mancha foliar gris por Cercospora",
        "common_rust_": "Roya común",
        "northern_leaf_blight": "Tizón foliar norteño",
        "esca_(black_measles)": "Esca (Sarampión negro)",
        "leaf_blight_(isariopsis_leaf_spot)": "Tizón foliar por Isariopsis",
        "haunglongbing_(citrus_greening)": "Huanglongbing / HLB (Enverdecimiento de los cítricos)",
        "leaf_scorch": "Quemadura de hoja"
    }

    parts = raw_class_name.split("___")
    raw_crop = parts[0].strip()
    raw_disease = parts[1].strip() if len(parts) > 1 else ""

    # Traducción del cultivo
    crop_clean = raw_crop.lower().replace(" ", "_")
    crop_display = crop_translations.get(crop_clean, crop_translations.get(raw_crop.lower(), raw_crop))

    # Traducción de la patología / estado
    disease_clean_under = raw_disease.lower().replace(" ", "_")
    disease_clean_space = raw_disease.lower().replace("_", " ")

    disease_display = disease_translations.get(
        disease_clean_space,
        disease_translations.get(
            disease_clean_under,
            raw_disease.replace("_", " ").title()
        )
    )

    return crop_display, disease_display

def predict_single_image(
    image_path="test-plant.jpg",
    model_path="models/plant_disease_mobilenetv2.keras",
    classes_path="models/class_names.json",
    top_k=5
):
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"No se encontró la imagen en: {img_file.resolve()}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado en: {model_path}. Ejecute scripts/train.py primero.")

    if not Path(classes_path).exists():
        raise FileNotFoundError(f"No se encontró el archivo de clases en: {classes_path}")

    # 1. Cargar clases
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # 2. Cargar modelo
    model = tf.keras.models.load_model(model_path)

    # 3. Cargar y preprocesar imagen
    img = Image.open(img_file).convert("RGB")
    img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # 4. Inferencia
    preds = model.predict(img_preprocessed, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1]

    best_idx = top_indices[0]
    best_raw_class = class_names[best_idx]
    best_conf = preds[best_idx] * 100
    crop_detected, condition_detected = parse_class_info(best_raw_class)

    # 5. Formatear y mostrar resultados
    print("\n" + "=" * 65)
    print("                    DIAGNÓSTICO DE LA PLANTA")
    print("=" * 65)
    print(f"  Archivo:  {img_file.name}")
    print(f"  Ruta:     {img_file.resolve()}\n")
    print(f"  🌱 Planta detectada:    {crop_detected}")
    print(f"  🔬 Estado / Patología:  {condition_detected}")
    print(f"  🎯 Nivel de Confianza:  {best_conf:.2f}%\n")
    print("  Distribución de probabilidades (Top):")
    print("  " + "-" * 61)

    for i in range(min(top_k, len(class_names))):
        idx = top_indices[i]
        raw_c = class_names[idx]
        crop, disease = parse_class_info(raw_c)
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
                "crop": parse_class_info(class_names[idx])[0],
                "condition": parse_class_info(class_names[idx])[1],
                "probability": float(preds[idx] * 100)
            }
            for idx in top_indices[:top_k]
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir estado/enfermedad de una planta")
    parser.add_argument("--image", type=str, default="test-plant.jpg", help="Ruta de la imagen de prueba")
    parser.add_argument("--model", type=str, default="models/plant_disease_mobilenetv2.keras", help="Ruta del modelo")
    parser.add_argument("--classes", type=str, default="models/class_names.json", help="Ruta de class_names.json")
    parser.add_argument("--top", type=int, default=5, help="Número de predicciones a mostrar")
    args = parser.parse_args()

    predict_single_image(
        image_path=args.image,
        model_path=args.model,
        classes_path=args.classes,
        top_k=args.top
    )
