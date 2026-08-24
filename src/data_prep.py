import sys
import os
import shutil
import random
import json
import argparse
from pathlib import Path

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Diccionario completo de mapeo, traducciones y equivalencia con PlantDoc
CLASS_METADATA = {
    "Apple___Apple_scab": {
        "crop_es": "Manzana",
        "disease_es": "Sarna del manzano",
        "plantdoc_class": "Apple Scab Leaf"
    },
    "Apple___Black_rot": {
        "crop_es": "Manzana",
        "disease_es": "Podredumbre negra",
        "plantdoc_class": "grape leaf black rot" # referencia fungica
    },
    "Apple___Cedar_apple_rust": {
        "crop_es": "Manzana",
        "disease_es": "Roya del manzano",
        "plantdoc_class": "Apple rust leaf"
    },
    "Apple___healthy": {
        "crop_es": "Manzana",
        "disease_es": "Saludable",
        "plantdoc_class": "Apple leaf"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop_es": "Maíz",
        "disease_es": "Mancha foliar gris por Cercospora",
        "plantdoc_class": "Corn Gray leaf spot"
    },
    "Corn_(maize)___Common_rust_": {
        "crop_es": "Maíz",
        "disease_es": "Roya común",
        "plantdoc_class": "Corn rust leaf"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop_es": "Maíz",
        "disease_es": "Tizón foliar norteño",
        "plantdoc_class": "Corn leaf blight"
    },
    "Corn_(maize)___healthy": {
        "crop_es": "Maíz",
        "disease_es": "Saludable",
        "plantdoc_class": None
    },
    "Grape___Black_rot": {
        "crop_es": "Vid (Uva)",
        "disease_es": "Podredumbre negra",
        "plantdoc_class": "grape leaf black rot"
    },
    "Grape___Esca_(Black_Measles)": {
        "crop_es": "Vid (Uva)",
        "disease_es": "Esca (Sarampión negro)",
        "plantdoc_class": None
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop_es": "Vid (Uva)",
        "disease_es": "Tizón foliar por Isariopsis",
        "plantdoc_class": None
    },
    "Grape___healthy": {
        "crop_es": "Vid (Uva)",
        "disease_es": "Saludable",
        "plantdoc_class": "grape leaf"
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop_es": "Pimiento",
        "disease_es": "Mancha bacteriana",
        "plantdoc_class": "Bell_pepper leaf spot"
    },
    "Pepper,_bell___healthy": {
        "crop_es": "Pimiento",
        "disease_es": "Saludable",
        "plantdoc_class": "Bell_pepper leaf"
    },
    "Potato___Early_blight": {
        "crop_es": "Papa",
        "disease_es": "Tizón temprano",
        "plantdoc_class": "Potato leaf early blight"
    },
    "Potato___Late_blight": {
        "crop_es": "Papa",
        "disease_es": "Tizón tardío",
        "plantdoc_class": "Potato leaf late blight"
    },
    "Potato___healthy": {
        "crop_es": "Papa",
        "disease_es": "Saludable",
        "plantdoc_class": None
    },
    "Tomato___Bacterial_spot": {
        "crop_es": "Tomate",
        "disease_es": "Mancha bacteriana",
        "plantdoc_class": "Tomato leaf bacterial spot"
    },
    "Tomato___Early_blight": {
        "crop_es": "Tomate",
        "disease_es": "Tizón temprano",
        "plantdoc_class": "Tomato Early blight leaf"
    },
    "Tomato___Late_blight": {
        "crop_es": "Tomate",
        "disease_es": "Tizón tardío",
        "plantdoc_class": "Tomato leaf late blight"
    },
    "Tomato___Leaf_Mold": {
        "crop_es": "Tomate",
        "disease_es": "Moho foliar",
        "plantdoc_class": "Tomato mold leaf"
    },
    "Tomato___Septoria_leaf_spot": {
        "crop_es": "Tomate",
        "disease_es": "Mancha foliar por Septoria",
        "plantdoc_class": "Tomato Septoria leaf spot"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop_es": "Tomate",
        "disease_es": "Ácaros / Arañuela roja",
        "plantdoc_class": None
    },
    "Tomato___Target_Spot": {
        "crop_es": "Tomate",
        "disease_es": "Mancha diana",
        "plantdoc_class": None
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop_es": "Tomate",
        "disease_es": "Virus del enrollamiento amarillo de la hoja (TYLCV)",
        "plantdoc_class": "Tomato leaf yellow virus"
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop_es": "Tomate",
        "disease_es": "Virus del mosaico del tomate (TMV)",
        "plantdoc_class": "Tomato leaf mosaic virus"
    },
    "Tomato___healthy": {
        "crop_es": "Tomate",
        "disease_es": "Saludable",
        "plantdoc_class": "Tomato leaf"
    }
}

def prepare_multicrop_dataset(
    source_dir="PlantVillage-Dataset/raw/color",
    target_dir="datasets/plantvillage_mvp2",
    models_dir="models",
    target_crops=("Apple", "Corn", "Grape", "Pepper", "Potato", "Tomato"),
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    print("=" * 70)
    print("  AGROVISION MVP 2 — PREPARACION Y MAPEO MULTICULTIVO (6 CULTIVOS)")
    print("=" * 70)

    random.seed(seed)
    src_path = Path(source_dir)
    target_path = Path(target_dir)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Directorio origen no encontrado: {src_path.resolve()}")

    # 1. Identificar clases correspondientes a los 6 cultivos objetivo
    all_dirs = sorted([d.name for d in src_path.iterdir() if d.is_dir()])
    selected_classes = [
        cls for cls in all_dirs
        if any(cls.lower().startswith(crop.lower()) for crop in target_crops)
    ]

    print(f"\n[+] Clases seleccionadas para los 6 cultivos ({len(selected_classes)} clases):")
    for cls in selected_classes:
        meta = CLASS_METADATA.get(cls, {})
        crop_name = meta.get("crop_es", "N/A")
        disease_name = meta.get("disease_es", "N/A")
        n_imgs = len(list((src_path / cls).glob("*.*")))
        print(f"    * [{crop_name}] {disease_name:<44} -> {n_imgs:4d} imagenes")

    # 2. Guardar mapeo oficial en models/class_mapping_mvp2.json
    mapping_payload = {
        "total_classes": len(selected_classes),
        "classes": selected_classes,
        "metadata": {cls: CLASS_METADATA.get(cls, {}) for cls in selected_classes}
    }
    mapping_file = models_path / "class_mapping_mvp2.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=4, ensure_ascii=False)
    print(f"\n[OK] Mapeo de clases y equivalencias guardado en: {mapping_file.resolve()}")

    # 3. Preparar directorios de salida
    splits = ["train", "validation", "test"]
    for s in splits:
        for cls in selected_classes:
            (target_path / s / cls).mkdir(parents=True, exist_ok=True)

    print("\n[+] Dividiendo dataset estratificado (70% train / 15% validation / 15% test)...")
    total_train, total_val, total_test = 0, 0, 0

    for cls in selected_classes:
        class_dir = src_path / cls
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img in train_imgs:
            dest = target_path / "train" / cls / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        for img in val_imgs:
            dest = target_path / "validation" / cls / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        for img in test_imgs:
            dest = target_path / "test" / cls / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

        total_train += len(train_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)

    print("\n[OK] Particion estratificada completada exitosamente:")
    print(f"    * Entrenamiento (train):     {total_train:5d} imagenes")
    print(f"    * Validacion (validation):  {total_val:5d} imagenes")
    print(f"    * Prueba (test):            {total_test:5d} imagenes")
    print(f"    * Total procesado:          {total_train + total_val + total_test:5d} imagenes")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparar dataset AgroVision MVP 2 (6 Cultivos)")
    parser.add_argument("--source", type=str, default="PlantVillage-Dataset/raw/color", help="Directorio origen")
    parser.add_argument("--target", type=str, default="datasets/plantvillage_mvp2", help="Directorio destino")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de reproducibilidad")
    args = parser.parse_args()

    prepare_multicrop_dataset(
        source_dir=args.source,
        target_dir=args.target,
        seed=args.seed
    )
