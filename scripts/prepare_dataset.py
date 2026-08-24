import sys
import os
import shutil
import random
import argparse
from pathlib import Path

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def prepare_dataset(
    source_dir="PlantVillage-Dataset/raw/color",
    target_dir="dataset",
    crops=("Potato", "Tomato"),
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    test_image_dest="test-plant.jpg"
):
    print("=" * 60)
    print("  PREPARACION Y FILTRADO DEL DATASET PLANTVILLAGE")
    print("=" * 60)

    random.seed(seed)
    src_path = Path(source_dir)
    target_path = Path(target_dir)

    if not src_path.exists():
        raise FileNotFoundError(f"El directorio origen no existe: {src_path.resolve()}")

    # 1. Inspeccionar clases disponibles dinámicamente
    all_classes = sorted([d.name for d in src_path.iterdir() if d.is_dir()])
    print(f"\n[+] Total de clases encontradas en {source_dir}: {len(all_classes)}")

    # 2. Filtrar clases para los cultivos objetivo (Potato y Tomato)
    selected_classes = [
        cls for cls in all_classes
        if any(cls.lower().startswith(crop.lower() + "___") or cls.lower().startswith(crop.lower()) for crop in crops)
    ]

    print(f"\n[+] Clases seleccionadas para {crops} ({len(selected_classes)} clases):")
    for cls in selected_classes:
        n_imgs = len(list((src_path / cls).glob("*.*")))
        print(f"    * {cls} -> {n_imgs} imagenes")

    if not selected_classes:
        raise ValueError("No se encontraron clases coincidentes con los cultivos especificados.")

    # 3. Preparar directorios de salida
    splits = ["train", "validation", "test"]
    for s in splits:
        for cls in selected_classes:
            (target_path / s / cls).mkdir(parents=True, exist_ok=True)

    print("\n[+] Dividiendo y distribuyendo imagenes (70% train / 15% validation / 15% test)...")
    total_train, total_val, total_test = 0, 0, 0
    test_image_candidate = None

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

        if test_imgs and test_image_candidate is None:
            test_image_candidate = test_imgs[0]

        total_train += len(train_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)

    print("\n[OK] Division completada exitosamente:")
    print(f"    * Entrenamiento (train):     {total_train} imagenes")
    print(f"    * Validacion (validation):  {total_val} imagenes")
    print(f"    * Prueba (test):            {total_test} imagenes")
    print(f"    * Total procesado:          {total_train + total_val + total_test} imagenes")

    # Copiar una imagen de prueba a la raíz
    if test_image_candidate and test_image_dest:
        dest_path = Path(test_image_dest)
        shutil.copy2(test_image_candidate, dest_path)
        print(f"\n[OK] Imagen de prueba inicial copiada a: {dest_path.resolve()}")
        print(f"    Clase real de la imagen de prueba: {test_image_candidate.parent.name}")

    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparar dataset PlantVillage para Papa y Tomate")
    parser.add_argument("--source", type=str, default="PlantVillage-Dataset/raw/color", help="Ruta origen de PlantVillage")
    parser.add_argument("--target", type=str, default="dataset", help="Ruta destino para train/validation/test")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para reproducibilidad")
    args = parser.parse_args()

    prepare_dataset(
        source_dir=args.source,
        target_dir=args.target,
        crops=("Potato", "Tomato"),
        seed=args.seed
    )
