import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
from src.train_kallpa import load_raw_image_and_label
from src.focal_loss import SolanaceaeFocalLoss

# Custom objects required for loading the model
CUSTOM_OBJECTS = {
    "SolanaceaeFocalLoss": SolanaceaeFocalLoss
}

def expected_calibration_error(y_true, y_pred, num_bins=10):
    """
    Calcula el Expected Calibration Error (ECE).
    y_true: vector 1D con las clases verdaderas.
    y_pred: matriz 2D con las probabilidades predichas (shape: N, num_classes).
    """
    confidences = np.max(y_pred, axis=1)
    predictions = np.argmax(y_pred, axis=1)
    accuracies = (predictions == y_true)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prob_in_bin = np.mean(in_bin)
        
        if prob_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
            
    return ece * 100.0  # Percentage

def build_test_dataset(manifest_path, batch_size=32):
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    paths = [r["path"] for r in records]
    labels = [r["class_idx"] for r in records]
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, labels

def evaluate_kallpa(
    model_path="models/LeafNet_kallpa.keras",
    manifest_dir="datasets/leafnet_kallpa",
    plantdoc_dir="PlantDoc-Dataset/test",
    batch_size=32
):
    print("=" * 80)
    print("  AGROVISION LeafNet-kallpa (MVP 7) — EVALUACIÓN MULTIDIMENSIONAL")
    print("=" * 80)
    
    m_dir = ROOT_DIR / manifest_dir
    model_file = ROOT_DIR / model_path
    
    if not model_file.exists():
        print(f"[!] Error: No se encontró el modelo {model_file}")
        return
        
    print(f"[+] Cargando modelo '{model_path}'...")
    model = tf.keras.models.load_model(model_file, compile=False)
    
    # 1. Evaluar PlantVillage Val (Control Lab)
    print("\n[+] Evaluando PlantVillage Val (Control Laboratorio)...")
    val_manifest = m_dir / "val_manifest.json"
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_records = json.load(f)
        
    pv_val_paths = [r["path"] for r in val_records if r["domain"] == "PlantVillage"]
    pv_val_labels = [r["class_idx"] for r in val_records if r["domain"] == "PlantVillage"]
    
    pv_ds = tf.data.Dataset.from_tensor_slices((pv_val_paths, pv_val_labels))
    pv_ds = pv_ds.map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    pv_preds = model.predict(pv_ds, verbose=1)
    pv_acc = np.mean(np.argmax(pv_preds, axis=1) == pv_val_labels) * 100
    
    # 2. Evaluar PlantWild Test
    print("\n[+] Evaluando PlantWild Test (Campo Real)...")
    pw_test_manifest = m_dir / "plantwild_test_manifest.json"
    pw_ds, pw_labels = build_test_dataset(pw_test_manifest, batch_size)
    pw_preds = model.predict(pw_ds, verbose=1)
    
    pw_top1 = np.mean(np.argmax(pw_preds, axis=1) == pw_labels) * 100
    pw_top3 = tf.keras.metrics.sparse_top_k_categorical_accuracy(pw_labels, pw_preds, k=3).numpy().mean() * 100
    pw_ece = expected_calibration_error(pw_labels, pw_preds)
    
    # 3. Evaluar PlantDoc (Benchmark Ciego)
    print("\n[+] Evaluando PlantDoc (Benchmark Ciego Externo)...")
    classes_path = ROOT_DIR / "models" / "class_names_mvp2.json"
    mapping_path = ROOT_DIR / "models" / "class_mapping_mvp2.json"
    with open(classes_path, "r", encoding="utf-8") as f:
        agro_classes = json.load(f)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)["metadata"]
        
    plantdoc_paths = []
    plantdoc_labels = []
    
    pd_dir = ROOT_DIR / plantdoc_dir
    if pd_dir.exists():
        for cls_folder in pd_dir.iterdir():
            if cls_folder.is_dir():
                pd_cls_name = cls_folder.name
                matched_idx = None
                for idx, ac in enumerate(agro_classes):
                    if mapping[ac].get("plantdoc_class") == pd_cls_name:
                        matched_idx = idx
                        break
                
                if matched_idx is not None:
                    for img_f in cls_folder.iterdir():
                        if img_f.is_file() and img_f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            plantdoc_paths.append(str(img_f))
                            plantdoc_labels.append(matched_idx)
                            
    if len(plantdoc_paths) > 0:
        pd_ds = tf.data.Dataset.from_tensor_slices((plantdoc_paths, plantdoc_labels))
        pd_ds = pd_ds.map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        pd_preds = model.predict(pd_ds, verbose=1)
        pd_top1 = np.mean(np.argmax(pd_preds, axis=1) == plantdoc_labels) * 100
        pd_top3 = tf.keras.metrics.sparse_top_k_categorical_accuracy(plantdoc_labels, pd_preds, k=3).numpy().mean() * 100
        
        potato_early = 14
        potato_late = 15
        tomato_early = 18
        tomato_late = 19
        
        pd_preds_argmax = np.argmax(pd_preds, axis=1)
        pd_labels = np.array(plantdoc_labels)
        
        is_potato_actual = (pd_labels == potato_early) | (pd_labels == potato_late)
        pred_tomato = (pd_preds_argmax == tomato_early) | (pd_preds_argmax == tomato_late)
        potato_to_tomato = np.sum(is_potato_actual & pred_tomato)
    else:
        pd_top1, pd_top3, potato_to_tomato = 0.0, 0.0, 0
    
    print("\n" + "=" * 60)
    print("  RESULTADOS OFICIALES LEAFNET-KALLPA (MVP 7)")
    print("=" * 60)
    print(f"PlantVillage Val Top-1:   {pv_acc:.2f}%")
    print(f"PlantWild Test Top-1:     {pw_top1:.2f}%")
    print(f"PlantWild Test Top-3:     {pw_top3:.2f}%")
    print(f"PlantWild Test ECE:       {pw_ece:.2f}% (Menor es mejor)")
    print(f"PlantDoc Top-1 (Ciego):   {pd_top1:.2f}%")
    print(f"PlantDoc Top-3 (Ciego):   {pd_top3:.2f}%")
    print(f"Domain Shift Gap (PV-PD): {pv_acc - pd_top1:.2f} pp")
    print(f"Confusión Papa -> Tomate: {potato_to_tomato} casos (PlantDoc)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    evaluate_kallpa()
