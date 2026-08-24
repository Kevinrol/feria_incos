import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Asegurar que la raíz del proyecto esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def evaluate_plantdoc_benchmark(
    plantdoc_dir="PlantDoc-Dataset/test",
    model_path="models/agrovision_mobilenetv2_mvp2.keras",
    classes_path="models/class_names_mvp2.json",
    mapping_path="models/class_mapping_mvp2.json",
    results_dir="results"
):
    print("=" * 80)
    print("  AGROVISION MVP 3 — AUDITORIA EXTERNA Y BENCHMARK CON PLANTDOC (CAMPO REAL)")
    print("=" * 80)

    plantdoc_path = Path(plantdoc_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    if not plantdoc_path.exists():
        raise FileNotFoundError(f"Directorio PlantDoc no encontrado en: {plantdoc_path.resolve()}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modelo MVP 2 no encontrado en: {model_path}")
    if not Path(classes_path).exists() or not Path(mapping_path).exists():
        raise FileNotFoundError("Archivos de clases o mapeo no encontrados en models/")

    # 1. Cargar clases y mapeo
    with open(classes_path, "r", encoding="utf-8") as f:
        model_classes = json.load(f)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
        meta = mapping_data.get("metadata", {})

    # Mapear carpeta PlantDoc -> clase oficial del modelo e índice
    plantdoc_to_model_idx = {}
    plantdoc_to_display = {}
    mapped_classes = []

    for idx, cls_name in enumerate(model_classes):
        m = meta.get(cls_name, {})
        pd_class = m.get("plantdoc_class")
        if pd_class and (plantdoc_path / pd_class).exists():
            plantdoc_to_model_idx[pd_class] = idx
            crop_es = m.get("crop_es", cls_name.split("___")[0])
            dis_es = m.get("disease_es", cls_name.split("___")[1])
            disp_label = f"{crop_es}: {dis_es}"
            plantdoc_to_display[pd_class] = disp_label
            mapped_classes.append((pd_class, idx, disp_label))

    print(f"\n[+] Clases mapeadas y encontradas en PlantDoc: {len(mapped_classes)} clases:")
    for pd_cls, idx, disp in mapped_classes:
        n_imgs = len(list((plantdoc_path / pd_cls).glob("*.*")))
        print(f"    * [{idx:02d}] {disp:<44} (Carpeta: '{pd_cls}') -> {n_imgs:2d} imgs")

    # 2. Cargar modelo
    print(f"\n[+] Cargando modelo entrenado desde '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    # 3. Recopilar y evaluar todas las imágenes de PlantDoc
    print("\n[+] Procesando imágenes de campo real y ejecutando inferencia...")
    y_true_indices = []
    y_pred_indices = []
    y_pred_top3 = []
    y_true_crops = []
    y_pred_crops = []
    results_per_image = []

    total_evaluated = 0

    for pd_cls, true_idx, disp_label in mapped_classes:
        folder = plantdoc_path / pd_cls
        img_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        true_crop = meta[model_classes[true_idx]]["crop_es"]

        for img_p in img_files:
            try:
                img = Image.open(img_p).convert("RGB")
                img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
                img_arr = np.array(img_resized, dtype=np.float32)
                img_batch = np.expand_dims(img_arr, axis=0)
                img_prep = preprocess_input(img_batch)

                preds = model.predict(img_prep, verbose=0)[0]
                top_indices = np.argsort(preds)[::-1]
                pred_idx = top_indices[0]
                pred_crop = meta[model_classes[pred_idx]]["crop_es"]

                y_true_indices.append(true_idx)
                y_pred_indices.append(pred_idx)
                y_pred_top3.append(top_indices[:3])
                y_true_crops.append(true_crop)
                y_pred_crops.append(pred_crop)

                total_evaluated += 1
                results_per_image.append({
                    "file": img_p.name,
                    "folder": pd_cls,
                    "true_class": model_classes[true_idx],
                    "predicted_class": model_classes[pred_idx],
                    "true_label": disp_label,
                    "predicted_label": f"{pred_crop}: {meta[model_classes[pred_idx]]['disease_es']}",
                    "confidence": float(preds[pred_idx] * 100),
                    "is_exact_match": bool(true_idx == pred_idx),
                    "is_crop_match": bool(true_crop == pred_crop),
                    "in_top3": bool(true_idx in top_indices[:3])
                })
            except Exception as e:
                print(f"[!] Error procesando {img_p.name}: {e}")

    # 4. Cálculo de Métricas Globales
    y_true_arr = np.array(y_true_indices)
    y_pred_arr = np.array(y_pred_indices)
    exact_matches = np.sum(y_true_arr == y_pred_arr)
    top1_accuracy = (exact_matches / total_evaluated) * 100.0

    top3_matches = sum(1 for true_i, top3 in zip(y_true_indices, y_pred_top3) if true_i in top3)
    top3_accuracy = (top3_matches / total_evaluated) * 100.0

    crop_matches = sum(1 for tc, pc in zip(y_true_crops, y_pred_crops) if tc == pc)
    crop_level_accuracy = (crop_matches / total_evaluated) * 100.0

    # Métricas por cultivo
    crops_list = sorted(list(set(y_true_crops)))
    crop_stats = {}
    for c in crops_list:
        indices = [i for i, tc in enumerate(y_true_crops) if tc == c]
        c_total = len(indices)
        c_exact = sum(1 for i in indices if y_true_indices[i] == y_pred_indices[i])
        c_crop_acc = sum(1 for i in indices if y_true_crops[i] == y_pred_crops[i])
        crop_stats[c] = {
            "total_images": c_total,
            "exact_disease_accuracy": float(c_exact / c_total * 100.0),
            "crop_identification_accuracy": float(c_crop_acc / c_total * 100.0)
        }

    # 5. Generar Matriz de Confusión Externa
    unique_classes_in_eval = sorted(list(set(y_true_indices)))
    eval_display_labels = [
        f"{meta[model_classes[idx]]['crop_es']}: {meta[model_classes[idx]]['disease_es']}"
        for idx in unique_classes_in_eval
    ]

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=unique_classes_in_eval)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(16, 13))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Greens',
        xticklabels=eval_display_labels,
        yticklabels=eval_display_labels,
        cbar_kws={'label': 'Proporción'},
        annot_kws={"size": 8}
    )
    plt.title(f'AgroVision MVP 3: Benchmark Externo en Campo Real (PlantDoc)\nTop-1 Accuracy: {top1_accuracy:.2f}% | Top-3 Accuracy: {top3_accuracy:.2f}% | Identificación de Cultivo: {crop_level_accuracy:.2f}%', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Clase Predicha por el Modelo', fontsize=11, fontweight='bold')
    plt.ylabel('Clase Real (PlantDoc)', fontsize=11, fontweight='bold')
    plt.xticks(rotation=60, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_plantdoc_path = os.path.join(results_dir, "confusion_matrix_plantdoc.png")
    plt.savefig(cm_plantdoc_path, dpi=300)
    plt.close()
    print(f"\n[OK] Matriz de confusión externa guardada en: {cm_plantdoc_path}")

    # 6. Reporte Comparativo (PlantVillage vs PlantDoc)
    # Cargar métricas internas previas
    internal_metrics_path = os.path.join(results_dir, "test_metrics_mvp2.json")
    internal_acc = 91.28
    if os.path.exists(internal_metrics_path):
        with open(internal_metrics_path, "r", encoding="utf-8") as f:
            internal_data = json.load(f)
            internal_acc = internal_data.get("test_accuracy", 0.9128) * 100.0

    domain_gap = internal_acc - top1_accuracy

    comparison_report = {
        "plantvillage_internal_test": {
            "environment": "Laboratorio (fondo controlado, luz uniforme)",
            "accuracy": float(internal_acc),
            "samples_evaluated": 5104
        },
        "plantdoc_external_benchmark": {
            "environment": "Campo real (fondos ruidosos, sombras, hojas superpuestas)",
            "top1_exact_accuracy": float(top1_accuracy),
            "top3_accuracy": float(top3_accuracy),
            "crop_identification_accuracy": float(crop_level_accuracy),
            "samples_evaluated": total_evaluated,
            "classes_evaluated": len(mapped_classes)
        },
        "domain_shift_analysis": {
            "generalization_gap_top1": float(domain_gap),
            "crop_recognition_gap": float(internal_acc - crop_level_accuracy)
        },
        "per_crop_breakdown": crop_stats
    }

    comparison_json_path = os.path.join(results_dir, "benchmark_comparison.json")
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=4, ensure_ascii=False)
    print(f"[OK] Reporte comparativo guardado en: {comparison_json_path}")

    # 7. Mostrar resultados por consola
    print("\n" + "=" * 80)
    print("                    RESULTADOS DE LA AUDITORIA EXTERNA (PLANTDOC)")
    print("=" * 80)
    print(f"  Total de imágenes de campo evaluadas:  {total_evaluated}")
    print(f"  🎯 Exactitud Exacta (Top-1 Match):       {top1_accuracy:.2f}%")
    print(f"  🎯 Exactitud en Top-3 Probabilidades:   {top3_accuracy:.2f}%")
    print(f"  🌱 Identificación Botánica del Cultivo:  {crop_level_accuracy:.2f}%\n")
    print("  Rendimiento Desglosado por Cultivo en Campo Real:")
    print("  " + "-" * 76)
    for crop, stats in crop_stats.items():
        print(f"   * {crop:<12} -> Exactitud Patológica: {stats['exact_disease_accuracy']:5.1f}% | Acierto de Planta: {stats['crop_identification_accuracy']:5.1f}% ({stats['total_images']:2d} fotos)")
    print("  " + "-" * 76)
    print(f"\n  🔬 BRECHA DE GENERALIZACION (Domain Shift):")
    print(f"     - PlantVillage Test Set (Laboratorio):   {internal_acc:.2f}%")
    print(f"     - PlantDoc Benchmark (Campo Real):       {top1_accuracy:.2f}% (Top-1) | {top3_accuracy:.2f}% (Top-3)")
    print(f"     - Brecha de Dominio (*Generalization Gap*): {domain_gap:+.2f}%")
    print("=" * 80 + "\n")

    return comparison_report

if __name__ == "__main__":
    evaluate_plantdoc_benchmark()
