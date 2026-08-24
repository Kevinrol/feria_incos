import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate_model_on_datasets(model_path, classes_path, mapping_path, plantvillage_test_dir, plantdoc_test_dir):
    # Cargar metadatos
    with open(classes_path, "r", encoding="utf-8") as f:
        model_classes = json.load(f)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
        meta = mapping_data.get("metadata", {})

    model = tf.keras.models.load_model(model_path)

    # 1. Evaluar en PlantVillage Test
    print(f"  [+] Evaluando en PlantVillage Test ({plantvillage_test_dir})...")
    pv_ds = tf.keras.utils.image_dataset_from_directory(
        plantvillage_test_dir,
        image_size=(224, 224),
        batch_size=32,
        label_mode="int",
        shuffle=False
    )
    pv_ds_prep = pv_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    pv_results = model.evaluate(pv_ds_prep, verbose=0)
    pv_loss, pv_acc = float(pv_results[0]), float(pv_results[1] * 100.0)

    # 2. Evaluar en PlantDoc Benchmark (20 clases)
    print(f"  [+] Evaluando en PlantDoc Benchmark ({plantdoc_test_dir})...")
    plantdoc_path = Path(plantdoc_test_dir)
    
    mapped_classes = []
    for idx, cls_name in enumerate(model_classes):
        m = meta.get(cls_name, {})
        pd_class = m.get("plantdoc_class")
        if pd_class and (plantdoc_path / pd_class).exists():
            crop_es = m.get("crop_es", cls_name.split("___")[0])
            dis_es = m.get("disease_es", cls_name.split("___")[1])
            mapped_classes.append((pd_class, idx, f"{crop_es}: {dis_es}", crop_es))

    y_true_indices, y_pred_indices, y_pred_top3 = [], [], []
    y_true_crops, y_pred_crops = [], []
    total_pd = 0

    for pd_cls, true_idx, disp_label, true_crop in mapped_classes:
        folder = plantdoc_path / pd_cls
        img_files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

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
                total_pd += 1
            except Exception:
                pass

    y_true_arr = np.array(y_true_indices)
    y_pred_arr = np.array(y_pred_indices)

    pd_top1 = float(np.mean(y_true_arr == y_pred_arr) * 100.0)
    pd_top3 = float(np.mean([t in top3 for t, top3 in zip(y_true_indices, y_pred_top3)]) * 100.0)
    pd_crop_acc = float(np.mean([tc == pc for tc, pc in zip(y_true_crops, y_pred_crops)]) * 100.0)

    # Desglose por cultivo
    crops_list = sorted(list(set(y_true_crops)))
    crop_stats = {}
    for c in crops_list:
        indices = [i for i, tc in enumerate(y_true_crops) if tc == c]
        c_total = len(indices)
        c_exact = sum(1 for i in indices if y_true_indices[i] == y_pred_indices[i])
        c_crop = sum(1 for i in indices if y_true_crops[i] == y_pred_crops[i])
        crop_stats[c] = {
            "exact_accuracy": float(c_exact / c_total * 100.0),
            "crop_accuracy": float(c_crop / c_total * 100.0),
            "samples": c_total
        }

    return {
        "plantvillage_top1": pv_acc,
        "plantvillage_loss": pv_loss,
        "plantdoc_top1": pd_top1,
        "plantdoc_top3": pd_top3,
        "plantdoc_crop_acc": pd_crop_acc,
        "plantdoc_samples": total_pd,
        "domain_shift_gap": float(pv_acc - pd_top1),
        "per_crop": crop_stats
    }

def run_comparison(
    model_a_path="models/agrovision_mobilenetv2_mvp2.keras",
    model_b_path="models/agrovision_mobilenetv2_mvp4.keras",
    classes_path="models/class_names_mvp2.json",
    mapping_path="models/class_mapping_mvp2.json",
    plantvillage_dir="datasets/plantvillage_mvp2/test",
    plantdoc_dir="PlantDoc-Dataset/test",
    results_dir="results"
):
    print("=" * 80)
    print("  AGROVISION — ESTUDIO COMPARATIVO EXPERIMENTAL: MODELO A (MVP 2) vs MODELO B (MVP 4)")
    print("=" * 80)

    os.makedirs(results_dir, exist_ok=True)

    print("\n[+] Evaluando Modelo A (MVP 2 - Baseline)...")
    res_a = evaluate_model_on_datasets(model_a_path, classes_path, mapping_path, plantvillage_dir, plantdoc_dir)

    print("\n[+] Evaluando Modelo B (MVP 4 - Augmentation Fuerte + Fine-Tuning)...")
    res_b = evaluate_model_on_datasets(model_b_path, classes_path, mapping_path, plantvillage_dir, plantdoc_dir)

    # 3. Generar Gráfica Comparativa de Barras
    metrics = ["PlantVillage Top-1", "PlantDoc Top-1", "PlantDoc Top-3", "Identif. Cultivo"]
    vals_a = [res_a["plantvillage_top1"], res_a["plantdoc_top1"], res_a["plantdoc_top3"], res_a["plantdoc_crop_acc"]]
    vals_b = [res_b["plantvillage_top1"], res_b["plantdoc_top1"], res_b["plantdoc_top3"], res_b["plantdoc_crop_acc"]]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(11, 6))
    rects1 = plt.bar(x - width/2, vals_a, width, label='Modelo A (MVP 2 - Baseline)', color='#4dabf7', edgecolor='black', lw=1.2)
    rects2 = plt.bar(x + width/2, vals_b, width, label='Modelo B (MVP 4 - Aug Fuerte)', color='#40c057', edgecolor='black', lw=1.2)

    plt.ylabel('Exactitud (%)', fontsize=12, fontweight='bold')
    plt.title('AgroVision: Comparativa Experimental de Mitigación del Domain Shift', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x, metrics, fontsize=11, fontweight='bold')
    plt.legend(fontsize=11)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()

    bars_save_path = os.path.join(results_dir, "model_comparison_bars.png")
    plt.savefig(bars_save_path, dpi=300)
    plt.close()
    print(f"\n[OK] Gráfica de barras comparativa guardada en: {bars_save_path}")

    # 4. Guardar JSON Comparativo
    delta_report = {
        "model_a_baseline": res_a,
        "model_b_mvp4": res_b,
        "delta": {
            "plantvillage_top1_delta": float(res_b["plantvillage_top1"] - res_a["plantvillage_top1"]),
            "plantdoc_top1_delta": float(res_b["plantdoc_top1"] - res_a["plantdoc_top1"]),
            "plantdoc_top3_delta": float(res_b["plantdoc_top3"] - res_a["plantdoc_top3"]),
            "plantdoc_crop_acc_delta": float(res_b["plantdoc_crop_acc"] - res_a["plantdoc_crop_acc"]),
            "domain_shift_reduction": float(res_a["domain_shift_gap"] - res_b["domain_shift_gap"])
        }
    }

    json_save_path = os.path.join(results_dir, "domain_shift_mitigation_comparison.json")
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(delta_report, f, indent=4, ensure_ascii=False)
    print(f"[OK] Reporte comparativo JSON guardado en: {json_save_path}")

    # 5. Imprimir tabla en consola
    print("\n" + "=" * 80)
    print("                    TABLA COMPARATIVA DE RESULTADOS EXPERIMENTALES")
    print("=" * 80)
    print(f" {'Métrica':<32} | {'Modelo A (MVP 2)':<16} | {'Modelo B (MVP 4)':<16} | {'Delta (Δ)':<10}")
    print(" " + "-" * 78)
    print(f" {'PlantVillage Test Top-1':<32} | {res_a['plantvillage_top1']:14.2f}% | {res_b['plantvillage_top1']:14.2f}% | {delta_report['delta']['plantvillage_top1_delta']:+6.2f}%")
    print(f" {'PlantDoc Benchmark Top-1':<32} | {res_a['plantdoc_top1']:14.2f}% | {res_b['plantdoc_top1']:14.2f}% | {delta_report['delta']['plantdoc_top1_delta']:+6.2f}%")
    print(f" {'PlantDoc Benchmark Top-3':<32} | {res_a['plantdoc_top3']:14.2f}% | {res_b['plantdoc_top3']:14.2f}% | {delta_report['delta']['plantdoc_top3_delta']:+6.2f}%")
    print(f" {'Identificación de Cultivo (Planta)':<32} | {res_a['plantdoc_crop_acc']:14.2f}% | {res_b['plantdoc_crop_acc']:14.2f}% | {delta_report['delta']['plantdoc_crop_acc_delta']:+6.2f}%")
    print(" " + "-" * 78)
    print(f" {'Brecha de Dominio (Domain Shift)':<32} | {res_a['domain_shift_gap']:14.2f} pp | {res_b['domain_shift_gap']:14.2f} pp | {delta_report['delta']['domain_shift_reduction']:+6.2f} pp")
    print("=" * 80 + "\n")

    return delta_report

if __name__ == "__main__":
    run_comparison()
