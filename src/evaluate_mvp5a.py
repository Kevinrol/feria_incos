import sys
import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def evaluate_mvp5a(
    model_path="models/agrovision_mobilenetv2_mvp4.keras",
    classes_path="models/class_names_mvp2.json",
    mapping_path="models/class_mapping_mvp2.json",
    crops_meta_path="results/mvp5_crops_metadata.json",
    results_dir="results"
):
    print("=" * 80)
    print("  AGROVISION MVP 5-A — EVALUACION COMPARATIVA ZERO-RETRAIN (RAW vs CROP)")
    print("=" * 80)

    results_path = Path(results_dir)

    with open(classes_path, "r", encoding="utf-8") as f:
        model_classes = json.load(f)
    with open(mapping_path, "r", encoding="utf-8") as f:
        meta = json.load(f).get("metadata", {})
    with open(crops_meta_path, "r", encoding="utf-8") as f:
        crops_meta = json.load(f)

    # Crear mapeo inverso de clase PlantDoc a índice de modelo
    pd_to_idx = {}
    for idx, cls_name in enumerate(model_classes):
        m = meta.get(cls_name, {})
        pd_cls = m.get("plantdoc_class")
        if pd_cls:
            pd_to_idx[pd_cls] = idx

    print(f"\n[+] Cargando Modelo MVP 4 (ZERO-RETRAIN) desde '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    total_images = len(crops_meta)
    raw_top1_correct = 0
    raw_top3_correct = 0
    raw_crop_correct = 0

    crop_top1_correct = 0
    crop_top3_correct = 0
    crop_crop_correct = 0

    # Solanáceas
    raw_p2t = 0
    raw_t2p = 0
    crop_p2t = 0
    crop_t2p = 0

    # Desglose por cultivo
    crops_all = ["Manzana", "Maíz", "Papa", "Pimiento", "Tomate", "Vid (Uva)"]
    per_crop_raw = {c: {"total": 0, "top1": 0, "crop_acc": 0} for c in crops_all}
    per_crop_crop = {c: {"total": 0, "top1": 0, "crop_acc": 0} for c in crops_all}

    # Cohorte sólo válidos
    valid_count = 0
    valid_raw_top1 = 0
    valid_crop_top1 = 0

    evaluation_records = []

    print(f"[+] Ejecutando inferencia sobre {total_images} pares de imágenes (Raw vs Crop)...")

    for item in crops_meta:
        true_crop = item["crop"]
        ground_truth = item["ground_truth"]
        status = item["localization_status"]
        orig_p = Path(item["original_path"])
        crop_p = ROOT_DIR / item["crop_path"]

        # Obtener true_idx
        pd_folder = orig_p.parent.name
        true_idx = pd_to_idx.get(pd_folder)

        # 1. Inferencia Raw
        img_raw = Image.open(orig_p).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        raw_arr = preprocess_input(np.expand_dims(np.array(img_raw, dtype=np.float32), axis=0))
        raw_preds = model.predict(raw_arr, verbose=0)[0]
        raw_top_indices = np.argsort(raw_preds)[::-1]
        raw_pred_idx = raw_top_indices[0]
        raw_pred_crop = meta[model_classes[raw_pred_idx]]["crop_es"]
        raw_pred_dis = meta[model_classes[raw_pred_idx]]["disease_es"]
        raw_conf = float(raw_preds[raw_pred_idx] * 100)

        # 2. Inferencia Crop
        img_crop = Image.open(crop_p).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        crop_arr = preprocess_input(np.expand_dims(np.array(img_crop, dtype=np.float32), axis=0))
        crop_preds = model.predict(crop_arr, verbose=0)[0]
        crop_top_indices = np.argsort(crop_preds)[::-1]
        crop_pred_idx = crop_top_indices[0]
        crop_pred_crop = meta[model_classes[crop_pred_idx]]["crop_es"]
        crop_pred_dis = meta[model_classes[crop_pred_idx]]["disease_es"]
        crop_conf = float(crop_preds[crop_pred_idx] * 100)

        # Evaluar Raw
        raw_is_top1 = bool(true_idx == raw_pred_idx)
        raw_is_top3 = bool(true_idx in raw_top_indices[:3])
        raw_is_crop = bool(true_crop == raw_pred_crop)

        if raw_is_top1: raw_top1_correct += 1
        if raw_is_top3: raw_top3_correct += 1
        if raw_is_crop: raw_crop_correct += 1

        # Evaluar Crop
        crop_is_top1 = bool(true_idx == crop_pred_idx)
        crop_is_top3 = bool(true_idx in crop_top_indices[:3])
        crop_is_crop = bool(true_crop == crop_pred_crop)

        if crop_is_top1: crop_top1_correct += 1
        if crop_is_top3: crop_top3_correct += 1
        if crop_is_crop: crop_crop_correct += 1

        # Tracking por cultivo
        per_crop_raw[true_crop]["total"] += 1
        if raw_is_top1: per_crop_raw[true_crop]["top1"] += 1
        if raw_is_crop: per_crop_raw[true_crop]["crop_acc"] += 1

        per_crop_crop[true_crop]["total"] += 1
        if crop_is_top1: per_crop_crop[true_crop]["top1"] += 1
        if crop_is_crop: per_crop_crop[true_crop]["crop_acc"] += 1

        # Solanáceas
        if true_crop == "Papa" and raw_pred_crop == "Tomate": raw_p2t += 1
        if true_crop == "Tomate" and raw_pred_crop == "Papa": raw_t2p += 1
        if true_crop == "Papa" and crop_pred_crop == "Tomate": crop_p2t += 1
        if true_crop == "Tomate" and crop_pred_crop == "Papa": crop_t2p += 1

        # Cohorte Válidos
        if status == "VALID_CROP":
            valid_count += 1
            if raw_is_top1: valid_raw_top1 += 1
            if crop_is_top1: valid_crop_top1 += 1

        # Categoría diferencial
        if raw_is_top1 and crop_is_top1:
            diff_cat = "ROBUST"
        elif (not raw_is_top1) and crop_is_top1:
            diff_cat = "FIXED"
        elif (not raw_is_top1) and (not crop_is_top1):
            diff_cat = "PERSISTENT"
        else:
            diff_cat = "REGRESSION"

        evaluation_records.append({
            "filename": item["filename"],
            "crop": true_crop,
            "ground_truth": ground_truth,
            "localization_status": status,
            "bbox_area_ratio": item["bbox_area_ratio"],
            "raw": {
                "predicted": f"{raw_pred_crop}: {raw_pred_dis}",
                "confidence": raw_conf,
                "top1_correct": raw_is_top1,
                "top3_correct": raw_is_top3,
                "crop_correct": raw_is_crop
            },
            "crop": {
                "predicted": f"{crop_pred_crop}: {crop_pred_dis}",
                "confidence": crop_conf,
                "top1_correct": crop_is_top1,
                "top3_correct": crop_is_top3,
                "crop_correct": crop_is_crop
            },
            "category": diff_cat
        })

    # Métricas Globales
    raw_top1_pct = (raw_top1_correct / total_images) * 100
    raw_top3_pct = (raw_top3_correct / total_images) * 100
    raw_crop_pct = (raw_crop_correct / total_images) * 100

    crop_top1_pct = (crop_top1_correct / total_images) * 100
    crop_top3_pct = (crop_top3_correct / total_images) * 100
    crop_crop_pct = (crop_crop_correct / total_images) * 100

    # Guardar JSON de Resultados
    comparison_summary = {
        "total_images": total_images,
        "valid_crops_count": valid_count,
        "localization_success_rate": (valid_count / total_images) * 100,
        "global_intent_to_diagnose": {
            "raw_top1": raw_top1_pct,
            "crop_top1": crop_top1_pct,
            "delta_top1": crop_top1_pct - raw_top1_pct,
            "raw_top3": raw_top3_pct,
            "crop_top3": crop_top3_pct,
            "delta_top3": crop_top3_pct - raw_top3_pct,
            "raw_crop_acc": raw_crop_pct,
            "crop_crop_acc": crop_crop_pct,
            "delta_crop_acc": crop_crop_pct - raw_crop_pct
        },
        "per_protocol_valid_crops_cohort": {
            "valid_samples": valid_count,
            "raw_top1_in_valid": (valid_raw_top1 / valid_count) * 100 if valid_count > 0 else 0,
            "crop_top1_in_valid": (valid_crop_top1 / valid_count) * 100 if valid_count > 0 else 0,
            "delta_valid_top1": ((valid_crop_top1 - valid_raw_top1) / valid_count) * 100 if valid_count > 0 else 0
        },
        "solanaceae_tracking": {
            "papa_to_tomate_raw": raw_p2t,
            "papa_to_tomate_crop": crop_p2t,
            "papa_to_tomate_delta": crop_p2t - raw_p2t,
            "tomate_to_papa_raw": raw_t2p,
            "tomate_to_papa_crop": crop_t2p,
            "tomate_to_papa_delta": crop_t2p - raw_t2p
        },
        "per_crop_breakdown": {
            c: {
                "samples": per_crop_raw[c]["total"],
                "raw_top1": (per_crop_raw[c]["top1"] / per_crop_raw[c]["total"]) * 100,
                "crop_top1": (per_crop_crop[c]["top1"] / per_crop_crop[c]["total"]) * 100,
                "delta_top1": ((per_crop_crop[c]["top1"] - per_crop_raw[c]["top1"]) / per_crop_raw[c]["total"]) * 100,
                "raw_crop_acc": (per_crop_raw[c]["crop_acc"] / per_crop_raw[c]["total"]) * 100,
                "crop_crop_acc": (per_crop_crop[c]["crop_acc"] / per_crop_crop[c]["total"]) * 100,
                "delta_crop_acc": ((per_crop_crop[c]["crop_acc"] - per_crop_raw[c]["crop_acc"]) / per_crop_raw[c]["total"]) * 100
            } for c in crops_all
        },
        "detailed_records": evaluation_records
    }

    comp_json_path = results_path / "mvp5a_comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=4, ensure_ascii=False)
    print(f"\n[OK] Datos comparativos guardados en: {comp_json_path}")

    # Generar Gráfico de Barras Comparativo
    plot_comparison(comparison_summary, results_path / "mvp5a_comparison.png")

    # Imprimir Tabla Resumen en Consola
    print("\n" + "=" * 85)
    print("                 AGROVISION — MVP 5-A EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 85)
    print(f" Métrica Evaluada               | MVP 4 (Raw Image) | MVP 5-A (Leaf Crop) | Variación (Δ) ")
    print(" " + "-" * 83)
    print(f" PlantDoc Top-1 (Global)        |           {raw_top1_pct:6.2f}% |             {crop_top1_pct:6.2f}% |    {crop_top1_pct - raw_top1_pct:+6.2f} pp")
    print(f" PlantDoc Top-3 (Sugerencias)   |           {raw_top3_pct:6.2f}% |             {crop_top3_pct:6.2f}% |    {crop_top3_pct - raw_top3_pct:+6.2f} pp")
    print(f" Reconocimiento Botánico Planta |           {raw_crop_pct:6.2f}% |             {crop_crop_pct:6.2f}% |    {crop_crop_pct - raw_crop_pct:+6.2f} pp")
    print(" " + "-" * 83)
    print(f" Top-1 en Cohorte Válida ({valid_count}/{total_images})  |           {(valid_raw_top1/valid_count)*100:6.2f}% |             {(valid_crop_top1/valid_count)*100:6.2f}% |    {((valid_crop_top1 - valid_raw_top1)/valid_count)*100:+6.2f} pp")
    print(" " + "-" * 83)
    print(" DESGLOSE POR CULTIVO (Top-1 Patológico):")
    for c in crops_all:
        r_t1 = (per_crop_raw[c]["top1"] / per_crop_raw[c]["total"]) * 100
        c_t1 = (per_crop_crop[c]["top1"] / per_crop_crop[c]["total"]) * 100
        print(f"  • {c:12s} ({per_crop_raw[c]['total']:2d} fotos)    |           {r_t1:6.2f}% |             {c_t1:6.2f}% |    {c_t1 - r_t1:+6.2f} pp")
    print(" " + "-" * 83)
    print(" SOLANACEAS (Confusión Cruzada):")
    print(f"  Papa -> Tomate:   MVP 4 Raw = {raw_p2t:2d}  |  MVP 5-A Crop = {crop_p2t:2d}  |  Δ = {crop_p2t - raw_p2t:+d}")
    print(f"  Tomate -> Papa:   MVP 4 Raw = {raw_t2p:2d}  |  MVP 5-A Crop = {crop_t2p:2d}  |  Δ = {crop_t2p - raw_t2p:+d}")
    print("=" * 85 + "\n")

def plot_comparison(summary, save_path):
    metrics = ["Top-1 Global", "Top-3 Global", "Reconoc. Planta", "Top-1 Cohorte Válida"]
    g = summary["global_intent_to_diagnose"]
    p = summary["per_protocol_valid_crops_cohort"]

    raw_vals = [g["raw_top1"], g["raw_top3"], g["raw_crop_acc"], p["raw_top1_in_valid"]]
    crop_vals = [g["crop_top1"], g["crop_top3"], g["crop_crop_acc"], p["crop_top1_in_valid"]]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, raw_vals, width, label='MVP 4 (Foto Original)', color='#E07A5F')
    bars2 = plt.bar(x + width/2, crop_vals, width, label='MVP 5-A (Hoja Aislada)', color='#2A9D8F')

    plt.ylabel('Exactitud (%)', fontsize=12)
    plt.title('AgroVision MVP 5-A — Comparativa Zero-Retrain: Foto Original vs Hoja Aislada', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x, metrics, fontsize=11)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)

    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Gráfica comparativa guardada en: {save_path}")

if __name__ == "__main__":
    evaluate_mvp5a()
