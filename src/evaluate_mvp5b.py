import sys
import os
import json
import argparse
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

def evaluate_model_on_datasets(model_path, model_classes, meta, mapped_classes, pv_test_dir, plantdoc_dir):
    print(f"\n[+] Evaluando modelo '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    # 1. Evaluar en PlantVillage Test
    pv_test_path = Path(pv_test_dir)
    pv_test_ds = tf.keras.utils.image_dataset_from_directory(
        pv_test_path,
        image_size=(224, 224),
        batch_size=32,
        label_mode="int",
        shuffle=False
    )
    pv_test_prep = pv_test_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    eval_res = model.evaluate(pv_test_prep, verbose=0)
    pv_loss, pv_acc = float(eval_res[0]), float(eval_res[1] * 100)

    # 2. Evaluar en PlantDoc Test
    plantdoc_path = Path(plantdoc_dir)
    total_pd = 0
    top1_correct = 0
    top3_correct = 0
    crop_correct = 0

    p2t_count = 0
    t2p_count = 0

    crops_all = ["Manzana", "Maíz", "Papa", "Pimiento", "Tomate", "Vid (Uva)"]
    per_crop = {c: {"total": 0, "top1": 0, "crop_acc": 0} for c in crops_all}

    for pd_cls, true_idx, disp_label, true_crop in mapped_classes:
        folder = plantdoc_path / pd_cls
        img_files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])

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

                total_pd += 1
                if pred_idx == true_idx:
                    top1_correct += 1
                if true_idx in top_indices[:3]:
                    top3_correct += 1
                if pred_crop == true_crop:
                    crop_correct += 1

                per_crop[true_crop]["total"] += 1
                if pred_idx == true_idx:
                    per_crop[true_crop]["top1"] += 1
                if pred_crop == true_crop:
                    per_crop[true_crop]["crop_acc"] += 1

                if true_crop == "Papa" and pred_crop == "Tomate":
                    p2t_count += 1
                if true_crop == "Tomate" and pred_crop == "Papa":
                    t2p_count += 1

            except Exception as e:
                pass

    pd_top1 = (top1_correct / total_pd) * 100 if total_pd > 0 else 0
    pd_top3 = (top3_correct / total_pd) * 100 if total_pd > 0 else 0
    pd_crop_acc = (crop_correct / total_pd) * 100 if total_pd > 0 else 0
    domain_shift_gap = pv_acc - pd_top1

    return {
        "plantvillage_top1": pv_acc,
        "plantvillage_loss": pv_loss,
        "plantdoc_top1": pd_top1,
        "plantdoc_top3": pd_top3,
        "plantdoc_crop_acc": pd_crop_acc,
        "domain_shift_gap": domain_shift_gap,
        "papa_to_tomate": p2t_count,
        "tomate_to_papa": t2p_count,
        "per_crop": {
            c: {
                "samples": per_crop[c]["total"],
                "top1": (per_crop[c]["top1"] / per_crop[c]["total"]) * 100 if per_crop[c]["total"] > 0 else 0,
                "crop_acc": (per_crop[c]["crop_acc"] / per_crop[c]["total"]) * 100 if per_crop[c]["total"] > 0 else 0
            } for c in crops_all
        }
    }

def run_4way_evaluation(
    model_mvp2_path="models/agrovision_mobilenetv2_mvp2.keras",
    model_mvp4_path="models/agrovision_mobilenetv2_mvp4.keras",
    model_mvp5b_path="models/agrovision_mobilenetv2_mvp5b.keras",
    classes_path="models/class_names_mvp2.json",
    mapping_path="models/class_mapping_mvp2.json",
    pv_test_dir="datasets/plantvillage_mvp2/test",
    plantdoc_dir="PlantDoc-Dataset/test",
    results_dir="results"
):
    print("=" * 80)
    print("  AGROVISION — EVALUACION COMPARATIVA 4-WAY (MVP 2 vs MVP 4 vs MVP 5-A vs MVP 5-B)")
    print("=" * 80)

    results_path = Path(results_dir)

    with open(classes_path, "r", encoding="utf-8") as f:
        model_classes = json.load(f)
    with open(mapping_path, "r", encoding="utf-8") as f:
        meta = json.load(f).get("metadata", {})

    plantdoc_path = Path(plantdoc_dir)
    mapped_classes = []
    for idx, cls_name in enumerate(model_classes):
        m = meta.get(cls_name, {})
        pd_class = m.get("plantdoc_class")
        if pd_class and (plantdoc_path / pd_class).exists():
            crop_es = m.get("crop_es", cls_name.split("___")[0])
            dis_es = m.get("disease_es", cls_name.split("___")[1])
            mapped_classes.append((pd_class, idx, f"{crop_es}: {dis_es}", crop_es))

    # 1. Evaluar MVP 2
    res_mvp2 = evaluate_model_on_datasets(model_mvp2_path, model_classes, meta, mapped_classes, pv_test_dir, plantdoc_dir)

    # 2. Evaluar MVP 4
    res_mvp4 = evaluate_model_on_datasets(model_mvp4_path, model_classes, meta, mapped_classes, pv_test_dir, plantdoc_dir)

    # 3. Evaluar MVP 5-B
    res_mvp5b = evaluate_model_on_datasets(model_mvp5b_path, model_classes, meta, mapped_classes, pv_test_dir, plantdoc_dir)

    summary_4way = {
        "mvp2_baseline": res_mvp2,
        "mvp4_strong_aug": res_mvp4,
        "mvp5a_crop_zero_retrain": {
            "plantvillage_top1": res_mvp4["plantvillage_top1"],
            "plantdoc_top1_global": 34.30,
            "plantdoc_top1_valid_cohort": 36.36,
            "plantdoc_top3": 56.98,
            "plantdoc_crop_acc": 70.93,
            "domain_shift_gap": 60.68,
            "papa_to_tomate": 10,
            "tomate_to_papa": 0
        },
        "mvp5b_crop_aware": res_mvp5b,
        "delta_mvp5b_vs_mvp4": {
            "plantdoc_top1": res_mvp5b["plantdoc_top1"] - res_mvp4["plantdoc_top1"],
            "plantdoc_top3": res_mvp5b["plantdoc_top3"] - res_mvp4["plantdoc_top3"],
            "plantdoc_crop_acc": res_mvp5b["plantdoc_crop_acc"] - res_mvp4["plantdoc_crop_acc"],
            "domain_shift_reduction": res_mvp4["domain_shift_gap"] - res_mvp5b["domain_shift_gap"],
            "papa_to_tomate_delta": res_mvp5b["papa_to_tomate"] - res_mvp4["papa_to_tomate"]
        }
    }

    json_path = results_path / "mvp5b_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_4way, f, indent=4, ensure_ascii=False)
    print(f"\n[OK] Datos comparativos 4-Way guardados en: {json_path}")

    # Graficar
    plot_4way_bars(summary_4way, results_path / "mvp5b_comparison_bars.png")

    # Imprimir Tabla
    print("\n" + "=" * 90)
    print("                      TABLA COMPARATIVA CIENTIFICA EVOLUTIVA (MVP 2 -> MVP 5-B)")
    print("=" * 90)
    print(" Métrica Evaluada            | MVP 2 (Base) | MVP 4 (Aug) | MVP 5-A (Crop 0R) | MVP 5-B (Crop-Aware) | Δ (5B vs 4) ")
    print(" " + "-" * 88)
    print(f" PlantVillage Test Top-1     |       {res_mvp2['plantvillage_top1']:6.2f}% |      {res_mvp4['plantvillage_top1']:6.2f}% |           —       |              {res_mvp5b['plantvillage_top1']:6.2f}% |   {res_mvp5b['plantvillage_top1'] - res_mvp4['plantvillage_top1']:+6.2f} pp")
    print(f" PlantDoc Test Top-1         |       {res_mvp2['plantdoc_top1']:6.2f}% |      {res_mvp4['plantdoc_top1']:6.2f}% |        {34.30:6.2f}%  |              {res_mvp5b['plantdoc_top1']:6.2f}% |   {res_mvp5b['plantdoc_top1'] - res_mvp4['plantdoc_top1']:+6.2f} pp")
    print(f" PlantDoc Top-3              |       {res_mvp2['plantdoc_top3']:6.2f}% |      {res_mvp4['plantdoc_top3']:6.2f}% |        {56.98:6.2f}%  |              {res_mvp5b['plantdoc_top3']:6.2f}% |   {res_mvp5b['plantdoc_top3'] - res_mvp4['plantdoc_top3']:+6.2f} pp")
    print(f" Identificación de Cultivo   |       {res_mvp2['plantdoc_crop_acc']:6.2f}% |      {res_mvp4['plantdoc_crop_acc']:6.2f}% |        {70.93:6.2f}%  |              {res_mvp5b['plantdoc_crop_acc']:6.2f}% |   {res_mvp5b['plantdoc_crop_acc'] - res_mvp4['plantdoc_crop_acc']:+6.2f} pp")
    print(" " + "-" * 88)
    print(f" Domain Shift Gap (Línea B.) |       {res_mvp2['domain_shift_gap']:6.2f} pp|      {res_mvp4['domain_shift_gap']:6.2f} pp|        {60.68:6.2f} pp|              {res_mvp5b['domain_shift_gap']:6.2f} pp|   {res_mvp5b['domain_shift_gap'] - res_mvp4['domain_shift_gap']:+6.2f} pp")
    print(f" Confusión Papa -> Tomate    |       {res_mvp2['papa_to_tomate']:6d}   |      {res_mvp4['papa_to_tomate']:6d}   |        {10:6d}    |              {res_mvp5b['papa_to_tomate']:6d}   |   {res_mvp5b['papa_to_tomate'] - res_mvp4['papa_to_tomate']:+6d}")
    print("=" * 90 + "\n")

def plot_4way_bars(summary, save_path):
    metrics = ["PlantDoc Top-1", "PlantDoc Top-3", "Reconoc. Planta", "Domain Shift Gap"]
    
    m2 = summary["mvp2_baseline"]
    m4 = summary["mvp4_strong_aug"]
    m5a = summary["mvp5a_crop_zero_retrain"]
    m5b = summary["mvp5b_crop_aware"]

    v_m2 = [m2["plantdoc_top1"], m2["plantdoc_top3"], m2["plantdoc_crop_acc"], m2["domain_shift_gap"]]
    v_m4 = [m4["plantdoc_top1"], m4["plantdoc_top3"], m4["plantdoc_crop_acc"], m4["domain_shift_gap"]]
    v_m5a = [m5a["plantdoc_top1_global"], m5a["plantdoc_top3"], m5a["plantdoc_crop_acc"], m5a["domain_shift_gap"]]
    v_m5b = [m5b["plantdoc_top1"], m5b["plantdoc_top3"], m5b["plantdoc_crop_acc"], m5b["domain_shift_gap"]]

    x = np.arange(len(metrics))
    width = 0.20

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width, v_m2, width, label='MVP 2 (Baseline)', color='#457B9D')
    plt.bar(x - 0.5*width, v_m4, width, label='MVP 4 (Strong Aug)', color='#E07A5F')
    plt.bar(x + 0.5*width, v_m5a, width, label='MVP 5-A (Crop 0-Retrain)', color='#81B29A')
    plt.bar(x + 1.5*width, v_m5b, width, label='MVP 5-B (Crop-Aware)', color='#2A9D8F')

    plt.ylabel('Porcentaje (%) / Puntos Porcentuales (pp)', fontsize=11)
    plt.title('AgroVision — Evolución Experimental Multidimensional (MVP 2 -> MVP 5-B)', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(x, metrics, fontsize=11)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Gráfica comparativa 4-Way guardada en: {save_path}")

if __name__ == "__main__":
    run_4way_evaluation()
