import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def evaluate_model_on_pv_test(model, pv_test_dir):
    pv_ds = tf.keras.utils.image_dataset_from_directory(
        pv_test_dir,
        image_size=(224, 224),
        batch_size=32,
        label_mode="int",
        shuffle=False
    )
    pv_prep = pv_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    res = model.evaluate(pv_prep, verbose=0)
    return float(res[1] * 100)

def evaluate_model_on_pw_test(model, pw_manifest_path):
    with open(pw_manifest_path, "r", encoding="utf-8") as f:
        pw_records = json.load(f)
    
    total = len(pw_records)
    if total == 0:
        return 0.0, 0.0
    
    correct_top1 = 0
    correct_top3 = 0
    
    for r in pw_records:
        img_p = r["path"]
        true_idx = r["class_idx"]
        try:
            with Image.open(img_p) as img:
                img = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                arr = preprocess_input(np.expand_dims(np.array(img, dtype=np.float32), axis=0))
                preds = model.predict(arr, verbose=0)[0]
                top_indices = np.argsort(preds)[::-1]
                
                if top_indices[0] == true_idx:
                    correct_top1 += 1
                if true_idx in top_indices[:3]:
                    correct_top3 += 1
        except Exception:
            pass
            
    top1 = (correct_top1 / total) * 100
    top3 = (correct_top3 / total) * 100
    return float(top1), float(top3)

def evaluate_model_on_plantdoc(model, model_classes, meta, plantdoc_dir):
    plantdoc_path = Path(plantdoc_dir)
    mapped_classes = []
    for idx, cls_name in enumerate(model_classes):
        m = meta.get(cls_name, {})
        pd_class = m.get("plantdoc_class")
        if pd_class and (plantdoc_path / pd_class).exists():
            crop_es = m.get("crop_es", cls_name.split("___")[0])
            dis_es = m.get("disease_es", cls_name.split("___")[1])
            mapped_classes.append((pd_class, idx, f"{crop_es}: {dis_es}", crop_es))

    total = 0
    top1_correct = 0
    top3_correct = 0
    crop_correct = 0
    
    # Solanáceas tracking
    potato_to_tomato = 0
    tomato_to_potato = 0
    
    per_crop_stats = defaultdict(lambda: {"total": 0, "top1": 0, "crop_correct": 0})
    confusion_records = []

    for pd_class, true_idx, disp_name, true_crop_es in mapped_classes:
        class_folder = plantdoc_path / pd_class
        for img_path in class_folder.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    arr = preprocess_input(np.expand_dims(np.array(img, dtype=np.float32), axis=0))
                    preds = model.predict(arr, verbose=0)[0]
                    top_indices = np.argsort(preds)[::-1]
                    
                    pred_idx = top_indices[0]
                    pred_name = model_classes[pred_idx]
                    pred_crop_es = meta[pred_name]["crop_es"]
                    
                    is_top1 = (pred_idx == true_idx)
                    is_top3 = (true_idx in top_indices[:3])
                    is_crop = (pred_crop_es == true_crop_es)
                    
                    total += 1
                    if is_top1: top1_correct += 1
                    if is_top3: top3_correct += 1
                    if is_crop: crop_correct += 1
                    
                    # Solanáceas
                    if true_crop_es == "Papa" and pred_crop_es == "Tomate":
                        potato_to_tomato += 1
                    elif true_crop_es == "Tomate" and pred_crop_es == "Papa":
                        tomato_to_potato += 1
                        
                    per_crop_stats[true_crop_es]["total"] += 1
                    if is_top1: per_crop_stats[true_crop_es]["top1"] += 1
                    if is_crop: per_crop_stats[true_crop_es]["crop_correct"] += 1
                    
                    confusion_records.append({
                        "image": img_path.name,
                        "true_idx": true_idx,
                        "pred_idx": int(pred_idx),
                        "true_class": disp_name,
                        "pred_class": model_classes[pred_idx]
                    })
            except Exception:
                pass

    metrics = {
        "total_images": total,
        "top1_acc": round((top1_correct / total) * 100, 2) if total > 0 else 0,
        "top3_acc": round((top3_correct / total) * 100, 2) if total > 0 else 0,
        "crop_acc": round((crop_correct / total) * 100, 2) if total > 0 else 0,
        "potato_to_tomato": potato_to_tomato,
        "tomato_to_potato": tomato_to_potato,
        "per_crop": {
            k: {
                "total": v["total"],
                "top1_acc": round((v["top1"] / v["total"]) * 100, 2) if v["total"] > 0 else 0,
                "crop_acc": round((v["crop_correct"] / v["total"]) * 100, 2) if v["total"] > 0 else 0
            } for k, v in per_crop_stats.items()
        },
        "confusion_records": confusion_records
    }
    return metrics

def run_evaluation(
    model_paths={
        "MVP 2 (Baseline)": "models/agrovision_mobilenetv2_mvp2.keras",
        "MVP 4 (Strong Aug)": "models/agrovision_mobilenetv2_mvp4.keras",
        "MVP 6 (In-the-Wild)": "models/agrovision_mobilenetv2_mvp6.keras"
    },
    pv_test_dir="datasets/plantvillage_mvp2/test",
    pw_manifest_path="datasets/agrovision_mvp6/plantwild_test_manifest.json",
    plantdoc_dir="PlantDoc-Dataset/test",
    classes_path="models/class_names_mvp2.json",
    mapping_path="models/class_mapping_mvp2.json",
    output_json="results/mvp6_comparison.json",
    output_plot="results/mvp6_comparison_bars.png"
):
    print("=" * 80)
    print("  AGROVISION MVP 6 — EVALUACION COMPARATIVA 3-DOMINIOS (PV vs PW vs PlantDoc)")
    print("=" * 80)

    with open(ROOT_DIR / classes_path, "r", encoding="utf-8") as f:
        model_classes = json.load(f)
    with open(ROOT_DIR / mapping_path, "r", encoding="utf-8") as f:
        meta = json.load(f).get("metadata", {})

    results = {}

    for name, rel_path in model_paths.items():
        m_file = ROOT_DIR / rel_path
        if not m_file.exists():
            print(f"[-] Saltando '{name}': archivo '{m_file}' no encontrado.")
            continue

        print(f"\n[+] Evaluando {name} ('{rel_path}')...")
        model = tf.keras.models.load_model(m_file)

        # 1. PlantVillage Test (Laboratorio)
        print("    • Evaluando en PlantVillage Test (5,104 fotos)...")
        pv_acc = evaluate_model_on_pv_test(model, ROOT_DIR / pv_test_dir)

        # 2. PlantWild Test (Campo Real In-The-Wild)
        print("    • Evaluando en PlantWild Test (1,116 fotos)...")
        pw_top1, pw_top3 = evaluate_model_on_pw_test(model, ROOT_DIR / pw_manifest_path)

        # 3. PlantDoc Benchmark (Campo Real Ciego Externo)
        print("    • Evaluando en PlantDoc Blind Benchmark (172 fotos)...")
        pd_res = evaluate_model_on_plantdoc(model, model_classes, meta, ROOT_DIR / plantdoc_dir)

        # Brechas de dominio
        lab_gap = pv_acc - pd_res["top1_acc"]
        real_gap = pw_top1 - pd_res["top1_acc"]

        results[name] = {
            "model_file": rel_path,
            "pv_test_top1": round(pv_acc, 2),
            "pw_test_top1": round(pw_top1, 2),
            "pw_test_top3": round(pw_top3, 2),
            "plantdoc_top1": pd_res["top1_acc"],
            "plantdoc_top3": pd_res["top3_acc"],
            "crop_identification": pd_res["crop_acc"],
            "potato_to_tomato": pd_res["potato_to_tomato"],
            "tomato_to_potato": pd_res["tomato_to_potato"],
            "lab_domain_shift_gap": round(lab_gap, 2),
            "real_generalization_gap": round(real_gap, 2),
            "per_crop": pd_res["per_crop"]
        }

    # Guardar resultados JSON
    with open(ROOT_DIR / output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n[OK] Resultados guardados en: {output_json}")

    # Imprimir Tabla Comparativa Oficial
    print_comparative_table(results)

    # Generar Gráfica Comparativa
    generate_comparison_plot(results, ROOT_DIR / output_plot)

def print_comparative_table(results):
    print("\n" + "=" * 95)
    print("                      TABLA COMPARATIVA CIENTIFICA MULTIDOMINIO (MVP 4 vs MVP 6)")
    print("=" * 95)
    header = f"{'Métrica Evaluada':<30} | {'MVP 2 (Base)':<13} | {'MVP 4 (Ref)':<13} | {'MVP 6 (In-Wild)':<15} | {'Δ (6 vs 4)':<10}"
    print(header)
    print("-" * 95)

    m4 = results.get("MVP 4 (Strong Aug)", {})
    m6 = results.get("MVP 6 (In-the-Wild)", {})
    m2 = results.get("MVP 2 (Baseline)", {})

    def get_val(m, key):
        return m.get(key, 0.0)

    rows = [
        ("PlantVillage Test Top-1 (Lab)", "pv_test_top1", "%"),
        ("PlantWild Test Top-1 (Real)", "pw_test_top1", "%"),
        ("PlantDoc Test Top-1 (Ciego)", "plantdoc_top1", "%"),
        ("PlantDoc Top-3", "plantdoc_top3", "%"),
        ("Identificación de Cultivo", "crop_identification", "%"),
        ("Brecha Laboratorio (PV-PD)", "lab_domain_shift_gap", " pp"),
        ("Brecha Mundo Real (PW-PD)", "real_generalization_gap", " pp"),
        ("Confusión Papa -> Tomate", "potato_to_tomato", "")
    ]

    for label, key, unit in rows:
        v2_str = f"{get_val(m2, key):.2f}{unit}" if key != "potato_to_tomato" else f"{int(get_val(m2, key))}"
        v4_str = f"{get_val(m4, key):.2f}{unit}" if key != "potato_to_tomato" else f"{int(get_val(m4, key))}"
        v6_str = f"{get_val(m6, key):.2f}{unit}" if key != "potato_to_tomato" else f"{int(get_val(m6, key))}"
        
        diff = get_val(m6, key) - get_val(m4, key)
        diff_str = f"{diff:+.2f}{unit}" if key != "potato_to_tomato" else f"{int(diff):+d}"
        
        print(f"{label:<30} | {v2_str:>13} | {v4_str:>13} | {v6_str:>15} | {diff_str:>10}")

    print("=" * 95 + "\n")

def generate_comparison_plot(results, save_path):
    models_list = list(results.keys())
    if not models_list:
        return

    metrics = [
        ("PlantVillage\nTest Top-1", "pv_test_top1"),
        ("PlantWild\nTest Top-1", "pw_test_top1"),
        ("PlantDoc\nTest Top-1", "plantdoc_top1"),
        ("PlantDoc\nTop-3", "plantdoc_top3"),
        ("Reconocimiento\nCultivo", "crop_identification")
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#95a5a6", "#2980b9", "#27ae60"]
    for i, m_name in enumerate(models_list):
        vals = [results[m_name].get(k, 0.0) for _, k in metrics]
        offset = (i - len(models_list)/2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=m_name, color=colors[i % len(colors)], edgecolor="black", alpha=0.85)
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title("Evolución Multidominio AgroVision: MVP 2 vs MVP 4 vs MVP 6", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Exactitud (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper left", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Gráfica comparativa guardada en: {save_path}")

if __name__ == "__main__":
    run_evaluation()
