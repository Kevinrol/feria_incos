import sys
import os
import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw

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

def create_annotated_error_image(img_path, true_label, pred_label, conf, top3_labels, save_path):
    orig_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    target_w = 600
    target_h = int((target_w / orig_w) * orig_h)
    img_resized = orig_img.resize((target_w, target_h), Image.Resampling.BILINEAR)

    banner_h = 120
    canvas = Image.new("RGB", (target_w, target_h + banner_h), color=(20, 24, 33))
    canvas.paste(img_resized, (0, banner_h))

    draw = ImageDraw.Draw(canvas)
    title_text = "AGROVISION MVP 4 ERROR DE DIAGNOSTICO (PLANTDOC)"
    real_text = f"REAL:      {true_label}"
    pred_text = f"PREDICHO:  {pred_label} (Confianza: {conf:.1f}%)"
    top3_text = f"Top 3: {', '.join(top3_labels[:3])}"

    draw.rectangle([(0, 0), (target_w, 24)], fill=(180, 40, 40))
    draw.text((10, 4), title_text, fill=(255, 255, 255))
    draw.text((10, 32), real_text, fill=(100, 220, 100))
    draw.text((10, 56), pred_text, fill=(255, 120, 120))
    draw.text((10, 84), top3_text, fill=(180, 200, 220))

    canvas.save(save_path, quality=90)

def run_error_analysis_mvp4(
    model_path="models/agrovision_mobilenetv2_mvp4.keras",
    classes_path="models/class_names_mvp2.json",
    mapping_path="models/class_mapping_mvp2.json",
    plantdoc_dir="PlantDoc-Dataset/test",
    results_dir="results"
):
    print("=" * 80)
    print("  AGROVISION MVP 4 — ANALISIS FORENSE DE ERRORES Y CONFUSION PAPA ↔ TOMATE")
    print("=" * 80)

    plantdoc_path = Path(plantdoc_dir)
    results_path = Path(results_dir)
    errors_base_dir = results_path / "errors_mvp4"
    errors_base_dir.mkdir(parents=True, exist_ok=True)

    crop_folders = {
        "Manzana": errors_base_dir / "manzana",
        "Maíz": errors_base_dir / "maiz",
        "Papa": errors_base_dir / "papa",
        "Pimiento": errors_base_dir / "pimiento",
        "Tomate": errors_base_dir / "tomate",
        "Vid (Uva)": errors_base_dir / "vid"
    }
    for folder in crop_folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    with open(classes_path, "r", encoding="utf-8") as f:
        model_classes = json.load(f)
    with open(mapping_path, "r", encoding="utf-8") as f:
        meta = json.load(f).get("metadata", {})

    mapped_classes = []
    for idx, cls_name in enumerate(model_classes):
        m = meta.get(cls_name, {})
        pd_class = m.get("plantdoc_class")
        if pd_class and (plantdoc_path / pd_class).exists():
            crop_es = m.get("crop_es", cls_name.split("___")[0])
            dis_es = m.get("disease_es", cls_name.split("___")[1])
            mapped_classes.append((pd_class, idx, f"{crop_es}: {dis_es}", crop_es))

    model = tf.keras.models.load_model(model_path)

    errors_list = []
    total_imgs, correct_imgs = 0, 0
    solanaceae_confusion_count = 0

    for pd_cls, true_idx, disp_label, crop_name in mapped_classes:
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
                pred_dis = meta[model_classes[pred_idx]]["disease_es"]
                pred_disp = f"{pred_crop}: {pred_dis}"
                conf = float(preds[pred_idx] * 100)

                top3_disp = [
                    f"{meta[model_classes[i]]['crop_es']}:{meta[model_classes[i]]['disease_es']}({preds[i]*100:.0f}%)"
                    for i in top_indices[:3]
                ]

                total_imgs += 1
                if true_idx == pred_idx:
                    correct_imgs += 1
                else:
                    # Detectar confusión cruzada de Solanáceas
                    is_solanaceae_cross = bool((crop_name == "Papa" and pred_crop == "Tomate") or (crop_name == "Tomate" and pred_crop == "Papa"))
                    if is_solanaceae_cross:
                        solanaceae_confusion_count += 1

                    err_info = {
                        "filename": img_p.name,
                        "crop": crop_name,
                        "real_label": disp_label,
                        "pred_label": pred_disp,
                        "confidence": conf,
                        "is_same_crop": bool(crop_name == pred_crop),
                        "is_solanaceae_cross": is_solanaceae_cross,
                        "top3": top3_disp
                    }
                    errors_list.append(err_info)

                    dest_folder = crop_folders.get(crop_name, errors_base_dir)
                    create_annotated_error_image(
                        img_path=img_p,
                        true_label=disp_label,
                        pred_label=pred_disp,
                        conf=conf,
                        top3_labels=top3_disp,
                        save_path=dest_folder / f"error_{img_p.stem}.jpg"
                    )

            except Exception as e:
                print(f"[!] Error con {img_p.name}: {e}")

    confusion_pairs = Counter([(e["real_label"], e["pred_label"]) for e in errors_list])
    sorted_confusions = confusion_pairs.most_common()

    # Guardar reporte de errores
    report_path = results_path / "error_comparison_report_mvp4.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# 🔬 Informe de Errores — AgroVision MVP 4 (Augmentation Fuerte)

* **Total fotos auditadas en PlantDoc:** `{total_imgs}`
* **Aciertos exactos:** `{correct_imgs}` (`{correct_imgs/total_imgs*100:.2f}%`)
* **Total de fallos:** `{len(errors_list)}`
* **Confusiones Cruzadas Papa ↔ Tomate (Solanáceas):** `{solanaceae_confusion_count}` casos (en MVP 2 fueron `10` casos).

## Top Errores en Modelo MVP 4:
| # | Real (PlantDoc) | Predicho (Modelo MVP 4) | Casos | % Error |
|---|---|---|---|---|
""")
        for idx, (pair, count) in enumerate(sorted_confusions[:10]):
            f.write(f"| {idx+1} | **{pair[0]}** | **{pair[1]}** | **{count}** fotos | `{count/len(errors_list)*100:.1f}%` |\n")

    print(f"\n[OK] Auditoría de errores MVP 4 guardada en: {report_path}")
    print(f"[OK] Imágenes anotadas guardadas en: {errors_base_dir}")
    print(f"     * Confusiones Solanáceas (Papa ↔ Tomate): {solanaceae_confusion_count} casos")

if __name__ == "__main__":
    run_error_analysis_mvp4()
