import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent

def audit_plantwild(
    plantwild_root="plantwild/plantwild",
    pv_classes_path="models/class_names_mvp2.json",
    pv_mapping_path="models/class_mapping_mvp2.json",
    output_dir="results/datasets"
):
    print("=" * 80)
    print("  FASE 1: AUDITORIA TECNICA Y CIENTIFICA DE PLANTWILD")
    print("=" * 80)

    pw_path = ROOT_DIR / plantwild_root
    images_dir = pw_path / "images"
    classes_file = pw_path / "classes.txt"
    trainval_file = pw_path / "trainval.txt"
    prompts_file = pw_path / "plantwild_prompts.json"
    url_file = pw_path / "url_record.json"

    out_path = ROOT_DIR / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Cargar clases de AgroVision (27 clases)
    with open(ROOT_DIR / pv_classes_path, "r", encoding="utf-8") as f:
        agro_classes = json.load(f)
    with open(ROOT_DIR / pv_mapping_path, "r", encoding="utf-8") as f:
        agro_meta = json.load(f).get("metadata", {})

    # 2. Cargar clases de PlantWild
    pw_classes = []
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        pw_classes.append((int(parts[0]), parts[1]))
                    else:
                        pw_classes.append((len(pw_classes), line))
    else:
        pw_class_names = sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
        pw_classes = list(enumerate(pw_class_names))

    print(f"[+] Total de clases registradas en PlantWild: {len(pw_classes)}")

    # 3. Analizar imágenes por clase, formatos y resoluciones
    class_stats = []
    total_images = 0
    format_counts = defaultdict(int)
    widths = []
    heights = []
    aspect_ratios = []

    print("[+] Analizando archivos de imagen, formatos y resoluciones...")
    for cls_id, cls_name in pw_classes:
        c_dir = images_dir / cls_name
        if not c_dir.exists():
            img_files = []
        else:
            img_files = [f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        
        count = len(img_files)
        total_images += count
        
        cls_widths = []
        cls_heights = []

        for img_f in img_files:
            format_counts[img_f.suffix.lower()] += 1
            try:
                with Image.open(img_f) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h if h > 0 else 1.0)
                    cls_widths.append(w)
                    cls_heights.append(h)
            except Exception as e:
                pass

        avg_w = float(np.mean(cls_widths)) if cls_widths else 0.0
        avg_h = float(np.mean(cls_heights)) if cls_heights else 0.0

        # Determinar si es saludable
        is_healthy = True if cls_name.endswith(" leaf") else False
        
        # Extraer cultivo base
        crop_word = cls_name.split()[0]
        if cls_name.startswith("bell pepper"):
            crop_word = "bell pepper"
        elif cls_name.startswith("grapevine"):
            crop_word = "grape"
        elif cls_name.startswith("cherry"):
            crop_word = "cherry"

        class_stats.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "crop": crop_word,
            "is_healthy": is_healthy,
            "image_count": count,
            "avg_width": round(avg_w, 1),
            "avg_height": round(avg_h, 1)
        })

    # 4. Analizar splits si existen (trainval.txt)
    split_counts = {"train": 0, "val": 0, "test": 0, "unknown": 0}
    if trainval_file.exists():
        with open(trainval_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("=")
                if len(parts) == 3:
                    mode = parts[2].strip()
                    if mode == "1":
                        split_counts["train"] += 1
                    elif mode == "2":
                        split_counts["val"] += 1
                    elif mode == "0":
                        split_counts["test"] += 1
                    else:
                        split_counts["unknown"] += 1

    # 5. Mapeo Taxonómico con AgroVision (27 Clases)
    agro_to_pw_map = {
        "Apple___Apple_scab": "apple scab",
        "Apple___Black_rot": "apple black rot",
        "Apple___Cedar_apple_rust": "apple rust",
        "Apple___healthy": "apple leaf",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "corn gray leaf spot",
        "Corn_(maize)___Common_rust_": "corn rust",
        "Corn_(maize)___Northern_Leaf_Blight": "corn northern leaf blight",
        "Corn_(maize)___healthy": "corn leaf",
        "Grape___Black_rot": "grape black rot",
        "Grape___Esca_(Black_Measles)": None,
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape leaf spot",
        "Grape___healthy": "grape leaf",
        "Pepper,_bell___Bacterial_spot": "bell pepper leaf spot",
        "Pepper,_bell___healthy": "bell pepper leaf",
        "Potato___Early_blight": "potato early blight",
        "Potato___Late_blight": "potato late blight",
        "Potato___healthy": "potato leaf",
        "Tomato___Bacterial_spot": "tomato bacterial leaf spot",
        "Tomato___Early_blight": "tomato early blight",
        "Tomato___Late_blight": "tomato late blight",
        "Tomato___Leaf_Mold": "tomato leaf mold",
        "Tomato___Septoria_leaf_spot": "tomato septoria leaf spot",
        "Tomato___Spider_mites Two-spotted_spider_mite": None,
        "Tomato___Target_Spot": None,
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato yellow leaf curl virus",
        "Tomato___Tomato_mosaic_virus": "tomato mosaic virus",
        "Tomato___healthy": "tomato leaf"
    }

    pw_class_dict = {item["class_name"]: item for item in class_stats}

    matched_classes = []
    unmatched_agro = []
    total_matched_images = 0

    for idx, agro_c in enumerate(agro_classes):
        meta_info = agro_meta.get(agro_c, {})
        crop_es = meta_info.get("crop_es", agro_c.split("___")[0])
        dis_es = meta_info.get("disease_es", agro_c.split("___")[1])
        pw_name = agro_to_pw_map.get(agro_c)

        if pw_name and pw_name in pw_class_dict:
            img_c = pw_class_dict[pw_name]["image_count"]
            total_matched_images += img_c
            matched_classes.append({
                "agro_idx": idx,
                "agro_class": agro_c,
                "crop_es": crop_es,
                "disease_es": dis_es,
                "plantwild_class": pw_name,
                "images_in_plantwild": img_c
            })
        else:
            unmatched_agro.append({
                "agro_idx": idx,
                "agro_class": agro_c,
                "crop_es": crop_es,
                "disease_es": dis_es,
                "reason": "No presente en taxonomía de PlantWild"
            })

    # Clases de PlantWild adicionales para nuestros 6 cultivos
    our_6_crops = ["apple", "corn", "grape", "grapevine", "pepper", "bell pepper", "potato", "tomato"]
    matched_pw_names = set([m["plantwild_class"] for m in matched_classes])
    
    additional_6_crops_classes = []
    new_species_classes = []

    for item in class_stats:
        c_name = item["class_name"]
        if c_name in matched_pw_names:
            continue
        
        is_our_crop = any(c_name.startswith(p) for p in our_6_crops)
        if is_our_crop:
            additional_6_crops_classes.append(item)
        else:
            new_species_classes.append(item)

    # 6. Guardar CSV de distribución de clases
    csv_path = out_path / "plantwild_class_distribution.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "crop", "is_healthy", "image_count", "avg_width", "avg_height", "matched_agrovision_class"])
        for item in class_stats:
            agro_equiv = ""
            for m in matched_classes:
                if m["plantwild_class"] == item["class_name"]:
                    agro_equiv = m["agro_class"]
                    break
            writer.writerow([
                item["class_id"],
                item["class_name"],
                item["crop"],
                "1" if item["is_healthy"] else "0",
                item["image_count"],
                item["avg_width"],
                item["avg_height"],
                agro_equiv
            ])
    print(f"[OK] CSV de distribución guardado en: {csv_path}")

    # 7. Guardar JSON de auditoría completo
    audit_json = {
        "dataset_name": "PlantWild",
        "location": str(pw_path),
        "total_images": total_images,
        "total_classes": len(pw_classes),
        "resolution_stats": {
            "min_width": int(np.min(widths)) if widths else 0,
            "max_width": int(np.max(widths)) if widths else 0,
            "mean_width": float(np.mean(widths)) if widths else 0,
            "median_width": float(np.median(widths)) if widths else 0,
            "min_height": int(np.min(heights)) if heights else 0,
            "max_height": int(np.max(heights)) if heights else 0,
            "mean_height": float(np.mean(heights)) if heights else 0,
            "median_height": float(np.median(heights)) if heights else 0,
            "mean_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else 0
        },
        "formats": dict(format_counts),
        "splits": split_counts,
        "agrovision_compatibility": {
            "total_agrovision_classes": len(agro_classes),
            "matched_classes_count": len(matched_classes),
            "unmatched_classes_count": len(unmatched_agro),
            "total_matched_images": total_matched_images,
            "matched_classes": matched_classes,
            "unmatched_agrovision_classes": unmatched_agro,
            "additional_classes_for_existing_crops": additional_6_crops_classes,
            "new_plant_species_count": len(new_species_classes)
        }
    }

    json_path = out_path / "plantwild_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_json, f, indent=4, ensure_ascii=False)
    print(f"[OK] JSON de auditoría guardado en: {json_path}")

    # 8. Generar Informe Markdown detallado
    md_path = out_path / "plantwild_audit.md"
    generate_markdown_report(audit_json, md_path)
    print(f"[OK] Informe Markdown guardado en: {md_path}")
    print("=" * 80 + "\n")

def generate_markdown_report(audit, save_path):
    res = audit["resolution_stats"]
    compat = audit["agrovision_compatibility"]
    splits = audit["splits"]

    md = f"""# 🔬 Auditoría Técnica y Taxonómica: Dataset PlantWild (In-the-Wild)

## 📌 1. Resumen Ejecutivo del Dataset

| Dimensión | Métrica Auditada | Significado Técnico |
|---|:---:|---|
| **Total de Imágenes** | **`{audit['total_images']:,}` fotos** | Dataset masivo tomado en condiciones de campo real (*in-the-wild*). |
| **Total de Clases** | **`{audit['total_classes']}` clases** | Cubre especies hortícolas, frutales, cereales y ornamentales. |
| **Formato de Imágenes** | **JPG (`{audit['formats'].get('.jpg', 0):,}`), JPEG (`{audit['formats'].get('.jpeg', 0):,}`), PNG (`{audit['formats'].get('.png', 0):,}`)** | Fotografías RGB nativas sin fondo sintético. |
| **Resolución Promedio** | **`{res['mean_width']:.0f} × {res['mean_height']:.0f}` px** | Resoluciones variadas (Mín: `{res['min_width']}×{res['min_height']}`, Máx: `{res['max_width']}×{res['max_height']}`). |
| **División (Splits)** | **Train: `{splits['train']:,}` \| Val: `{splits['val']:,}` \| Test: `{splits['test']:,}`** | 80% Train/Val (70% Train, 10% Val), 20% Test predefinido. |
| **Compatibilidad con AgroVision** | **`{compat['matched_classes_count']}` de 27 clases (`{compat['matched_classes_count']/27*100:.1f}%`)** | **`{compat['total_matched_images']:,}` imágenes reales directas** para nuestro catálogo actual. |

---

## 🎯 2. Mapeo y Coincidencia Directa con AgroVision (24 Clases)

Estas **24 clases** coinciden 1-a-1 en cultivo y patología con nuestro modelo de 27 clases, aportando un total de **`{compat['total_matched_images']:,}` imágenes de campo real**:

| Agro ID | Cultivo | Patología en AgroVision (PlantVillage) | Clase en PlantWild | Fotos en PlantWild |
|:---:|---|---|---|:---:|
"""
    for m in compat["matched_classes"]:
        md += f"| **{m['agro_idx']:02d}** | {m['crop_es']} | *{m['disease_es']}* | `{m['plantwild_class']}` | **`{m['images_in_plantwild']}`** |\n"

    md += f"""
---

## ❌ 3. Clases de AgroVision Sin Correspondencia en PlantWild (Solo 3 Clases)

| Agro ID | Cultivo | Patología | Causa |
|:---:|---|---|---|
"""
    for u in compat["unmatched_agrovision_classes"]:
        md += f"| **{u['agro_idx']:02d}** | {u['crop_es']} | *{u['disease_es']}* | `{u['reason']}` |\n"

    md += """
---

## 🌿 4. Clases Adicionales en PlantWild para Nuestros 6 Cultivos

PlantWild incluye **patologías de campo adicionales** para los 6 cultivos estratégicos de AgroVision que podrían incorporarse en versiones futuras:

| Cultivo | Patología Adicional en PlantWild | Fotos Disponibles |
|---|---|:---:|
"""
    for a in compat["additional_classes_for_existing_crops"]:
        md += f"| **{a['crop'].capitalize()}** | `{a['class_name']}` | **`{a['image_count']}`** |\n"

    md += f"""
---

## 🌍 5. Nuevas Especies de Cultivos en PlantWild ({compat['new_plant_species_count']} Clases)

PlantWild amplía el espectro agronómico con **{compat['new_plant_species_count']} clases** pertenecientes a otros cultivos agrícolas:
* **Hortalizas:** Brócoli, Repollo, Zanahoria, Coliflor, Apio, Pepino, Berenjena, Ajo, Jengibre, Lechuga, Calabaza, Calabacín.
* **Frutales y Berries:** Plátano, Arándano, Cereza, Cítricos, Melocotón, Ciruela, Frambuesa, Fresa.
* **Cultivos Industriales y Cereales:** Café, Arroz, Soja, Tabaco, Frijol/Alubia, Albahaca.

---

## 💡 6. Conclusiones de Utilidad Inmediata para AgroVision

1. **Aporte Cuantitativo Masivo:** Obtenemos **`{compat['total_matched_images']:,}` fotografías de campo real** perfectamente etiquetadas para 24 de nuestras 27 clases.
2. **Superación del Domain Shift:** A diferencia de PlantVillage (fondos de laboratorio homogéneos), PlantWild contiene ruido real, luz solar natural, variaciones de escala e iluminación de campo.
3. **Resguardo de Integridad:** PlantWild puede utilizarse como dataset de entrenamiento / fine-tuning in-the-wild, **manteniendo PlantDoc estrictamente separado y ciego como test benchmark final**.
"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md)

if __name__ == "__main__":
    audit_plantwild()
