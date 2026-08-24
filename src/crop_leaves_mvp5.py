import sys
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def detect_leaf_region(img_bgr):
    """
    Localizador foliar analítico no contaminado basado en:
    1. Índice de Exceso de Verde (ExG = 2G - R - B) para saliencia vegetativa.
    2. Espacio de color HSV (rango de hojas: verdes, amarillas cloróticas y pardas necróticas).
    3. Segmentación morfológica con filtrado de componentes conectados.
    4. Bounding Box envolvente con padding del 5%.
    """
    h, w = img_bgr.shape[:2]
    total_area = h * w

    # 1. Calcular Exceso de Verde (ExG)
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    denom = r + g + b + 1e-5
    r_norm = r / denom
    g_norm = g / denom
    b_norm = b / denom
    exg = 2 * g_norm - r_norm - b_norm

    # 2. Máscara HSV para tejido foliar (Verde vivo + amarilleo clorótico + marchitamiento marrón)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (20, 25, 25), (95, 255, 255))
    mask_yellow_brown = cv2.inRange(hsv, (10, 30, 30), (25, 255, 200))
    hsv_mask = cv2.bitwise_or(mask_green, mask_yellow_brown)

    # 3. Combinar ExG + HSV
    exg_uint8 = np.clip((exg + 1) * 127.5, 0, 255).astype(np.uint8)
    _, exg_thresh = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined_mask = cv2.bitwise_or(exg_thresh, hsv_mask)

    # 4. Operaciones morfológicas para consolidar la región foliar
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. Encontrar contornos
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0.0, "LOCALIZATION_FAILURE", 0.0

    # Seleccionar el contorno foliar más significativo o fusionar racimos cercanos
    valid_contours = [c for c in contours if cv2.contourArea(c) > (0.03 * total_area)]

    if not valid_contours:
        # Fallback al contorno más grande
        largest_c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_c)
        area_ratio = (bw * bh) / total_area
        conf = float(cv2.contourArea(largest_c) / (bw * bh + 1e-5))
        status = "LOW_CONFIDENCE" if area_ratio >= 0.08 else "LOCALIZATION_FAILURE"
        return [int(x), int(y), int(bw), int(bh)], float(area_ratio), status, float(conf)

    # Unir contornos válidos para capturar la hoja completa (incluso si la necrosis corta el verde)
    all_points = np.vstack(valid_contours)
    x, y, bw, bh = cv2.boundingRect(all_points)

    # Añadir padding del 5%
    pad_x = int(bw * 0.05)
    pad_y = int(bh * 0.05)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    final_bw = x2 - x1
    final_bh = y2 - y1
    area_ratio = float((final_bw * final_bh) / total_area)
    leaf_pixels = np.sum([cv2.contourArea(c) for c in valid_contours])
    conf = float(np.clip(leaf_pixels / (final_bw * final_bh + 1e-5), 0.1, 1.0))

    if 0.10 <= area_ratio <= 0.98 and conf >= 0.35:
        status = "VALID_CROP"
    elif area_ratio < 0.10:
        status = "LOCALIZATION_FAILURE"
    else:
        status = "LOW_CONFIDENCE"

    return [int(x1), int(y1), int(final_bw), int(final_bh)], area_ratio, status, conf

def run_leaf_cropping(
    plantdoc_dir="PlantDoc-Dataset/test",
    classes_path="models/class_names_mvp2.json",
    mapping_path="models/class_mapping_mvp2.json",
    results_dir="results"
):
    print("=" * 80)
    print("  AGROVISION MVP 5-A — PROCESAMIENTO Y LOCALIZACION DE HOJAS (ZERO DATA LEAKAGE)")
    print("=" * 80)

    plantdoc_path = Path(plantdoc_dir)
    results_path = Path(results_dir)
    crops_base_dir = results_path / "mvp5_crops"
    detections_dir = results_path / "mvp5_detections"
    crops_base_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)

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

    crops_metadata = []
    status_counts = {"VALID_CROP": 0, "LOW_CONFIDENCE": 0, "LOCALIZATION_FAILURE": 0}

    print(f"\n[+] Procesando 172 imágenes de PlantDoc con Localizador de Saliencia Foliar...")

    for pd_cls, true_idx, disp_label, true_crop in mapped_classes:
        folder = plantdoc_path / pd_cls
        img_files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])

        # Subcarpeta por cultivo en crops (normalizado sin tildes)
        crop_folder_name = (
            true_crop.lower()
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        crop_save_dir = crops_base_dir / crop_folder_name
        crop_save_dir.mkdir(parents=True, exist_ok=True)

        for img_p in img_files:
            img_bgr = cv2.imread(str(img_p))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            bbox, area_ratio, status, conf = detect_leaf_region(img_bgr)
            status_counts[status] += 1

            # Generar Crop y Bounding Box Overlay
            vis_img = img_bgr.copy()

            if bbox is not None and status in ["VALID_CROP", "LOW_CONFIDENCE"]:
                x, y, bw, bh = bbox
                crop_img = img_bgr[y:y+bh, x:x+bw]
                crop_rel_path = f"results/mvp5_crops/{crop_folder_name}/{img_p.name}"
                cv2.imwrite(str(crop_save_dir / img_p.name), crop_img)

                # Color del BBox (Verde para VALID_CROP, Amarillo para LOW_CONFIDENCE)
                color = (0, 220, 0) if status == "VALID_CROP" else (0, 200, 240)
                cv2.rectangle(vis_img, (x, y), (x+bw, y+bh), color, 3)
                label_txt = f"{status} ({area_ratio*100:.1f}%)"
                cv2.putText(vis_img, label_txt, (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Si falló la localización, se guarda la imagen original como fallback de evaluación
                crop_img = img_bgr
                crop_rel_path = f"results/mvp5_crops/{crop_folder_name}/{img_p.name}"
                cv2.imwrite(str(crop_save_dir / img_p.name), crop_img)
                cv2.putText(vis_img, "LOCALIZATION_FAILURE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 240), 2)

            # Guardar visualización de detección
            cv2.imwrite(str(detections_dir / f"det_{img_p.name}"), vis_img)

            crops_metadata.append({
                "filename": img_p.name,
                "crop": true_crop,
                "ground_truth": disp_label,
                "original_path": str(img_p),
                "crop_path": crop_rel_path,
                "localization_status": status,
                "bbox": bbox if bbox else [0, 0, w, h],
                "bbox_area_ratio": area_ratio,
                "confidence": conf
            })

    # Guardar metadatos JSON
    meta_path = results_path / "mvp5_crops_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(crops_metadata, f, indent=4, ensure_ascii=False)

    total = len(crops_metadata)
    success_rate = (status_counts["VALID_CROP"] / total) * 100

    print("\n" + "=" * 80)
    print("           AGROVISION MVP 5-A — RESUMEN DE LOCALIZACION FOLIAR")
    print("=" * 80)
    print(f" Total imágenes procesadas:        {total}")
    print(f" 🟢 VALID_CROP (Hoja Aislada):     {status_counts['VALID_CROP']:3d} ({status_counts['VALID_CROP']/total*100:.1f}%)")
    print(f" 🟡 LOW_CONFIDENCE (Baja Certeza): {status_counts['LOW_CONFIDENCE']:3d} ({status_counts['LOW_CONFIDENCE']/total*100:.1f}%)")
    print(f" 🔴 LOCALIZATION_FAILURE:          {status_counts['LOCALIZATION_FAILURE']:3d} ({status_counts['LOCALIZATION_FAILURE']/total*100:.1f}%)")
    print(" " + "-" * 78)
    print(f" 🎯 LOCALIZATION SUCCESS RATE:      {success_rate:.2f}%")
    print(f" 📁 Recortes guardados en:          {crops_base_dir}")
    print(f" 📁 Visualizaciones de control en:  {detections_dir}")
    print(f" 📄 Metadatos exportados en:        {meta_path}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_leaf_cropping()
