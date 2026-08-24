import sys
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def create_mvp5a_visual_card(img_raw_path, img_crop_path, ground_truth, raw_pred, raw_conf, crop_pred, crop_conf, category, save_path):
    raw_img = Image.open(img_raw_path).convert("RGB")
    crop_img = Image.open(img_crop_path).convert("RGB")

    target_size = (300, 300)
    raw_resized = raw_img.resize(target_size, Image.Resampling.BILINEAR)
    crop_resized = crop_img.resize(target_size, Image.Resampling.BILINEAR)

    card_w = 640
    card_h = 460
    canvas = Image.new("RGB", (card_w, card_h), color=(20, 24, 33))

    cat_colors = {
        "FIXED": (40, 167, 69),        # Verde (Éxito de la intervención)
        "ROBUST": (30, 130, 230),       # Azul (Acierto consistente)
        "PERSISTENT": (220, 53, 69),    # Rojo (Fallo estructural)
        "REGRESSION": (240, 140, 30)    # Naranja (Regresión)
    }
    header_color = cat_colors.get(category, (100, 100, 100))

    draw = ImageDraw.Draw(canvas)

    # Banner superior
    draw.rectangle([(0, 0), (card_w, 28)], fill=header_color)
    draw.text((10, 6), f"AGROVISION MVP 5-A AUDITORIA — CATEGORIA: {category}", fill=(255, 255, 255))

    # Textos de diagnóstico
    draw.text((10, 38), f"Condición Real (Ground Truth):  {ground_truth}", fill=(120, 230, 120))
    draw.text((10, 62), f"MVP 4 (Foto Original):           {raw_pred} ({raw_conf:.1f}%)", fill=(240, 200, 120))
    draw.text((10, 86), f"MVP 5-A (Hoja Aislada):          {crop_pred} ({crop_conf:.1f}%)", fill=(130, 200, 255))
    draw.text((10, 110), f"Archivo: {Path(img_raw_path).name}", fill=(180, 180, 180))

    # Pegar imágenes lado a lado
    canvas.paste(raw_resized, (15, 140))
    canvas.paste(crop_resized, (325, 140))

    draw.text((80, 442), "FOTO ORIGINAL (RAW)", fill=(200, 200, 200))
    draw.text((400, 442), "HOJA AISLADA (CROP)", fill=(200, 200, 200))

    canvas.save(save_path, quality=90)

def analyze_errors_mvp5a(
    comp_json_path="results/mvp5a_comparison.json",
    crops_meta_path="results/mvp5_crops_metadata.json",
    results_dir="results"
):
    print("=" * 80)
    print("  AGROVISION MVP 5-A — AUDITORIA FORENSE Y REPORTE CIENTIFICO")
    print("=" * 80)

    results_path = Path(results_dir)
    vis_dir = results_path / "mvp5_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    with open(comp_json_path, "r", encoding="utf-8") as f:
        comp = json.load(f)
    with open(crops_meta_path, "r", encoding="utf-8") as f:
        meta_dict = {item["filename"]: item for item in json.load(f)}

    records = comp["detailed_records"]
    total = len(records)

    cat_counts = {"FIXED": 0, "ROBUST": 0, "PERSISTENT": 0, "REGRESSION": 0}
    for r in records:
        cat_counts[r["category"]] += 1

    print("\n[+] Generando tarjetas visuales comparativas en 'results/mvp5_visualizations/'...")
    for idx, r in enumerate(records):
        fn = r["filename"]
        meta_item = meta_dict.get(fn, {})
        orig_p = Path(meta_item.get("original_path", ""))
        crop_p = ROOT_DIR / meta_item.get("crop_path", "")

        cat = r["category"]

        # Generar tarjetas para FIXED, REGRESSION y muestra de PERSISTENT/ROBUST
        if cat in ["FIXED", "REGRESSION"] or (idx % 4 == 0):
            if orig_p.exists() and crop_p.exists():
                save_p = vis_dir / f"{cat.lower()}_{orig_p.stem}.jpg"
                create_mvp5a_visual_card(
                    img_raw_path=orig_p,
                    img_crop_path=crop_p,
                    ground_truth=r["ground_truth"],
                    raw_pred=r["raw"]["predicted"],
                    raw_conf=r["raw"]["confidence"],
                    crop_pred=r["crop"]["predicted"],
                    crop_conf=r["crop"]["confidence"],
                    category=cat,
                    save_path=save_p
                )

    # Generar Informe Markdown
    report_path = results_path / "mvp5a_error_report.md"
    generate_report_md(comp, cat_counts, report_path)
    print(f"\n[OK] Informe final de MVP 5-A guardado en: {report_path}")

def generate_report_md(comp, cat_counts, save_path):
    total = comp["total_images"]
    loc_rate = comp["localization_success_rate"]
    g = comp["global_intent_to_diagnose"]
    p = comp["per_protocol_valid_crops_cohort"]
    sol = comp["solanaceae_tracking"]
    crops = comp["per_crop_breakdown"]

    md = f"""# 🔬 AgroVision — Informe del Experimento MVP 5-A (Aislamiento Foliar Zero-Retrain)

Este informe documenta los resultados del experimento **MVP 5-A**. Evaluamos el impacto puro de aislar la región foliar mediante detección y recorte (*Bounding Box Crop*) sobre las **172 imágenes de campo real de PlantDoc**, utilizando el modelo **MobileNetV2 (MVP 4) 100% intacto y sin reentrenar**.

---

## 🎯 1. Métricas de Localización Foliar

| Métrica de Localización | Conteo | Porcentaje | Significado Experimental |
|---|:---:|:---:|---|
| **🟢 Tasa de Éxito de Localización (*Success Rate*)** | **`{comp['valid_crops_count']}`** fotos | **`{loc_rate:.2f}%`** | Región foliar clara detectada con relación de área óptima ($10\% \\le \\text{{Ratio}} \\le 98\%$). |
| **🟡 Baja Certeza / Fallo de Localización** | **`{total - comp['valid_crops_count']}`** fotos | **`{100 - loc_rate:.2f}%`** | Fondos con ruido masivo, hojas excesivamente pequeñas o oclusión severa. |

---

## 📊 2. Tabla Comparativa de Rendimiento Diagnóstico

| Métrica Evaluada | **MVP 4 (Foto Original)** | **MVP 5-A (Hoja Aislada)** | **Variación ($\Delta$)** | Interpretación Científica |
|---|:---:|:---:|:---:|---|
| **PlantDoc Top-1 (Global)** | `{g['raw_top1']:.2f}%` | **`{g['crop_top1']:.2f}%`** | **`{g['delta_top1']:+6.2f} pp`** | Evaluación *Intent-to-Diagnose* sobre las 172 fotos. |
| **PlantDoc Top-3 (Sugerencias)** | `{g['raw_top3']:.2f}%` | **`{g['crop_top3']:.2f}%`** | **`{g['delta_top3']:+6.2f} pp`** | Presencia de la patología real entre las 3 principales opciones. |
| **Identificación Botánica de Planta** | `{g['raw_crop_acc']:.2f}%` | **`{g['crop_crop_acc']:.2f}%`** | **`{g['delta_crop_acc']:+6.2f} pp`** | Acierto en la especie de cultivo frente a fondos reales. |
| **Top-1 en Cohorte Válida ({comp['valid_crops_count']} fotos)** | `{p['raw_top1_in_valid']:.2f}%` | **`{p['crop_top1_in_valid']:.2f}%`** | **`{p['delta_valid_top1']:+6.2f} pp`** | Rendimiento puro sobre hojas donde la localización fue exitosa. |

---

## 🔬 3. Auditoría Diferencial en las 4 Categorías

| Categoría | Cantidad | % Total | Significado Científico |
|---|:---:|:---:|---|
| 🟢 **Fixed (Recuperados por Crop)** | **`{cat_counts['FIXED']}`** fotos | **`{cat_counts['FIXED']/total*100:.1f}%`** | Casos donde eliminar el suelo, dedos o ramas **permitió al modelo diagnosticar correctamente**. |
| 🔵 **Robust (Aciertos Consistentes)** | **`{cat_counts['ROBUST']}`** fotos | **`{cat_counts['ROBUST']/total*100:.1f}%`** | Casos claros donde ambos enfoques acertaron. |
| 🔴 **Persistent (Fallos Estructurales)** | **`{cat_counts['PERSISTENT']}`** fotos | **`{cat_counts['PERSISTENT']/total*100:.1f}%`** | Casos que fallaron en ambos; el error se debe a la similitud intrínseca de los síntomas fúngicos o resolución. |
| 🟠 **Regression (Regresiones)** | **`{cat_counts['REGRESSION']}`** fotos | **`{cat_counts['REGRESSION']/total*100:.1f}%`** | Casos donde el recorte eliminó contexto foliar relevante o truncó la lesión. |

---

## 🍅 4. Tracking de Solanáceas (Papa $\leftrightarrow$ Tomate)

| Confusión Cruzada | MVP 4 (Foto Original) | MVP 5-A (Hoja Aislada) | Variación ($\Delta$) |
|---|:---:|:---:|:---:|
| **Papa $\rightarrow$ Tomate** | **`{sol['papa_to_tomate_raw']}`** fotos | **`{sol['papa_to_tomate_crop']}`** fotos | **`{sol['papa_to_tomate_delta']:+d}`** |
| **Tomate $\rightarrow$ Papa** | **`{sol['tomate_to_papa_raw']}`** fotos | **`{sol['tomate_to_papa_crop']}`** fotos | **`{sol['tomate_to_papa_delta']:+d}`** |

---

## 🌿 5. Desglose por Cultivo

| Cultivo Evaluado | Muestras | MVP 4 Raw Top-1 | MVP 5-A Crop Top-1 | $\Delta$ Top-1 | MVP 4 Especie | MVP 5-A Especie | $\Delta$ Especie |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for c, d in crops.items():
        md += f"| **{c}** | {d['samples']} | `{d['raw_top1']:.1f}%` | **`{d['crop_top1']:.1f}%`** | `{d['delta_top1']:+5.1f} pp` | `{d['raw_crop_acc']:.1f}%` | **`{d['crop_crop_acc']:.1f}%`** | `{d['delta_crop_acc']:+5.1f} pp` |\n"

    md += r"""
---

## 🎯 6. Conclusiones y Decisión Metodológica para el MVP 5-B

### Análisis de la Hipótesis Experimental ($H_0$ vs $H_1$):
* **Impacto del Aislamiento Foliar Puro:** La eliminación del fondo mediante Bounding Box modificó el rendimiento global de """ + f"`{g['raw_top1']:.2f}%` a `{g['crop_top1']:.2f}%` ($\\Delta = {g['delta_top1']:+5.2f}\\text{{ pp}}$)." + r"""
* **Decisión para MVP 5-B:**
  * Si $\Delta > 0$: La evidencia empírica respalda que el fondo era un factor limitante, justificando el **MVP 5-B (Reentrenamiento con Crops)** para que los filtros convolucionales se especialicen exclusivamente en patrones foliares aislados.
  * Si $\Delta \le 0$: La evidencia demuestra que MobileNetV2 preentrenado con fondos artificiales no adapta sus escalas de textura automáticamente sin un reentrenamiento explícito.
"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md)

if __name__ == "__main__":
    analyze_errors_mvp5a()
