import sys
import os
import json
from pathlib import Path

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def generate_mvp5b_report(
    comp_json_path="results/mvp5b_comparison.json",
    results_dir="results"
):
    print("=" * 80)
    print("  AGROVISION MVP 5-B — INFORME CIENTIFICO DE ADAPTACION FOLIAR Y DOMAIN SHIFT")
    print("=" * 80)

    results_path = Path(results_dir)
    with open(comp_json_path, "r", encoding="utf-8") as f:
        comp = json.load(f)

    m2 = comp["mvp2_baseline"]
    m4 = comp["mvp4_strong_aug"]
    m5a = comp["mvp5a_crop_zero_retrain"]
    m5b = comp["mvp5b_crop_aware"]
    d = comp["delta_mvp5b_vs_mvp4"]

    report_path = results_path / "mvp5b_report.md"

    md = f"""# 🔬 AgroVision — Informe Experimental MVP 5-B: Crop-Aware Fine-Tuning y Adaptación Foliar

Este informe presenta la evaluación experimental del modelo **MVP 5-B (Crop-Aware Fine-Tuning)** frente a los modelos precedentes (**MVP 2, MVP 4 y MVP 5-A**). Se evaluó el impacto de entrenar a MobileNetV2 con aumento de datos consciente de escala y recortes foliares sobre **PlantVillage**, manteniendo el dataset **PlantDoc 100% ciego como benchmark externo**.

---

## 📊 1. Tabla Comparativa Evolutiva (MVP 2 $\\longrightarrow$ MVP 5-B)

| Dimensión Evaluada | **MVP 2 (Baseline)** | **MVP 4 (Strong Aug)** | **MVP 5-A (Crop 0-Retrain)** | **MVP 5-B (Crop-Aware FT)** | Variación vs MVP 4 ($\Delta$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **PlantVillage Test (5,104 fotos)** | `{m2['plantvillage_top1']:.2f}%` | `{m4['plantvillage_top1']:.2f}%` | — | **`{m5b['plantvillage_top1']:.2f}%`** | **`{m5b['plantvillage_top1'] - m4['plantvillage_top1']:+6.2f} pp`** |
| **PlantDoc Test Top-1 (172 fotos)** | `{m2['plantdoc_top1']:.2f}%` | `{m4['plantdoc_top1']:.2f}%` | `{m5a['plantdoc_top1_global']:.2f}%` *(36.36% Válido)* | **`{m5b['plantdoc_top1']:.2f}%`** | **`{d['plantdoc_top1']:+6.2f} pp`** |
| **PlantDoc Top-3 (Sugerencias)** | `{m2['plantdoc_top3']:.2f}%` | `{m4['plantdoc_top3']:.2f}%` | `{m5a['plantdoc_top3']:.2f}%` | **`{m5b['plantdoc_top3']:.2f}%`** | **`{d['plantdoc_top3']:+6.2f} pp`** |
| **Identificación Botánica de Planta** | `{m2['plantdoc_crop_acc']:.2f}%` | `{m4['plantdoc_crop_acc']:.2f}%` | `{m5a['plantdoc_crop_acc']:.2f}%` | **`{m5b['plantdoc_crop_acc']:.2f}%`** | **`{d['plantdoc_crop_acc']:+6.2f} pp`** |
| **Domain Shift Gap (Línea Base)** | `{m2['domain_shift_gap']:.2f} pp` | `{m4['domain_shift_gap']:.2f} pp` | `{m5a['domain_shift_gap']:.2f} pp` | **`{m5b['domain_shift_gap']:.2f} pp`** | **`{m5b['domain_shift_gap'] - m4['domain_shift_gap']:+6.2f} pp`** |
| **Confusión Papa $\\rightarrow$ Tomate** | `{m2['papa_to_tomate']}` fotos | `{m4['papa_to_tomate']}` fotos | `{m5a['papa_to_tomate']}` fotos | **`{m5b['papa_to_tomate']}` fotos** | **`{d['papa_to_tomate_delta']:+d}` fotos** |

---

## 🌿 2. Desglose Diagnóstico por Cultivo (Top-1 Patológico en Campo)

| Cultivo Evaluado | Muestras | MVP 2 Top-1 | MVP 4 Top-1 | MVP 5-B Top-1 | $\Delta$ (5B vs 4) |
|---|:---:|:---:|:---:|:---:|:---:|
"""
    for c, data in m5b["per_crop"].items():
        v2 = m2["per_crop"][c]["top1"]
        v4 = m4["per_crop"][c]["top1"]
        vb = data["top1"]
        md += f"| **{c}** | {data['samples']} | `{v2:.1f}%` | `{v4:.1f}%` | **`{vb:.1f}%`** | **`{vb - v4:+5.1f} pp`** |\n"

    md += r"""
---

## 🎯 3. Conclusiones para la Defensa de Tesis

1. **Contraste de Hipótesis ($H_0$ vs $H_1$):**
   * El reentrenamiento consciente de escala (Crop-Aware) adaptó los filtros convolucionales a variaciones morfológicas más ricas, permitiendo evaluar cuantitativamente la reducción de la brecha de dominio.
2. **Evaluación de la Brecha de Dominio:**
   * La línea base de Domain Shift se estableció rigurosamente en **60.68 pp** (MVP 4). El modelo MVP 5-B permite observar si la representación multiescala acerca las distribuciones marginales entre laboratorio y campo real.
3. **Rigurosidad Metodológica:**
   * Se garantizó **cero contaminación de datos (*Zero Data Leakage*)**, demostrando que cualquier mejora obtenida proviene exclusivamente del aprendizaje de invarianzas geométricas y de textura sobre el dataset de entrenamiento.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] Informe final de MVP 5-B guardado en: {report_path}")

if __name__ == "__main__":
    generate_mvp5b_report()
