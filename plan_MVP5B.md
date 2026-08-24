# AgroVision MVP 5-B — Plan Experimental: Crop-Aware Fine-Tuning y Adaptación Foliar

## 🎯 Objetivo General
Determinar si un **Fine-Tuning con representación foliar multiescala** sobre el dataset de entrenamiento (PlantVillage) permite al modelo MobileNetV2 adaptarse a la escala espacial de recortes y primeros planos, mejorando su capacidad diagnóstica en condiciones de campo real (**PlantDoc**), manteniendo este último **100% ciego del entrenamiento**.

---

## 1. Marco Teórico e Hipótesis Científica

### H₀ — Hipótesis Nula
El Fine-Tuning con imágenes foliares recortadas y variación de escala no produce una mejora estadísticamente significativa en la generalización sobre PlantDoc respecto a MVP 4.

### H₁ — Hipótesis Alternativa
El Fine-Tuning con imágenes foliares recortadas mejora la generalización sobre PlantDoc al reducir el desajuste (*scale mismatch*) entre la representación utilizada durante el entrenamiento (hojas completas sobre fondo artificial) y la utilizada en inferencia (primeros planos y hojas aisladas).

---

## 2. Modelos a Comparar

| Modelo | Pipeline de Entrenamiento | Evaluación PlantDoc | Rol Metodológico |
|---|---|---|---|
| **MVP 2** | PlantVillage + Augmentation suave | 33.14% Top-1 | **Baseline Inicial** |
| **MVP 4** | PlantVillage + Augmentation fuerte | 34.30% Top-1 | **Baseline Mitigado (Gap = 60.68 pp)** |
| **MVP 5-A** | MVP 4 intacto (Zero-Retrain) + Crop | 34.30% Top-1 *(36.36% en Cohorte Válida)* | **Aislamiento Puro** |
| **MVP 5-B** | **Crop-Aware Fine-Tuning en PlantVillage** | **¿XX.XX%?** | **Adaptación Foliar Activa** |

> 🚨 **Regla Inquebrantable:** PlantDoc **NUNCA** entra al entrenamiento de MVP 5-B. Ni imágenes, ni etiquetas, ni recortes derivados de PlantDoc. Permanece exclusivamente como benchmark externo ciego.

---

## 3. Estrategia de Aumento de Datos en PlantVillage (*Crop-Aware Pipeline*)

En lugar de generar recortes estáticos que preserven la limpieza artificial de PlantVillage, se diseña un pipeline dinámico en Keras (`src/augmentations_crop_aware.py`):
1. **Zoom Multiescala Extendido (`RandomZoom`):** Rango $[-40\%, +40\%]$ para simular hojas que llenan el encuadre y recortes cerrados sobre zonas de lesión.
2. **Traslación Agresiva (`RandomTranslation`):** Rango $[-15\%, +15\%]$ para simular encuadres descentrados.
3. **Rotación Calibrada (`RandomRotation`):** $\pm 19.8^\circ$ ($0.055 \times 360^\circ$).
4. **Variación Fotométrica:** Contraste ($\pm 20\%$) y Brillo ($\pm 20\%$).

---

## 4. Criterios de Evaluación y Línea Base

### Criterio Primario:
* **$\text{PlantDoc Top-1 (MVP 5-B)} > 34.30\%$** (Superar el récord de MVP 4).

### Criterios Secundarios:
* **PlantDoc Top-3:** $> 56.40\%$.
* **Identificación de Cultivo:** $> 70.35\%$.
* **Confusión Solanáceas:** Reducción de los $10$ casos de `Papa -> Tomate`.
* **Reducción de la Brecha de Dominio (*Domain Shift Gap*):**
  $$\text{Gap}_{\text{MVP 5-B}} = \text{Top1}_{\text{PlantVillage}} - \text{Top1}_{\text{PlantDoc}} < \mathbf{60.68\text{ pp}}$$
* **Preservación en Laboratorio:** Mantener $\text{PlantVillage Test} \ge 92.0\%$.

---

## 5. Estructura de Scripts y Flujo de Trabajo

```text
src/
├── augmentations_crop_aware.py  # Pipeline Sequential de aumento multiescala
├── train_mvp5b.py               # Entrenamiento en 2 fases con EarlyStopping
├── evaluate_mvp5b.py            # Evaluación 4-Way (MVP 2 vs 4 vs 5-A vs 5-B)
└── analyze_errors_mvp5b.py      # Auditoría diferencial y tracking Solanáceas

results/
├── mvp5b_comparison.json        # Registro completo de métricas
├── mvp5b_comparison_bars.png    # Gráfico de barras comparativo 4-Way
├── training_history_mvp5b.png   # Curvas de convergencia
└── mvp5b_report.md              # Informe científico final para defensa
```
