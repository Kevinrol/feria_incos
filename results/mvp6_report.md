# 🔬 Informe Científico Experimental — MVP 6: In-the-Wild Domain Adaptation

**Proyecto:** AgroVision — Sistema de Diagnóstico Fitosanitario de Precisión  
**Fecha:** Agosto 2026  
**Autor:** Equipo de Investigación de Inteligencia Artificial AgroVision  
**Fase:** MVP 6 — Adaptación de Dominio Multidominio (Laboratorio + Mundo Real)  

---

## 🎯 Resumen Ejecutivo

El experimento **MVP 6** abordó de forma frontal el problema del *Domain Shift* descubierto en MVP 3 y no resuelto por las aumentaciones sintéticas (MVP 4 / MVP 5-B). Mediante la incorporación de **5,625 fotografías de campo real procedentes de PlantWild** compatibles con las 27 clases canónicas de AgroVision, se entrenó una arquitectura adaptativa con **muestreo balanceado por dominio (50% PlantVillage / 50% PlantWild por lote)**.

Los resultados demuestran la **mayor victoria empírica del proyecto hasta la fecha**:
1. **PlantDoc (Benchmark Ciego Externo In-the-Wild):** El Top-1 alcanzó un récord histórico de **`42.44%`** ($\mathbf{+8.14\text{ pp}}$ frente a MVP 4 y $\mathbf{+9.30\text{ pp}}$ frente a MVP 2).
2. **PlantDoc Top-3:** Alcanzó **`71.51%`** ($\mathbf{+15.11\text{ pp}}$ frente al 56.40% de MVP 4), permitiendo que en más de 7 de cada 10 casos reales la patología correcta esté en la terna de sospecha clínica.
3. **Identificación de Cultivo en PlantDoc:** Aumentó a **`77.33%`** ($\mathbf{+6.98\text{ pp}}$).
4. **Dominio In-the-Wild (PlantWild Test):** Saltó de 18.01% a **`47.58%`** ($\mathbf{+29.57\text{ pp}}$).
5. **Preservación de Laboratorio (PlantVillage Test):** Se mantuvo en **`93.40%`**, confirmando que el modelo **no sufrió olvido catastrófico** del conocimiento de laboratorio.
6. **Desbloqueo de Papa y Solanáceas:** La confusión crítica `Papa -> Tomate` se redujo en un **70%** (de 10 casos a solo 3), elevando la identificación de Papa de 6.67% a **`73.33%`**.

---

## 📊 Tabla Comparativa Científica (3 Dominios Evaluados)

| Métrica Evaluada | MVP 2 (Baseline) | MVP 4 (Strong Aug) | MVP 5-B (Crop FT) | MVP 6 (In-the-Wild) | Δ Absoluta (6 vs 4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PlantVillage Test Top-1 (Lab)** | 91.28% | 94.98% | **95.02%** | 93.40% | *-1.58 pp* |
| **PlantWild Test Top-1 (Mundo Real)** | 17.38% | 18.01% | 15.20% | **47.58%** | **+29.57 pp** |
| **PlantWild Test Top-3** | 36.47% | 38.44% | 33.80% | **73.30%** | **+34.86 pp** |
| **PlantDoc Test Top-1 (Ciego Externo)** | 33.14% | 34.30% | 26.74% | **42.44%** | **+8.14 pp** |
| **PlantDoc Top-3** | 53.49% | 56.40% | 48.84% | **71.51%** | **+15.11 pp** |
| **Identificación de Cultivo (PlantDoc)** | 69.19% | 70.35% | 61.05% | **77.33%** | **+6.98 pp** |
| **Brecha Shift Lab (PV - PD)** | 58.14 pp | 60.68 pp | 68.28 pp | **50.96 pp** | **-9.72 pp** |
| **Brecha Generalización Real (PW - PD)**| -15.76 pp | -16.29 pp | -11.54 pp | **5.14 pp** | **Paridad Real** |
| **Confusión Papa $\rightarrow$ Tomate** | 10 / 15 | 10 / 15 | 11 / 15 | **3 / 15** | **-70.0%** |

---

## 📈 Desglose por Cultivo en el Benchmark Ciego (PlantDoc)

| Cultivo Evaluado | Total Fotos | MVP 4 Top-1 | MVP 6 Top-1 | Δ Top-1 | MVP 4 Cultivo Acc | MVP 6 Cultivo Acc | Δ Cultivo Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Manzana** | 29 | 31.03% | **55.17%** | **+24.14 pp** | 34.48% | **86.21%** | **+51.73 pp** |
| **Maíz** | 25 | 40.00% | **60.00%** | **+20.00 pp** | 100.00% | **100.00%** | 0.00 pp |
| **Vid (Uva)** | 18 | 55.56% | **61.11%** | **+5.55 pp** | 77.78% | **83.33%** | **+5.55 pp** |
| **Pimiento** | 16 | 43.75% | **37.50%** | -6.25 pp | 56.25% | **50.00%** | -6.25 pp |
| **Papa** | 15 | 6.67% | **33.33%** | **+26.66 pp** | 6.67% | **73.33%** | **+66.66 pp** |
| **Tomate** | 69 | 31.88% | **28.99%** | -2.89 pp | 89.86% | **71.01%** | -18.85 pp |

---

## 🔬 Análisis de Hallazgos Clave

### 1. Desbloqueo del Reconocimiento de Papa
En MVP 2, MVP 4 y MVP 5-B, el cultivo de Papa estaba completamente bloqueado en campo real: solo 1 de 15 imágenes era reconocida correctamente y 10 eran confundidas con Tomate. En **MVP 6**, la identificación correcta del cultivo de Papa subió de 6.67% a **`73.33%`**, y la exactitud patológica Top-1 subió de 6.67% a **`33.33%`**. La inclusión de fotos reales de Papa de PlantWild proporcionó la variabilidad morfológica foliar necesaria para que la red aprendiera a diferenciar ambas especies solanáceas en condiciones agrestes.

### 2. Generalización In-the-Wild Verdadera (PW vs PD Gap)
En MVP 2 y MVP 4, los modelos fallaban tanto en PlantWild (17-18%) como en PlantDoc (33-34%), reflejando que eran detectores ciegos fuera del fondo gris/blanco de laboratorio. En MVP 6:
- PlantWild Test: **`47.58%`**
- PlantDoc Test: **`42.44%`**
- **Brecha de Generalización Real:** Solo **`5.14 pp`**. Esto demuestra que el modelo ha aprendido representaciones invariantes a la iluminación solar, suelo y desorden botánico que se transfieren consistentemente a un dataset externo nunca visto durante el entrenamiento.

### 3. Veredicto Metodológico
Se cumple la **Condición Verde (🟢 Mejora Real)** definida en el protocolo experimental:
$$\text{PlantDoc Top-1} \uparrow (+8.14\text{ pp}), \quad \text{PlantWild Top-1} \uparrow (+29.57\text{ pp}), \quad \text{PlantVillage} \approx 93.40\% \ge 92\%$$

---

## 🚀 Conclusiones y Próximos Pasos

1. **MVP 6 queda consolidado como el modelo principal de clasificación (Backbone Oficial de AgroVision).**
2. **PlantSeg (Fase de Detección/Segmentación):** Con el clasificador adaptado a texturas reales, el siguiente paso estratégico para superar la cota del 50-60% en PlantDoc es acoplar la **segmentación/localización foliar con PlantSeg** (entrenando un detector de hojas YOLO/Mask-RCNN sobre las 77,394 máscaras) para eliminar ruido periférico de fondo antes de alimentar a MVP 6.
