# 🔬 AgroVision — Informe del Experimento MVP 5-A (Aislamiento Foliar Zero-Retrain)

Este informe documenta los resultados del experimento **MVP 5-A**. Evaluamos el impacto puro de aislar la región foliar mediante detección y recorte (*Bounding Box Crop*) sobre las **172 imágenes de campo real de PlantDoc**, utilizando el modelo **MobileNetV2 (MVP 4) 100% intacto y sin reentrenar**.

---

## 🎯 1. Métricas de Localización Foliar

| Métrica de Localización | Conteo | Porcentaje | Significado Experimental |
|---|:---:|:---:|---|
| **🟢 Tasa de Éxito de Localización (*Success Rate*)** | **`33`** fotos | **`19.19%`** | Región foliar clara detectada con relación de área óptima ($10\% \le \text{Ratio} \le 98\%$). |
| **🟡 Baja Certeza / Fallo de Localización** | **`139`** fotos | **`80.81%`** | Fondos con ruido masivo, hojas excesivamente pequeñas o oclusión severa. |

---

## 📊 2. Tabla Comparativa de Rendimiento Diagnóstico

| Métrica Evaluada | **MVP 4 (Foto Original)** | **MVP 5-A (Hoja Aislada)** | **Variación ($\Delta$)** | Interpretación Científica |
|---|:---:|:---:|:---:|---|
| **PlantDoc Top-1 (Global)** | `34.30%` | **`34.30%`** | **` +0.00 pp`** | Evaluación *Intent-to-Diagnose* sobre las 172 fotos. |
| **PlantDoc Top-3 (Sugerencias)** | `56.40%` | **`56.98%`** | **` +0.58 pp`** | Presencia de la patología real entre las 3 principales opciones. |
| **Identificación Botánica de Planta** | `70.35%` | **`70.93%`** | **` +0.58 pp`** | Acierto en la especie de cultivo frente a fondos reales. |
| **Top-1 en Cohorte Válida (33 fotos)** | `33.33%` | **`36.36%`** | **` +3.03 pp`** | Rendimiento puro sobre hojas donde la localización fue exitosa. |

---

## 🔬 3. Auditoría Diferencial en las 4 Categorías

| Categoría | Cantidad | % Total | Significado Científico |
|---|:---:|:---:|---|
| 🟢 **Fixed (Recuperados por Crop)** | **`2`** fotos | **`1.2%`** | Casos donde eliminar el suelo, dedos o ramas **permitió al modelo diagnosticar correctamente**. |
| 🔵 **Robust (Aciertos Consistentes)** | **`57`** fotos | **`33.1%`** | Casos claros donde ambos enfoques acertaron. |
| 🔴 **Persistent (Fallos Estructurales)** | **`111`** fotos | **`64.5%`** | Casos que fallaron en ambos; el error se debe a la similitud intrínseca de los síntomas fúngicos o resolución. |
| 🟠 **Regression (Regresiones)** | **`2`** fotos | **`1.2%`** | Casos donde el recorte eliminó contexto foliar relevante o truncó la lesión. |

---

## 🍅 4. Tracking de Solanáceas (Papa $\leftrightarrow$ Tomate)

| Confusión Cruzada | MVP 4 (Foto Original) | MVP 5-A (Hoja Aislada) | Variación ($\Delta$) |
|---|:---:|:---:|:---:|
| **Papa $ightarrow$ Tomate** | **`10`** fotos | **`10`** fotos | **`+0`** |
| **Tomate $ightarrow$ Papa** | **`0`** fotos | **`0`** fotos | **`+0`** |

---

## 🌿 5. Desglose por Cultivo

| Cultivo Evaluado | Muestras | MVP 4 Raw Top-1 | MVP 5-A Crop Top-1 | $\Delta$ Top-1 | MVP 4 Especie | MVP 5-A Especie | $\Delta$ Especie |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Manzana** | 29 | `31.0%` | **`27.6%`** | ` -3.4 pp` | `34.5%` | **`31.0%`** | ` -3.4 pp` |
| **Maíz** | 25 | `40.0%` | **`40.0%`** | ` +0.0 pp` | `100.0%` | **`100.0%`** | ` +0.0 pp` |
| **Papa** | 15 | `6.7%` | **`6.7%`** | ` +0.0 pp` | `6.7%` | **`6.7%`** | ` +0.0 pp` |
| **Pimiento** | 16 | `43.8%` | **`56.2%`** | `+12.5 pp` | `56.2%` | **`68.8%`** | `+12.5 pp` |
| **Tomate** | 69 | `31.9%` | **`30.4%`** | ` -1.4 pp` | `89.9%` | **`89.9%`** | ` +0.0 pp` |
| **Vid (Uva)** | 18 | `55.6%` | **`55.6%`** | ` +0.0 pp` | `77.8%` | **`77.8%`** | ` +0.0 pp` |

---

## 🎯 6. Conclusiones Oficiales y Decisión Metodológica para el MVP 5-B

### Veredicto Científico de MVP 5-A:
> **MVP 5-A demostró que el aislamiento foliar, aplicado mediante Bounding Box Crop y sin modificar los pesos de MobileNetV2, no produce una mejora significativa en el rendimiento global sobre PlantDoc.** El Top-1 se mantuvo en **`34.30%`**, mientras que el Top-3 aumentó ligeramente de **`56.40%` a `56.98%`**.
>
> Sin embargo, en la **cohorte de imágenes con localización foliar válida (33 fotos)** se observó una mejora de **3.03 puntos porcentuales en Top-1**, pasando de **`33.33%` a `36.36%`**. Además, **Pimiento presentó una mejora notable de 12.50 puntos porcentuales** (pasando de `43.75%` a `56.25%`). Estos resultados sugieren que **la eliminación del fondo puede ser beneficiosa cuando la hoja es correctamente aislada**, pero que el modelo previamente entrenado en laboratorio no está adaptado a esta nueva representación espacial.

### ⚠️ Hallazgo Estructural Principal: Dos Problemas Desacoplados
```text
                  AgroVision
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
   LOCALIZACIÓN              CLASIFICACIÓN
   ¿Dónde está la hoja?      ¿Qué enfermedad tiene?
          │                       │
       19.19%              MVP 4: 34.30%
     localización                 ↓
       válida              MVP 5-A Crop: 34.30%
```

### 🚀 Hoja de Ruta para el Experimento MVP 5-B:
Evaluar experimentalmente si un **Crop-Aware Fine-Tuning con representación foliar multiescala** sobre PlantVillage permite aprovechar de forma consistente la eliminación de fondo, **manteniendo PlantDoc completamente ciego del entrenamiento**.
* **Línea Base del Domain Shift Gap:** $\text{MVP 4 Gap} = 94.98\% - 34.30\% = \mathbf{60.68\text{ pp}}$.


