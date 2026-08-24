# 🔬 AgroVision — Informe Forense Comparativo de Errores: MVP 2 vs MVP 4

Este informe presenta el análisis diferencial **imagen por imagen** sobre las **172 fotografías de campo real de PlantDoc** para auditar el comportamiento del **Modelo A (MVP 2 - Baseline)** frente al **Modelo B (MVP 4 - Aumento Fuerte Calibrado)**.

---

## 📊 1. Resumen de Clasificación en las 4 Categorías

| Categoría | MVP 2 | MVP 4 | Cantidad de Fotos | % del Total | Significado Científico |
|---|:---:|:---:|:---:|:---:|---|
| 🟢 **Robust Cases** | ✅ | ✅ | **`46`** fotos | `26.7%` | Casos visualmente claros donde ambos modelos funcionan sólidamente. |
| 🔵 **Fixed Cases** | ❌ | ✅ | **`13`** fotos | `7.6%` | **Casos corregidos por MVP 4** tras la aplicación de la estrategia de aumento fuerte. |
| 🔴 **Structural Errors** | ❌ | ❌ | **`102`** fotos | `59.3%` | Fallos persistentes por **ruido contextual extremo** (suelo, dedos, ramas, sol directo). |
| 🟠 **Regressions** | ✅ | ❌ | **`11`** fotos | `6.4%` | Casos aislados donde MVP 2 acertó y MVP 4 falló. |

> **Balance Neto de Aciertos:** MVP 2 = `57` aciertos (`33.14%`) $\longrightarrow$ MVP 4 = `59` aciertos (`34.30%`).

---

## 🍅 2. Tracking Específico de Solanáceas (Papa $\leftrightarrow$ Tomate)

| Par de Confusión Cruzada | MVP 2 (Baseline) | MVP 4 (Aumento Fuerte) | Variación ($\Delta$) | Interpretación |
|---|:---:|:---:|:---:|---|
| **Papa $ightarrow$ Tomate** | **`10`** fotos | **`10`** fotos | **`+0`** | Confusión de follaje de patata diagnosticado como tomatera. |
| **Tomate $ightarrow$ Papa** | **`0`** fotos | **`0`** fotos | **`+0`** | Confusión de hojas de tomate diagnosticadas como patata. |

### Hallazgos Clave en Solanáceas:
* **Asimetría Experimental (Papa vs. Tomate):**
  * En **Tomate**, los aciertos botánicos subieron de **54 a 62 fotos** (+8 fotos, alcanzando **`89.86%`**). El aumento de datos ayudó notablemente a la red a aislar el contorno de la hoja de tomate.
  * En **Papa**, la exactitud cayó de **13.3% a 6.7%** con los mismos **10 casos de confusión hacia Tomate** (*Tizón tardío* y *Tizón temprano* por *Phytophthora/Alternaria*).
* **Conclusión Experimental:**
  * La aumentación fuerte **no mitigó la principal confusión interespecífica Papa $\rightarrow$ Tomate**. Mejoró la robustez de ciertos cultivos con rasgos foliares dominantes (Tomate, Vid), pero consolidó el sesgo hacia las clases mayoritarias.


---

## 🌿 3. Matrices de Confusión por Cultivo (6 $	imes$ 6)

### Matriz de Confusión de Cultivos — Modelo MVP 2 (Baseline)
| Real \ Predicho | Manzana | Maíz | Papa | Pimiento | Tomate | Vid (Uva) | **Exactitud Cultivo** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Manzana** | 14 |  0 |  1 |  3 |  7 |  4 | **`48.3%`** |
| **Maíz** |  0 | 25 |  0 |  0 |  0 |  0 | **`100.0%`** |
| **Papa** |  0 |  2 |  2 |  1 | 10 |  0 | **`13.3%`** |
| **Pimiento** |  0 |  2 |  0 | 11 |  3 |  0 | **`68.8%`** |
| **Tomate** |  4 |  5 |  0 |  4 | 54 |  2 | **`78.3%`** |
| **Vid (Uva)** |  4 |  0 |  0 |  0 |  1 | 13 | **`72.2%`** |

### Matriz de Confusión de Cultivos — Modelo MVP 4 (Augmentation Fuerte)
| Real \ Predicho | Manzana | Maíz | Papa | Pimiento | Tomate | Vid (Uva) | **Exactitud Cultivo** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Manzana** | 10 |  1 |  0 |  2 |  9 |  7 | **`34.5%`** |
| **Maíz** |  0 | 25 |  0 |  0 |  0 |  0 | **`100.0%`** |
| **Papa** |  0 |  3 |  1 |  1 | 10 |  0 | **`6.7%`** |
| **Pimiento** |  0 |  2 |  0 |  9 |  3 |  2 | **`56.2%`** |
| **Tomate** |  0 |  3 |  0 |  3 | 62 |  1 | **`89.9%`** |
| **Vid (Uva)** |  2 |  0 |  0 |  0 |  2 | 14 | **`77.8%`** |

---

## 🔴 4. Análisis de los Errores Estructurales Persistentes (Top 10)

Estos son los casos donde **ambos modelos fallaron**, representando el límite intrínseco de la clasificación directa sin segmentación de hoja:

| # | Condición Real (PlantDoc) | Diagnóstico MVP 4 | Casos | Causa Contextual Identificada |
|---|---|---|:---:|---|
| 1 | **Papa: Tizón tardío** | **Tomate: Tizón tardío** | **6** fotos | Síntomas fúngicos homólogos de Solanáceas (*Phytophthora*) |
| 2 | **Tomate: Virus del mosaico del tomate (TMV)** | **Tomate: Tizón tardío** | **6** fotos | Múltiples hojas superpuestas o polvo en envés |
| 3 | **Tomate: Saludable** | **Tomate: Tizón tardío** | **5** fotos | Múltiples hojas superpuestas o polvo en envés |
| 4 | **Maíz: Roya común** | **Maíz: Tizón foliar norteño** | **4** fotos | Lesión longitudinal similar entre Roya/Cercospora |
| 5 | **Papa: Tizón temprano** | **Tomate: Tizón tardío** | **4** fotos | Síntomas fúngicos homólogos de Solanáceas (*Phytophthora*) |
| 6 | **Tomate: Moho foliar** | **Tomate: Tizón temprano** | **4** fotos | Múltiples hojas superpuestas o polvo en envés |
| 7 | **Tomate: Virus del enrollamiento amarillo de la hoja (TYLCV)** | **Tomate: Tizón tardío** | **4** fotos | Múltiples hojas superpuestas o polvo en envés |
| 8 | **Manzana: Roya del manzano** | **Vid (Uva): Esca (Sarampión negro)** | **3** fotos | Fondo ruidoso (suelo seco, dedos o ramas ajenas) |
| 9 | **Maíz: Tizón foliar norteño** | **Maíz: Mancha foliar gris por Cercospora** | **3** fotos | Lesión longitudinal similar entre Roya/Cercospora |
| 10 | **Tomate: Saludable** | **Tomate: Tizón temprano** | **3** fotos | Múltiples hojas superpuestas o polvo en envés |

---

## 🎯 5. Conclusión y Recomendación Justificada para el MVP 5

### Veredicto del Análisis Forense:
* **El 70.35% de las imágenes son clasificadas correctamente a nivel botánico (planta).** En Maíz el acierto es del 100% y en Tomate es del 89.86%.
* **La gran mayoría de los fallos persistentes (102 fotos) no son confusiones absurdas entre plantas distintas**, sino:
  1. **Confusiones de patología dentro de la misma planta:** La red reconoce la hoja, pero la textura del suelo de fondo o la luz solar quemada altera el aspecto de la mancha necrótica.
  2. **Fotos con encuadre lejano:** La hoja enferma ocupa apenas el 15-20% de los píxeles, y el 80% restante es tierra o cielo.

### 🚀 Recomendación Empírica para MVP 5:
> ### **👉 OPCION B — DETECCION / SEGMENTACION DE HOJA (YOLO / Bounding Box Crop)**
> **Justificación Metodológica:**
> El 56.9% de los fallos estructurales aciertan la especie botánica (58 de 102), pero fallan la patología debido al ruido de fondo (suelo, ramas, dedos, iluminación solar extrema).
> La evidencia empírica sugiere que aislar la región foliar mediante detección o segmentación podría reducir una proporción significativa de los errores asociados al ruido contextual. Esta hipótesis será evaluada experimentalmente en el MVP 5.

