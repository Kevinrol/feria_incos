# AgroVision — Especificaciones del Modelo, Preprocesamiento y Taxonomía Fitosanitaria

Documento técnico del sistema de visión artificial **AgroVision** para la detección y diagnóstico fitosanitario de cultivos agrícolas.

---

## 📌 1. Información General del Modelo (MVP 6)

* **Nombre del Modelo:** AgroVision MobileNetV2 (MVP 6 — *In-the-Wild Domain Adaptation*)
* **Archivo de Pesos:** `models/agrovision_mobilenetv2_mvp6.keras`
* **Arquitectura Base:** MobileNetV2 preentrenado en ImageNet con *fine-tuning* progresivo multidominio.
* **Datasets de Entrenamiento:**
  * **PlantVillage:** 27 clases canónicas en condiciones controladas de laboratorio.
  * **PlantWild:** 24 clases compatibles (5,625 imágenes reales de campo *in-the-wild*).
* **Benchmark de Evaluación Ciego Externo:** PlantDoc (172 imágenes de campo real, 100% ciego).
* **Rendimiento Clave:**
  * **PlantVillage Test:** `93.40%` (Preservación de laboratorio sin olvido catastrófico).
  * **PlantDoc Top-1 (Campo Ciego):** `42.44%` (+8.14 pp vs MVP 4).
  * **PlantDoc Top-3:** `71.51%` (+15.11 pp vs MVP 4).
  * **Identificación de Especie Botánica en Campo:** `77.33%`.

---

## ⚙️ 2. Pipeline de Preprocesamiento de Imágenes

El pipeline de preparación y preprocesamiento de datos se ejecuta de manera idéntica en entrenamiento e inferencia (exceptuando las capas estocásticas de *Data Augmentation* exclusivas de entrenamiento):

```
[ Imagen Cruda (JPG/PNG) ]
           │
           ▼
1. Lectura y Decodificación RGB (`tf.io.decode_image`)
           │
           ▼
2. Redimensionamiento a 224 × 224 px (`tf.image.resize`)
           │
           ▼
3. Conversión a float32 en rango [0.0, 255.0]
           │
           ▼
4. Aumento Fuerte de Datos (Strong Augmentation V2 - Solo Train)
           │
           ▼
5. Normalización MobileNetV2 a rango [-1.0, +1.0] (`preprocess_input`)
           │
           ▼
6. Ingesta por Lote Balanceado (50% PlantVillage / 50% PlantWild)
```

### Detalle Paso a Paso:

1. **Lectura y Decodificación:**
   * Carga de bytes desde disco y decodificación forzada a 3 canales de color (RGB), descartando canales alfa o animaciones.
2. **Redimensionamiento Espacial (*Resizing*):**
   * Ajuste bilinear/bicúbico a resolución estándar de **$224 \times 224 \times 3$**.
3. **Casting de Tipo de Datos:**
   * Conversión a tensores `tf.float32` con valores de píxel en $[0.0, 255.0]$.
4. **Pipeline de Aumento Fuerte de Datos (*Strong Data Augmentation V2* - En Entrenamiento):**
   * **`RandomFlip("horizontal_and_vertical")`:** Inversión aleatoria horizontal y vertical.
   * **`RandomRotation(factor=0.055, fill_mode="reflect")`:** Rotación aleatoria en un rango de $\approx \pm 20^\circ$ ($\pm 19.8^\circ$).
   * **`RandomZoom(height_factor=(-0.20, 0.20), width_factor=(-0.20, 0.20))`:** Zoom aleatorio de $\pm 20\%$ (tanto acercamiento como alejamiento).
   * **`RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10))`:** Desplazamiento horizontal/vertical de $\pm 10\%$ para simular hojas descentradas en campo.
   * **`RandomContrast(factor=0.20)`:** Escala de contraste en el rango $[0.80, 1.20]$ (simula días nublados o sol directo).
   * **`RandomBrightness(factor=0.20, value_range=(0.0, 255.0))`:** Ajuste de brillo en $\pm 20\%$ (simula sombras duras o destellos solares en la cutícula foliar).
5. **Normalización Específica para MobileNetV2:**
   * Se aplica la transformación estándar:
     $$\text{pixel}_{\text{norm}} = \frac{\text{pixel}}{127.5} - 1.0 \quad \in [-1.0, +1.0]$$
6. **Muestreo Multidominio Balanceado (*Balanced Domain Sampling*):**
   * Mediante `tf.data.Dataset.sample_from_datasets`, cada lote (*batch*) se compone en un **50% de fotos de laboratorio (PlantVillage)** y un **50% de fotos reales de campo (PlantWild)**, mitigando el sesgo de laboratorio (*Domain Shift*).

---

## 🌱 3. Especies Vegetales / Plantas Detectadas (6 Cultivos)

AgroVision está especializado en **6 cultivos agrícolas y hortícolas**:

1. 🍎 **Manzana** (*Apple* / *Malus domestica*)
2. 🌽 **Maíz** (*Corn / Maize* / *Zea mays*)
3. 🍇 **Vid / Uva** (*Grape* / *Vitis vinifera*)
4. 🫑 **Pimiento / Pimiento Morrón** (*Bell Pepper* / *Capsicum annuum*)
5. 🥔 **Papa / Patata** (*Potato* / *Solanum tuberosum*)
6. 🍅 **Tomate** (*Tomato* / *Solanum lycopersicum*)

---

## 🔬 4. Catálogo Taxonómico de Patologías Fitosanitarias (27 Clases Canónicas)

El sistema discrimina entre **21 patologías** (hongos, bacterias, virus y plagas) y **6 estados saludables**:

| ID | Nombre Canónico de Clase (Código Modelo) | Cultivo | Patología / Diagnóstico en Español | Tipo de Agente Causal |
| :---: | :---| :--- | :---| :---|
| **00** | `Apple___Apple_scab` | Manzana | Sarna del manzano | Hongo (*Venturia inaequalis*) |
| **01** | `Apple___Black_rot` | Manzana | Podredumbre negra | Hongo (*Botryosphaeria obtusa*) |
| **02** | `Apple___Cedar_apple_rust` | Manzana | Roya del manzano / del cedro | Hongo (*Gymnosporangium juniperi-virginianae*) |
| **03** | `Apple___healthy` | Manzana | **Saludable** | Sin patología evidente |
| **04** | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Maíz | Mancha foliar gris por Cercospora | Hongo (*Cercospora zeae-maydis*) |
| **05** | `Corn_(maize)___Common_rust_` | Maíz | Roya común del maíz | Hongo (*Puccinia sorghi*) |
| **06** | `Corn_(maize)___Northern_Leaf_Blight` | Maíz | Tizón foliar norteño | Hongo (*Exserohilum turcicum*) |
| **07** | `Corn_(maize)___healthy` | Maíz | **Saludable** | Sin patología evidente |
| **08** | `Grape___Black_rot` | Vid | Podredumbre negra de la vid | Hongo (*Guignardia bidwellii*) |
| **09** | `Grape___Esca_(Black_Measles)` | Vid | Esca (Sarampión negro) | Complejo fúngico (*Phaeomoniella*, *Fomitiporia*) |
| **10** | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | Vid | Tizón foliar por Isariopsis | Hongo (*Pseudocercospora cladosporioides*) |
| **11** | `Grape___healthy` | Vid | **Saludable** | Sin patología evidente |
| **12** | `Pepper,_bell___Bacterial_spot` | Pimiento | Mancha bacteriana | Bacteria (*Xanthomonas campestris pv. vesicatoria*) |
| **13** | `Pepper,_bell___healthy` | Pimiento | **Saludable** | Sin patología evidente |
| **14** | `Potato___Early_blight` | Papa | Tizón temprano | Hongo (*Alternaria solani*) |
| **15** | `Potato___Late_blight` | Papa | Tizón tardío | Oomiceto (*Phytophthora infestans*) |
| **16** | `Potato___healthy` | Papa | **Saludable** | Sin patología evidente |
| **17** | `Tomato___Bacterial_spot` | Tomate | Mancha bacteriana | Bacteria (*Xanthomonas vesicatoria / perforans*) |
| **18** | `Tomato___Early_blight` | Tomate | Tizón temprano | Hongo (*Alternaria solani*) |
| **19** | `Tomato___Late_blight` | Tomate | Tizón tardío | Oomiceto (*Phytophthora infestans*) |
| **20** | `Tomato___Leaf_Mold` | Tomate | Moho foliar | Hongo (*Passalora fulva*) |
| **21** | `Tomato___Septoria_leaf_spot` | Tomate | Mancha foliar por Septoria | Hongo (*Septoria lycopersici*) |
| **22** | `Tomato___Spider_mites Two-spotted_spider_mite` | Tomate | Ácaros / Arañuela roja de dos manchas | Plaga / Ácaro (*Tetranychus urticae*) |
| **23** | `Tomato___Target_Spot` | Tomate | Mancha diana | Hongo (*Corynespora cassiicola*) |
| **24** | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Tomate | Virus del enrollamiento amarillo de la hoja | Virus (*TYLCV*) |
| **25** | `Tomato___Tomato_mosaic_virus` | Tomate | Virus del mosaico del tomate | Virus (*TMV*) |
| **26** | `Tomato___healthy` | Tomate | **Saludable** | Sin patología evidente |

---

## 💻 5. Instrucciones de Inferencia Rápida

Para predecir una imagen utilizando el pipeline y modelo oficial:

```bash
# Diagnóstico con modelo MVP 6
python predict.py --image test-plant.jpg --model models/agrovision_mobilenetv2_mvp6.keras --top 5
```
