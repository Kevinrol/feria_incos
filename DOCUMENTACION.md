# 🌿 AgroVision — Documentación del Proyecto
## Sistema de Diagnóstico de Enfermedades en Plantas mediante Inteligencia Artificial

> **Para quién es este documento:**  
> Este documento está escrito para cualquier persona — no necesitas saber programación ni inteligencia artificial para entenderlo. Explicamos todo desde cero.

---

## 📋 Tabla de Contenidos

1. [¿Qué es AgroVision?](#1-qué-es-agrovision)
2. [¿Qué problema resuelve?](#2-qué-problema-resuelve)
3. [¿Qué es la Inteligencia Artificial?](#3-qué-es-la-inteligencia-artificial)
4. [¿Qué es el Aprendizaje Profundo (Deep Learning)?](#4-qué-es-el-aprendizaje-profundo-deep-learning)
5. [¿Cómo ve una imagen una computadora?](#5-cómo-ve-una-imagen-una-computadora)
6. [¿Con qué datos fue entrenado el sistema?](#6-con-qué-datos-fue-entrenado-el-sistema)
7. [¿Cómo se entrenó el modelo?](#7-cómo-se-entrenó-el-modelo)
8. [Los modelos del proyecto](#8-los-modelos-del-proyecto)
9. [¿Cómo usar el sistema?](#9-cómo-usar-el-sistema)
10. [Resultados y precisión](#10-resultados-y-precisión)
11. [Limitaciones honestas](#11-limitaciones-honestas)
12. [Glosario de términos](#12-glosario-de-términos)

---

## 1. ¿Qué es AgroVision?

**AgroVision** es un sistema de inteligencia artificial que puede analizar una fotografía de una planta y diagnosticar si está enferma, y en caso de estarlo, identificar qué enfermedad tiene.

Funciona de manera similar a cómo un médico mira síntomas en un paciente, pero en este caso:
- El **"médico"** es un programa de computadora entrenado con miles de imágenes
- El **"paciente"** es una planta
- Los **"síntomas"** son las manchas, colores y texturas que aparecen en las hojas

**¿Qué plantas puede diagnosticar?**

| Planta | Enfermedades que detecta |
|---|---|
| 🥔 **Papa** | Tizón tardío, Tizón temprano, Sana |
| 🍅 **Tomate** | Tizón tardío, Tizón temprano, Mancha bacteriana, Moho foliar, Mancha de Septoria, Ácaro araña, Mancha diana, Virus enrollado, Virus del mosaico, Sana |
| 🍎 **Manzana** | Sarna, Podredumbre negra, Roya, Sana |
| 🌽 **Maíz** | Mancha gris, Roya común, Tizón del norte, Sano |
| 🍇 **Uva** | Podredumbre negra, Esca, Tizón foliar, Sana |
| 🌶️ **Pimiento** | Mancha bacteriana, Sano |

---

## 2. ¿Qué problema resuelve?

En el Perú y en toda Latinoamérica, las enfermedades en los cultivos representan una de las principales causas de pérdidas económicas para los agricultores.

**🔴 Diagnóstico tardío:** Un agricultor puede notar que sus plantas están enfermas, pero no saber exactamente qué enfermedad tienen ni cómo tratarla. Para cuando llega un agrónomo, la enfermedad ya se extendió.

**🔴 Falta de acceso a expertos:** En zonas rurales alejadas, no siempre hay agrónomos disponibles para evaluar los cultivos en persona.

**🔴 Uso incorrecto de fungicidas:** Sin un diagnóstico correcto, los agricultores usan productos que no corresponden a la enfermedad real, gastando dinero y dañando el suelo.

**✅ La solución de AgroVision:** Con solo tomar una foto con el celular y ejecutar el programa, se puede obtener un diagnóstico en segundos, identificando la enfermedad con el nombre técnico correcto.

---

## 3. ¿Qué es la Inteligencia Artificial?

Imagina que tienes un amigo que nunca olvida nada. Si le muestras miles de fotos de manzanas diciéndole "esto es una manzana", y miles de fotos de naranjas diciéndole "esto es una naranja", eventualmente tu amigo aprenderá a distinguirlas solo con mirarlas, incluso si nunca antes había visto esa manzana o naranja en particular.

La **Inteligencia Artificial (IA)** funciona de manera similar, pero en lugar de un amigo humano, es un programa de computadora que:

1. Recibe muchos ejemplos de algo (en nuestro caso, miles de fotos de plantas enfermas y sanas)
2. Encuentra patrones y características que distinguen cada caso
3. Aprende a identificar nuevos casos que nunca ha visto antes

> 💡 **Analogía simple:** La IA es como enseñarle a un niño a reconocer animales mostrándole muchas fotos. Después de ver suficientes ejemplos, el niño reconocerá un perro aunque nunca haya visto a ese perro en particular.

### ¿La IA "piensa" como los humanos?

**No.** La IA no piensa ni siente. Es un sistema matemático muy sofisticado que encuentra patrones en los números. Una imagen, para una computadora, no es "una hoja con manchas marrones" — es una tabla de millones de números que representan el color de cada píxel.

---

## 4. ¿Qué es el Aprendizaje Profundo (Deep Learning)?

El **Aprendizaje Profundo** (en inglés: *Deep Learning*) es una técnica específica dentro de la Inteligencia Artificial que imita, de forma muy simplificada, cómo funciona el cerebro humano.

### La inspiración: el cerebro humano

El cerebro humano está compuesto por miles de millones de **neuronas** — células que se conectan entre sí y transmiten señales eléctricas. Cuando aprendemos algo nuevo, estas conexiones se fortalecen o debilitan.

El Aprendizaje Profundo usa **redes neuronales artificiales**: programas matemáticos que imitan esta estructura en capas interconectadas.

### ¿Cómo funciona una Red Neuronal?

Imagina una fábrica con varias líneas de producción en serie:

```
IMAGEN DE ENTRADA
       ↓
┌─────────────────────────────────────────────┐
│  CAPA 1: Detecta bordes y líneas            │
│  (¿Hay una línea horizontal aquí?)          │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  CAPA 2: Detecta formas básicas             │
│  (¿Hay un círculo o una mancha aquí?)       │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  CAPA 3: Detecta texturas                   │
│  (¿Hay una textura rugosa o lisa?)          │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  CAPA 4: Detecta patrones complejos         │
│  (¿Hay manchas necróticas con borde amarillo?)│
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  CAPA FINAL: Decide la clase                │
│  → "Papa - Tizón tardío (87% confianza)"    │
└─────────────────────────────────────────────┘
```

La palabra **"Profundo"** viene de que hay muchas capas (en nuestro modelo hay más de 100 capas internas). Cuantas más capas, más cosas complejas puede aprender el modelo.

### ¿Qué tipo de aprendizaje usa AgroVision?

AgroVision usa **Aprendizaje Supervisado** — el tipo más común de aprendizaje automático.

**¿Qué significa supervisado?**

Es como enseñar con un libro de respuestas:
- Se le muestra una imagen al modelo
- Se le dice la respuesta correcta: "Esto es tizón tardío en papa"
- Si el modelo se equivoca, se le corrige ajustando sus parámetros internos
- Se repite esto millones de veces hasta que el modelo aprende

La "supervisión" es que siempre hay un humano (o una base de datos creada por humanos) que dice cuál es la respuesta correcta durante el entrenamiento.

---

## 5. ¿Cómo ve una imagen una computadora?

### Píxeles y números

Una imagen digital está formada por **píxeles** (puntos de color). Cada píxel tiene 3 valores numéricos:
- **R** (Rojo): un número del 0 al 255
- **G** (Verde): un número del 0 al 255
- **B** (Azul): un número del 0 al 255

Una imagen de 224×224 píxeles (el tamaño que usa AgroVision) contiene:
> **224 × 224 × 3 = 150,528 números**

Para la computadora, una fotografía de una hoja enferma es simplemente una tabla de 150,528 números. La magia del Deep Learning está en aprender qué combinaciones de estos números corresponden a qué enfermedades.

### ¿Por qué 224×224?

Los modelos de AgroVision redimensionan todas las imágenes a 224×224 píxeles antes de procesarlas. Esto es un equilibrio entre:
- **Tamaño suficiente** para ver los detalles de la enfermedad
- **Tamaño manejable** para que la computadora lo procese rápido

---

## 6. ¿Con qué datos fue entrenado el sistema?

Un modelo de Deep Learning es tan bueno como los datos con los que fue entrenado. AgroVision usó dos fuentes de datos complementarias:

### 📚 Fuente 1: PlantVillage Dataset

**¿Qué es?** Una base de datos académica creada por investigadores de la Universidad de Penn State (USA) con imágenes de plantas enfermas tomadas en condiciones controladas de laboratorio.

| Característica | Detalle |
|---|---|
| **Creadores** | Universidad de Penn State — EEUU |
| **Total de imágenes** | ~87,000 imágenes |
| **Enfermedades** | 38 clases de enfermedades en 14 tipos de plantas |
| **Condición** | Fotos en fondo blanco, iluminación perfecta |
| **Uso** | Entrenamiento principal |

**¿Cómo se ven estas imágenes?**  
Fotos muy "limpias" tomadas en laboratorio: la hoja está sobre un fondo blanco uniforme, con buena iluminación. Son perfectas para enseñar al modelo qué aspecto tiene cada enfermedad en condiciones ideales.

### 🌿 Fuente 2: PlantDoc Dataset

**¿Qué es?** Un conjunto de imágenes de plantas enfermas tomadas en condiciones reales del campo — fotos con fondos complejos, diferentes iluminaciones, múltiples hojas visibles.

| Característica | Detalle |
|---|---|
| **Creadores** | Investigadores de India y EE.UU. |
| **Total de imágenes** | ~2,600 imágenes |
| **Condición** | Fotos reales de campo, condiciones variables |
| **Uso** | Mejora de generalización (para que el modelo funcione en condiciones reales) |

### ¿Por qué usar dos fuentes?

El problema con usar solo PlantVillage es que el modelo aprendería a diagnosticar enfermedades solo en fotos perfectas de laboratorio. En el campo real, las fotos son diferentes: hay sombras, el fondo no es blanco, hay múltiples hojas.

Al combinar ambas fuentes, el modelo aprende tanto los síntomas detallados (PlantVillage) como a reconocerlos en condiciones reales (PlantDoc).

### Estadísticas del dataset de entrenamiento

```
┌─────────────────────────────────────────────────┐
│         DATASET HÍBRIDO AGROVISION              │
├─────────────────────────────────────────────────┤
│  Total imágenes de entrenamiento: 27,654        │
│  Total imágenes de validación:     5,623        │
│  PlantVillage (lab):              ~23,700 imgs  │
│  PlantDoc (campo real):            ~3,954 imgs  │
│  Clases / Enfermedades:               27        │
└─────────────────────────────────────────────────┘
```

> **Nota:** La validación es el "examen final": imágenes que el modelo **nunca vio** durante el entrenamiento. Si el modelo funciona bien con estas imágenes nuevas, confirmamos que realmente aprendió y no solo "memorizó" los ejemplos.

---

## 7. ¿Cómo se entrenó el modelo?

El entrenamiento de LeafNet-kallpa (el modelo más avanzado de AgroVision) siguió un proceso de **3 fases progresivas**. Esta técnica se llama **Transfer Learning** (Aprendizaje por Transferencia).

### ¿Qué es Transfer Learning?

Imagina que quieres enseñarle a alguien a identificar enfermedades en papa. En lugar de empezar desde cero (enseñándole primero qué es una planta, luego qué es una hoja, etc.), puedes contratar a alguien que **ya sabe de biología** y solo entrenarle específicamente en enfermedades de papa.

El Transfer Learning hace lo mismo:
1. Se parte de un modelo que ya "sabe ver imágenes" (entrenado en millones de fotos generales)
2. Se especializa ese conocimiento para identificar enfermedades en plantas

El modelo base que usamos se llama **EfficientNetB0** — fue desarrollado por Google y entrenado con 14 millones de imágenes generales del mundo. Ya sabe detectar formas, texturas, colores y objetos.

### Las 3 Fases de Entrenamiento

```
FASE 1: CALENTAMIENTO (3 épocas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• El modelo base (EfficientNetB0) está CONGELADO
• Solo se entrena la "cabeza" nueva (las últimas capas)
• Como contratar a un experto en biología y solo
  enseñarle los nombres de las enfermedades
• Duración: ~20 minutos
• Resultado: El modelo aprende las 27 clases básicamente

FASE 2: AJUSTE SUAVE (10 épocas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Se descongelan las últimas 40 capas del modelo base
• El modelo empieza a adaptar su visión específicamente
  para hojas de plantas (en lugar de objetos generales)
• Como cuando el experto en biología empieza a notar
  detalles específicos de las plantas
• Duración: ~1.5 horas
• Resultado: Precisión mejora significativamente

FASE 3: AJUSTE FINO COMPLETO (20 épocas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• TODAS las capas del modelo se entrenan juntas
• Ajuste muy suave para no olvidar lo ya aprendido
• Se da más importancia a las clases de papa (3x más peso)
• Se usa más imágenes de campo real (60%) que de lab (40%)
• Duración: ~2 horas
• Resultado: Modelo final especializado

TOTAL: ~4-5 horas de entrenamiento en CPU
```

### ¿Qué es una época?

Una **época** (en inglés: *epoch*) es cuando el modelo procesa **todos** los ejemplos de entrenamiento una vez completa.

- Con 27,654 imágenes de entrenamiento y una época, el modelo vio las 27,654 imágenes una vez
- Con 33 épocas totales, vio cada imagen ~33 veces
- Cada vez que las ve, ajusta un poco sus parámetros internos para mejorar

### ¿Cuántos parámetros tiene el modelo?

El modelo LeafNet-kallpa tiene **4,389,566 parámetros** (aproximadamente 4.4 millones).

¿Qué es un parámetro? Son los "valores aprendidos" dentro de la red neuronal — como las "configuraciones" internas que el modelo ajusta durante el entrenamiento. Cada parámetro es un número decimal que el modelo modifica millones de veces hasta que aprende a diagnosticar correctamente.

### Técnicas especiales usadas

**🔀 Data Augmentation (Aumento de Datos)**  
Para evitar que el modelo "memorice" exactamente las fotos del dataset, se generan versiones modificadas de cada imagen:
- Se voltean horizontalmente y verticalmente
- Se rotan ligeramente
- Se cambia el brillo y contraste
- Se hace zoom in/out

Esto hace que el modelo vea "nuevas" imágenes y aprenda características más generales.

**✂️ CutMix**  
Se mezclan partes de dos imágenes diferentes en una sola. Por ejemplo: parte de una hoja con tizón tardío + parte de una hoja sana. El modelo aprende a ser más robusto con imágenes parciales o mixtas.

**⚖️ Class Weighting (Peso de Clases)**  
Como hay muchas más imágenes de tomate que de papa en el dataset, el modelo tendería naturalmente a "favorecer" al tomate. Para compensar, se le da **3 veces más importancia** a los errores en clases de papa durante el entrenamiento.

**🎯 Focal Loss**  
Una función matemática especial que hace que el modelo se concentre más en los ejemplos difíciles (los que se equivoca más) en lugar de los fáciles que ya domina.

---

## 8. Los modelos del proyecto

AgroVision tiene una historia de iteraciones — empezamos con modelos simples y fuimos mejorando. Cada versión se llama "MVP" (Mínimo Producto Viable):

| Modelo | Arquitectura | Parámetros | Precisión Lab | Precisión Campo |
|---|---|---|---|---|
| MVP 1 | MobileNetV2 | ~3.4M | ~85% | ~50% |
| MVP 2 | MobileNetV2 | ~3.4M | ~87% | ~52% |
| MVP 4 | MobileNetV2 | ~3.4M | ~87% | ~55% |
| MVP 5b | MobileNetV2 | ~3.4M | ~88% | ~54% |
| MVP 6 | MobileNetV2 | ~3.4M | ~88% | ~58% |
| **MVP 7 — LeafNet-kallpa** | **EfficientNetB0** | **~4.4M** | **~84%** | **~80% Top-3** |

### ¿Por qué "LeafNet-kallpa"?

- **LeafNet**: "Leaf" = hoja en inglés. Red neuronal para hojas.
- **Kallpa**: Palabra **quechua** que significa **fuerza, energía, potencia**. El nombre refleja el contexto andino del proyecto, orientado a ayudar a agricultores de la región.

### Formatos de exportación

El modelo final está disponible en varios formatos para diferentes usos:

| Archivo | Tamaño | Uso |
|---|---|---|
| `LeafNet_kallpa.keras` | 35.8 MB | Investigación y mejora del modelo |
| `LeafNet_kallpa.tflite` | 17.4 MB | Dispositivos móviles (calidad completa) |
| `LeafNet_kallpa_fp16.tflite` | 8.8 MB | Celulares (menor tamaño, ~misma calidad) |
| `LeafNet_kallpa_int8.tflite` | 4.9 MB | Dispositivos muy limitados |

---

## 9. ¿Cómo usar el sistema?

### Opción 1 — Diagnóstico simple (un solo modelo)

```bash
python src/predict.py --image "foto_de_mi_planta.jpg" --model models/LeafNet_kallpa.keras
```

**Resultado de ejemplo:**
```
=================================================================
                AGROVISION — DIAGNOSTICO DE LA PLANTA
=================================================================
  Planta detectada:     Papa
  Estado / Patologia:   Tizon tardio
  Nivel de Confianza:   87.3%

  Distribucion de probabilidades (Top):
   [1] Papa - Tizon tardio              87.30%
   [2] Tomate - Tizon tardio             7.12%
   [3] Papa - Tizon temprano             3.21%
=================================================================
```

### Opción 2 — Diagnóstico de alta precisión (TTA + Ensemble)

Para obtener la mayor precisión posible, se puede usar el modo avanzado que combina dos modelos y analiza cada imagen 8 veces:

```bash
python src/predict_tta_ensemble.py --image "foto_de_mi_planta.jpg"
```

Este modo:
- Analiza la imagen 8 veces con pequeñas variaciones
- Combina las predicciones de MVP6 + LeafNet-kallpa
- Promedia todos los resultados para un diagnóstico más robusto

### Consejos para mejores resultados

Para que el modelo funcione bien, la foto debe:
- ✅ Mostrar claramente las hojas afectadas
- ✅ Tener buena iluminación (luz natural o artificial uniforme)
- ✅ Estar enfocada (sin blur)
- ✅ Mostrar las manchas o síntomas claramente
- ❌ No usar fotos muy oscuras o con sombras fuertes
- ❌ No usar fotos donde la hoja esté muy lejos
- ❌ No usar fotos muy borrosas

---

## 10. Resultados y precisión

### ¿Qué significa "precisión" en este contexto?

Cuando decimos que el modelo tiene 84% de precisión, significa que de cada 100 imágenes que le muestras, acierta correctamente en ~84 de ellas.

### Métricas oficiales del LeafNet-kallpa

Las métricas se evaluaron con imágenes que el modelo **nunca vio** durante el entrenamiento:

```
PlantVillage (fotos de laboratorio):
  → Top-1:  83.98%  (acierta en el primer intento el 84% de las veces)
  → Top-3: ~97%     (la respuesta está en las 3 primeras opciones)

PlantDoc (fotos reales de campo — las más difíciles):
  → Top-1:  51.16%  (fotos imperfectas, más difíciles)
  → Top-3:  81.98%  (la respuesta está en las 3 primeras opciones el 82%)
```

### ¿Por qué hay diferencia entre lab y campo?

Esta es la principal limitación del sistema. Las fotos de laboratorio son "perfectas" — fondo blanco, buena luz. Las fotos del campo tienen ruido, sombras, fondo complejo. El modelo que aprendió en fotos perfectas tiene más dificultad con fotos imperfectas.

> **Importante:** El **Top-3 en campo real es 82%** — esto significa que si el sistema te da las 3 enfermedades más probables, la enfermedad real estará entre esas 3 opciones el 82% de las veces. En la práctica, un agrónomo puede usar esta lista de 3 opciones como punto de partida para su diagnóstico.

---

## 11. Limitaciones honestas

Es importante ser transparentes sobre lo que el sistema **no puede hacer** o donde puede fallar:

### ⚠️ Casos de confusión conocidos

**Tizón tardío: Papa vs Tomate**  
El hongo *Phytophthora infestans* causa tizón tardío tanto en papa como en tomate. Los síntomas visuales son casi idénticos. Incluso patólogos humanos con años de experiencia a veces necesitan análisis de laboratorio para diferenciarlos. El modelo comete errores en este caso específico.

**Solución práctica:** Si el modelo dice "tizón tardío" — ya sea en papa o tomate — el tratamiento con fungicidas específicos para *Phytophthora* es similar en ambos casos.

### ⚠️ El sistema no reemplaza a un agrónomo

El sistema es una **herramienta de apoyo**, no un reemplazo. Ante una enfermedad grave o pérdidas importantes, siempre se debe consultar con un profesional.

### ⚠️ Funciona mejor con fotos de calidad

Como se mencionó, el modelo tiene mejor rendimiento con fotos bien tomadas. Una foto borrosa, muy oscura o donde no se ven bien los síntomas dará resultados menos confiables.

### ⚠️ Solo detecta las 27 enfermedades para las que fue entrenado

Si una planta tiene una enfermedad que no está en las 27 clases del modelo, el modelo dará la respuesta más cercana que conoce, que puede ser incorrecta.

---

## 12. Glosario de Términos

| Término | Definición simple |
|---|---|
| **Inteligencia Artificial (IA)** | Programas que aprenden de ejemplos para tomar decisiones |
| **Machine Learning** | Rama de la IA donde los programas aprenden solos de datos |
| **Deep Learning** | Técnica de ML usando redes neuronales con muchas capas |
| **Red Neuronal** | Programa matemático inspirado en el cerebro, con capas de "neuronas" artificiales |
| **Neurona Artificial** | Unidad matemática básica que recibe valores, los procesa y pasa el resultado |
| **Capa (Layer)** | Grupo de neuronas artificiales que procesan la información juntas |
| **Parámetros** | Los valores internos del modelo que se ajustan durante el entrenamiento |
| **Entrenamiento** | Proceso de mostrarle al modelo miles de ejemplos para que aprenda |
| **Época (Epoch)** | Una pasada completa por todos los datos de entrenamiento |
| **Validación** | Evaluación del modelo con imágenes que nunca vio |
| **Transfer Learning** | Adaptar un modelo ya entrenado para una nueva tarea específica |
| **Dataset** | Conjunto de datos (imágenes + etiquetas) usado para entrenar |
| **Precisión (Accuracy)** | Porcentaje de aciertos del modelo |
| **Top-1** | La primera predicción del modelo es la correcta |
| **Top-3** | La respuesta correcta está entre las 3 primeras predicciones |
| **EfficientNetB0** | Arquitectura de red neuronal desarrollada por Google, eficiente y precisa |
| **MobileNetV2** | Arquitectura ligera de Google, diseñada para funcionar en celulares |
| **TFLite** | Formato optimizado de TensorFlow para correr modelos en dispositivos móviles |
| **Data Augmentation** | Técnica de generar variaciones de las imágenes para mejorar el aprendizaje |
| **Ensemble** | Combinar las predicciones de varios modelos para mayor precisión |
| **TTA** | Test-Time Augmentation: analizar la misma imagen múltiples veces con variaciones |
| **Confianza** | Qué tan seguro está el modelo de su predicción (en porcentaje) |
| **PlantVillage** | Dataset de ~87,000 imágenes de plantas enfermas de la U. Penn State |
| **PlantDoc** | Dataset de ~2,600 fotos reales de plantas en campo abierto |
| **Phytophthora infestans** | Hongo causante del tizón tardío en papa y tomate |
| **Domain Shift** | Diferencia entre el tipo de imágenes usadas para entrenar vs las del mundo real |
| **Aprendizaje Supervisado** | Tipo de IA donde cada ejemplo de entrenamiento tiene etiqueta con la respuesta correcta |
| **Kallpa** | Palabra quechua que significa fuerza, energía, potencia |

---

## Información Técnica del Proyecto

| Aspecto | Detalle |
|---|---|
| **Lenguaje de programación** | Python 3.12 |
| **Framework de IA** | TensorFlow 2.x / Keras |
| **Arquitectura principal** | EfficientNetB0 (Google, 2019) |
| **Tipo de aprendizaje** | Supervisado — Clasificación multiclase |
| **Número de clases** | 27 enfermedades/estados |
| **Imágenes de entrenamiento** | 27,654 |
| **Imágenes de validación** | 5,623 |
| **Hardware de entrenamiento** | CPU (Intel, sin GPU) |
| **Tiempo de entrenamiento** | ~4-5 horas por entrenamiento completo |
| **Repositorio** | github.com/Kevinrol/feria_incos |

---

*Documento elaborado para el proyecto AgroVision — Feria INCOS*  
*Última actualización: Agosto 2026*
