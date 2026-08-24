import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Asegurar que la raíz del proyecto esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def evaluate_internal(
    dataset_dir="datasets/plantvillage_mvp2",
    models_dir="models",
    results_dir="results",
    batch_size=32
):
    print("=" * 75)
    print("  AGROVISION MVP 2 — EVALUACION EN TEST INTERNO (27 CLASES, 5,104 IMAGENES)")
    print("=" * 75)

    model_path = os.path.join(models_dir, "agrovision_mobilenetv2_mvp2.keras")
    classes_path = os.path.join(models_dir, "class_names_mvp2.json")
    mapping_path = os.path.join(models_dir, "class_mapping_mvp2.json")
    test_dir = os.path.join(dataset_dir, "test")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Archivo de clases no encontrado en: {classes_path}")

    # 1. Cargar metadatos
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    meta_dict = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            meta_dict = mapping_data.get("metadata", {})

    display_names = []
    for c in class_names:
        m = meta_dict.get(c, {})
        crop = m.get("crop_es", c.split("___")[0])
        dis = m.get("disease_es", c.split("___")[1].replace("_", " ").title())
        display_names.append(f"{crop}: {dis}")

    print(f"\n[+] Evaluando modelo con {len(class_names)} clases.")

    # 2. Cargar test dataset
    print(f"\n[+] Cargando test set desde '{test_dir}'...")
    test_raw_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    test_ds = test_raw_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # 3. Cargar modelo
    print(f"\n[+] Cargando modelo desde '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    # 4. Evaluación global
    eval_results = model.evaluate(test_ds, verbose=1)
    test_loss, test_acc = eval_results[0], eval_results[1]

    # 5. Predicciones completas
    print("\n[+] Generando predicciones para matriz de confusion y métricas...")
    y_true, y_pred_probs = [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 6. Reporte de Clasificación
    print("\n" + "=" * 75)
    print("  REPORTE DE CLASIFICACION DETALLADO POR CLASE (MVP 2)")
    print("=" * 75)
    report = classification_report(y_true, y_pred, target_names=display_names, digits=4)
    print(report)

    # 7. Matriz de Confusión Normalizada (27x27)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(18, 15))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=display_names,
        yticklabels=display_names,
        cbar_kws={'label': 'Proporcion'},
        annot_kws={"size": 7}
    )
    plt.title(f'AgroVision MVP 2: Matriz de Confusion (27 Clases) - Test Accuracy: {test_acc*100:.2f}%', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Clase Predicha', fontsize=12, fontweight='bold')
    plt.ylabel('Clase Real', fontsize=12, fontweight='bold')
    plt.xticks(rotation=60, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_save_path = os.path.join(results_dir, "confusion_matrix_mvp2.png")
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"\n[OK] Matriz de confusion guardada en: {cm_save_path}")

    # Guardar métricas en JSON
    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "total_test_samples": int(len(y_true)),
        "report_text": report
    }
    metrics_json_path = os.path.join(results_dir, "test_metrics_mvp2.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[OK] Metricas de evaluacion guardadas en: {metrics_json_path}")
    print("\n" + "=" * 75)
    print(f"  TEST ACCURACY GLOBAL (MVP 2): {test_acc * 100:.2f}% | TEST LOSS: {test_loss:.4f}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar AgroVision MVP 2")
    parser.add_argument("--dataset", type=str, default="datasets/plantvillage_mvp2", help="Directorio del dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano de batch")
    args = parser.parse_args()

    evaluate_internal(
        dataset_dir=args.dataset,
        batch_size=args.batch_size
    )
