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

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def format_display_name(raw_name):
    parts = raw_name.split("___")
    crop = "Tomate" if parts[0] == "Tomato" else ("Papa" if parts[0] == "Potato" else parts[0])
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
    if disease.lower() == "healthy":
        disease = "Saludable"
    return f"{crop}: {disease}"

def evaluate_model(
    dataset_dir="dataset",
    models_dir="models",
    results_dir="results",
    batch_size=32
):
    print("=" * 70)
    print("  EVALUACION DEL MODELO EN EL CONJUNTO TEST (EXCLUSIVO)")
    print("=" * 70)

    model_path = os.path.join(models_dir, "plant_disease_mobilenetv2.keras")
    classes_path = os.path.join(models_dir, "class_names.json")
    test_dir = os.path.join(dataset_dir, "test")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Archivo de clases no encontrado en: {classes_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Directorio de test no encontrado en: {test_dir}")

    # 1. Cargar clases
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    print(f"\n[+] Clases esperadas ({len(class_names)}):")
    for idx, c in enumerate(class_names):
        print(f"    [{idx:02d}] {c} ({format_display_name(c)})")

    # 2. Cargar conjunto de test (sin shuffle)
    print(f"\n[+] Cargando conjunto de prueba desde '{test_dir}'...")
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
    print(f"\n[+] Cargando modelo entrenado desde '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    # 4. Evaluación global
    print("\n[+] Calculando perdida y precision global en Test...")
    eval_results = model.evaluate(test_ds, verbose=1)
    test_loss = eval_results[0]
    test_acc = eval_results[1]

    # 5. Obtener predicciones y etiquetas reales
    print("\n[+] Generando predicciones para todas las muestras de Test...")
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.append(preds)
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 6. Reporte de Clasificación
    print("\n" + "=" * 70)
    print("  REPORTE DE CLASIFICACION DETALLADO POR CLASE")
    print("=" * 70)
    display_names = [format_display_name(c) for c in class_names]
    report = classification_report(y_true, y_pred, target_names=display_names, digits=4)
    print(report)

    # 7. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Greens',
        xticklabels=display_names,
        yticklabels=display_names,
        cbar_kws={'label': 'Proporcion'}
    )
    plt.title(f'Matriz de Confusion Normalizada - Test Set (Accuracy: {test_acc*100:.2f}%)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Clase Predicha', fontsize=12, fontweight='bold')
    plt.ylabel('Clase Real', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_save_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"\n[OK] Matriz de confusion guardada en: {cm_save_path}")

    # 8. Análisis de Confusiones Más Frecuentes
    print("\n" + "=" * 70)
    print("  ANALISIS DE CONFUSIONES MAS FRECUENTES (Errores del modelo)")
    print("=" * 70)
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((class_names[i], class_names[j], cm[i, j], cm_norm[i, j] * 100))

    confusions.sort(key=lambda x: x[2], reverse=True)

    if confusions:
        for true_cls, pred_cls, count, pct in confusions[:10]:
            print(f"  * Real: {format_display_name(true_cls):<28} -> Predicha: {format_display_name(pred_cls):<28} | Errores: {count:3d} ({pct:.1f}%)")
    else:
        print("  * Excelente: No se encontraron confusiones en el conjunto de prueba.")

    # Guardar métricas en JSON
    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "total_test_samples": int(len(y_true)),
        "report_text": report
    }
    metrics_json_path = os.path.join(results_dir, "test_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)

    print("\n" + "=" * 70)
    print(f"  TEST ACCURACY GLOBAL: {test_acc * 100:.2f}% | TEST LOSS: {test_loss:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluar modelo MobileNetV2 en test set")
    parser.add_argument("--dataset", type=str, default="dataset", help="Directorio del dataset")
    parser.add_argument("--models", type=str, default="models", help="Directorio de modelos")
    parser.add_argument("--results", type=str, default="results", help="Directorio de resultados")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano de batch")
    args = parser.parse_args()

    evaluate_model(
        dataset_dir=args.dataset,
        models_dir=args.models,
        results_dir=args.results,
        batch_size=args.batch_size
    )
