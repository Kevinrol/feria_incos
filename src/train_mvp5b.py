import sys
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.augmentations_crop_aware import get_crop_aware_augmentation

def build_crop_aware_model(num_classes=27):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3), name="input_layer")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.25, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs, outputs, name="AgroVision_MobileNetV2_MVP5B_CropAware")
    return model, base_model

def train_mvp5b(
    data_dir="datasets/plantvillage_mvp2",
    model_save_path="models/agrovision_mobilenetv2_mvp5b.keras",
    history_save_path="models/training_history_mvp5b.json",
    plot_save_path="results/training_history_mvp5b.png",
    batch_size=32,
    epochs_stage1=10,
    epochs_stage2=10,
    patience=3
):
    print("=" * 80)
    print("  AGROVISION MVP 5-B — CROP-AWARE FINE-TUNING (ADAPTACION FOLIAR)")
    print("=" * 80)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Directorios no encontrados en '{data_dir}'.")

    # 1. Cargar datasets de PlantVillage
    print(f"\n[+] Cargando dataset de entrenamiento: {train_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=42
    )

    print(f"[+] Cargando dataset de validación: {val_dir}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"[+] Total de clases configuradas: {num_classes}")

    # 2. Pipeline de Data Augmentation Crop-Aware
    crop_aware_augmentation = get_crop_aware_augmentation()
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(crop_aware_augmentation(x, training=True)), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(buffer_size=AUTOTUNE)

    # 3. Construir el Modelo
    model, base_model = build_crop_aware_model(num_classes=num_classes)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    # 4. FASE 1: Feature Extraction
    print("\n" + "=" * 80)
    print(f"  FASE 1: EXTRACCION DE CARACTERISTICAS (Base Congelada, {epochs_stage1} épocas máx)")
    print("=" * 80)

    cb_stage1 = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=cb_stage1
    )

    # 5. FASE 2: Crop-Aware Fine-Tuning
    print("\n" + "=" * 80)
    print(f"  FASE 2: CROP-AWARE FINE-TUNING (Descongelando 30 capas, {epochs_stage2} épocas máx)")
    print("=" * 80)

    base_model.trainable = True
    # Descongelar las últimas 30 capas
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb_stage2 = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    initial_epoch = len(history1.history["loss"])
    total_epochs = initial_epoch + epochs_stage2

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=cb_stage2
    )

    # 6. Unificar y Guardar Historial
    full_history = {
        "accuracy": history1.history["accuracy"] + history2.history["accuracy"],
        "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
        "loss": history1.history["loss"] + history2.history["loss"],
        "val_loss": history1.history["val_loss"] + history2.history["val_loss"],
        "stage1_epochs": initial_epoch,
        "stage2_epochs": len(history2.history["loss"])
    }

    with open(history_save_path, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=4)
    print(f"\n[OK] Historial de entrenamiento guardado en: {history_save_path}")

    # 7. Graficar Curvas
    plot_training_history(full_history, initial_epoch, plot_save_path)
    print(f"[OK] Gráfico de curvas guardado en: {plot_save_path}")
    print(f"[OK] Modelo MVP 5-B guardado con éxito en: {model_save_path}")
    print("=" * 80 + "\n")

def plot_training_history(history, split_epoch, save_path):
    epochs_range = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(14, 5))

    # Exactitud
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["accuracy"], label="Entrenamiento (Crop-Aware)", color="#2A9D8F", lw=2)
    plt.plot(epochs_range, history["val_accuracy"], label="Validación", color="#E76F51", lw=2)
    plt.axvline(x=split_epoch + 0.5, color="gray", linestyle="--", label="Inicio Fine-Tuning")
    plt.title("Exactitud (Accuracy) — AgroVision MVP 5-B", fontsize=12, fontweight="bold")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["loss"], label="Pérdida Entrenamiento", color="#2A9D8F", lw=2)
    plt.plot(epochs_range, history["val_loss"], label="Pérdida Validación", color="#E76F51", lw=2)
    plt.axvline(x=split_epoch + 0.5, color="gray", linestyle="--", label="Inicio Fine-Tuning")
    plt.title("Pérdida (Loss) — AgroVision MVP 5-B", fontsize=12, fontweight="bold")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar MobileNetV2 con Crop-Aware Fine-Tuning (MVP 5-B)")
    parser.add_argument("--epochs1", type=int, default=10, help="Épocas Fase 1 (Feature Extraction)")
    parser.add_argument("--epochs2", type=int, default=10, help="Épocas Fase 2 (Fine-Tuning)")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--patience", type=int, default=3, help="Paciencia para EarlyStopping")
    args = parser.parse_args()

    train_mvp5b(
        epochs_stage1=args.epochs1,
        epochs_stage2=args.epochs2,
        batch_size=args.batch_size,
        patience=args.patience
    )
