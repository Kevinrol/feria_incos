import sys
import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

def build_data_pipeline(dataset_dir="dataset", img_size=(224, 224), batch_size=32):
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Directorios no encontrados en '{dataset_dir}'. Ejecute primero prepare_dataset.py.")

    print(f"\n[+] Cargando conjuntos desde '{dataset_dir}' con batch_size={batch_size}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"\n[OK] Clases detectadas dinamicamente ({len(class_names)}):")
    for idx, name in enumerate(class_names):
        print(f"    [{idx:02d}] {name}")

    # Data Augmentation exclusivo para entrenamiento
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ], name="data_augmentation")

    # Map preprocess_input & augmentation
    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

def build_model(num_classes, img_size=(224, 224)):
    print("\n[+] Construyendo modelo con MobileNetV2 pre-entrenado en ImageNet...")
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Congelado para Feature Extraction

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3), name="input_layer")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, name="dropout_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="PlantVillage_MobileNetV2")
    return model, base_model

def plot_history(history_stage1, history_stage2, save_path="results/training_history.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Combinar historias
    acc = history_stage1.history.get('accuracy', []) + history_stage2.history.get('accuracy', [])
    val_acc = history_stage1.history.get('val_accuracy', []) + history_stage2.history.get('val_accuracy', [])
    loss = history_stage1.history.get('loss', []) + history_stage2.history.get('loss', [])
    val_loss = history_stage1.history.get('val_loss', []) + history_stage2.history.get('val_loss', [])

    epochs_stage1 = len(history_stage1.history.get('accuracy', []))
    total_epochs = len(acc)

    plt.figure(figsize=(12, 5))

    # Subplot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, total_epochs + 1), acc, label='Train Accuracy', color='#2b8a3e', lw=2)
    plt.plot(range(1, total_epochs + 1), val_acc, label='Val Accuracy', color='#1c7ed6', lw=2)
    plt.axvline(x=epochs_stage1, color='gray', linestyle='--', label='Inicio Fine-Tuning')
    plt.title('Evolucion de la Exactitud (Accuracy)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Subplot Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, total_epochs + 1), loss, label='Train Loss', color='#e03131', lw=2)
    plt.plot(range(1, total_epochs + 1), val_loss, label='Val Loss', color='#f08c00', lw=2)
    plt.axvline(x=epochs_stage1, color='gray', linestyle='--', label='Inicio Fine-Tuning')
    plt.title('Evolucion de la Perdida (Loss)', fontsize=12, fontweight='bold')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[OK] Grafico de evolucion guardado en: {save_path}")

def train(
    dataset_dir="dataset",
    models_dir="models",
    results_dir="results",
    batch_size=32,
    stage1_epochs=8,
    stage2_epochs=6
):
    print("=" * 60)
    print("  ENTRENAMIENTO MOBILENETV2 - RECONOCIMIENTO DE PLANTAS")
    print("=" * 60)

    set_seed(42)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "plant_disease_mobilenetv2.keras")
    classes_save_path = os.path.join(models_dir, "class_names.json")
    history_save_path = os.path.join(models_dir, "training_history.json")

    # 1. Pipeline de datos
    train_ds, val_ds, class_names = build_data_pipeline(dataset_dir=dataset_dir, batch_size=batch_size)

    # Guardar nombres de clases en json
    with open(classes_save_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4, ensure_ascii=False)
    print(f"[OK] Nombres de clases guardados en: {classes_save_path}")

    # 2. Modelo
    model, base_model = build_model(num_classes=len(class_names))

    # Callbacks
    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    # -----------------------------------------------------------
    # ETAPA 1: Feature Extraction
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ETAPA 1: FEATURE EXTRACTION (Base congelada, lr=1e-4)")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=callbacks_stage1
    )

    # -----------------------------------------------------------
    # ETAPA 2: Fine-Tuning
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ETAPA 2: FINE-TUNING (Ultimas 30 capas descongeladas, lr=1e-5)")
    print("=" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_count = sum([w.shape.num_elements() for w in model.trainable_weights])
    non_trainable_count = sum([w.shape.num_elements() for w in model.non_trainable_weights])
    print(f"[+] Parametros entrenables en fine-tuning: {trainable_count:,}")
    print(f"[+] Parametros congelados: {non_trainable_count:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        callbacks=callbacks_stage2
    )

    # Guardar métricas de entrenamiento
    combined_history = {
        "stage1": {k: [float(val) for val in v] for k, v in history1.history.items()},
        "stage2": {k: [float(val) for val in v] for k, v in history2.history.items()}
    }
    with open(history_save_path, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=4)
    print(f"\n[OK] Historial de entrenamiento guardado en: {history_save_path}")

    # Guardar gráfico de curvas
    plot_history(history1, history2, save_path=os.path.join(results_dir, "training_history.png"))
    print(f"\n[OK] Modelo final optimizado guardado en: {model_save_path}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar MobileNetV2 en PlantVillage")
    parser.add_argument("--dataset", type=str, default="dataset", help="Directorio del dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano de batch")
    parser.add_argument("--epochs1", type=int, default=8, help="Epocas etapa 1 (Feature extraction)")
    parser.add_argument("--epochs2", type=int, default=6, help="Epocas etapa 2 (Fine-tuning)")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        stage1_epochs=args.epochs1,
        stage2_epochs=args.epochs2
    )
