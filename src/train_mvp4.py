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

# Asegurar que la raíz del proyecto esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.augmentations_v2 import get_strong_data_augmentation_layer

def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

def build_data_pipeline(dataset_dir="datasets/plantvillage_mvp2", img_size=(224, 224), batch_size=32):
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Directorios no encontrados en '{dataset_dir}'. Ejecute primero src/data_prep.py.")

    print(f"\n[+] Cargando conjuntos desde '{dataset_dir}' (batch_size={batch_size})...")
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
    print(f"\n[OK] Clases detectadas para MVP 4 ({len(class_names)} clases):")
    for idx, name in enumerate(class_names):
        print(f"    [{idx:02d}] {name}")

    strong_augmentation = get_strong_data_augmentation_layer()

    # Preprocesamiento con Aumento Fuerte en Entrenamiento
    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(strong_augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

def build_model(num_classes, img_size=(224, 224)):
    print("\n[+] Construyendo arquitectura MobileNetV2 para MVP 4 (pesos ImageNet)...")
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3), name="input_layer")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.25, name="dropout_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="AgroVision_MobileNetV2_MVP4")
    return model, base_model

def plot_history(history_stage1, history_stage2, save_path="results/training_history_mvp4.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    acc = history_stage1.history.get('accuracy', []) + history_stage2.history.get('accuracy', [])
    val_acc = history_stage1.history.get('val_accuracy', []) + history_stage2.history.get('val_accuracy', [])
    loss = history_stage1.history.get('loss', []) + history_stage2.history.get('loss', [])
    val_loss = history_stage1.history.get('val_loss', []) + history_stage2.history.get('val_loss', [])

    epochs_stage1 = len(history_stage1.history.get('accuracy', []))
    total_epochs = len(acc)

    plt.figure(figsize=(13, 5))

    # Exactitud
    plt.subplot(1, 2, 1)
    plt.plot(range(1, total_epochs + 1), acc, label='Train Accuracy (Aug Fuerte)', color='#2b8a3e', lw=2)
    plt.plot(range(1, total_epochs + 1), val_acc, label='Val Accuracy', color='#1c7ed6', lw=2)
    plt.axvline(x=epochs_stage1, color='gray', linestyle='--', label='Inicio Fine-Tuning')
    plt.title('AgroVision MVP 4: Evolución de Exactitud (Accuracy)', fontsize=12, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(range(1, total_epochs + 1), loss, label='Train Loss', color='#e03131', lw=2)
    plt.plot(range(1, total_epochs + 1), val_loss, label='Val Loss', color='#f08c00', lw=2)
    plt.axvline(x=epochs_stage1, color='gray', linestyle='--', label='Inicio Fine-Tuning')
    plt.title('AgroVision MVP 4: Evolución de Pérdida (Loss)', fontsize=12, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[OK] Gráfico de curvas guardado en: {save_path}")

def train_mvp4(
    dataset_dir="datasets/plantvillage_mvp2",
    models_dir="models",
    results_dir="results",
    batch_size=32,
    max_epochs1=10,
    max_epochs2=10,
    patience=3
):
    print("=" * 75)
    print("  AGROVISION MVP 4 — ENTRENAMIENTO ROBUSTO CON AUMENTACIÓN FUERTE")
    print("=" * 75)

    set_seed(42)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "agrovision_mobilenetv2_mvp4.keras")
    history_save_path = os.path.join(models_dir, "training_history_mvp4.json")

    # 1. Cargar pipeline de datos
    train_ds, val_ds, class_names = build_data_pipeline(dataset_dir=dataset_dir, batch_size=batch_size)

    # 2. Construir modelo
    model, base_model = build_model(num_classes=len(class_names))

    # -----------------------------------------------------------
    # ETAPA 1: Feature Extraction con EarlyStopping Dinámico
    # -----------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"  ETAPA 1: FEATURE EXTRACTION (Hasta {max_epochs1} épocas, patience={patience}, lr=1e-4)")
    print("=" * 75)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs1,
        callbacks=callbacks_stage1
    )

    # -----------------------------------------------------------
    # ETAPA 2: Fine-Tuning con EarlyStopping Dinámico
    # -----------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"  ETAPA 2: FINE-TUNING (Hasta {max_epochs2} épocas, patience={patience}, lr=1e-5)")
    print("=" * 75)

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable_count = sum([w.shape.num_elements() for w in model.trainable_weights])
    non_trainable_count = sum([w.shape.num_elements() for w in model.non_trainable_weights])
    print(f"[+] Parámetros entrenables: {trainable_count:,} | Parámetros congelados: {non_trainable_count:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs2,
        callbacks=callbacks_stage2
    )

    # Guardar historiales combinados
    combined_history = {
        "stage1": {k: [float(val) for val in v] for k, v in history1.history.items()},
        "stage2": {k: [float(val) for val in v] for k, v in history2.history.items()}
    }
    with open(history_save_path, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=4)
    print(f"\n[OK] Historial de entrenamiento guardado en: {history_save_path}")

    # Graficar curvas
    plot_history(history1, history2, save_path=os.path.join(results_dir, "training_history_mvp4.png"))
    print(f"\n[OK] Modelo MVP 4 guardado con éxito en: {model_save_path}")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar AgroVision MobileNetV2 MVP 4")
    parser.add_argument("--dataset", type=str, default="datasets/plantvillage_mvp2", help="Ruta del dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--epochs1", type=int, default=10, help="Máximo de épocas Feature Extraction")
    parser.add_argument("--epochs2", type=int, default=10, help="Máximo de épocas Fine-Tuning")
    parser.add_argument("--patience", type=int, default=3, help="Paciencia para EarlyStopping")
    args = parser.parse_args()

    train_mvp4(
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        max_epochs1=args.epochs1,
        max_epochs2=args.epochs2,
        patience=args.patience
    )
