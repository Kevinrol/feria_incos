import os
import sys
import json
import time
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
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.augmentations_v2 import get_strong_data_augmentation_layer

def set_seed(seed=42):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

class MultiDomainValidationCallback(tf.keras.callbacks.Callback):
    """
    Callback para evaluar y monitorear por separado el rendimiento en:
    1. PlantVillage Validation (Laboratorio)
    2. PlantWild Validation (Campo Real In-The-Wild)
    """
    def __init__(self, pv_val_ds, pw_val_ds):
        super().__init__()
        self.pv_val_ds = pv_val_ds
        self.pw_val_ds = pw_val_ds
        self.pv_accs = []
        self.pw_accs = []

    def on_epoch_end(self, epoch, logs=None):
        pv_res = self.model.evaluate(self.pv_val_ds, verbose=0)
        pw_res = self.model.evaluate(self.pw_val_ds, verbose=0)
        
        pv_acc = float(pv_res[1] * 100)
        pw_acc = float(pw_res[1] * 100)
        
        self.pv_accs.append(pv_acc)
        self.pw_accs.append(pw_acc)
        
        print(f"\n  [Domain Breakdown] Epoch {epoch+1:02d} -> PV Val Acc: {pv_acc:.2f}% | PW Val Acc: {pw_acc:.2f}%")

def load_raw_image_and_label(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img.set_shape([224, 224, 3])
    img = tf.cast(img, tf.float32)
    return img, label

def build_balanced_datasets(manifest_dir="datasets/agrovision_mvp6", batch_size=32):
    m_dir = ROOT_DIR / manifest_dir
    train_manifest_p = m_dir / "train_manifest.json"
    val_manifest_p = m_dir / "val_manifest.json"

    if not train_manifest_p.exists() or not val_manifest_p.exists():
        raise FileNotFoundError(f"Manifiestos no encontrados en '{m_dir}'. Ejecute primero src/data_prep_mvp6.py.")

    with open(train_manifest_p, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(val_manifest_p, "r", encoding="utf-8") as f:
        val_records = json.load(f)

    # 1. Separar por dominio para el entrenamiento balanceado
    pv_train_paths = [r["path"] for r in train_records if r["domain"] == "PlantVillage"]
    pv_train_labels = [r["class_idx"] for r in train_records if r["domain"] == "PlantVillage"]

    pw_train_paths = [r["path"] for r in train_records if r["domain"] == "PlantWild"]
    pw_train_labels = [r["class_idx"] for r in train_records if r["domain"] == "PlantWild"]

    print(f"\n[+] Registros de Entrenamiento:")
    print(f"    • PlantVillage (Laboratorio): {len(pv_train_paths):,} imágenes")
    print(f"    • PlantWild (Campo Real):     {len(pw_train_paths):,} imágenes")

    # Dataset PlantVillage Train (en [0, 255])
    ds_pv_tr = tf.data.Dataset.from_tensor_slices((pv_train_paths, pv_train_labels))
    ds_pv_tr = ds_pv_tr.shuffle(buffer_size=5000, seed=42).repeat()
    ds_pv_tr = ds_pv_tr.map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Dataset PlantWild Train (en [0, 255])
    ds_pw_tr = tf.data.Dataset.from_tensor_slices((pw_train_paths, pw_train_labels))
    ds_pw_tr = ds_pw_tr.shuffle(buffer_size=2000, seed=42).repeat()
    ds_pw_tr = ds_pw_tr.map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    # Muestreo Balanceado 50% PV / 50% PW por batch
    aug_layer = get_strong_data_augmentation_layer()
    
    train_ds = tf.data.Dataset.sample_from_datasets(
        [ds_pv_tr, ds_pw_tr],
        weights=[0.5, 0.5],
        seed=42
    )
    train_ds = train_ds.batch(batch_size)
    # Aumentación sobre [0, 255] seguida de preprocess_input a [-1, 1]
    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(aug_layer(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # Pasos por época: 1 ciclo completo sobre el dominio real balanceado (50% PV / 50% PW)
    steps_per_epoch = (len(pw_train_paths) * 2) // batch_size
    print(f"[+] Pasos por época calibrados: {steps_per_epoch} batches ({steps_per_epoch * batch_size:,} fotos/época)")

    # 2. Validación Combinada y Desglosada (con preprocess_input)
    val_paths = [r["path"] for r in val_records]
    val_labels = [r["class_idx"] for r in val_records]
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    pv_val_paths = [r["path"] for r in val_records if r["domain"] == "PlantVillage"]
    pv_val_labels = [r["class_idx"] for r in val_records if r["domain"] == "PlantVillage"]
    pv_val_ds = tf.data.Dataset.from_tensor_slices((pv_val_paths, pv_val_labels)).map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    pv_val_ds = pv_val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    pw_val_paths = [r["path"] for r in val_records if r["domain"] == "PlantWild"]
    pw_val_labels = [r["class_idx"] for r in val_records if r["domain"] == "PlantWild"]
    pw_val_ds = tf.data.Dataset.from_tensor_slices((pw_val_paths, pw_val_labels)).map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    pw_val_ds = pw_val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    print(f"\n[+] Registros de Validación:")
    print(f"    • PlantVillage Val: {len(pv_val_paths):,} imágenes")
    print(f"    • PlantWild Val:    {len(pw_val_paths):,} imágenes")
    print(f"    • Total Val:        {len(val_paths):,} imágenes")

    return train_ds, val_ds, pv_val_ds, pw_val_ds, steps_per_epoch

def train_mvp6(
    base_model_path="models/agrovision_mobilenetv2_mvp4.keras",
    save_model_path="models/agrovision_mobilenetv2_mvp6.keras",
    history_json_path="models/training_history_mvp6.json",
    history_plot_path="results/training_history_mvp6.png",
    phase1_epochs=8,
    phase2_epochs=12,
    batch_size=32
):
    print("=" * 80)
    print("  AGROVISION MVP 6 — ENTRENAMIENTO DE ADAPTACION DE DOMINIO IN-THE-WILD")
    print("=" * 80)

    set_seed(42)

    # 1. Cargar datasets balanceados
    train_ds, val_ds, pv_val_ds, pw_val_ds, steps_per_epoch = build_balanced_datasets(batch_size=batch_size)

    # 2. Cargar modelo inicial MVP 4 intacto
    base_model_file = ROOT_DIR / base_model_path
    if not base_model_file.exists():
        raise FileNotFoundError(f"Modelo base '{base_model_file}' no encontrado.")

    print(f"\n[+] Cargando pesos iniciales desde '{base_model_file}'...")
    model = tf.keras.models.load_model(base_model_file)
    model._name = "AgroVision_MobileNetV2_MVP6"

    # Localizar backbone convolucional
    base_backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_backbone = layer
            break

    domain_cb = MultiDomainValidationCallback(pv_val_ds, pw_val_ds)

    # =========================================================================
    # FASE 1: Adaptación Suave de Cabeza Clasificadora (Base Congelada, lr=1e-4)
    # =========================================================================
    print(f"\n" + "=" * 60)
    print(f"  FASE 1: ADAPTACION SUAVE (Base Congelada, lr=1e-4, {phase1_epochs} épocas)")
    print(f"=" * 60)

    if base_backbone:
        base_backbone.trainable = False
    else:
        for layer in model.layers[:-2]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    t0 = time.time()
    history_phase1 = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=phase1_epochs,
        callbacks=[domain_cb],
        verbose=1
    )

    # =========================================================================
    # FASE 2: Fine-Tuning Multidominio (Últimas 30 Capas, lr=1e-5)
    # =========================================================================
    print(f"\n" + "=" * 60)
    print(f"  FASE 2: FINE-TUNING MULTIDOMINIO (Últimas 30 Capas, lr=1e-5, {phase2_epochs} épocas)")
    print(f"=" * 60)

    if base_backbone:
        base_backbone.trainable = True
        for layer in base_backbone.layers[:-30]:
            layer.trainable = False
        print(f"[+] Capas de backbone descongeladas: {sum(1 for l in base_backbone.layers if l.trainable)} de {len(base_backbone.layers)}")
    else:
        for layer in model.layers[-30:]:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ROOT_DIR / save_model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

    history_phase2 = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=phase1_epochs + phase2_epochs,
        initial_epoch=phase1_epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, domain_cb],
        verbose=1
    )

    total_time = time.time() - t0
    print(f"\n[OK] Entrenamiento MVP 6 finalizado en {total_time/60:.1f} minutos.")

    # Guardar modelo final si no lo guardó el checkpoint
    if not (ROOT_DIR / save_model_path).exists():
        model.save(ROOT_DIR / save_model_path)
    print(f"[OK] Modelo MVP 6 guardado en: {save_model_path}")

    # =========================================================================
    # Consolidar Historial y Gráficas
    # =========================================================================
    full_history = {
        "accuracy": history_phase1.history.get("accuracy", []) + history_phase2.history.get("accuracy", []),
        "val_accuracy": history_phase1.history.get("val_accuracy", []) + history_phase2.history.get("val_accuracy", []),
        "loss": history_phase1.history.get("loss", []) + history_phase2.history.get("loss", []),
        "val_loss": history_phase1.history.get("val_loss", []) + history_phase2.history.get("val_loss", []),
        "pv_val_accuracy": domain_cb.pv_accs,
        "pw_val_accuracy": domain_cb.pw_accs,
        "total_training_time_minutes": round(total_time / 60, 2)
    }

    with open(ROOT_DIR / history_json_path, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=4)
    print(f"[OK] Historial guardado en: {history_json_path}")

    # Generar gráfica de curvas con desglose de dominios
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    epochs_range = range(1, len(full_history["accuracy"]) + 1)

    # 1. Exactitud
    axes[0].plot(epochs_range, full_history["accuracy"], "o-", label="Train Accuracy (Balanced 50/50)", color="#2b5c8f", lw=2)
    axes[0].plot(epochs_range, full_history["val_accuracy"], "s--", label="Val Accuracy (Combined)", color="#e67e22", lw=2)
    if full_history["pv_val_accuracy"]:
        axes[0].plot(epochs_range, full_history["pv_val_accuracy"], "^-.", label="PV Val (Laboratorio)", color="#27ae60", lw=1.8)
    if full_history["pw_val_accuracy"]:
        axes[0].plot(epochs_range, full_history["pw_val_accuracy"], "v-.", label="PW Val (Campo Real)", color="#8e44ad", lw=1.8)
    axes[0].axvline(x=phase1_epochs, color="gray", linestyle=":", label="Inicio Fine-Tuning (Fase 2)")
    axes[0].set_title("Evolución de Exactitud por Dominio (MVP 6)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Épocas", fontsize=11)
    axes[0].set_ylabel("Exactitud (%)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right")

    # 2. Pérdida
    axes[1].plot(epochs_range, full_history["loss"], "o-", label="Train Loss", color="#2b5c8f", lw=2)
    axes[1].plot(epochs_range, full_history["val_loss"], "s--", label="Val Loss", color="#e74c3c", lw=2)
    axes[1].axvline(x=phase1_epochs, color="gray", linestyle=":", label="Inicio Fine-Tuning (Fase 2)")
    axes[1].set_title("Evolución de Pérdida (Loss) Multidominio (MVP 6)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Épocas", fontsize=11)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(ROOT_DIR / history_plot_path, dpi=300)
    plt.close()
    print(f"[OK] Gráfica de entrenamiento guardada en: {history_plot_path}")

if __name__ == "__main__":
    train_mvp6()
