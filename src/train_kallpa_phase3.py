"""
train_kallpa_phase3.py
======================
Retoma el entrenamiento de LeafNet-kallpa desde el checkpoint guardado,
ejecutando SOLO la Fase 3 (fine-tuning global) con mejoras de precision:

Mejoras vs version anterior:
  1. Class weighting: papa recibe 3x mas peso (balancear 3 clases vs 10 tomate)
  2. Sampling 60/40 PW/PV: mas peso a imagenes reales de campo (PlantDoc)
  3. Augmentacion mas agresiva: simula fotos reales (sombras, ruido)
  4. LR Cosine Annealing: convergencia mas suave
  5. EarlyStopping patience=6: mas tiempo para que el modelo mejore
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.augmentations_v3 import get_strong_data_augmentation_layer_v3, cutmix, apply_saturation
from src.focal_loss import SolanaceaeFocalLoss

NUM_CLASSES = 27

# Indices de clases de papa en class_names_mvp2.json
POTATO_CLASS_INDICES = [14, 15, 16]  # Potato___Early_blight, Late_blight, healthy


class MultiDomainValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, pv_val_ds, pw_val_ds):
        super().__init__()
        self.pv_val_ds = pv_val_ds
        self.pw_val_ds = pw_val_ds

    def on_epoch_end(self, epoch, logs=None):
        pv_res = self.model.evaluate(self.pv_val_ds, verbose=0)
        pw_res = self.model.evaluate(self.pw_val_ds, verbose=0)
        pv_acc = float(pv_res[1] * 100)
        pw_acc = float(pw_res[1] * 100)
        print(f"\n  [Domain] Epoch {epoch+1:02d} -> PlantVillage(lab): {pv_acc:.1f}% | PlantDoc(campo): {pw_acc:.1f}%")


def get_field_augmentation_layer():
    """Augmentacion extra agresiva para simular fotos reales de campo."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(factor=0.10, fill_mode="reflect"),
        layers.RandomZoom(height_factor=(-0.25, 0.25), fill_mode="reflect"),
        layers.RandomTranslation(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15), fill_mode="reflect"),
        layers.RandomContrast(factor=0.35),
        layers.RandomBrightness(factor=0.35, value_range=(0.0, 255.0)),
    ], name="field_augmentation")


def load_raw_image_and_label(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img.set_shape([224, 224, 3])
    img = tf.cast(img, tf.float32)
    return img, label


def build_datasets(manifest_dir="datasets/leafnet_kallpa", batch_size=32):
    m_dir = ROOT_DIR / manifest_dir
    with open(m_dir / "train_manifest.json", "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(m_dir / "val_manifest.json", "r", encoding="utf-8") as f:
        val_records = json.load(f)

    pv_train_paths  = [r["path"] for r in train_records if r["domain"] == "PlantVillage"]
    pv_train_labels = [r["class_idx"] for r in train_records if r["domain"] == "PlantVillage"]
    pw_train_paths  = [r["path"] for r in train_records if r["domain"] == "PlantWild"]
    pw_train_labels = [r["class_idx"] for r in train_records if r["domain"] == "PlantWild"]

    print(f"  Dataset: {len(pv_train_paths)} PlantVillage + {len(pw_train_paths)} PlantDoc")

    ds_pv = (tf.data.Dataset
             .from_tensor_slices((pv_train_paths, pv_train_labels))
             .shuffle(8000, seed=7).repeat()
             .map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE))

    ds_pw = (tf.data.Dataset
             .from_tensor_slices((pw_train_paths, pw_train_labels))
             .shuffle(3000, seed=7).repeat()
             .map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE))

    aug_v3    = get_strong_data_augmentation_layer_v3()
    aug_field = get_field_augmentation_layer()

    # 40% lab + 60% campo real -> mejor generalizacion
    train_ds = tf.data.Dataset.sample_from_datasets([ds_pv, ds_pw], weights=[0.40, 0.60], seed=7)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(lambda x, y: apply_saturation(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (aug_v3(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (aug_field(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: cutmix(x, y, probability=0.4), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = max((len(pw_train_paths) * 3) // batch_size, 200)

    val_paths  = [r["path"]      for r in val_records]
    val_labels = [r["class_idx"] for r in val_records]
    val_ds = (tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
              .map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size)
              .map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)), num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))

    def make_domain_val(records, domain):
        paths  = [r["path"]      for r in records if r["domain"] == domain]
        labels = [r["class_idx"] for r in records if r["domain"] == domain]
        return (tf.data.Dataset.from_tensor_slices((paths, labels))
                .map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    pv_val_ds = make_domain_val(val_records, "PlantVillage")
    pw_val_ds = make_domain_val(val_records, "PlantWild")

    return train_ds, val_ds, pv_val_ds, pw_val_ds, steps_per_epoch


def compute_class_weights(manifest_dir="datasets/leafnet_kallpa"):
    m_dir = ROOT_DIR / manifest_dir
    with open(m_dir / "train_manifest.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    counts = np.zeros(NUM_CLASSES, dtype=np.float32)
    for r in records:
        counts[r["class_idx"]] += 1

    total = counts.sum()
    base_weights = total / (NUM_CLASSES * counts + 1e-6)
    base_weights = base_weights / base_weights.mean()

    # Boost a clases de papa
    for idx in POTATO_CLASS_INDICES:
        base_weights[idx] *= 3.0

    class_weight_dict = {i: float(w) for i, w in enumerate(base_weights)}

    classes_path = ROOT_DIR / "models/class_names_mvp2.json"
    if classes_path.exists():
        class_names = json.loads(classes_path.read_text())
        print("\n  Pesos de clase (papa con boost 3x):")
        for i in POTATO_CLASS_INDICES:
            if i < len(class_names):
                print(f"    [{i}] {class_names[i]:<45} peso: {class_weight_dict[i]:.2f}x")

    return class_weight_dict


def run_phase3(
    checkpoint_path="models/LeafNet_kallpa.keras",
    save_path="models/LeafNet_kallpa.keras",
    batch_size=32,
    phase3_epochs=20,
):
    print("=" * 80)
    print("  LeafNet-kallpa -- FASE 3 MEJORADA (Precision General + Papa)")
    print("=" * 80)

    model_file = ROOT_DIR / checkpoint_path
    if not model_file.exists():
        raise FileNotFoundError(f"No se encontro el checkpoint: {model_file}")

    print("\n[+] Cargando modelo checkpoint...")
    model = tf.keras.models.load_model(str(model_file), compile=False)
    print(f"    Parametros: {model.count_params():,}")

    print("\n[+] Preparando datasets...")
    train_ds, val_ds, pv_val_ds, pw_val_ds, steps_per_epoch = build_datasets(batch_size=batch_size)
    print(f"    steps_per_epoch: {steps_per_epoch}")

    print("\n[+] Calculando pesos de clase...")
    class_weights = compute_class_weights()

    # Descongelar todas las capas
    model.trainable = True

    loss_fn = SolanaceaeFocalLoss(label_smoothing=0.05, alpha=0.85, gamma=2.0)

    total_steps = steps_per_epoch * phase3_epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-5,
        decay_steps=total_steps,
        alpha=3e-8 / 3e-5,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(ROOT_DIR / save_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    cb_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )
    cb_domain = MultiDomainValidationCallback(pv_val_ds, pw_val_ds)
    cb_csv = tf.keras.callbacks.CSVLogger(
        str(ROOT_DIR / "models/kallpa_phase3_history.csv"), append=False
    )

    print("\n[+] Iniciando FASE 3 MEJORADA...")
    print(f"    LR: 3e-5 cosine decay -> 3e-8 en {phase3_epochs} epocas")
    print(f"    Sampling: 40% PlantVillage + 60% PlantDoc")
    print(f"    Papa boost: 3x en funcion de perdida\n")

    start = time.time()
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=phase3_epochs,
        class_weight=class_weights,
        callbacks=[cb_checkpoint, cb_early, cb_domain, cb_csv],
        verbose=1,
    )
    elapsed = (time.time() - start) / 3600

    best_val_acc  = max(history.history.get("val_accuracy", [0])) * 100
    best_val_loss = min(history.history.get("val_loss", [99]))
    epochs_run    = len(history.history["val_loss"])

    print("\n" + "=" * 80)
    print("  FASE 3 COMPLETADA")
    print("=" * 80)
    print(f"  Epocas ejecutadas:  {epochs_run}/{phase3_epochs}")
    print(f"  Mejor val_accuracy: {best_val_acc:.2f}%")
    print(f"  Mejor val_loss:     {best_val_loss:.4f}")
    print(f"  Tiempo total:       {elapsed:.1f}h")
    print(f"  Modelo guardado:    {ROOT_DIR / save_path}")
    print("=" * 80)

    hist_path = ROOT_DIR / "models/kallpa_phase3_history.json"
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"  Historial:          {hist_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LeafNet-kallpa Phase 3 mejorada")
    parser.add_argument("--checkpoint", type=str, default="models/LeafNet_kallpa.keras")
    parser.add_argument("--save",       type=str, default="models/LeafNet_kallpa.keras")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch",      type=int, default=32)
    args = parser.parse_args()

    run_phase3(
        checkpoint_path=args.checkpoint,
        save_path=args.save,
        batch_size=args.batch,
        phase3_epochs=args.epochs,
    )
