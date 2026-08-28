import os
import sys
import json
import time
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

# Configurar stdout para UTF-8 en Windows
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

def set_seed(seed=7):
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

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
        
        print(f"\n  [Domain Breakdown] Epoch {epoch+1:02d} -> PV Val Acc: {pv_acc:.2f}% | PW Val Acc: {pw_acc:.2f}%")

def load_raw_image_and_label(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img.set_shape([224, 224, 3])
    img = tf.cast(img, tf.float32) # [0, 255]
    return img, label

def build_balanced_datasets(manifest_dir="datasets/leafnet_kallpa", batch_size=32):
    m_dir = ROOT_DIR / manifest_dir
    train_manifest_p = m_dir / "train_manifest.json"
    val_manifest_p = m_dir / "val_manifest.json"

    with open(train_manifest_p, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(val_manifest_p, "r", encoding="utf-8") as f:
        val_records = json.load(f)

    pv_train_paths = [r["path"] for r in train_records if r["domain"] == "PlantVillage"]
    pv_train_labels = [r["class_idx"] for r in train_records if r["domain"] == "PlantVillage"]
    pw_train_paths = [r["path"] for r in train_records if r["domain"] == "PlantWild"]
    pw_train_labels = [r["class_idx"] for r in train_records if r["domain"] == "PlantWild"]

    ds_pv_tr = tf.data.Dataset.from_tensor_slices((pv_train_paths, pv_train_labels)).shuffle(5000, seed=7).repeat().map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    ds_pw_tr = tf.data.Dataset.from_tensor_slices((pw_train_paths, pw_train_labels)).shuffle(2000, seed=7).repeat().map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    aug_layer = get_strong_data_augmentation_layer_v3()
    
    train_ds = tf.data.Dataset.sample_from_datasets([ds_pv_tr, ds_pw_tr], weights=[0.5, 0.5], seed=7).batch(batch_size)
    
    # Apply saturation manually and then aug layer
    train_ds = train_ds.map(lambda x, y: apply_saturation(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    # Apply cutmix
    train_ds = train_ds.map(lambda x, y: cutmix(x, y, probability=0.5), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = (len(pw_train_paths) * 2) // batch_size

    # Validation
    val_paths = [r["path"] for r in val_records]
    val_labels = [r["class_idx"] for r in val_records]
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    # EfficientNet expects [0, 255] and one-hot labels for evaluating properly if we used one-hot in training
    val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), 27)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    pv_val_paths = [r["path"] for r in val_records if r["domain"] == "PlantVillage"]
    pv_val_labels = [r["class_idx"] for r in val_records if r["domain"] == "PlantVillage"]
    pv_val_ds = tf.data.Dataset.from_tensor_slices((pv_val_paths, pv_val_labels)).map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    pv_val_ds = pv_val_ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), 27)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    pw_val_paths = [r["path"] for r in val_records if r["domain"] == "PlantWild"]
    pw_val_labels = [r["class_idx"] for r in val_records if r["domain"] == "PlantWild"]
    pw_val_ds = tf.data.Dataset.from_tensor_slices((pw_val_paths, pw_val_labels)).map(load_raw_image_and_label, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    pw_val_ds = pw_val_ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), 27)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, pv_val_ds, pw_val_ds, steps_per_epoch

def build_kallpa_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    backbone = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="swish")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(27, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="LeafNet_kallpa")
    return model, backbone

def train_kallpa(
    save_model_path="models/LeafNet_kallpa.keras",
    history_json_path="models/training_history_kallpa.json",
    batch_size=32
):
    print("=" * 80)
    print("  AGROVISION LeafNet-kallpa (MVP 7) — ENTRENAMIENTO 3 FASES (EFFICIENTNET-B0)")
    print("=" * 80)

    set_seed(7)
    train_ds, val_ds, pv_val_ds, pw_val_ds, steps_per_epoch = build_balanced_datasets(batch_size=batch_size)
    model, backbone = build_kallpa_model()
    domain_cb = MultiDomainValidationCallback(pv_val_ds, pw_val_ds)

    loss_fn = SolanaceaeFocalLoss(label_smoothing=0.1)

    # FASE 1: Warmup
    print("\n[+] FASE 1: WARMUP (3 Épocas, Base Congelada, lr=1e-4)")
    backbone.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss_fn, metrics=["accuracy"])
    model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds, epochs=3, callbacks=[domain_cb], verbose=1)

    # FASE 2: Fine-Tuning Suave
    print("\n[+] FASE 2: FINE-TUNING SUAVE (10 Épocas, Últimas 40 Capas, lr=1e-4 -> 5e-6)")
    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=1e-4, first_decay_steps=10*steps_per_epoch, t_mul=2.0, m_mul=0.9, alpha=0.05)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=loss_fn, metrics=["accuracy"])
    
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(str(ROOT_DIR / save_model_path), monitor="val_loss", save_best_only=True)
    cb_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    
    model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds, epochs=13, initial_epoch=3, callbacks=[cb_checkpoint, cb_early, domain_cb], verbose=1)

    # FASE 3: Fine-Tuning Global Completo
    print("\n[+] FASE 3: FINE-TUNING GLOBAL COMPLETO (15 Épocas, Todas las Capas, lr=1e-5 -> 1e-7)")
    backbone.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=loss_fn, metrics=["accuracy"])
    
    cb_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1)
    
    model.fit(train_ds, steps_per_epoch=steps_per_epoch, validation_data=val_ds, epochs=28, initial_epoch=13, callbacks=[cb_checkpoint, cb_early, cb_reduce_lr, domain_cb], verbose=1)

if __name__ == "__main__":
    train_kallpa()
