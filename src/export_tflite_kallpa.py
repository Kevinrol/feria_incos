import os
import sys
import tensorflow as tf
from pathlib import Path

# Configurar stdout para UTF-8 en Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.focal_loss import SolanaceaeFocalLoss

# Custom objects required for loading the model
CUSTOM_OBJECTS = {
    "SolanaceaeFocalLoss": SolanaceaeFocalLoss
}

def export_tflite_kallpa(model_path="models/LeafNet_kallpa.keras", output_dir="models"):
    print("=" * 80)
    print("  AGROVISION LeafNet-kallpa (MVP 7) — EXPORTACIÓN TFLITE PARA FLUTTER")
    print("=" * 80)
    
    m_dir = ROOT_DIR / output_dir
    model_file = ROOT_DIR / model_path
    
    if not model_file.exists():
        print(f"[!] Error: No se encontró el modelo {model_file}")
        return
        
    print(f"[+] Cargando modelo Keras '{model_path}'...")
    model = tf.keras.models.load_model(model_file, compile=False)
    
    # Exportar Float32 (Máxima Precisión)
    print("\n[+] Exportando TFLite Float32 (Precisión Completa)...")
    converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = converter_f32.convert()
    f32_path = m_dir / "LeafNet_kallpa.tflite"
    with open(f32_path, "wb") as f:
        f.write(tflite_f32)
    print(f"    ✓ Guardado en: {f32_path} ({len(tflite_f32) / 1024 / 1024:.2f} MB)")
    
    # Exportar Float16 (Balance)
    print("\n[+] Exportando TFLite Float16 (Reducción 50%)...")
    converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_f16.target_spec.supported_types = [tf.float16]
    tflite_f16 = converter_f16.convert()
    f16_path = m_dir / "LeafNet_kallpa_fp16.tflite"
    with open(f16_path, "wb") as f:
        f.write(tflite_f16)
    print(f"    ✓ Guardado en: {f16_path} ({len(tflite_f16) / 1024 / 1024:.2f} MB)")
    
    # Exportar INT8 (Gama Baja)
    # Nota: INT8 Quantization requiere un Representative Dataset para calibrar las activaciones.
    # Como es un modelo para móviles, podemos usar un generador dummy o omitirlo si solo queremos
    # quantización dinámica. Aquí usaremos Dynamic Range Quantization.
    print("\n[+] Exportando TFLite Dynamic Range (Quantization Ligera)...")
    converter_dyn = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dyn = converter_dyn.convert()
    dyn_path = m_dir / "LeafNet_kallpa_int8.tflite"
    with open(dyn_path, "wb") as f:
        f.write(tflite_dyn)
    print(f"    ✓ Guardado en: {dyn_path} ({len(tflite_dyn) / 1024 / 1024:.2f} MB)")

    print("\n[✓] ¡Exportación completada exitosamente!")
    
if __name__ == "__main__":
    export_tflite_kallpa()
