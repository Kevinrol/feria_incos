import tensorflow as tf
from tensorflow.keras import layers

def get_crop_aware_augmentation():
    """
    Pipeline de aumento de datos Crop-Aware y Multiescala para MVP 5-B.
    
    Parámetros documentados para reproducibilidad científica:
    - RandomZoom((-0.40, 0.40)): Simula recortes cerrados sobre zonas de lesión y cambios
      de distancia de captura (escala entre 60% y 140%).
    - RandomTranslation((-0.15, 0.15)): Simula encuadres descentrados de la hoja.
    - RandomRotation(0.055): 0.055 * 360° ≈ 19.8° (aproximadamente ±20°).
    - RandomContrast(0.20): Escala de contraste [0.80, 1.20].
    - RandomBrightness(0.20): Desplazamiento de brillo ±20% en rango [0.0, 255.0].
    - RandomFlip("horizontal"): Invarianza a la orientación de la hoja.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.055),  # ~19.8° ≈ ±20°
        layers.RandomZoom(height_factor=(-0.40, 0.40), width_factor=(-0.40, 0.40)),
        layers.RandomTranslation(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15)),
        layers.RandomContrast(factor=0.20),
        layers.RandomBrightness(factor=0.20, value_range=(0.0, 255.0))
    ], name="crop_aware_data_augmentation")

if __name__ == "__main__":
    aug = get_crop_aware_augmentation()
    print("[OK] Pipeline de aumento Crop-Aware para MVP 5-B inicializado correctamente:")
    aug.summary()
