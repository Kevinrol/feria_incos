import tensorflow as tf
from tensorflow.keras import layers

def get_strong_data_augmentation_layer():
    """
    Pipeline de Aumento de Datos Fuerte (Strong Data Augmentation) para AgroVision MVP 4.
    Diseñado para forzar a la red a ser invariante al fondo, oclusiones, encuadres
    descentrados y variaciones extremas de luz solar de campo.

    Detalle exacto de hiperparámetros (Keras API):
    ---------------------------------------------
    1. RandomFlip("horizontal_and_vertical"):
       - Invierte aleatoriamente las imágenes tanto horizontal como verticalmente.
    
    2. RandomRotation(factor=0.055):
       - Factor 0.055 representa una fracción de 2*pi (vuelta completa de 360°).
       - Rango de rotación: [-0.055 * 360°, +0.055 * 360°] ≈ [-19.8°, +19.8°] ≈ ±20°.
    
    3. RandomZoom(height_factor=(-0.20, 0.20), width_factor=(-0.20, 0.20)):
       - Rango de zoom explícito de [-20%, +20%].
       - Valores negativos aplican zoom-in (acercamiento) y positivos aplican zoom-out (alejamiento).
    
    4. RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)):
       - Desplaza la imagen aleatoriamente hasta un ±10% de su altura y anchura.
       - Simula hojas que no están perfectamente centradas en el encuadre.
    
    5. RandomContrast(factor=0.20):
       - Escala el contraste en el intervalo [1.0 - 0.20, 1.0 + 0.20] = [0.80, 1.20].
       - Simula días nublados (bajo contraste) o sol directo con sombras duras (alto contraste).
    
    6. RandomBrightness(factor=0.20):
       - Modifica el brillo sumando o restando hasta un ±20% del rango de intensidad.
       - Simula destellos solares en la cutícula de las hojas o tomas en sombra.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="aug_v2_random_flip"),
        layers.RandomRotation(factor=0.055, fill_mode="reflect", name="aug_v2_random_rotation"),
        layers.RandomZoom(height_factor=(-0.20, 0.20), width_factor=(-0.20, 0.20), fill_mode="reflect", name="aug_v2_random_zoom"),
        layers.RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10), fill_mode="reflect", name="aug_v2_random_translation"),
        layers.RandomContrast(factor=0.20, name="aug_v2_random_contrast"),
        layers.RandomBrightness(factor=0.20, value_range=(0.0, 255.0), name="aug_v2_random_brightness")
    ], name="agrovision_strong_data_augmentation_v2")
