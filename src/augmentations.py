import tensorflow as tf
from tensorflow.keras import layers

def get_data_augmentation_layer():
    """
    Pipeline de aumento de datos robusto para AgroVision:
    Simula variaciones naturales de ángulo, zoom y luz sin distorsionar
    la morfología foliar de las plantas.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="aug_random_flip"),
        layers.RandomRotation(0.15, name="aug_random_rotation"),
        layers.RandomZoom(0.15, name="aug_random_zoom"),
        layers.RandomContrast(0.15, name="aug_random_contrast"),
        layers.RandomBrightness(factor=0.15, name="aug_random_brightness")
    ], name="agrovision_data_augmentation")
