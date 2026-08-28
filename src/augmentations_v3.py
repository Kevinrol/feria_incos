import tensorflow as tf
from tensorflow.keras import layers

def get_strong_data_augmentation_layer_v3():
    """
    Pipeline de Aumento de Datos Fuerte V3 para LeafNet-kallpa (MVP 7).
    Aumenta la agresividad de contraste/brillo y añade saturación.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="aug_v3_random_flip"),
        layers.RandomRotation(factor=0.055, fill_mode="reflect", name="aug_v3_random_rotation"),
        layers.RandomZoom(height_factor=(-0.20, 0.20), width_factor=(-0.20, 0.20), fill_mode="reflect", name="aug_v3_random_zoom"),
        layers.RandomTranslation(height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10), fill_mode="reflect", name="aug_v3_random_translation"),
        layers.RandomContrast(factor=0.25, name="aug_v3_random_contrast"),
        # Note: In TF 2.10+, RandomBrightness can be used directly or as a layer.
        layers.RandomBrightness(factor=0.25, value_range=(0.0, 255.0), name="aug_v3_random_brightness"),
    ], name="agrovision_strong_data_augmentation_v3")

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

@tf.function
def get_box(lambda_value, image_size):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)
    cut_w = tf.cast(tf.cast(image_size[1], tf.float32) * cut_rat, tf.int32)
    cut_h = tf.cast(tf.cast(image_size[0], tf.float32) * cut_rat, tf.int32)
    
    cx = tf.random.uniform((), minval=0, maxval=image_size[1], dtype=tf.int32)
    cy = tf.random.uniform((), minval=0, maxval=image_size[0], dtype=tf.int32)
    
    bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, image_size[1])
    bby1 = tf.clip_by_value(cy - cut_h // 2, 0, image_size[0])
    bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, image_size[1])
    bby2 = tf.clip_by_value(cy + cut_h // 2, 0, image_size[0])
    
    return bby1, bbx1, bby2, bbx2

@tf.function
def cutmix(images, labels, probability=0.5, alpha=1.0, num_classes=27):
    """
    Aplica CutMix a un lote de imágenes y etiquetas.
    - images: Tensor de shape (batch_size, H, W, C)
    - labels: Tensor de shape (batch_size,) o (batch_size, num_classes)
    """
    batch_size = tf.shape(images)[0]
    
    # Decidir si aplicamos CutMix a este batch
    if tf.random.uniform(()) > probability:
        # If labels are not one-hot, make them one-hot
        if len(tf.shape(labels)) == 1:
            labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
        return images, labels
        
    # Obtener índices permutados
    permutation = tf.random.shuffle(tf.range(batch_size))
    
    # Permutar imágenes y etiquetas
    images_2 = tf.gather(images, permutation)
    if len(tf.shape(labels)) == 1:
        labels_1 = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
        labels_2 = tf.one_hot(tf.cast(tf.gather(labels, permutation), tf.int32), num_classes)
    else:
        labels_1 = labels
        labels_2 = tf.gather(labels, permutation)
        
    # Muestrear lambda de distribución Beta
    lambda_value = sample_beta_distribution(1, alpha, alpha)[0]
    
    image_size = tf.shape(images)[1:3]
    bby1, bbx1, bby2, bbx2 = get_box(lambda_value, image_size)
    
    # Crear la máscara del recorte
    mask_y = tf.range(image_size[0])
    mask_y = tf.logical_and(mask_y >= bby1, mask_y < bby2)
    mask_x = tf.range(image_size[1])
    mask_x = tf.logical_and(mask_x >= bbx1, mask_x < bbx2)
    
    mask = tf.logical_and(
        tf.expand_dims(mask_y, 1), 
        tf.expand_dims(mask_x, 0)
    )
    mask = tf.expand_dims(mask, -1)
    mask = tf.cast(mask, images.dtype)
    
    # Mezclar las imágenes
    images_mixed = images * (1 - mask) + images_2 * mask
    
    # Ajustar lambda exacto de acuerdo a la caja
    box_w = tf.cast(bbx2 - bbx1, tf.float32)
    box_h = tf.cast(bby2 - bby1, tf.float32)
    exact_lambda = 1.0 - (box_w * box_h) / tf.cast(image_size[0] * image_size[1], tf.float32)
    
    # Mezclar las etiquetas
    labels_mixed = exact_lambda * labels_1 + (1 - exact_lambda) * labels_2
    
    return images_mixed, labels_mixed

def apply_saturation(images, labels):
    # Aplica Random Saturation como preprocesamiento manual (solo en train)
    # factor = 0.2 -> Rango [0.8, 1.2]
    factor = tf.random.uniform((), 0.8, 1.2)
    # tf.image.adjust_saturation expects images in [0, 1] for best results but works on raw pixels if properly bounded
    images = tf.image.adjust_saturation(images, factor)
    return images, labels
