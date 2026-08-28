import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="src.focal_loss")
class SolanaceaeFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.8, gamma=2.0, penalty_weight=0.2, label_smoothing=0.1, num_classes=27, name="solanaceae_focal_loss", reduction="sum_over_batch_size", **kwargs):
        super().__init__(name=name, reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.penalty_weight = penalty_weight
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        
        # Indices for Solanaceae confusion classes based on MVP2 mapping
        self.potato_early = 14
        self.potato_late = 15
        self.tomato_early = 18
        self.tomato_late = 19

    def call(self, y_true, y_pred):
        # y_true is expected to be one-hot or cutmix mixed probabilities
        y_true_smoothed = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / self.num_classes)
        
        # Clip predictions to prevent NaNs
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        
        # Standard Cross Entropy Loss
        ce_loss = -y_true_smoothed * tf.math.log(y_pred)
        
        # Focal Loss term
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = self.alpha * tf.pow(1.0 - pt, self.gamma)
        
        # Base focal loss
        focal_loss = focal_weight * ce_loss
        base_loss = tf.reduce_sum(focal_loss, axis=-1)
        
        # Solanaceae penalty
        # Since y_true can be mixed, we use the probability of the class as the mask weight
        is_potato_early = y_true[:, self.potato_early]
        is_potato_late = y_true[:, self.potato_late]
        is_tomato_early = y_true[:, self.tomato_early]
        is_tomato_late = y_true[:, self.tomato_late]
        
        # Probabilities of the confusing classes
        prob_tomato_early = y_pred[:, self.tomato_early]
        prob_tomato_late = y_pred[:, self.tomato_late]
        prob_potato_early = y_pred[:, self.potato_early]
        prob_potato_late = y_pred[:, self.potato_late]
        
        # Calculate penalties weighted by the actual presence of the true class
        penalty_pe_te = is_potato_early * prob_tomato_early
        penalty_pl_tl = is_potato_late * prob_tomato_late
        penalty_te_pe = is_tomato_early * prob_potato_early
        penalty_tl_pl = is_tomato_late * prob_potato_late
        
        total_penalty = penalty_pe_te + penalty_pl_tl + penalty_te_pe + penalty_tl_pl
        
        final_loss = base_loss + (self.penalty_weight * total_penalty)
        return tf.reduce_mean(final_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "penalty_weight": self.penalty_weight,
            "label_smoothing": self.label_smoothing,
            "num_classes": self.num_classes,
        })
        return config
