
import tensorflow as tf

_policy = tf.keras.mixed_precision.global_policy()

class SmoothL1(tf.losses.Loss):
    def __init__(self, delta):
        super().__init__(reduction="none", name="SmoothL1Loss")
        self._delta = delta
    def call(self, y_true, y_pred):
        difference = tf.abs(y_true - y_pred)
        loss = tf.where(tf.less(difference, self._delta), 0.5 * difference**2, difference - 0.5)
        return tf.reduce_sum(loss, axis=-1)

class BalanceL1(tf.losses.Loss):
    def __init__(self, delta, alpha = 0.5, gamma=1.5):
        super().__init__(reduction="none", name="BalanceL1")
        self._delta = delta
        self._alpha = alpha
        self._gamma = gamma
    def call(self, y_true, y_pred):
        difference = tf.abs(y_true - y_pred)
        
        b = tf.math.exp(self._gamma/self._alpha) - 1
        loss = tf.where(tf.less(difference, self._delta), \
            self._alpha/b*(b*difference + 1) * tf.math.log(b*difference/self._delta + 1) - self._alpha*difference,\
            self._gamma * difference + self._gamma/b - self._alpha * self._delta)
        return tf.reduce_sum(loss, axis=-1)

class FocalLoss(tf.losses.Loss):
    def __init__(self, alpha, gamma, mode = "normal"):
        super().__init__(reduction="none", name="FocalLoss")
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        pt = tf.math.exp(-cross_entropy)                                      
        alpha = y_true*self._alpha + (1-y_true)*(1-self._alpha)

        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

class QFocalLoss(tf.losses.Loss):
    def __init__(self, alpha, gamma):
        super().__init__(reduction="none", name="QFocalLoss")
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        pred_prob = tf.sigmoid(y_pred)

        loss = tf.pow(tf.abs(y_true - pred_prob), self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

class MultiBoxLoss(tf.losses.Loss):
    def __init__(self, config):
        super().__init__(reduction="none", name="MultiBoxLoss")
        self._clf_loss = FocalLoss(alpha = config['training_config']["ClfLoss"]["Alpha"], gamma=config['training_config']["ClfLoss"]["Gamma"])
        self._box_loss = SmoothL1(delta = config['training_config']["BoxLoss"]["Delta"])

        self._num_classes = config['training_config']["num_classes"]
        self._cls_loss_weight = config['training_config']["ClfLoss"]["Weight"]
        self._loc_loss_weight = config['training_config']["BoxLoss"]["Weight"]

    def call(self, y_true, y_pred):
        box_labels = y_true[..., :4]
        box_predictions = y_pred[..., :4]
        
        cls_labels = tf.one_hot(
            tf.cast(y_true[..., 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=_policy.compute_dtype)
        cls_predictions = y_pred[..., 4:]
        
        iou_label = y_true[..., 5:6]

        positive_mask = tf.cast(tf.greater(y_true[..., 4], -1.0), _policy.compute_dtype)
        ignore_mask = tf.equal(y_true[..., 4], -2.0)

        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        #clf_loss = self._clf_loss(cls_labels*iou_label, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        
        clf_loss = tf.where(ignore_mask, 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)

        normalizer = tf.reduce_sum(positive_mask)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss), normalizer*4.0)

        #normalizer = tf.maximum(tf.reduce_sum(positive_mask, axis=-1), 1)
        #clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_lossk, axis=-1), normalizer)
        #box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_lossk, axis=-1), normalizer*4.0)

        return clf_loss*self._cls_loss_weight, box_loss*self._loc_loss_weight, normalizer