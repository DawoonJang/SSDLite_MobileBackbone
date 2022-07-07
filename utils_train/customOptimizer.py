import tensorflow.keras.backend as K
import tensorflow as tf

class GCSGD(tf.keras.optimizers.SGD):
    def get_gradients(self, loss, params):
        
        grads = [grad - tf.reduce_mean(grad, axis=list(range(len(grad.shape)-1)), keep_dims=True) if len(grad.shape) > 1 else grad for grad in K.gradients(loss, params)]
        
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [tf.keras.optimizers.clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads