import tensorflow as tf

from mlp import MLP

##

class SHIFT_LAYER_PARAMETRIC(tf.keras.layers.Layer):
    def __init__(self,
                 dim,
                 flow_range = [-1.0, 1.0],
                 dims_hidden = [100],
                 hidden_activation = tf.nn.silu,
                 ):
        super().__init__()
        
        self.shift_ = MLP(dims_outputs = [dim],
                          outputs_activations = None,
                          dims_hidden = dims_hidden,
                          hidden_activation = hidden_activation,
                          name = 'shift_layer')

        Min, Max = flow_range
        self.wrap_ = lambda x: tf.math.floormod(x - Min, Max-Min) + Min

    def forward(self, x, cond, drop_rate = 0.0):
        return self.wrap_(x + self.shift_(cond, drop_rate=drop_rate)[0])

    def inverse(self, x, cond, drop_rate = 0.0):
        return self.wrap_(x - self.shift_(cond, drop_rate=drop_rate)[0])


class SHIFT_LAYER_STATIC(tf.keras.layers.Layer):
    def __init__(self,
                 flow_range = [-1.0, 1.0],
                 ):
        super().__init__()
        
        Min, Max = flow_range
        self.shift = 0.5*(Max-Min)
        self.wrap_ = lambda x: tf.math.floormod(x - Min, Max-Min) + Min

    def forward(self, x):
        return self.wrap_(x + self.shift)

    def inverse(self, x):
        return self.wrap_(x - self.shift)

