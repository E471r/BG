import tensorflow as tf

#

class MLP(tf.keras.layers.Layer):
    def __init__(self,
                 dims_outputs : list,
                 outputs_activations : list = None,
                 dims_hidden : list = [100],
                 hidden_activation = tf.nn.silu,
                 **kwargs):
        super().__init__(**kwargs)

        ''' MLP : class : Multilayer Perceptron.

        Inputs:
            dims_outputs : list of ints.
            outputs_activations : list of functions, or None (default: unconstrained linear outputs).
            dims_hidden : list of ints. [The length of this list determines how many hidden layers.]
            hidden_activation : non-linear function. [After each hidden later this non-linearity is applied.]

        '''

        n_hidden_layers = len(dims_hidden)
        n_output_layers = len(dims_outputs)

        if outputs_activations is None: outputs_activations = ['linear']*n_output_layers
        else: pass

        self.hidden_layers = [tf.keras.layers.Dense(dims_hidden[i], activation = hidden_activation) for i in range(n_hidden_layers)]
        self.output_layers = [tf.keras.layers.Dense(dims_outputs[j], activation = outputs_activations[j]) for j  in range(n_output_layers)]

    def call(self, x, drop_rate = 0.0):
        '''
        Inputs:
            x : (m,d) shaped tensor. 
                d, and the (d,dims_hidden[0]) shaped weights of the first hidden layer, become defined after the first call. 
            drop_rate : float in range [0,1]. 
                Default is 0.0, and always set to zero (from outside) when evaluating. 
                During training around 0.1 is approximate heuristic.
        Output:
            ys : list of tensors with shapes (m,dims_outputs[i]) for every output layer i.
        '''
        for layer in self.hidden_layers:
            x = layer(x)
        if drop_rate > 0.0: x = tf.keras.layers.Dropout(rate = drop_rate)(x, training=True) # drop_rate set to zero (from outside) every time when evaluating.
        else: pass
        ys = [layer(x) for layer in self.output_layers]
        return ys

