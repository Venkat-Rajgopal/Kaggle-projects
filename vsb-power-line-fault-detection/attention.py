# attention class
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K


class Attention(Layer):

    def __init__(self, step_dim, w_regularizer=None, b_regularizer=None,
                 w_constraint=None, b_constraint=None, bias=True, **kwargs):

        self.supports_masking = True
        # weight initializer
        self.init = initializers.get('glorot_uniform')  # initializers.glorot_uniform()

        self.w_regularizer = regularizers.get(w_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.w_constraint = constraints.get(w_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init, name='{}_w'.format(self.name),
                                 regularizer=self.w_regularizer,
                                 constraint=self.w_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.w, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

    def get_config(self):
        config = {
            'step_dim': self.step_dim,
            'w_regularizer': self.w_regularizer,
            'w_constraint': self.w_constraint,
            'b_regularizer': self.b_regularizer,
            'b_constraint': self.b_constraint,
            'bias': self.bias
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
