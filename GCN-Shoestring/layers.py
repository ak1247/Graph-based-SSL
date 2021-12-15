

import numpy as np
import tensorflow as tf


## Initializations
def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

## Global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

##
def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

## Dot produt
def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    def __init__(self,**kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}
        self.sparse_inputs = False
    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    def __init__(self,input_dim,output_dim,placeholders,\
                        dropout,bias=False,**kwargs):
        super(Dense,self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.bias = bias

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()
    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphConvolution(Layer):
    def __init__(self,input_dim,output_dim,placeholders,dropout=0.,sparse_inputs=False,\
            act=tf.nn.relu,bias=False,featureless=False, **kwargs):
        super(GraphConvolution,self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
    

def cosine_similarity(inputs, output_dim, placeholders, k_ratio):

    # get the number of unlabeled data needed for adjusting centroid
    assert k_ratio <= 1 and k_ratio >= 0, 'k_ratio is greater than 1 or less than 0'
    k = tf.math.multiply(tf.reduce_sum(tf.cast(placeholders['unlabels_mask'], tf.float32)), k_ratio)
    k = tf.cast(k, tf.int32)

    label = tf.cast(placeholders['labels'], tf.float32)
    placeholders['labels_mask'].set_shape([None])
    placeholders['unlabels_mask'].set_shape([None])

    # inputs = tf.Variable(inputs,dtype=tf.float32,validate_shape=False)
    # inputs.set_shape([None])

    # training_set, query, unlabeled = tf.split(inputs, [int(idx * fraction), idx - int(idx * fraction), -1], 0)
    training_set = tf.boolean_mask(inputs, placeholders['labels_mask'])
    unlabeled = tf.boolean_mask(inputs, placeholders['unlabels_mask'])
    training_label = tf.boolean_mask(label, placeholders['labels_mask'])

    # Average vector based on the training set (input)
    total = tf.linalg.matmul(tf.transpose(training_set), training_label)
    weight = tf.math.reciprocal(tf.reduce_sum(label, 0))
    protos = tf.transpose(tf.linalg.matmul(total, tf.diag(weight)))

    if k_ratio != 0:

        unlabeled_prob = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.nn.l2_normalize(unlabeled, dim=1), 1), tf.nn.l2_normalize(protos, dim=1)), 2)
        prob_top, prob_idx = tf.nn.top_k(unlabeled_prob, k, sorted=False)
        prob_sm = tf.nn.softmax(prob_top)
        prob_row_idx = tf.tile(tf.range(tf.shape(unlabeled_prob)[0])[:, tf.newaxis], (1, k))
        scatter_idx = tf.stack([prob_row_idx, prob_idx], axis=-1)
        unlabeled_prob = tf.transpose(tf.scatter_nd(scatter_idx, prob_top, tf.shape(unlabeled_prob)))
    

        unlabeled_weight = tf.math.add(tf.reduce_sum(unlabeled_prob, 0), tf.reduce_sum(training_label, 0))
        unlabeled_weight = tf.linalg.matmul(unlabeled_prob, tf.diag(tf.math.reciprocal(unlabeled_weight)))
        unlabeled_weight = tf.unstack(tf.expand_dims(tf.transpose(tf.unstack(unlabeled_weight, axis=1)), axis=1), axis=2)
        unlabeled_difference = tf.math.subtract(tf.expand_dims(unlabeled, axis=1), tf.expand_dims(protos, axis=0))
        unlabeled_difference = tf.transpose(tf.unstack(unlabeled_difference, axis=1), perm=[0, 2, 1])
        change = tf.reshape(tf.linalg.matmul(unlabeled_difference, unlabeled_weight), [output_dim, output_dim])
        protos = tf.math.add(change, protos)

    logits = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.nn.l2_normalize(inputs, dim=1), 1), tf.nn.l2_normalize(protos, dim=1)), 2)

