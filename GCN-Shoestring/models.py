

import tensorflow as tf
## My functions
from layers import Dense,GraphConvolution
from utils import masked_softmax_cross_entropy,masked_accuracy

class Model(object):
    def __init__(self,**kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
    
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None


    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def _build(self):
        raise NotImplementedError

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError
        
## GCN
class GCN(Model):
    def __init__(self, placeholders,input_dim,hidn_dim,learning_rate,weight_decay=5e-4, **kwargs):
        super(GCN, self).__init__(**kwargs)
        
        self.placeholders = placeholders
        self.inputs = placeholders['features']
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.input_dim = input_dim
        self.hidn_dim = hidn_dim
        self.weight_decay = weight_decay
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, 
            self.placeholders['labels'],self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])


        
    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                    output_dim=self.hidn_dim,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    dropout=True,
                                    sparse_inputs=True,
                                    logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.hidn_dim,
                                    output_dim=self.output_dim,
                                    placeholders=self.placeholders,
                                    act=lambda x:x,
                                    dropout=True,
                                    logging=self.logging))
                    
    def predict(self):
        return tf.nn.softmax(self.outputs)


## Shoestring
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
    training_set = tf.compat.v1.boolean_mask(inputs, placeholders['labels_mask'])
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

    return logits

class Shoestring(Model):
    def __init__(self, placeholders,input_dim,hidn_dim,learning_rate,weight_ML,weight_decay=5e-4, **kwargs):
        super(Shoestring, self).__init__(**kwargs)
        
        self.placeholders = placeholders
        self.inputs = placeholders['features']
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.input_dim = input_dim
        self.hidn_dim = hidn_dim
        self.weight_decay = weight_decay
        self.weight_ML = weight_ML
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]

        self.outs = None
        self.k_ratio = 0.

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],self.placeholders['labels_mask'])

        # Metric learning loss
        self.loss += self.weight_ML * masked_softmax_cross_entropy(self.outs, self.placeholders['labels'], self.placeholders['labels_mask'])
   

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])


        
    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                    output_dim=self.hidn_dim,
                                    placeholders=self.placeholders,
                                    act=tf.nn.relu,
                                    dropout=True,
                                    sparse_inputs=True,
                                    logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.hidn_dim,
                                    output_dim=self.output_dim,
                                    placeholders=self.placeholders,
                                    act=lambda x:x,
                                    dropout=True,
                                    logging=self.logging))
        self.outs = cosine_similarity(self.outputs, self.output_dim, self.placeholders, self.k_ratio)
                    
    def predict(self):
        return tf.nn.softmax(self.outputs)