import time
import tensorflow as tf
from os import cpu_count
## My functions
from utils import *
from models import GCN,Shoestring


### Hyper-parameters
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
dataset_name = 'citeseer'
model_name = 'gcn'

LEARNING_RATE = 0.01
DROPOUT = 0.5
LAMBDA = 0.01
HIDDEN_DIM = 16
EPOCHS = 200
EARLY_STOP = 10



### Data
## Load Data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_name)

## Preprocessing
features = preprocess_features(features)
if model_name == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(model_name))

###
## Define placeholders
labels_mask = train_mask.astype(np.int32)
unlabels_mask = test_mask.astype(np.int32)
y_train = y_train.astype(np.int32)
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    #'unlabels_mask': tf.placeholder(tf.int32),
    #'labels_mask': tf..placeholder_with_default(labels_mask, shape=(None), name='labels_mask'),
    #'unlabels_mask': tf..placeholder_with_default(unlabels_mask, shape=(None), name='unlabels_mask'),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

## Model
model = model_func(placeholders,features[2][1],HIDDEN_DIM,LEARNING_RATE,LAMBDA,logging=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


### Training
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

cost_val = []

print("Training starts..........")
for epoch in range(EPOCHS):
    t = time.time()

    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']:DROPOUT})

    ## Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    ## Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    # if epoch > EARLY_STOP and cost_val[-1] > np.mean(cost_val[-(EARLY_STOP):-1]):
    #     print("Early stopping...")
    #     break
print("--- Training Done! ---")

## Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))








