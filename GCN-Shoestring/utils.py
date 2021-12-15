
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import tensorflow as tf
import scipy.sparse as sp
from scipy import sparse

### Functions
## Load Data
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def split_dataset(labels, train_size, test_size, validation_size, validate=True, shuffle=True):
    idx = np.arange(len(labels))
    idx_test = []
    if shuffle:
        np.random.shuffle(idx)
    
    assert train_size > 0, "train size must bigger than 0."
    no_class = labels.shape[1]  # number of class
    #train_size =[ 1, 4, 8, 20, 20, 20, 20]
    train_size = [train_size for i in range(labels.shape[1])]
    idx_train = []
    count = [0 for i in range(no_class)]
    label_each_class = train_size
    next = 0
    for i in idx:
        if count == label_each_class:
            break
        next += 1
        for j in range(no_class):
            if labels[i, j] and count[j] < label_each_class[j]:
                idx_train.append(i)
                count[j] += 1
                break
        else:
            idx_test.append(i)

    idx_test = np.array(idx_test, dtype=idx.dtype)
    if validate:
        if validation_size:
            assert next + validation_size < len(idx), "Too many train data, no data left for validation."
            idx_val = idx[next:next + validation_size]
            next = next + validation_size
        else:
            idx_val = idx[next:]

        assert next < len(idx), "Too many train and validation data, no data left for testing."
        if test_size:
            assert next + test_size < len(idx)
            idx_test = idx[-test_size:]
        else:
            idx_test = np.concatenate([idx_test, idx[next:]])
    else:
        assert next < len(idx), "Too many train data, no data left for testing."
        if test_size:
            assert next + test_size < len(idx)
            idx_test = idx[-test_size:]
        else:
            idx_test = np.concatenate([idx_test, idx[next:]])
        idx_val = idx_test
        
        idx_train = np.array(idx_train)
    return idx_train, idx_val, idx_test

def load_data(dataset_str):
    print("Loading data ...")
    ## Reading data files
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    print("Label data:: Train:{}, Test:{}".format(x.shape,tx.shape))
    print("All data  :: Train:{}".format(allx.shape))
    print("Links     :: {}".format(len(graph)))

    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    ## Data preparation
    features = sparse.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    idx_train, idx_val, idx_test = split_dataset(labels,train_size=5,
         test_size=None,validation_size=500,validate=False,shuffle=True)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    
## Preprocessing
def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

# Features
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

# Adj matrix
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sparse.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

## Metrics
def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
        
##
def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict