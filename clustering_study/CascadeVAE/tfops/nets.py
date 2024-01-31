import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import tensorflow as tf

slim = tf.contrib.slim
#=============================================================================================================================================#
def encoder1_64(x, output_dim, output_nonlinearity=None, scope="ENC", reuse=False):
    nets_dict = dict()
    nets_dict['input'] = x
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }
# normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            biases_initializer=tf.zeros_initializer()):
            drop = slim.dropout(nets_dict['input'], keep_prob=0.5)
            nets_dict['fc0'] = slim.fully_connected(drop, 100, activation_fn=tf.nn.relu, scope='fc0')
            nets_dict['fc1'] = slim.fully_connected(nets_dict['fc0'], 100, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,scope='fc1')
            nets_dict['fc2'] = slim.fully_connected(nets_dict['fc1'], 100, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,scope='fc2')
            nets_dict['fc3'] = slim.fully_connected(nets_dict['fc2'], 100, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,scope='fc3')
            nets_dict['fc4'] = slim.fully_connected(nets_dict['fc3'], 10, activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, scope='fc4')
            nets_dict['output'] = slim.fully_connected(nets_dict['fc4'], output_dim, activation_fn=output_nonlinearity, normalizer_fn=slim.batch_norm,scope = "output")
            return nets_dict

def decoder1_64(z, scope="DEC", output_channel=5000, reuse=False):
    nets_dict = dict()
    nets_dict['input'] = z
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            biases_initializer=tf.zeros_initializer()):
            nets_dict['h'] = slim.fully_connected(nets_dict['input'], 10, activation_fn=tf.nn.relu, scope="h")
            nets_dict['fc0'] = slim.fully_connected(nets_dict['h'], 100, activation_fn=tf.nn.relu,  scope = "fc0")
            nets_dict['fc1'] = slim.fully_connected(nets_dict['fc0'], 100, activation_fn=tf.nn.relu,  scope = "fc1")
            nets_dict['fc2'] = slim.fully_connected(nets_dict['fc1'], 100, activation_fn=tf.nn.relu, scope='fc2')
            nets_dict['fc3'] = slim.fully_connected(nets_dict['fc2'], 100, activation_fn=tf.nn.relu, scope='fc3')
            nets_dict['output'] = slim.fully_connected(nets_dict['fc3'], output_channel, activation_fn=tf.nn.relu, scope='output')
            return nets_dict

