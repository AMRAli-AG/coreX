import os
import sys
import numpy as np
import tfsnippet as spt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def _layer_norm(inputs, scope=None, *args, **kwargs):
    with tf.variable_scope(scope or 'LayerNorm', reuse=tf.AUTO_REUSE):
        n_dims = inputs.get_shape()[-1].value or tf.shape(inputs)[-1]
        gamma = tf.get_variable('gamma', shape=[n_dims], initializer=tf.ones_initializer())
        beta = tf.get_variable('beta', shape=[n_dims], initializer=tf.zeros_initializer())
        mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        return gamma * (inputs - mean) / tf.sqrt(variance + 1e-6) + beta

class MockLayers(object):
    layer_norm = staticmethod(_layer_norm)

class MockFramework(object):
    def add_arg_scope(self, func): return func
    def arg_scope(self, *args, **kwargs):
        class DummyContextManager:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        return DummyContextManager()

class MockContrib(object):
    def __init__(self):
        self.rnn = self
        self.framework = MockFramework()
        self.layers = MockLayers()
    def static_bidirectional_rnn(self, *args, **kwargs):
        return [tf.zeros_like(args[2][0])] * len(args[2]), None, None

mock_contrib = MockContrib()
tf.contrib = mock_contrib
sys.modules['tensorflow.contrib'] = mock_contrib
sys.modules['tensorflow.contrib.framework'] = mock_contrib.framework
sys.modules['tensorflow.contrib.layers'] = mock_contrib.layers
sys.modules['tensorflow.contrib.rnn'] = mock_contrib

if not hasattr(tf, 'log'): tf.log = tf.math.log
if not hasattr(tf, 'exp'): tf.exp = tf.math.exp
if not hasattr(tf, 'sqrt'): tf.sqrt = tf.math.sqrt

import tensorflow as tf_base
for attr in dir(tf):
    if attr.startswith('__') or attr == 'compat':
        continue
    try:
        setattr(tf_base, attr, getattr(tf, attr))
    except (AttributeError, TypeError):
        pass
tf_base.contrib = mock_contrib


class Config:
    window_length = 120
    z_dim = 64
    x_dim = 75
    std_epsilon = 1e-4

config = Config()

mean_q_mlp = lambda x: tf.layers.dense(x, config.z_dim, activation=None, name='z_mean')
std_q_mlp = lambda x: tf.layers.dense(x, config.z_dim, activation=None, name='z_std')

import sys
sys.path.append('.')
from omni_anomaly.recurrent_distribution import RecurrentDistribution

# Input placeholder
input_q = tf.zeros([64, config.window_length, 512])

q_z_dist = RecurrentDistribution(
    input_q=input_q,
    mean_q_mlp=mean_q_mlp,
    std_q_mlp=std_q_mlp,
    z_dim=config.z_dim,
    window_length=config.window_length
)

posterior_flow = spt.layers.planar_normalizing_flows(20, name='posterior_flow')

from tfsnippet.distributions import FlowDistribution
q_z_dist_flow = FlowDistribution(q_z_dist, posterior_flow)

x = q_z_dist_flow.sample(n_samples=None, group_ndims=1)
print("x shape:", x.shape)
print("x.group_ndims:", x.group_ndims)
print("q_z_dist.log_prob(x.tensor, 0) shape:", q_z_dist.log_prob(x.tensor, 0).shape)
print("q_z_dist.log_prob(x.tensor, 1) shape:", q_z_dist.log_prob(x.tensor, 1).shape)
try:
    log_px = x.log_prob()
    print("x log_prob shape:", log_px.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
