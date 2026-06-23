# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from argparse import ArgumentParser

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ExpConfig
from omni_anomaly.utils import get_data
from omni_anomaly.model import OmniAnomaly
from tfsnippet.utils import get_variables_as_dict

def main():
    config = ExpConfig()
    config.restore_dir = 'model_coreX_v2_optimized'
    config.max_epoch = 0
    config.test_batch_size = 25
    config.test_n_z = 10
    config.window_length = 120
    config.x_dim = 75
    config.z_dim = 64
    config.posterior_flow_type = 'nf'
    config.adm_layers = 3
    config.rnn_num_hidden = 256
    config.dense_dim = 256
    config.use_connected_z_q = True
    config.use_connected_z_p = True

    # 1. Load Data
    print("--- Loading data ---")
    (x_train, _), (x_test, y_test) = get_data(dataset=config.dataset)
    
    # Clip sensors if necessary
    if x_test.shape[1] > config.x_dim:
        x_test = x_test[:, :config.x_dim]

    # 2. Find interesting window indices (Normal and Anomalous)
    print("--- Selecting representative sequences ---")
    window_length = config.window_length
    
    normal_indices = []
    anomaly_indices = []
    
    # We look for windows that are fully normal, and windows that contain anomalies
    for i in range(window_length, len(x_test)):
        window_labels = y_test[i - window_length:i]
        if np.sum(window_labels) == 0:
            if len(normal_indices) < 5:
                normal_indices.append(i - window_length)
        elif np.sum(window_labels) > 10:  # has significant anomaly
            if len(anomaly_indices) < 5:
                anomaly_indices.append(i - window_length)
        if len(normal_indices) >= 5 and len(anomaly_indices) >= 5:
            break

    indices_to_export = []
    for idx in normal_indices:
        indices_to_export.append((idx, "Normal"))
    for idx in anomaly_indices:
        indices_to_export.append((idx, "Anomaly"))

    # 3. Build Model Tensors
    print("--- Building TensorFlow Graph ---")
    input_x = tf.placeholder(tf.float32, shape=[None, config.window_length, config.x_dim], name='input_x')
    
    with tf.variable_scope('model') as model_vs:
        model = OmniAnomaly(config=config, name="model")
        
        # Build the graph EXACTLY as main.py does (Trainer builds loss first, creating all nested scopes perfectly)
        from omni_anomaly.training import Trainer
        trainer = Trainer(
            model=model, model_vs=model_vs,
            max_epoch=1, batch_size=config.batch_size,
            valid_batch_size=config.test_batch_size,
            initial_lr=0.001, lr_anneal_epochs=10,
            lr_anneal_factor=0.75, grad_clip_norm=5.0, valid_step_freq=100
        )
        
        # Now get the inference tensors for dashboard
        q_net = model.vae.variational(x=input_x, n_z=config.test_n_z, posterior_flow=model._posterior_flow)
        
        # Get mean and standard deviation from the sampled distribution
        z_sample_tensor = q_net['z'].tensor # Post NF sample [n_z, batch, window, z_dim]
        z_mean_tensor = tf.reduce_mean(z_sample_tensor, axis=0)
        z_std_tensor = tf.math.reduce_std(z_sample_tensor, axis=0)
        
        # Decoder P
        p_net = model.vae.model(z=q_net['z'], x=input_x, n_z=config.test_n_z)
        x_reconstructed_mean = p_net['x'].distribution.mean
        
        # Log Probability (reconstruction score) per sensor
        r_prob_tensor = p_net['x'].log_prob(group_ndims=0)
        
        # Extract AD Discrepancy
        dis_min_tensor = tf.constant(0.0)
        if hasattr(model, '_discrepancy_min_list') and model._discrepancy_min_list:
            dis_sum = tf.add_n(model._discrepancy_min_list) / len(model._discrepancy_min_list)
            dis_min_tensor = dis_sum
            
            dis_weight = tf.nn.softmax(-dis_sum, axis=-1)
            r_prob_tensor = r_prob_tensor * tf.expand_dims(dis_weight, -1)

    # 4. Restore Checkpoint and Run Evaluation
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    
    checkpoint_path = os.path.abspath(config.restore_dir)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
    
    # Load causal adjacency matrix
    causal_matrix_path = getattr(config, 'causal_adj_matrix_path', 'causal_adj_matrix.npy')
    try:
        causal_matrix = np.load(causal_matrix_path).tolist()
    except Exception as e:
        print(f"WARNING: Could not load causal matrix: {e}")
        causal_matrix = []

    export_data = {
        "threshold": -5.3517, # Set from recent run
        "causal_matrix": causal_matrix,
        "sequences": []
    }

    print("--- Running Inference ---")
    with tf.Session(config=config_proto) as sess:
        sess.as_default()
        if latest_ckpt:
            saver = tf.train.Saver()
            saver.restore(sess, latest_ckpt)
            print("--- Checkpoint restored successfully ---")
        else:
            print("--- WARNING: Checkpoint not found, using random weights ---")
            sess.run(tf.global_variables_initializer())

        # For each sequence, fetch the intermediate outputs
        for idx, label_type in indices_to_export:
            # Slice the window
            window_x = x_test[idx:idx + window_length]
            window_x_batch = np.expand_dims(window_x, axis=0)
            
            # Fetch tensors
            fetches = {
                'z_mean': z_mean_tensor,
                'z_std': z_std_tensor,
                'z_sample': z_sample_tensor,
                'x_reconstructed': x_reconstructed_mean,
                'r_prob': r_prob_tensor,
                'dis_min': dis_min_tensor
            }
            
            results = sess.run(fetches, feed_dict={input_x: window_x_batch})
            
            z_mean = results['z_mean']
            z_std = results['z_std']
            z_sample = results['z_sample']
            x_recon = results['x_reconstructed']
            r_prob = results['r_prob']
            dis_min = results['dis_min']
            
            # Clean up dimensions
            if z_mean.ndim == 4: # [n_z, batch, window, z_dim]
                z_mean = np.mean(z_mean, axis=0)
                z_std = np.mean(z_std, axis=0)
                z_sample = np.mean(z_sample, axis=0)
            if x_recon.ndim == 4: # [n_z, batch, window, x_dim]
                x_recon = np.mean(x_recon, axis=0)
            if r_prob.ndim == 4:
                r_prob = np.mean(r_prob, axis=0)
            
            if dis_min.ndim == 3 and dis_min.shape[0] > 1: # if n_z
                 dis_min = np.mean(dis_min, axis=0)
                
            z_mean = z_mean[0] # [window, z_dim]
            z_std = z_std[0]
            z_sample = z_sample[0]
            x_recon = x_recon[0] # [window, x_dim]
            r_prob = r_prob[0] # [window, x_dim]
            
            if hasattr(dis_min, "shape") and len(dis_min.shape) >= 2:
                 dis_min = dis_min[0] # [window]
            else:
                 dis_min = np.zeros(window_length)
            
            # Take last point values for z representations if it's per window
            if z_mean.ndim == 2:
                z_mean_last = z_mean[-1]
                z_std_last = z_std[-1]
                z_sample_last = z_sample[-1]
            else:
                z_mean_last = z_mean
                z_std_last = z_std
                z_sample_last = z_sample
                
            # Score per sensor at the last timestep
            r_prob_last = r_prob[-1] # [x_dim]
            # Convert log probability to anomaly score (negative log probability)
            sensor_anomaly_scores = -r_prob_last
            total_anomaly_score = np.mean(sensor_anomaly_scores)
            
            # Structure sequence data
            seq_data = {
                "id": f"Seq_{idx}_{label_type}",
                "label": label_type,
                "input_x": window_x.tolist(), # [120, 75]
                "z_mean": z_mean_last.tolist(), # [64]
                "z_std": z_std_last.tolist(), # [64]
                "z_sample": z_sample_last.tolist(), # [64]
                "x_reconstructed": x_recon.tolist(), # [120, 75]
                "sensor_anomaly_scores": sensor_anomaly_scores.tolist(), # [75]
                "discrepancy": dis_min.tolist(), # [120]
                "total_anomaly_score": float(total_anomaly_score)
            }
            export_data["sequences"].append(seq_data)
            print(f"Exported Sequence {idx} ({label_type}) - Score: {total_anomaly_score:.4f}")

    # Create dashboard folder and write JSON
    os.makedirs('dashboard', exist_ok=True)
    with open('dashboard/data.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    print("--- Saved dashboard data to dashboard/data.json ---")

if __name__ == '__main__':
    main()
