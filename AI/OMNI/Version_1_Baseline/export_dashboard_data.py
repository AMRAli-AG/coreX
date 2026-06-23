# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ExpConfig
from omni_anomaly.utils import get_data
from omni_anomaly.model import OmniAnomaly
from tfsnippet.utils import get_variables_as_dict

def main():
    config = ExpConfig()
    config.restore_dir = 'model_coreX_v1'
    config.max_epoch = 0
    config.test_batch_size = 25
    config.test_n_z = 10

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
    # y_test has length equal to the number of points.
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
        
        # Get tensors from the graph
        # Encoder Q
        q_net = model.vae.variational(x=input_x, n_z=config.test_n_z)
        
        # Get mean and standard deviation from the distribution
        z_mean_tensor = q_net['z'].distribution.mean
        z_std_tensor = q_net['z'].distribution.std
        z_sample_tensor = q_net['z'].tensor
        
        # Decoder P
        p_net = model.vae.model(z=q_net['z'], x=input_x, n_z=config.test_n_z)
        x_reconstructed_mean = p_net['x'].distribution.mean
        x_reconstructed_std = p_net['x'].distribution.std
        
        # Log Probability (reconstruction score) per sensor
        r_prob_tensor = p_net['x'].log_prob(group_ndims=0)

    # 4. Restore Checkpoint and Run Evaluation
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    
    checkpoint_path = os.path.abspath(config.restore_dir)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
    
    export_data = {
        "threshold": 0.7871, # Suggested threshold from the run
        "sequences": []
    }

    print("--- Running Inference ---")
    with tf.Session(config=config_proto) as sess:
        sess.as_default()
        if latest_ckpt:
            var_dict = get_variables_as_dict(model_vs)
            saver = tf.train.Saver(var_list=var_dict)
            saver.restore(sess, latest_ckpt)
            print("--- Checkpoint restored successfully ---")
        else:
            print("--- WARNING: Checkpoint not found, using random weights ---")
            sess.run(tf.global_variables_initializer())

        # For each sequence, fetch the intermediate outputs
        for idx, label_type in indices_to_export:
            # Slice the window
            window_x = x_test[idx:idx + window_length]
            window_x_batch = np.expand_dims(window_x, axis=0) # [1, 50, 36]
            
            # Fetch tensors
            fetches = {
                'z_mean': z_mean_tensor,
                'z_std': z_std_tensor,
                'z_sample': z_sample_tensor,
                'x_reconstructed': x_reconstructed_mean,
                'r_prob': r_prob_tensor
            }
            
            results = sess.run(fetches, feed_dict={input_x: window_x_batch})
            
            # Unpack results (since batch size is 1)
            # Shapes:
            # z_mean: [n_z, batch, window, z_dim] or [batch, window, z_dim] depending on model
            # Let's inspect the shapes returned to clean them up properly.
            z_mean = results['z_mean']
            z_std = results['z_std']
            z_sample = results['z_sample']
            x_recon = results['x_reconstructed']
            r_prob = results['r_prob']
            
            # Clean up dimensions
            # If n_z is used, it might add an extra dimension at axis 0
            if z_mean.ndim == 4: # [n_z, batch, window, z_dim]
                z_mean = np.mean(z_mean, axis=0)
                z_std = np.mean(z_std, axis=0)
                z_sample = np.mean(z_sample, axis=0)
            if x_recon.ndim == 4: # [n_z, batch, window, x_dim]
                x_recon = np.mean(x_recon, axis=0)
            if r_prob.ndim == 4:
                r_prob = np.mean(r_prob, axis=0)
                
            # Now we have batch dimension at 0
            z_mean = z_mean[0] # [window, z_dim] or [z_dim]
            z_std = z_std[0]
            z_sample = z_sample[0]
            x_recon = x_recon[0] # [window, x_dim]
            r_prob = r_prob[0] # [window, x_dim]
            
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
                "input_x": window_x.tolist(), # [50, 36]
                "z_mean": z_mean_last.tolist(), # [32]
                "z_std": z_std_last.tolist(), # [32]
                "z_sample": z_sample_last.tolist(), # [32]
                "x_reconstructed": x_recon.tolist(), # [50, 36]
                "sensor_anomaly_scores": sensor_anomaly_scores.tolist(), # [36]
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
