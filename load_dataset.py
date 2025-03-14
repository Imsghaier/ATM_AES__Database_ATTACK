# this is a simple script to download the data  to better understant what is stored exactly in the dataset
import h5py
import numpy as np

def load_dataset(file_path):
    with h5py.File(file_path, 'r') as f:
        profiling_traces = np.array(f['Profiling_traces/traces'])
        profiling_labels = np.array(f['Profiling_traces/labels'])
        attack_traces = np.array(f['Attack_traces/traces'])
        attack_labels = np.array(f['Attack_traces/labels'])

    # Normalize the data
    profiling_traces = profiling_traces / np.max(np.abs(profiling_traces))
    attack_traces = attack_traces / np.max(np.abs(attack_traces))

    # Reshape the data to add the channel dimension
    profiling_traces = profiling_traces.reshape((profiling_traces.shape[0], profiling_traces.shape[1], 1))
    attack_traces = attack_traces.reshape((attack_traces.shape[0], attack_traces.shape[1], 1))

    return profiling_traces, profiling_labels, attack_traces, attack_labels

# Example usage
if __name__ == '__main__':
    file_path = 'C:/Users/sghai/PythonProject1/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_data/ASCAD_databases/ASCAD.h5'
    profiling_traces, profiling_labels, attack_traces, attack_labels = load_dataset(file_path)
    print("Dataset loaded successfully.")
    print("Profiling Traces Shape:", profiling_traces.shape)
    print("Profiling Labels Shape:", profiling_labels.shape)
    print("Attack Traces Shape:", attack_traces.shape)
    print("Attack Labels Shape:", attack_labels.shape)