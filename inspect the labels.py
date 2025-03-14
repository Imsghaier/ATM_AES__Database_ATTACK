
# i used this script to analyse the  how the labels are stored to  acess them correctly with my model trainnig and my attack script
import h5py
import numpy as np

def load_and_inspect_labels(file_path, num_samples=1000):
    with h5py.File(file_path, 'r') as f:
        profiling_labels_all = np.array(f['Profiling_traces/labels'])

    print(f"Total labels shape: {profiling_labels_all.shape}")

    # Print labels for each byte
    for byte_index in range(16):
        print(f"\nLabels for key byte {byte_index}:")
        if profiling_labels_all.ndim == 1:
            labels = profiling_labels_all[:num_samples]  # Adjust this if you need specific key byte information
        else:
            labels = profiling_labels_all[:num_samples, byte_index]
        print(labels)

# Path to the dataset
file_path = 'C:/Users/sghai/PythonProject1/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_data/ASCAD_databases/ASCAD.h5'

# Load and inspect labels
load_and_inspect_labels(file_path)
