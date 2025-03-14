#my  script attack  is aimimg the version ASCAD/ATMEGA_AES__fixed_key
#  this attack is  using the  trained  cnn  the  models  by the script  train model   in order to reveal the used key bytes one by one
import h5py
import numpy as np
from keras.models import load_model
from scipy import stats

def load_attack_dataset(file_path, target_byte):
    with h5py.File(file_path, 'r') as f:
        metadata = f['Attack_traces/metadata']
        plaintext = np.array([metadata[i]['plaintext'][target_byte] for i in range(len(metadata))])
        key = np.array([metadata[i]['key'][target_byte] for i in range(len(metadata))])
        traces = np.array(f['Attack_traces/traces'])

    # Normalize the data by scaling it between -1 and 1
    traces = traces / np.max(np.abs(traces))

    # Reshape the data to add the channel dimension (1)
    traces = traces.reshape((traces.shape[0], traces.shape[1], 1))
    return traces, plaintext, key

def predict_key_byte(model, traces):
    # Use the model to predict the key byte for each trace
    predictions = model.predict(traces)
    # Get the most likely predicted key byte for each trace
    predicted_key = np.argmax(predictions, axis=1)

    # Get the mode (most common value) of the predicted key bytes
    mode_result = stats.mode(predicted_key, axis=None, keepdims=True)

    if mode_result.count[0] == 0:
        raise ValueError("No mode found in predicted keys.")

    # The most common predicted key byte
    most_common_key = mode_result.mode[0]

    return most_common_key

def recover_full_key(file_path):
    recovered_key = []
    # Loop through all 16 bytes of the key
    for byte_index in range(16):
        print(f"Recovering key byte {byte_index}...")

        # Load the attack dataset for the current byte
        attack_traces, plaintext, key = load_attack_dataset(file_path, byte_index)

        # Load the trained model for the current byte
        model_path = f'my_vgg16_model_byte_{byte_index}.keras'
        cnn_model = load_model(model_path)

        # Predict the key byte
        most_common_key = predict_key_byte(cnn_model, attack_traces)
        recovered_key.append(most_common_key)

        print(f"Recovered key byte {byte_index}: {most_common_key}")

    return recovered_key

# Path to the dataset
file_path = 'C:/Users/sghai/PythonProject1/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_data/ASCAD_databases/ASCAD.h5'

# Recover the full key
recovered_key = recover_full_key(file_path)
print(f"Recovered AES key: {recovered_key}")

