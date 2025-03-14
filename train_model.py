import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
import os

# Since the VGG model has many layers and parameters, which require significant processing power for training,
# GPU acceleration is needed to make the training process much faster and more efficient compared to using a CPU alone.

# This script is designed to set up and verify the environment necessary for TensorFlow to utilize GPU acceleration
# with CUDA and cuDNN libraries. It configures the appropriate paths for CUDA and cuDNN in the system's environment variables,
# ensures TensorFlow can dynamically allocate GPU memory, and verifies the TensorFlow installation and its GPU setup
# by performing a simple matrix multiplication operation.

# The versions of TensorFlow, CUDA, and cuDNN used were chosen based on the insights provided by the community of the ASCAD database.

# Set CUDA and cuDNN paths
os.environ['CUDA_PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\libnvvp'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\include'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\extras\\CUPTI\\lib64'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# The architecture and norms of the CNN VGG model were taken from the article "Study of Deep Learning Techniques for
# Side-Channel Analysis and Introduction to ASCAD Database," which states that the VGG model was the best model
# for revealing relevant results of the key.

def create_vgg16_model(input_shape):
    with tf.device('/GPU:0'):  # Specify to use the GPU
        model = Sequential()
        model.add(Input(shape=input_shape))

        # Block 1: 2 Conv1D layers + BatchNorm + MaxPooling + Dropout
        # Batch Normalization helps stabilize and accelerate training by normalizing inputs for each layer.
        # Dropout prevents overfitting by randomly dropping neurons during training.
        model.add(Conv1D(filters=64, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=64, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))

        # Block 2: 2 Conv1D layers + BatchNorm + MaxPooling + Dropout
        model.add(Conv1D(filters=128, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=128, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))

        # Block 3: 3 Conv1D layers + BatchNorm + MaxPooling + Dropout
        model.add(Conv1D(filters=256, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=256, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=256, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))

        # Block 4: 3 Conv1D layers + BatchNorm + MaxPooling + Dropout
        model.add(Conv1D(filters=512, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=512, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=512, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))

        # Block 5: 3 Conv1D layers + BatchNorm + MaxPooling + Dropout
        model.add(Conv1D(filters=512, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=512, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=512, kernel_size=11, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.25))

        # Classification Block: Flatten + 2 Dense layers
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(256, activation='softmax'))  # 256 classes for key byte values 0-255

        # RMSprop optimizer with a learning rate of 1e-5
        optimizer = RMSprop(learning_rate=1e-5)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_profiling_dataset(file_path, target_byte):
    with h5py.File(file_path, 'r') as f:
        metadata = f['Profiling_traces/metadata']
        plaintext = np.array([metadata[i]['plaintext'][target_byte] for i in range(len(metadata))])
        key = np.array([metadata[i]['key'][target_byte] for i in range(len(metadata))])
        traces = np.array(f['Profiling_traces/traces'])

    # Normalize the data by dividing by the max absolute value to scale between -1 and 1
    traces = traces / np.max(np.abs(traces))

    # Reshape the data to add the channel dimension (1)
    traces = traces.reshape((traces.shape[0], traces.shape[1], 1))

    # Compute labels for the target byte using XOR between plaintext and key
    labels = np.bitwise_xor(plaintext, key)
    labels = to_categorical(labels, num_classes=256)  # Convert labels to categorical (one-hot encoding)  to get better results

    return traces, labels

file_path = 'C:/Users/sghai/PythonProject1/ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_data/ASCAD_databases/ASCAD.h5'

# Loop through all 16 bytes of the key
for byte_index in range(16):

    print(f"Training model for key byte {byte_index}...")
    profiling_traces, profiling_labels = load_profiling_dataset(file_path, target_byte=byte_index)
    input_shape = (profiling_traces.shape[1], 1)
    vgg16_model = create_vgg16_model(input_shape)

    # Callbacks for EarlyStopping and ModelCheckpoint
    # EarlyStopping stops training when validation loss stops improving, preventing overfitting
    # ModelCheckpoint saves the best model based on validation loss
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=f'my_vgg16_model_byte_{byte_index}.keras', monitor='val_loss', save_best_only=True)

    # Train the model with callbacks
    vgg16_model.fit(profiling_traces, profiling_labels, epochs=75, batch_size=200, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])
    print(f"Model for key byte {byte_index} saved successfully.")
