
# since VGG model have many layers and parameters, which require significant processing power for training than a gpu accelaration is needed for  making the training process much faster and more efficient compared to using a CPU alone
#This script is designed to set up and verify the environment necessary for TensorFlow to utilize GPU acceleration with CUDA and cuDNN libraries.
# It configures the appropriate paths for CUDA and cuDNN in the system's environment variables, ensures TensorFlow can dynamically allocate GPU memory,
# The script also verifies the TensorFlow installation and its GPU setup by performing a simple matrix multiplication operation.
#  the versions of the used  tensorflow , cuda , cudnn were  chosen based on the the insights provided by the community of the ASCAD databae

import os

# Set CUDA and cuDNN paths
os.environ['CUDA_PATH'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\libnvvp'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\include'
os.environ['PATH'] += ';C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\extras\\CUPTI\\lib64'

# Print PATH for debugging
print("Updated PATH:", os.environ['PATH'])

# Verify ptxas existence in PATH
print("ptxas in PATH:", os.system("where ptxas"))

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Verify TensorFlow and GPU setup
print("TensorFlow Version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Perform a basic TensorFlow operation to check GPU functionality
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)

print("Matrix a:\n", a.numpy())
print("Matrix b:\n", b.numpy())
print("Matrix c (a * b):\n", c.numpy())


