from keras.models import load_model

# Load the trained model
model_path = 'C:/Users/sghai/PythonProject1/.venv/my_vgg16_model_byte_0.keras'
model = load_model(model_path)

# Display the model summary
model.summary()

# Print the details of each layer
for layer in model.layers:
    print(f"Layer name: {layer.name}, Layer type: {layer.__class__.__name__}, Output shape: {layer.output_shape}")

# Print the shape of the weights for each layer
weights = model.get_weights()
for i, weight in enumerate(weights):
    print(f"Layer {i} weights shape: {weight.shape}")

#  print the model configuration
config = model.get_config()
print(config)
