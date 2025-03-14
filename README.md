
# Side-Channel Analysis Using a VGG16-Inspired CNN Model  

## Overview  
This project implements a deep learning-based solution using a Convolutional Neural Network (CNN) model inspired by the VGG16 architecture. While originally designed for image processing, this adapted model is tailored to handle **one-dimensional side-channel traces**.  

The primary aim is to train the model on profiling traces to uncover the relationship between side-channel measurements and the cryptographic key. Once trained, the model performs a **side-channel attack**, predicting the secret AES key byte by byte from attack traces.  

## Dataset  
The project utilizes the **ASCAD dataset**, which contains side-channel measurements suitable for evaluating the performance of machine learning and deep learning techniques in key recovery.  

## Objectives  
- Adapt the VGG16-inspired CNN architecture for one-dimensional data.  
- Train the model on profiling traces to learn key-related patterns.  
- Predict the secret AES key from attack traces with high accuracy.  
- Demonstrate the feasibility and effectiveness of deep learning in side-channel analysis.  

## Features  
- **Deep Learning**: Utilizes a CNN model inspired by VGG16 for side-channel data processing.  
- **AES Key Recovery**: Performs byte-wise prediction of cryptographic keys.  
- **ASCAD Integration**: Demonstrates practical application with a standard dataset.  

## Requirements  
- Python 3.x  
- TensorFlow / PyTorch  
- NumPy  
- Matplotlib  
## Usage  
1. Clone this repository:  
   git clone <repository_url>
  
2. Install the required dependencies:  
   pip install -r requirements.txt
 
3. Train the model using profiling traces:
   python train_model.py.
5. run the  script of  the attack  


## Results  
The model demonstrates the viability of deep learning-based key recovery techniques in side-channel analysis and highlights the importance of secure cryptographic implementation 
