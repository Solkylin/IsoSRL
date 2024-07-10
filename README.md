# Isolated Sign Language Recognition System

## Project Overview
This project aims to recognize isolated sign language words using deep learning models. It supports various models, including R3D, R2+1D, and LSTM+Resnet, and is suitable for educational and communication assistance scenarios. Users can recognize sign language by uploading videos, recording videos, or using real-time recognition.

## Requirements
- Python >= 3.8.18
- PyTorch >= 1.10.1, <= 2.2.1 (GPU version recommended for better performance)

## Installation Guide

### Create a Python Virtual Environment
```bash
conda create -n pytorch python=3.8.18
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
Note: If you encounter dependency conflicts, specify the correct version based on the error message, e.g.`pip install torch==1.10.1`。

## Running the Project

### Run Frontend
Start the frontend interface:
```bash
streamlit run app.py
```

### Training and Testing
Select the model for training and testing in the `train.py` file:
```python
# Select model
# model = R3D()
# model = R2Plus1D()
# model = LSTMResnet()
```
Set the dataset path:
```python
# Dataset path
data_path = "../SLR_Dataset/CSL_Isolated/color_video_25000"
label_path = '../SLR_Dataset/CSL_Isolated/dictionary.txt'
```
Note: Modify the dataset path in train.py according to your actual situation.

## Project Structure
```plaintext
.
├── app.py          # Main code for frontend
├── dataset.py      # Dataset class code for sign language data
├── model.py        # Model definitions
├── prediction.py   # Code for predicting a single video
├── real.py         # Main code for real-time recognition
├── demo.py         # Demo for real-time recognition (no frontend)
├── test.py         # Testing code (confusion matrix and accuracy)
├── tool.py         # Utility code
├── train.py        # Training code
├── data/           # Directory for uploaded videos and other necessary data
├── logs/           # Log files directory
├── static/         # Resource files directory
├── video/          # Directory for intermediate frame images
└── weight/         # Directory for trained weights
```
## Notes
I apologize for any inconvenience this may cause!
