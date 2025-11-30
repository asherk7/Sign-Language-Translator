# Real-Time Sign Language Detection

This project implements a real-time sign language detection system using computer vision and deep learning. The goal is to recognize American Sign Language (ASL) gestures from a live camera feed and translate them into text, enabling more accessible communication between signers and non-signers.

## Overview

The system uses a convolutional neural network trained on two open-source ASL datasets to classify static hand gestures corresponding to letters of the English alphabet. A lightweight model architecture is used to achieve a balance between accuracy and inference speed, making the system suitable for real-time applications on standard hardware.

## Features

* Real-time sign detection from a webcam or video input
* Hand region extraction using MediaPipe
* Classification using a fine-tuned MobileNetV2 model
* Optimized inference through ONNX export
* Evaluation on combined Kaggle ASL datasets

## Datasets

The model was trained on two public datasets:

* **ASL Alphabet Dataset** (87,000 images)
* **American Sign Language Dataset** (166,000 images)

Both datasets contain labeled images of hand gestures representing the 26 English letters, along with “space” and “nothing” classes. The “delete” class was removed for consistency. Images were resized to 224×224 and combined into a unified dataset with an 80/10/10 training, validation, and testing split.

## Model

The core model is a pretrained **MobileNetV2** fine-tuned for 28 output classes.
Training was performed in two stages:

1. Training the classifier head with frozen base weights
2. Fine-tuning the entire network with a lower learning rate

The model uses cross-entropy loss and the Adam optimizer, with a cosine learning rate scheduler.

## Evaluation

Performance is evaluated using accuracy, F1-score, and inference latency.
The model achieves high accuracy on static test data and maintains real-time performance during live webcam inference.

## Usage
To run the sign language detection system, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/asherk7/Sign-Language-Translator.git
    cd Sign-Language-Translator
    ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the pretrained model weights and place them in the appropriate directory.
4. Run the real-time detection script:
   ```bash
   python realtime.py
   ```

## Credits

Developed by **Asher Khan**, **Hamza Abou Jaib**, and **Mehdi Syed** for CS/SE 4AL3 at McMaster University.
