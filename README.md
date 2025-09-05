# S7-MA4144-Neural Networks and Fuzzy Logic

This repository contains coursework and projects for the Neural Networks and Fuzzy Logic course (MA4144) in Semester 7.

## Contents

### Project 1 - Basics
- **Files**: `MA4144InClass1(Basics).ipynb`, `MA4144InClass1(Basics).html`, `MA4144InClass1(Basics).pdf`
- **Data**: `DiabetesTrain.csv`, `DiabetesTest.csv`
- **Description**: Introduction to neural networks fundamentals and basic implementations

### Project 2 - Neural Networks
- **Files**: `MA4144InClass2(NNet).ipynb`, `MA4144InClass2(NNet).html`, `MA4144InClass2(NNet).pdf`
- **Data**: `train_mnist.npz`
- **Description**: Implementation of Multi-Layer Perceptron (MLP) for digit classification
- **Features**:
  - Custom neural network implementation from scratch
  - Feedforward and backpropagation algorithms
  - Binary classification for digits (1 vs 5, 7 vs 9)
  - Hyperparameter tuning and validation
  - Training/validation loss visualization

## Key Implementations

### Neural Network Components
- **Activation Functions**: Sigmoid, ReLU
- **Loss Function**: Mean Squared Error
- **Optimization**: Gradient Descent with mini-batches
- **Architecture**: Customizable hidden layers

### Models Trained
- `model_1_5`: Binary classifier for digits 1 vs 5
- `model_7_9`: Binary classifier for digits 7 vs 9

## Usage

1. Open the Jupyter notebooks in your preferred environment
2. Ensure you have the required dependencies installed:
   - NumPy
   - Matplotlib
   - Pandas (for Project 1)
   - Scikit-learn (for evaluation metrics)

3. Run the cells sequentially to:
   - Load and preprocess data
   - Implement neural network components
   - Train models
   - Evaluate performance

## Results

The implemented MLP achieves high accuracy on MNIST digit classification tasks with proper hyperparameter tuning.

## Course Information

- **Course**: MA4144 - Neural Networks and Fuzzy Logic
- **Semester**: 7
- **Academic Year**: 2024-2025

## Files Structure

```
├── Project 1/
│   ├── MA4144InClass1(Basics).ipynb
│   ├── MA4144InClass1(Basics).html
│   ├── MA4144InClass1(Basics).pdf
│   ├── DiabetesTrain.csv
│   ├── DiabetesTest.csv
│   └── pandoc.zip
├── Project 2/
│   ├── MA4144InClass2(NNet).ipynb
│   ├── MA4144InClass2(NNet).html
│   ├── MA4144InClass2(NNet).pdf
│   └── train_mnist.npz
├── rules.txt
├── README.md
└── .gitignore
```
