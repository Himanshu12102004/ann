# Multilayer Perceptron Model (Function Approximator)

This repository contains a custom implementation of a Multilayer Perceptron (MLP) in C++, designed as a function approximator. The MLP model is built from scratch, featuring backpropagation, various activation functions, and gradient descent algorithms.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Bonus](#modification-of-layer-configuration)

## Introduction

The Multilayer Perceptron (MLP) is a fundamental neural network model used for a variety of tasks, including classification, regression, and function approximation. This project implements a fully functional MLP from scratch, showcasing my understanding of neural networks and C++ programming.

## Features

- **Customizable Architecture**: Define the number of layers and neurons per layer.
- **Backpropagation**: Efficient gradient descent-based learning algorithm.
- **Activation Functions**: Supports various activation functions including sigmoid, and ReLU.
- **Function Approximation**: Accurate approximation of complex non-linear functions.
- **Learning Rate Adjustment**: Fine-tune the learning rate for optimal performance.

## Getting Started
1. Clone the repository:

   ```bash
   git clone https://github.com/Himanshu12102004/ann.git
   cd ann
   ```
## Usage
### How to Start

1. **Generate a Dataset**:

   - Modify, compile and run the data generation program. Adjust the C++ version as needed:

     ```bash
     g++ -std=c++11 generateData.cpp -o generateData
     ./generateData
     ```

2. **Prepare the ANN**:

   - Open `main.cpp` and set the mode in the `Ann` constructor to `initialize`.

   ![Screenshot](https://github.com/user-attachments/assets/33a98587-7b59-4903-b29b-d7047a89b9d9)

3. **Segregate Training and Testing Data**:

   - Compile and run the main program to read and segregate the training and testing data:

     ```bash
     g++ -std=c++11 main.cpp -o main
     ./main
     ```

4. **Train The model**
   - After segregation train the data set by changing the mode to `training` and run the command below:
     ```bash
     g++ -std=c++11 main.cpp -o main
     ./main
     ```
5. **Test The model**
   - After training the ann test it by changing the mode to `training` and run the command below:
     ```bash
     g++ -std=c++11 main.cpp -o main
     ./main
     ```
6. **Want intraction?**
   - After training you may set the mode to `production` in order to have a interactive and real time input output sequence and run the command below:
     ```bash
     g++ -std=c++11 main.cpp -o main
     ./main
     ```

## Modification of layer configuration

**Modify the code shown in the picture to modify the configuration of the MLP**
<img width="1440" alt="Screenshot 2024-09-03 at 5 56 14 PM" src="https://github.com/user-attachments/assets/981d1aa7-2fc7-488d-9286-928f5772dd6e">

1. **Generate a Dataset**:
   - Modify, compile and run the data generation program. Adjust the C++ version as needed:
   
     ```bash
     g++ -std=c++11 generateData.cpp -o generateData
     ./generateData
     ```
2. **Prepare the ANN**:
   - Open `main.cpp` and set the mode in the `Ann` constructor to `initialize`.
   
   ![Screenshot](https://github.com/user-attachments/assets/33a98587-7b59-4903-b29b-d7047a89b9d9)
3. **Segregate Training and Testing Data**:
   - Compile and run the main program to read and segregate the training and testing data:

     ```bash
     g++ -std=c++11 main.cpp -o main
     ./main
     ```
4. **Train The model**
   - After segregation train the data set by changing the mode to `training` and run the command below:
     ```bash
     g++ -std=c++11 main.cpp -o main
     ./main
     ```
5. **Test The model**
   - After training the ann test it by changing the mode to `training` and run the command below:
     ```bash
     g++ -std=c++11 main.cpp -o main
     ./main
     ```
6. **Want intraction?**
   - After training you may set the mode to `production` in order to have a interactive and real time input output sequence and run the command below:
     ```bash
     g++ -std=c++11 main.cpp -o main
     ./main
     ```
## Modification of layer configuration
  **Modify the code shown in the picture to modify the configuration of the MLP**
<img width="1440" alt="Screenshot 2024-09-03 at 5 56 14 PM" src="https://github.com/user-attachments/assets/981d1aa7-2fc7-488d-9286-928f5772dd6e">
