[![](https://img.shields.io/badge/Deep__Learning-ANN-red)]()
[![](https://img.shields.io/badge/Python-Run_Code-blue?logo=Python)]()
[![](https://img.shields.io/badge/Tensorflow-3d3b3b?logo=Tensorflow)]()
[![](https://img.shields.io/badge/Keras-3d3b3b?logo=Keras)]()


# Customer Churn Rate Analysis with Artificial Neural Networks

**Why:** To estimate the churn rate of a bank customer in order to identify customers that are likely to leave the company and try to encourage them to stay by various marketing tools.

**How:** Using customer behavour data to train Artifical Neural Network (ANN) to predict the probability of each customer leaving the company.


<br>
<p align="center">
    <img src="https://github.com/TatevKaren/artificial-neural-network-business_case_study/blob/main/Churning_Customers.png?raw=true"
  width=699" height="400">
</p> 

<br>

## Case Study 
In this Business Case Study we predict the churn rate of bank customers. In order to learn about bank's customers we use Artificial Neural Networks (ANN). From the millions of customers we have randomly selected 10K bank customers and we use their characteristics/features to train an ANN model to predict the probability of each customer leaving the bank. 

The <a href ="https://github.com/TatevKaren/artificial-neural-network-business_case_study/blob/main/Artificial_Neural_Networks_Case_Study-2.pdf"> Case Study Paper </a> and <a href ="https://github.com/TatevKaren/artificial-neural-network-business_case_study/blob/main/Artificial_Neural_Network_Case_Study.py"> Python Code</a> contain the followin information<br>


 - Problem statement
 - Data overview
 - Data Preprocessing
 - Model building
 - ANN Initialization
 - Model compiling
 - Model fitting
 - Example prediction

## Data (Bank Customer Data)
To train the ANN model we use a training data of randomly selected 10K bank customers. The data consists of teh following varaible per bank customer:

- CustomerId Surname
- Credit Score
- Geography 
- Gender
- Age
- Tenure Balance
- Num Of Products at bank
- Has a Credit Card (binary variable)
- Is Active Member
- EstimatedSalary 

# Methodology

## Artificial Neural Networks (ANN)

An artificial neural network is an interconnected group of nodes, inspired by a simplification of neurons in a brain. The functionality of neural networks is often compared to the one of the multiple linear regression, where one uses multiple input features, also called independent variables, to predict the output variable, the dependent variable. 
In case of Neural Network we also use input features, referred as **Input Layer Neurons** to get information and learn about the outcome variable, referred as **Output Layer**.

The main difference between such regression and Neural Network is that, in the case the former the process runs in one iteration by minimizing the sum of the squared residuals (similar to cost function), whereas in case of Neural Network there is an intermediate step portrayed by the **Hidden Layer Neurons** which are used to get signals from the input layers and learn about the observations over and over again until the goal is achieved, the cost is minimized and no improvement is possible. So, one can say that ANNs are much more sophisticated than multiple linear regression.

<p align="center">
<img src="https://github.com/TatevKaren/artificial-neural-network-business_case_study/raw/main/ANN_layers.png?raw=true"
  alt=""width="500" height="250">
</p>

<br>

## Model Evaluation
Important evaluation steps, described in detail in the Case Study Paper, that help the ANN model to train and make accurate predictions such as:
- Loss Functions for ANN (Binary Cross Entroopy)
- Loss Function Optimizers (Adam Optimizer)
- Activation Functions (Rectifier and Sigmoid)

The combination of the Rectifier and Sigmoid activation functions is quite popular and this exact combination will be used in this case study as well, given that our goal is to estimate the probability that the customer will leave the bank. Given that the output variable is binary we use cost function Binary Cross Entropy. More specifically, the following model settings have been used to train the ANN model.

- Activation function for Hidden Layer: Rectifier
- Activation function for Output Layer: Sigmoid
- Optimization method: Adam Optimizer
- Cost function: Binary Cross Entropy
- Number of epochs: 100
- Batch size: 25

## Prediction in Python

<p align="left">
<img src="ANN_Case_Study_Sample_Output_1.png?raw=true"
  alt=""width="600" height="250">
</p>


<br>















