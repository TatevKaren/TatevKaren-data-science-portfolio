# Recurrent Neural Network Pricing Model

**Why:**  To determine the future value of a stock or a financial instrument traded (publicly) on an exchange to gain significant profit or avoid losses given efficient-market hypothesis. Although exact price prediction is nearly impossible, but approximate price estimation is possible and can positively impact investor's buying strategy.<br><br>
**How:** Using the past 5y historical stock price data and Recurrent Neural Network(RNN) with 5 LSTM layers combined with Time Series Analysis to predict the upward and downward trends in the future stock prices.<br>



[![](https://img.shields.io/badge/Deep__Learning-RNN-brightgreen)]()
[![](https://img.shields.io/badge/Python-Run_Code-blue?logo=Python)]()
[![](https://img.shields.io/badge/Tensorflow-3d3b3b?logo=Tensorflow)]()
[![](https://img.shields.io/badge/Keras-3d3b3b?logo=Keras)]()
[![](https://img.shields.io/badge/Price__Prediction-Google__Stock-yellow?logo=Google)]()
[![](https://img.shields.io/badge/Price__Prediction-Bitcoin-yellow?logo=Bitcoin)]()


<br><br><br>

# Case Study with 2 applications

In this case study we use Deep Learning, Recurrent Neural Networks with Long Short-Term Memory(LSTM) layers to predict the price of the a financial instrument. We apply it in two different cases: 

- Google Stock Price Prediction
- Bitcoin Price Prediction

We combine Time Series Analysis with RNN model with LSTM layers to predict the price using the financial products' historical data of last 5 years (2016-2020). LSTM is more sophisticated version of RNN which addresses the Vanishing Gradient Problem that RNNs often suffer from. 
<br><br>
## Application 1: Google Stock Price Prediction
We have used <a href = "https://github.com/TatevKaren/recurrent-neural-network-pricing-model/tree/main/data">Google stock price data</a>, publicly available and downloaded from Yahoo Finance. For training the model we have used stock prices for the period of 2016-2020 and used it to predict stock prices for the January of 2021. 
<br>
You can download the <a href = "https://github.com/TatevKaren/recurrent-neural-network-pricing-model/blob/main/data/Google_Stock_Price_Trainset.csv">Training Data here</a><br>
You can download the <a href = "https://github.com/TatevKaren/recurrent-neural-network-pricing-model/blob/main/data/Google_Stock_Price_Testset.csv">Test Data here</a><br><br>
Google Stock Price Development Graph (Training Data)
<p align="left">
  <img src="https://github.com/TatevKaren/recurrent-neural-network-pricing-model/blob/main/Google Stock Price Development.png?raw=true"
  width="500" height="300">
</p>
<br>

**Data Preprocessing**

<br>
In order to prepare the data to train and test RNN model we have performed certain data preprocessing steps using Tensorflow, Keras, Pandas and Scikit-Learn libraries. Detailed info about data preprocessing can be found in <a href = "https://github.com/TatevKaren/recurrent-neural-network-pricing-model/blob/main/Recurrent_Neural_Networks_Case_Study.pdf"> Case Study Paper</a>
<p align="left">
  <img src="https://github.com/TatevKaren/recurrent-neural-network-pricing-model/blob/main/data/Data transformation.png?raw=true"
  width="380" height="220">
</p>
<br>
<br>

**Prediction Results**

<br>
Comparing the real Google stock values and predicted Google stock values that are generated using RNN model, for the test period, the first month of 2021. RNN based on 5 LSTMs was able to properly predict all upward and downward trends as we see that the red line corresponding to the predicted stock prices follows the same pattern as the blue line which corresponds the real stock prices. 
<br>
<p align="left">
  <img src="https://github.com/TatevKaren/recurrent-neural-network-pricing-model/blob/main/Prediction_Results.png?raw=true"
  width="600" height="400">
</p>
<br>
<br>

<br>






## Application 2: Bitcoin Price Prediction



<br><br><br>
# Explaining Methodology

## Recurrent Neural Networks

What RNN does is that it translates the provided inputs to a machine readable vectors. Then the system processes each of this sequence of vectors one by one, moving from very first vector to the next one in a sequential order. While processing, the system passes the information through the hidden state (memory) to the next step of the sequence. Once the hidden state has collected all the existing information in the system until time period t, it is ready to move towards the next step and in this newer step the hidden step is classified as previous hidden state defined. Find more detailed description about eache of these processes in the <a href = "https://github.com/TatevKaren/recurrent-neural-network-stock-price-predicition-case-study/blob/main/Recurrent_Neural_Networks_Case_Study.pdf"> Case Study Paper</a>
<p align="left">
  <img src="https://miro.medium.com/max/1120/1*o-Cq5U8-tfa1_ve2Pf3nfg.gif?raw=true"
  width="800" height="200">
 </p>
<br>

## Long Short-Term Memory (LSTMs)

In the very initial epoch of RNN, the weights are randomly chosen values and in case this chosen values are very small, multiplying them with recurrent weight many times, the gradient becomes less and less and at some point the gradient will vanish because the lower the gradient is, the harder is to update weight which means the slower will be the process. Moreover, there is a domino effect and one improperly calculated weight effects the calculation of the remaining weights and makes them inaccurate as well, given that they all are related. Hence, the entire training of the network will be inaccurate and some part of an important information will be lost in the process and will not be kept in the short-term memory.<br>
LSTMs have the same information flow as usual RNNs with the sequential information transition where data propagates forward through sequential steps. The difference between the usual RNN and LSTM is the set of operations that are performed on the passed information and the input value in that specific step. These set of various operations gives the LSTM the opportunity to keep or forget parts of the information that flows into that particular step. The information in LSTMs flows through its gates which are LSTMs’ main concepts; forget gate, input gate, cell state, and output gate. All these processes are described in <a href = "https://github.com/TatevKaren/recurrent-neural-network-stock-price-predicition-case-study/blob/main/Recurrent_Neural_Networks_Case_Study.pdf"> Case Study Paper</a>
<br>
<p align="left">
  <img src="https://github.com/TatevKaren/recurrent-neural-network-pricing-model/blob/main/methods/LSTM.png?raw=true"
  width="400" height="300">
</p>
<br>
<p align="left">.
<br>


## Activation Functions (Sigmoid and Hyperbolic Tangent)

The Hyperbolic Tangent Activation Function (Tanh function) is used because it is one of the activation functions that transforms vectors in such a way that all the elements of this vector fall in the range of [-1,1]. Using this in LSTM we avoid having too large or too small values that are transmitted through the system and cause problems. The Sigmoid function transforms vectors in such a way that all the elements of this vector fall in the range of [0,1] which helps to regulate the network by updating or forgetting data because any number getting multiplied by 0 is 0, causing values to disappear or to be “forgotten” and any number multiplied by 1 is the same value therefore that value stays the same or is “kept”. More about this in <a href = "https://github.com/TatevKaren/recurrent-neural-network-pricing-model/blob/main/Recurrent_Neural_Networks_Case_Study.pdf"> Case Study Paper</a>
<br>
<p align="left">
  <img src="https://github.com/TatevKaren/recurrent-neural-network-pricing-model/blob/main/methods/Hyperbolic_Tangent_Activation_Function.png?raw=true"
  width="270" height="250">
  <img src="https://github.com/TatevKaren/recurrent-neural-network-pricing-model/blob/main/methods/Sigmoid_Activation_Function.png?raw=true"
  width="280" height="250">
</p>
<a href = "https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21"> Image Source </a>
<br>
<br>


