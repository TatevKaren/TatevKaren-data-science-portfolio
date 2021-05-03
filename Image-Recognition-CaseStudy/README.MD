[![](https://img.shields.io/badge/Deep__Learning-CNN-red)]()
[![](https://img.shields.io/badge/Python-Run_Code-blue?logo=Python)]()
[![](https://img.shields.io/badge/Tensorflow-3d3b3b?logo=Tensorflow)]()
[![](https://img.shields.io/badge/Keras-3d3b3b?logo=Keras)]()
[![](https://img.shields.io/badge/Image__Recognition-Cat_Dog_Images-yellow)]()

# Image Recognition with Convolutional Neural Networks

**Why:** To identify the class which an image belongs a dog image class or a cat image class.

**How:** Using 8K images of dogs and cats to train Convolutional Neural Network(CNN) to predict whether the input image is a dog image or a cat image.

<br>
<p href = "https://github.com/TatevKaren/convolutional-neural-network-image_recognition_case_study/blob/main/Convolutional_Neural_Networks_Case_Study.pdf">
    <img src="https://miro.medium.com/max/962/1*MBSM_G12XN105sEHsJ6C3A.png?raw=true"
  width=800" height="500">
</p> 
                         
## Case Study 
This is a Computer Vision Case Study with an Image recognition model that classifies an image to a binary class. Image recognition model based on Convolutional Neural Network (CNN) to identify from an image whether it is dog or a cat image. In this case study we use 8000 images of dogs and cats to train and test a Convolutional Neural Network (CNN) model that takes an image as an input and give as an output a class of 0 (cat) or 1 (dog) suggesting whether it is a dog or a cat picture. This image recognition model is based on CNN. 

The <a href ="https://github.com/TatevKaren/convolutional-neural-network-image_recognition_case_study/blob/main/Convolutional_Neural_Networks_Case_Study.pdf"> Case Study Paper </a> and <a href ="https://github.com/TatevKaren/convolutional-neural-network-image_recognition_case_study/blob/main/Convolutional_Neural_Network_Case_Study.py" > Python Code</a> contain the followin information<br>

 - Problem statement
 - Data overview
 - Data Preprocessing
 - Model building
 - CNN Initialization
 - Model compiling
 - Model fitting
 - Example prediction

<br>

## Training Data

We use training data consisting of 8000 images of dogs and cats to train the CNN model. Here are few examples of such images:<br><br>
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/dog.31.jpg?raw=true"
  width="180" height="180">
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/dog.24.jpg?raw=true"
  width="180" height="180">   
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/cat.12.jpg?raw=true"
  width="180" height="180">    
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/cat.1.jpg?raw=true"
  width="180" height="180">
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/dog.4.jpg?raw=true"
  width="180" height="180">     
</p>
<br><br>

## Model Application and Evaluation
To test the accuracy of the trained model we picked a pair of images and used the trained model to predict the class of each of these pair of two images, one of whiich is a dog image and the other one is a cate image. We would like to know the probability of each of this images belonging to a Cat class and Dog class. This will help us to evaluate the trained and tested CNN model to observe to which class does the model classify the following pictures:
<br>
<p align="left">
  <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/cat_or_dog_1.jpg?raw=true"
  width="300" height="200">
</p>
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/data/cat_or_dog_2.jpg?raw=true"
  width="300" height="200">
</p>
<br><br>

After compiling the model the CNN model accurately classified the first picture to a dog class and the second picture to a cat class. Following is a snapshot of a Python output. In the lower part you can see the predicted class of the first image and the second image, respectively.
<br><br>
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/Prediction_Snapshot.png?raw=true"
  width="1100" height="550">
</p>




# Methodology

## Convolutional Neural Networks (CNN)
This case study is based on CNN model and the <a href="https://github.com/TatevKaren/convolutional-neural-network-image_recognition_case_study/blob/main/Convolutional_Neural_Network_Case_Study-2.pdf">Case Study Paper</a> includes detailed description of all the steps and processes that CNN's include such as:
- Convolutional Operation
- Pooling
- Flattening
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/cnn_summary.png?raw=true"
  width="500" height="200">
</p>
<br>

## Model Evaluation
Important evaluation steps, described in detail in <a href="https://github.com/TatevKaren/computer-vision-case-study/blob/main/Convolutional_Neural_Networks_Case_Study.pdf"> Case Study Paper </a> , that help the CNN model to train and make accurate predictions such as:
- Loss Functions for CNN (SoftMax and Cross-Entropy)
- Loss Function Optimizers (SGD and Adam Optimizer)
- Activation Functions (Rectifier and Sigmoid)
<br>

## Where to find details about DL libraries: Tensorflow & Keras 
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/Keras ImageDataGenerator Library.png?raw=true"
  width="500" height="300">
</p>
<p>
    <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/Keras ImageDataGenerator Library2.png?raw=true"
  width="500" height="270">
</p>
<p>
   <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/Keras load_img function.png?raw=true"
  width="450" height="80">
   <img src="https://github.com/TatevKaren/dog_cat_image_recognition_cnn/blob/main/sources/Keras load_to_array function.png?raw=true"
  width="450" height="80">
</p>
Check out more information <a href = "https://keras.io/api/preprocessing/image/#imagedatagenerator-class"> here</a> 

<br><br>
