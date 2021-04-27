#----------------------- Artificial Neural Network for classification --------------------#
#importing required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


#----------------------- Data Pre-processing ----------------------#
# Checking the tensorflow version
print(tf.__version__)

# Loading the data
bank_data = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")

# Taking  all rows and all columns in the data except the last column as X (feature matrix)
#the row numbers and customer id's are not necessary for the modelling so we get rid of and start with credit score
X = bank_data.iloc[:,3:-1].values
print("Independent variables are:", X)
#taking all rows but only the last column as Y(dependent variable)
y = bank_data.iloc[:, -1].values
print("Dependent variable is:", y)


# Transforming the gender variable, labels are chosen randomly
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
print(X)

# Transforming the geography column variable, labels are chosen randomly, the ct asks for argument [1] the index of the target vb
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#printing the dimensions of each of those snapshots to see amount of rows and columns i each of them
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Data Scaling/normalization of the features that will go to the NN
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#----------------------- Building the model -----------------------#

# Initializing the ANN by calling the Sequential class fromm keras of Tensorflow
ann = tf.keras.models.Sequential()

#----------------------------------------------------------------------------------
# Adding "fully connected" INPUT layer to the Sequential ANN by calling Dense class
#----------------------------------------------------------------------------------
# Number of Units = 6 and Activation Function = Rectifier
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))


#----------------------------------------------------------------------------------
# Adding "fully connected" SECOND layer to the Sequential AMM by calling Dense class
#----------------------------------------------------------------------------------
# Number of Units = 6 and Activation Function = Rectifier
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))


#----------------------------------------------------------------------------------
# Adding "fully connected" OUTPUT layer to the Sequential ANN by calling Dense class
#----------------------------------------------------------------------------------
# Number of Units = 1 and Activation Function = Sigmoid
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#----------------------- Training the model -----------------------#
# Compiling the ANN
# Type of Optimizer = Adam Optimizer, Loss Function =  crossentropy for binary dependent variable, and Optimization is done w.r.t. accuracy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN model on training set  (fit method always the same)
# batch_size = 32, the default value, number of epochs  = 100
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#----------------------- Evaluating the Model ---------------------#
# the goal is to use this ANN model to predict the probability of the customer leaving the bank
# Predicting the churn probability for single observations

#Geography: French
#Credit Score:600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#with Credit Card
#Active member
#Estimated Salary: $50000

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
# this customer has 4% chance to leave the bank


#show the vector of predictions and real values
#probabilities
y_pred_prob = ann.predict(X_test)

#probabilities to binary
y_pred = (y_pred_prob > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

#Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix", confusion_matrix)
print("Accuracy Score", accuracy_score(y_test, y_pred))
