<br>

<p align="center">
   <a href="https://drive.google.com/file/d/1f1M6DQ8kGGUbRRcbJ94e6ac3Yia4tsoe/view">
        <img src="https://img.shields.io/badge/ACM%2FSIGAPP-SAP2021-yellow?logo=ACM"></a>
   <a href="https://drive.google.com/file/d/1f1M6DQ8kGGUbRRcbJ94e6ac3Yia4tsoe/view">
        <img src="https://img.shields.io/badge/Recommender%20System-LDA--LFM-red"></a>
   <a href="https://drive.google.com/file/d/1f1M6DQ8kGGUbRRcbJ94e6ac3Yia4tsoe/view">
        <img src="https://img.shields.io/badge/Data-Cross_Sectional-green"></a> 
   <a href="https://drive.google.com/file/d/1f1M6DQ8kGGUbRRcbJ94e6ac3Yia4tsoe/view">
        <img src="https://img.shields.io/badge/Amazon.com-23__datasets-important?logo=Amazon"></a> 
   <a href="https://drive.google.com/file/d/1f1M6DQ8kGGUbRRcbJ94e6ac3Yia4tsoe/view">
        <img src="https://img.shields.io/badge/Tensorflow-3d3b3b?logo=Tensorflow"></a>
   <a href="https://drive.google.com/file/d/1f1M6DQ8kGGUbRRcbJ94e6ac3Yia4tsoe/view">
        <img src="https://img.shields.io/badge/Python-Run__Step__by__Step-blue?logo=Python"></a>

        
</p>
<br>

# LDA-LFM Recommender System 
**This work is based on a paper preseted and published at 36th ACM/SIGAPP Symposium on Applied Computing (SAC 2021)**
<br><br>
<p> <img href ="https://medium.com/mlearning-ai/using-customer-and-product-features-in-recommender-systems-2734258873cf" src="https://miro.medium.com/max/1120/1*OAHUmDf7SkFFeYF6Bsa5EQ.png?raw=true"
  alt="Multivariate Linear Regression plot"
  width="900" height="300"> </p>
  <br> <br>
  
**Why:** Most of the existing recommender systems are based only on the rating data, and they ignore other sources of information that might increase the quality of recommendations, such as textual reviews or user and item characteristics. Moreover, the majority of those systems are applicable only on small datasets and are unable to handle large datasets. <br><br>

**How:** We propose a new hybrid recommender algorithm that combines a rating modelling technique (i.e., Latent Factor Model) with a topic modelling method based on textual reviews (i.e., Latent Dirichlet Allocation), and we extend the algorithm such that it allows adding extra user- and item-specific information to the system. We evaluated the performance of the algorithm using Amazon.com datasets. <br><br>

<br>
 
 ## Abstract
The increased popularity of e-commerce has led to an increase in both the number of onlineconsumers and products.  Deciding which product to buy from a large variety of productsmight be overwhelming.  Recommender systems (RS) are able to solve this problem of in-formation overload and help online buyers,  by suggesting them the right products to buy,resulting in higher customer satisfaction and customer loyalty.  Correspondingly, companiesget more opportunities to introduce new products and suggest them to the right customers,resulting in increased sales.  In order to make good recommendations, we must predict howa customer will respond to unseen product, based on the user feedback that often comes inthe form of a rating and review text.  However, traditional RS often use only the rating data,while ignoring the rich source of information present in users’ textual reviews and other typesof data.  Moreover, they are solely applicable to small datasets and unable to handle datasetswith millions of reviews.  

In this study, we designed a recommender system that combinesthe Latent Factor Model, based on rating data, with the Latent Dirichlet Allocation method,based on review data, applicable on both small and large data types.  We also investigatedthe  effect  of  adding  user-  and  item-  characteristics  on  the  performance  of  this  hybrid  rec-ommender  system.   Firstly,  we  found  that  adding  textual  reviews  to  the  RS  increases  itsprediction accuracy.  Secondly,  we observe that adding extra features to the recommendersystem increases the quality of its recommendations, which is especially true for medium andlarge datasets.  
  <br><br>
  
  - Check Out <a href="https://medium.com/mlearning-ai/using-customer-and-product-features-in-recommender-systems-2734258873cf">Medium Article </a>, a blog post explaining a part of the proposed algorithm
  - Check Out <a href="https://www.youtube.com/watch?v=i_j59gQTU6w&t=42s&ab_channel=TatevKaren">Conference Presentation Youtube Video</a>
  - Check Out <a href="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/ConferencePresentation_SAC2021.pdf"> Conference Presentation Slides</a>
  - Check Out <a href="https://drive.google.com/file/d/1f1M6DQ8kGGUbRRcbJ94e6ac3Yia4tsoe/view"> Published Research Paper</a>
  
  <br>
  <br>
  
  
# Amazon Review Big Data
In this expeiment, we used collection of datasets provided corresponding to the 23 product categories supplied by **Amazon.com Review Data**. This data without duplicates was prepared by <a href = "https://jmcauley.ucsd.edu/data/amazon/">Julian McAuley</a>. It consists of **142.8 million** product reviews and a metadata for 9.4 million products, spanning a period of 18 years, from May 1996 to July 2014 
<br>
  <p> <img src="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Figures/10_30_rating_review_figure.png?raw=true?raw=true"
  width="600" height="400"> </p>
This figure visualizes the percentage of ratings and reviews per item. Brown (dark grey for black and white print) bars represent the percentage of items with less than 10 ratings per category. Green (light grey for black and white print) bars rep- resent the percentage of items having on average more than 30 words per product category.
  <br>
    <br>
  <br>
 
 
# Python Code
To reproduce the results obtained in this analysis you can execute following Python coded step-by-step. Note that many of the libraries such as **Tensorflow** need to be preinstalled on your computer. You can use the following syntax in your command line to install the desired Python package:

```Python
# Check the versions of Pip3 and Python
python3 --version
pip3 --version

# Install necessary python packages.
sudo apt-get install python3-pip python3-dev -y

# Upgrade pipo this is 
pip3 install --upgrade pip

# Install tensorflow (cpu only)
pip3 install tensorflow
```

<br>
To replicate the analysis including, cleaning and processing the data and generating recommendations with 4 different baseline moodels and with proposed method, you can use the following Python scrips. The order need to be kept intact to avoid errors and the **Amazon.com** data can be found in <a href = "https://jmcauley.ucsd.edu/data/amazon/">Julian McAuley</a> as mentioned earlier. You can choose out oof 23 different datasets each representing a dataset of a product category of Amazon E-commerce.

<br>

- Step 1: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Required%20Libraries.py"> Importing Required Python Libraries </a>
- Step 2: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Data%20Preparation.py"> Data Preparation</a>
- Step 3: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Data%20Transformation.py"> Data Transformation</a>
- Step 4: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Data%20Transformation%20NLP.py"> Data Transformation Natural Language Preprocessing (NLP)</a>
- Step 5: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Sparse%20Matrix%20Generation.py"> Data Transformation (Sparse Matrix Generation)</a>
- Step 6: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Offset%20Recommender%20System.py"> Offset Recommender Algorithm</a>
- Step 7: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Baseline%20Recommender%20System.py"> Baseline Recommender Algorithm</a>
- Step 8: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Standard%20Matrix%20Factorization%20Recommender%20System.py"> Standard Matrix Factorization Recommender Algorithm </a>
- Step 9: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20LDAFirst%20Recommender%20System.py"> LDAFirst Recommender Algorithm</a>
- Step 10: <a href = "https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Pyton-Code/LDA-LFM:%20Proposed%20LDA-LFM%20algorithm.py"> LDA-LFM Recommender Algorithm</a>

 
  <br>
    <br>
  <br> 
 
 # Model Performance Visualizatiion 
 <br>
 
 ## Results 1
 Performance of **proposed LDA-LFM recommender system** compared to **Offset** and **Baseline Recommender Model** in terms of Mean Squared Error (MSE)
 <p> <img src="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Figures/Results%201.png?raw=true"
  width="650" height="400"> </p>
  <br>

**Interpretation**

This figure shows the percentage decrease or improvement in MSE when comparing the proposed method LDA-LFM to the Offset model and BRM. The green bars show the percentage decrrease in MSE or improvement in prediction accuracy when using LDA-LFM instead of the Offset method, whereas the gray bars show the percentage decrease in MSE or improvement in the accuracy of the predictions when using LDA-LFM instead of the Baseline Rating Model. What can see is that LDA-LFM improves the prediction accuracy with at least 10% compared to the Offset method for all datasets. When looking at the green bars we observe that the proposed method improves the accuracy of the predictions with at least 15% compared to the BRM model for most of the product datasets, and for the larger datasets this improvement can be more than 30% and even rich the 40% improvement. 
  <br>
  <br><br><br>
  
 ## Results 2
   
 Performance of **proposed LDA-LFM recommender system** compared to standard **Matrix Factorization** and **LDAFirst model** in terms of Mean Squared Error (MSE)
   <br><p> <img src="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Figures/Picture%202.png?raw=true"
  width="650" height="450"> </p>
  <br> 
  
**Interpretation**

The figure shows the percentage decrease in MSE or improvement in prediction accuracy when comparing the proposed method LDA-LFM to the standard LFM model and LDAFirst model. The orange bars show the percentage decrease in MSE when using LDA-LFM instead of the Standard LFM method, whereas the red bars show the percentage decrease in MSE when using LDA-LFM instead of the LDAFirst. What the figure shows is that in both cases the majority of bars lie above the 0 level indication positive effect on the prediction accuracy when using the proposed method. <br>
Looking at the orange bars one can see that for datasets except for smallest 3, Musical Instruments, Amazon Instant Video, Digital Music the LDA-LFM leads to an increase in the prediction accuracy, in the case of Kindle Store dataset this improvement is 14%, which means that the proposed algorithm leads to 14% improvement in the prediction accuracy compared to the standard LFM model for this dataset. When looking at the red bars we observe that for almost all datasets except of the smallest 3 and Patio, Lawn and Garden, the LDA-LFM leads to improvement in the prediction accuracy compared to the LDAFirst method, and in some cases such as Kindle Store the improvement is 14%. 
So, we see that the proposed model leads to an improvement in the prediction accuracy compared to the all 4 baseline cases for medium sized and large datasets. 
  
 <br> <br><br><br>
 ## Results 3

 Performance of **proposed LDA-LFM recommender system** compared to itself when adding 1-4 **extra features** in terms of Mean Squared Error (MSE)

  <p> <img src="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/LDA-LFM-Recommender-System/Figures/Picture%203.png?raw=true"
  width="650" height="400"> </p>
  <br>
  
**Interpretation**

This figure shows the improvement in the prediction accuracy of the base LDA-LFM model with its 4 different variations when adding 1-4 extra features, the red bars correspond to the case when 1 extra feature  is added to the model, orange when 2 extra features are added, gray when 3 features are added and green when 4 extra features are added. As we can see there are many positive bars which means that for all those cases it holds that adding extra  features leads to higher prediction accuracy in terms of MSE. For almost all datasets we see positive effects or not very significant positive or negative effects. The green bars corresponding to the case where the most amount of extra features had been added are more visible suggesting that adding more features help to improve the prediction accuracy of the LDA-LFM model more. 

  <br>
  <br>

