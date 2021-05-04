# Multivariate Statistics Case Study: US City Ranking

[![](https://img.shields.io/badge/RStudio-Code__Run-blue?logo=RStudio)]()
[![](https://img.shields.io/badge/Multivariate-Statistics-red)]()
[![](https://img.shields.io/badge/Statistics-PCA-yellow)]()
[![](https://img.shields.io/badge/Statistics-FA-green)]()
[![](https://img.shields.io/badge/Statistics-CCA-purple)]()



**Why:** To identify the top US cities with highest capital, urban growth and development for purposes for everyone looking to relocate to or within US for various purposes.

**How:** Using Places Rated Almanac data to find the rankings of the cities in United States based on a single combination of 9 rating variables using Principal Components Analysis (PCA) and Factor Analysis (FA). We also use Canonical Correlation Analysis (CCA) to get more insights in this data and investigate the correlation between two sets of rating variables (if existing).

This Case Study in the field of Multivariate Statistics aims to find the rankings of the cities in United States based on a single combination of 9 rating variables using multivariate techniques: Principal Components Analysis (PCA) and Factor Analysis (FA). Moreover, we will also use Canonical Correlation Analysis (CCA) to get more insight of this data and investigate the correlation between two sets of rating variables (if existing). We aim to find the linear combination of rating variables that would maximally explain the variation of the data and rank the U.S. cities according to this new rating criterion.
- <a href = "https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/MultivariateStatistics_CaseStudy_PCA_FA_CCA.pdf">Case Study Paper (pdf)</a>
- <a href= "https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/MultivariateStat_PCA_FA_CCA_code.R">R Code</a>

<br> 
Check out my blog post about <a href = "https://towardsdatascience.com/fundamentals-of-statistics-for-data-scientists-and-data-analysts-69d93a05aae7">Fundamentals of Statistics for Data Scientists and Data Analysts in Towards Data Science</a> which conatins part of the methodology used in this work.
<br>

# Data (Places Rated Almanac)
The data “Places Rated Almanac” (Boyer and Savageau, 1985) contains 9 variables constructed for 329 metropolitan areas of the U.S. These composite variables are Climate, Housing Cost, Health care, Crime, Transportation, Education, Arts, Recreation and Personal Economy. More information about the latest snapshot of this data can be found <a href = "https://www.americansforthearts.org/by-program/reports-and-data/legislation-policy/naappd/places-rated-almanac-your-guide-to-finding-the-best-places-to-live-in-america-0">here</a>.



# Methodology Dimensionality Reduction 

## Principal Component Analysis
Principal Component Analysis or PCA is a dimensionality reduction technique that is very often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller set that still contains most of the information or the variation in the original large dataset. Correspondingly, the eigenvalues of the covraiance matrix show the variance explained by a particular data field out of the total variance. The idea behind PCA is to create new (independent) variables, called Principal Components, that are a linear combination of the existing variable.
<p align="center">
<img src="https://github.com/TatevKaren/Multivariate-Statistics/blob/main/PCA_plot.png?raw=true"
  alt="Principle Component Analysis"
  width="450" height="350"></p>
  
Above you can see the biplot where on X-axes and on Y-axes the first and the secont Principal Components, respectively, are visualized. We have scaled the data and from the plot we can observe that Crime and Education have the smallest margins and the remaining variables show substantial variations. 
  
Then using Elbow Rule or Kaiser Rule, you can determine the number of principal components that optimally summarize the data without losing too much information. It is also important to look at the proportion of total variation (PRTV) that is explained by each principal component to decide whether it is beneficial to include or to exclude it. 
<p align="center">
<img src="https://github.com/TatevKaren/Multivariate-Statistics/blob/main/Elbow_rule_%25varc_explained.png?raw=true"
  alt="Principle Component Analysis: Elbow Rule % Variation Explained"
  width="450" height="350">
</p>
First three principal components explain 63.10% of the total variation in data. Applying the ”elbow rule” it can be seen that one can optimally retain 3 components.  
<br>

## Factor Analysis
Factor analysis or FA is another statistical method for dimensionality reduction. It is one of the most commonly used inter-dependency techniques and is used when the relevant set of variables shows a systematic inter-dependence and the objective is to find out the latent factors that create a commonality. A factor model is as a series of multiple regressions, predicting each of the features variables from the values of the unobservable common factors.

Each variable has k of its own common factors, and these are related to the observations via factor loading matrix for a single observation as follows: In factor analysis, the factors are calculated to maximize between-group variance while minimizing in-group variance. They are factors because they group the underlying variables. Unlike the PCA, in FA the data needs to be normalized, given that FA assumption that the dataset follows Normal Distribution.

## Canoniical Correlation Analysis

Figures showing clear correlation between X and Y scores for the first canonical correlation which is not the case for the second factor, where one can observe some correlation but not significant. This shows that the first canonical correlation is important but second one is not.
<p align="left">
<img src="https://github.com/TatevKaren/Multivariate-Statistics/blob/main/CCA_plot2.png?raw=true"
  alt="Canonical Correlation Plot 2"
  width="400" height="350">
<img src="https://github.com/TatevKaren/Multivariate-Statistics/blob/main/CCA_plot1.png?raw=true"
  alt="Canonical Correlation Plot 1"
  width="400" height="350">
</p>


##




<br>
