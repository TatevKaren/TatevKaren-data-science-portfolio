<br><br>
<p align="center">
   <a href="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/AdvancedStatistics_OLS_MM_SIMI_CaseStudy.pdf"> 
        <img src="https://img.shields.io/badge/Case_Study-Missing_Data-red"></a>
   <a href="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/AdvancedStatistics_OLS_MM_SIMI_CaseStudy.pdf"> 
        <img src="https://img.shields.io/badge/Data-Cross_Sectional-green"></a>
   <a href="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/AdvancedStatistics_OLS_MM_SIMI_CaseStudy.pdf"> 
        <img src="https://img.shields.io/badge/Regressions-OLS_MM-blue"></a>
   <a href="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/AdvancedStatistics_OLS_MM_SIMI_CaseStudy.pdf"> 
        <img src="https://img.shields.io/badge/-Run_Code-3d3b3b?logo=RStudio"></a>
   <a href="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/AdvancedStatistics_OLS_MM_SIMI_CaseStudy.pdf"> 
        <img src="https://img.shields.io/badge/Missing_Data-Single_Imputation-important"></a>  
    <a href="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/AdvancedStatistics_OLS_MM_SIMI_CaseStudy.pdf"> 
        <img src="https://img.shields.io/badge/Missing_Data-Multiple_Imputation-yellow"></a> 

</p>
<br><br> 


# Advanced Marketing Models (Missing Data Mechanisms)

**Why:** Most of the statistical methods and algorithms require complete data and usig data with missing observations or entires can produce unreliable results.Therefore, it is important to know the reason for missingness in the data, it’s effect on the analysis and how these missiing data entir3es can be imputed. <br><br>

**How:** Using Boston housing data with a model-based simulation to perform Ordinary Least Squares (OLS) and Method of Moments (MM) estimations when applying Single Imputation (SI) or Multiple Imputation (MI) imputation techniques while artificially adding missing data with 3 different missing data mechanisms Missing At Random(MAR), Missing Completely At Random(MCAR) and Missing Not At Random(MNAR).<br>

<br>
<p align="center">
    <img src="https://atrium.ai/wp-content/uploads/elementor/thumbs/data_missing1-ot56ma5z1cf33uufbml86573m1k4wz1wgsnfz63vk8.jpg?raw=true"
  width=750" height="370">
</p> 

<br>
<br>

## Case Study 
In this Case Study we explore various missing data mechanisms such as MCAR, MNAR, and MAR. How missing data can be detected and additionally, we investigate the impact of these different type of missing data on the estimation results, where we use Ordinary Least Squares(OLS) and Method of Moments(MM) estimation as examples of estimation techniques.     

The <a href="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/AdvancedStatistics_OLS_MM_SIMI_CaseStudy.pdf"> Case Study Paper </a> and <a href="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/Single_vs_Multiple_Imputation.R">R Code</a> contain the following information<br>
- Missing data detection
- Missing data mechanisms (MNAR, MCAR, MAR)
- Missing data imputation techniques (Single Imputation, Multiple Imputation)
- Data Simulation to ingest missing data with MNAR,MCAR,MAR mechanism
- OLS and MM regressions
- Impact of Missing Data on OLS regression results
- Impact of Missing Data on Robust MM regression results


<br>

## Data
In this analysis we use the Boston housing data introduced by Harrison and Rubinfeld (1978) later corrected by Gilley and Kelley Pace (1996) that can be found <a href = "https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html">here</a>. The data cntains the following variables: 
 - CRIM: per capita crime rate by town
- ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: proportion of non-retail business acres per town.
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX:nitric oxides concentration (parts per 10 million)
- RM: average number of rooms per dwelling
- AGE: proportion of owner-occupied units built prior to 1940
- DIS: weighted distances to five Boston employment centres
- RAD: index of accessibility to radial highways
- TAX: full-value property-tax rate per $10,000
- PTRATIO: pupil-teacher ratio by town
- LSTAT: % lower status of the population
- MEDV: Median value of owner-occupied homes in $1000's

<br>

## Regression Model
<p align="center">
    <img src="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/Missing_Data_Regression_Model.png?raw=true"
  width=850" height="120">
</p> 

<br>

## Model Evaluation
To evaluate the mode we use **Mean Squared Error (MSE)**, **Bias**, 


<br>

## Case Study Findings
We have seen that increasing amount of missingness in data leads to an increase in mean MSE for both OLS and MM estimation results. Moreover,we observe that mean bias is 0 in OLS regression for all missingness rates which is not the case in MM estimation.

<br>





