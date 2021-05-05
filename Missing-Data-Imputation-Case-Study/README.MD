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
To evaluate the mode we use **Mean Squared Error (MSE)**, **Bias**, **Mean Standard Error**, **CR** based on 95% Confidence Intervals or stated differently, a combined performance measure for both coefficient estimates and standard errors.

Following table shows sample regression output.
<p align="center">
    <img src="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/LS_MM_RegressionResults.png?raw=true"
  width=850" height="120">
</p> 

Following table shows sample Single Imputation output.
<p align="center">
    <img src="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/SingleImputation_Results.png?raw=true"
  width=850" height="120">
</p> 



<br>

## Case Study Findings
In this simulation study based on Boston Housing data showed that when data is MCAR both OLS and MM estimates are unbiased. Both SI and MI lead to accurate results when missingness proportion is small(e.g. 10%), but since MI requires much more computation time, SI is preferred for this reason. When missingness amount is larger SI underestimates the mean standard errors in OLS and in MM, while MI performs well therefore is preferred. 

When sample size is larger SI leads to accurate results for OLS even when 20% of data is missing. MAR data leads to biased estimates for small samples especially when single imputation has been used and biased decreases when MI is used instead, here also OLS outperforms MM. When n is large and MI is used OLS estimates are unbiased and accrate unless data contains 50% missing values. 

Under MNAR both SI and MI performs poorly for both OLS and MM except when only small amount of data is missing. When n is large MI leads to accurate and almost unbiased OLS estimates. So, SI performs well in terms of computation time for MAR and MCAR data mechanism for OLS regression when small amount of data is missing otherwise MI outperforms SI. However, when data is MNAR only for small missingness and large sample size MI leads to accurate results for OLS. Finally, we found that significant amount of outliers (25%) effect the OLS estimation results significantly for both SI and MI but when for MCAR and MAR data MI leads to accurate results for MM estimation.
<br>





