
# Advanced Marketing Models (FastMCD)

**Why:** Linear Discriminant Analysis (LDA) is one of the most widely-used classification methods for predicting qualitative response variables but it is highly sensitive to the outliers and it produces unreliable results in case the data is contaminated. To improve the robustness of the LDA, it’s necessary to consider robust estimators for means and covaraince matrices. The robustified LDA employs robust estimators for location and scatter to limit the impact of outliers in the data.
<br>

**How:** Using FastMCD algorithm to compute robust estimators for location and scatter parameters for robustified LDA where we code the FastMCD function/algorithm manually in R and verify it's results with a FastMCD function from R library.<br>

<p align="center">
   <a align="center"> <img src="https://thumbs.dreamstime.com/b/funny-cartoon-circus-strong-man-muscular-athlete-lifts-barbell-retro-sportsman-mustache-flat-vector-guy-character-heavy-116376139.jpg" width="500" height="500"></a>
</p>


- <a href="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/AdvancedStatistics_FastMCD_CaseStudy.pdf"> Case Study Paper</a>
- <a href="https://github.com/TatevKaren/mathematics-statistics-for-data-science/blob/main/FastMCD.R">R Code</a> 

# Methodology

## Linear Discriminant Analysis
Linear discriminant analysis (LDA) is a generalization of Fisher's linear discriminant, which is a statistical method to find a linear combination of features that characterizes/separates two or more classes of objects. The resulting combination may be used as a linear classifier. LDA is closely related to analysis of variance (ANOVA) and regression analysis, which also attempt to express one (dependent) variable as a linear combination of other (independent) variables. However, ANOVA uses a continuous dependent variable and categorical independent variables, whereas LDA uses a categorical dependent variable (classes of LDA) and continuous independent variables. 

Logistic regression and Probit regression are more similar to LDA than ANOVA is, as they also explain a categorical (dependent) variable by the values of continuous (independent) variables. The key difference between Logistic Regression/Probit regression and LDA is the assumption about the probability distribution about the explanatory (independent) variables. In case of LDA , fundamental assumtion is that the independent variables are normally distributed. This can be checked by looking at the probability distribution of the variables. 
