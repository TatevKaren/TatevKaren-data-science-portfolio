# Top-N Movie Recommender

**Why:** For a given movie subscriber, determine the top N movies that this user will likely be interested in and if recommended he/she will watch these movies as well.

**How:** Using user's past rating data to train Item-Item Collaborative Filtering to predict top N movies the user is likely to assign a high rating.

<br>

<p align="center">
    <img src="https://miro.medium.com/max/1132/1*N0-ikjPv4RUVvS-6KCgLPg.jpeg?raw=true"
  width=699" height="400">
</p> 
                         
## Case Study 
In this case study we build a simple movie Recommender Systems that generates top N movie recommendations when a single input movie is provided.

The <a href="https://github.com/TatevKaren/data-science-popular-algorithms/blob/main/TopN_MovieRecommender.pdf"> Case Study Paper</a> and <a href="https://github.com/TatevKaren/DataScience/blob/main/TopN_MovieRecommender.py">Python Code</a> contain the followin information<br>

- Data processing
- Descriptive Statistics
- Collaborative Filtering (User-User and Item-Item)
- Item-based Collaborative Filtering
- K-Nearest Neighbors
- Model Evaluation (Cosine Similarity)

# Data (Movie Lens
In this case study we use the MovieLens dataset consisting of 20000263 ratings and 465564 tag applications across 27278 movies spanning a period of 20 years, from January 1995 to March 2015 1. This data was generated and used by Harper and Konstan (2015). Each row in the Rating data is a tuple with 4 components or variables and each row in the Movie data is a tuple consisting of 3 components or variables. Data consists of the following variables:
- userId
- movieId
- rating
- timestamp
- movieId
- title
- genre

# Methodology
To generate movie recommendations we usee Collaborativ Filtering algoritm. The idea behind this approach is that similar users are ex- pected to assign similar ratings to the same item. In case of Item-based CF, the system finds similar items and assuming that similar items will be rated in a similar way by the same person, it predicts the rating corresponding to a user assigned to that item. Usu- ally, the Item-based CF is preferred over the User-based CF because users can change their preferences and choices (aging, change of environment) whereas items (relatively) does not change over time.

Assuming that, if a user likes a certain movie, then she may like similar movies in the future, we will focus on Item-Item CF recommender systems in order to find similar movies corresponding to a movie to make recommendations. To find similar movies, we use Cosidefine the concept of similarity or neighborhoods. One can select all the neighbors or select random neighbors to be in the neighborhood. One of the most popular way of finding similar items in the field of recommender systems is in terms of K-Nearest Neighbor (KNN) approach, which selects top-K neighbors that are most similar to the target item.

<p align="center">
    <img src="https://github.com/TatevKaren/TatevKaren-data-science-portfolio/blob/main/Movie_Recommender/Movie%20Similarity.png?raw=true"
  width=450" height="250">
</p> 
<p align="center">
<img src="https://github.com/TatevKaren/TatevKaren-data-science-portfolio/blob/main/Movie_Recommender/Recommender%20Model%20Evaluation.png?raw=true"
  width=700" height="800">
</p> 
                              
<p align="center">
<img src="https://github.com/TatevKaren/TatevKaren-data-science-portfolio/blob/main/Movie_Recommender/Recommender%20Model%20Sample%20Output.png?raw=true"
  width=600" height="200">
</p> 





