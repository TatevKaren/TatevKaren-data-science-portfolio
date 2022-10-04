
<br>
<p align="center">
   
   <a href="">
        <img src="https://img.shields.io/badge/-Spotify-success"></a>
   <a href="">
        <img src="https://img.shields.io/badge/Case%20Study-Statistical%20Inference-orange"></a>
   <a href="">
        <img src="https://img.shields.io/badge/Case%20Study-Product%20Data%20Science-yellow"></a>
  
  <a href="">
        <img src="https://img.shields.io/badge/-Success%20Metrics-ff69b4"></a>
  <a href="">
        <img src="https://img.shields.io/badge/-Data%20Visualization-green"></a>
  <a href="">
        <img src="https://img.shields.io/badge/Programming-Python-blue"></a>
  
  <a href="">
        <img src="https://img.shields.io/badge/-Feature%20Engineering%20-yellowgreen"></a>
  
  
  
</p>
<br>

# Case Study: What-makes-playlist-successful

**Why** To find out which features maake playlist successful, to identify such playlists aand recommend them to the users to improve customer satisfaction and engagement. 

**How** Use EDA (Exploratory Data Analysis) and Simple Machine Learning to identify the features related to the sucessful playlists.

Python Code: <a href = "https://github.com/TatevKaren/What-makes-playlist-successful/blob/main/spotify_analysis.py"> here </a>

Medium Blog: <a href = "https://towardsdatascience.com/spotify-data-science-case-study-what-makes-a-playlist-successful-28fec482c523"> here </a>

<br>
<br>

---

This is an End-to-end Data Science Case Study with Tips and Python Implementation for Real Life Business Problem: What Makes Spotify Playlist Successful

***Content***
- Defining Business Goal and Technical Goal
- Game Plan
- Defining Success Metrics
- Short Term Success Metrics
- Long Term Success Metrics
- Feature Engineering: Candidates for Successful Playlists
- Data Exploration and Visualization
- Data Modelling
- Conclusion: Answering the Case Study Question

---

## Step 1: Defining Project Goal
The Business Goal here will be to provide the best playlists to Spotify users to :
- 1: increase user satisfaction (decrease the search time and improving the quality of the search results or recommender system's results), 
- 2: increase usage 
- 3: increase the revenue

So, we want to quantify what makes playlist successful. This we can do by defining a target success metrics which will measure the successfullness of the playlist.​Secondly we want to identify key-indicators of successful playlists, which are features differentiating successful playlists from unsuccessful ones.
By identifying what factors make a playlist successful according to the customers and knowing how to measure it. One can then improve the playlist search results or recommender systems for these playlists. The company can even personalize them, such that we provide playlists to the customers that they most likely will be interested in.Hence, we want Spotify users to listen to more playlists and longer. We want to quantify what makes playlist successful. This we can do by defining a target success metrics which will measure the successfullness of the playlist.
Image Source: Andrea Piacquadio


## Step 2: Game Plan or Strategy
The next thing you need to do is to come up with a strategy for conducting your case study with a goal to answer the question asked initially, which is "What Makes Playlist Successful?" and to achieve the objectives stated in the previous step. We will start by looking at the data and at some descriptive statistics to learn more about our data. ​Secondly, we will look at some candidate success metrics, that can help us measure the success of the playlist.
We want to identify key-indicators of successful playlists, which are features differentiating successful playlists from unsuccessful ones.
<img width="483" alt="Screenshot 2022-09-30 at 6 14 33 PM" src="https://user-images.githubusercontent.com/76843403/193312975-0b91a4eb-04bc-4440-9087-ddf7707b23eb.png">


## Step 3: Data and Descriptive Statistics
This is how the raw data looks like, in total 403.366 row (playlists) and 25 columns (features).​ Here we see the top 5 and bottom 5 observations in the data as well as first 4 and last 4 columns.​
Image Source: The Authorremoving outliers from the data
checking for missing value.

<img width="607" alt="Screenshot 2022-09-30 at 6 15 12 PM" src="https://user-images.githubusercontent.com/76843403/193313065-c550d775-0c70-4dd6-8ae9-ea9bbcde400a.png">

**Descriptive Statistics for Some Features**
Let us look at some descriptive statistics. Note that, Spotify's 399 playlists have been excluded from the data when doing calculations.This part is esential to remove the outliers from the data.
We see that there are around 315K unique owners of playlists. On average, playlist has 202 tracks, 88 albums, and 84 artists. We also see, that on average a playlist has quite short title, with only 2 tokens. 
Image Source: The AuthorMoreover, we see that on average playlist has 21 streams while only 12 of these streams are for more than 30seconds. So, on average, 57% of todays playlists streams end up being listened to for more than 30 seconds. This is something that we want to change and increase this percentage.
When we look at the usage related feature statistics, specifically at DAU, its AVG, MIN, MAX we see that there is a large variance in the daily active usage per playlist. There are some playlists with very large number of DAU while there are some with very few or not at all. Similar picture can be seen from Weekly and Monthly active usage. 

![1*_TKgqTU_sb2h9e7HVMsFoA](https://user-images.githubusercontent.com/76843403/193313130-68fa33e6-2525-4092-b5cd-b0b9caa4bfd9.png)


## Step 4: Defining Success Metrics
Let's now discuss a very important part in this analysis and case study, which is defining success metrics to measure the success of the playlist. ​The question is when can we call the playlist successful? Playlist is successful when many users listen to it and listen to it for longer time.​​There are many possible metrics which we can use to measure the succuss of the playlist and also at multiple levels. ​
When can we say that the playlist is successful?
Users listen to the successful playlists a lot
Users listen to the successful playlist for a longer time (longer than 30 seconds)

**What are possible levels for success?**
- Success at playlist level
- Success at the playlist owner level


For ***Short term metrics***, we can use for example the DAUs or WAUs, we can use the number of streams with more than 30s today. Few long term success metrics for playlist can be the MAU, the monthly active users, the Monthlystreams30s, or the monthly owner streams. 
- DAUs: Number of daily active users with streams more than 30 seconds from the playlist
- WAUs: Number of weekly active users with streams more than 30 seconds from the playlist
- Streams30s: Number of streams from playlist today with more than 30s

***Long Term Success Metrics***
- MAUs: Number of monthly active users with a stream over 30 seconds from the playlist this month. Or total number of active users from the playlist in the last 2 months (mau_both_months)
- Monthly_streams30s: Number of streams over 30 seconds by playlist
- Monthly_owner_streams30s: Number of streams over 30 seconds by playlist owner this month

### Chosen Metric
For simplicity, today I chose for WAU as success metrics, which is a good silver lining between the short term and long term performance of a playlist. However, ideally, we want to come up with a custom metrics that contains both short and long term performance of the playlist, and also the owners performance. ​For example, combination of DAU, WAU, MAU, streams30 today, monthly streams and monthly owner streams.​
Ideally, we want to come up with a custom metrics that contains both short and long term performance of the playlist, and also the owners performance. ​For example, combination of DAU, WAU, MAU, streams30 today, monthly streams and monthly owner streams.​


## Step 5: Candidate Features
Now when we know how to measure the performance of the playlist, lets look at few features which might be related to the success of the playlist. Here, I categorised them in 4 groups: Playlist Variety, Categorical Descriptors, Playlist Owners features, and Playlist quality ratios.
Image Source: The AuthorIn terms of Variety of the Playlist, we can look at the number of tracks, number of artists and number of albums of the playlist. In terms of categorical variables describing the playlist, we can look at the Leading Mood or Leading genre of a playlist.
We can also use Sentiment Analysis to measure how positive or negative the tokens of the playlist are.​
One can look at the number of playlists the owner has, which is a proxy of the effort of the owner, ​or create a binary variable indicating whether owner of the playlist has more than 1 playlist which is a proxy of the activity or the engagement of the owner. Finally, one can use some ratios describing the playlist.​
- Number of Tracks
- Number of Artists
- Number of Albums

***Playlist Categorical Features***
- Moods: Leading Mood
- Genres: Leading Genres
- Tokens Sentiment (Senitment Analysis for tokens)

***Playlist Owner Descriptors***
- Number of playlists per owner (proxy for owner effort)
- Owner has more than 1 playlist (proxy for owners activity)

***Playlist Ratios***
- Streams30s to Streams Ratio
- Streams30 to Monthly Streams Ratio
- Monthly_owner_stream30s to monthly_stream30s Ratio

You can use Sentiment Analysis to measure how positive or negative the tokens of the playlist are.​

## Step 6: Data Preparation
Let's now quickly review few data preparation steps before moving on with the Data Exploration and Visualization parts.​
Firstly, we need to be aware of outliers. One of outliers I mentioned earlier was about Spotify owned playlists. But beside this there are also other few other extreme values that have very large streams and usage and we want to exclude them to discover some patters for the average playlists which are more common. Therefore, we remove all playlists in the 99th percentile in the weekly usage.​​
- Checking for Outliers
- Removing 399 Playlists of Spotify with above than average stream
- Removing the rows from the 99th percentile

Next, we check for missing value. Although there are no null values, there are missing values data points in variables especially in Genres and Moods variables​​.
Checking for Missing Values
There are no null values in the data but there are missing values, especially in Mood, Genre variables

Finally, We convert the string or txt representation of the tokens to the list of tokens to use it for counting number of tokens and for performing sentiment analysis, where the playlists with negative sentiment will get low values and playlists with positive sentiment will get high values​.
Information Retrieval and Sentiment Analysis for Tokens
Converting text to a list of strings
Sentiment Score low: Negative vs high: Positive (NLP on tokens)


## Step 7: Data Exploration & Visualization
Let's start with the data exploration and visualization of the data.
Playlists Leading Genre
Here is the histogram of the Leading Genre of the playlist and we see that there are 3 clear Leading Genres with the largest number of playlists. ​
These are the most occurring playlist genres and what we can do is to create binary variable for these 3 genres "Indie Rock", "Rap" and "Pop" and use them as independent variables in a predictive model that will help to discover causation in the genre and the performance of the playlist.​
3 most popular leading genres across all playlists: "Indie Rock", "Rap", "Pop"
Playlists from these 3 genres are more successful?
Binary features from these 3 for genres for future predictive model
<img width="592" alt="Screenshot 2022-09-30 at 6 16 13 PM" src="https://user-images.githubusercontent.com/76843403/193313237-a96d574d-a7c9-42ec-adfb-7b3f3d7b7543.png">
<img width="600" alt="Screenshot 2022-09-30 at 6 16 32 PM" src="https://user-images.githubusercontent.com/76843403/193313282-92c11ae7-042e-4979-8071-4579ec1efbcf.png">

<img width="600" alt="Screenshot 2022-09-30 at 6 16 44 PM" src="https://user-images.githubusercontent.com/76843403/193313318-fffda8cc-176c-458d-8bcb-73bd8ae91025.png">
<img width="554" alt="Screenshot 2022-09-30 at 6 16 56 PM" src="https://user-images.githubusercontent.com/76843403/193313373-ec49ed06-bded-40c4-bc04-f42c6f2bea88.png">

<img width="573" alt="Screenshot 2022-09-30 at 6 17 08 PM" src="https://user-images.githubusercontent.com/76843403/193313396-a0676148-5295-4437-924d-8a592e93ee26.png">
<img width="557" alt="Screenshot 2022-09-30 at 6 17 20 PM" src="https://user-images.githubusercontent.com/76843403/193313429-a1b9ffcc-9296-4f55-9deb-76c301152c72.png">
<img width="555" alt="Screenshot 2022-09-30 at 6 17 33 PM" src="https://user-images.githubusercontent.com/76843403/193313461-fdf207f4-74f5-4faf-b212-daca12625b8a.png">
<img width="575" alt="Screenshot 2022-09-30 at 6 17 49 PM" src="https://user-images.githubusercontent.com/76843403/193313507-8370d7f7-ead5-40ff-ad10-db5fb5ea0b4b.png">

## Step 8: Simple Data Modelling
From Data Exploratory and Visualization analysis, we came up with a list of possible variables that showed a relationship with the success of the playlist. But these variables show a correlation with the success of the playlist.
We need to do one more step to find out whether these and many other features not only are related but also have a statistically significant impact on the success of the playlist since correlation is not causation. Which we will do by using linear regression and this is just a very simple version of it.
For this, we need to create a sample of playlists and the sample size can be determined based on the confidence level, which I chose as 99%​, margin of error of 1% aand this resulted to 16K playlists. ​
Image Source: The AuthorFollowing are the results of running multiple Linear Regression with OLS estimation when using some of the earlier mentioned features and some new features as independent variables and the WAU as the dependent variable. ​
What we can see is that Pop genre has a statistically significant and negative effect on the WAU of the playlist. Also, Number of albums and Number of genres have statistically significant and positive impact on the WAU of playlist. This verifies our earlier findings that Variety matters. ​
- Pop Genre: significant and negative effect ​
- Streams30s: significant and positive effect ​
- Num_albums: significant and positive effect​
- Num_genres: significant and positive effect ​
- Users: significant and positive effect ​

<img width="608" alt="Screenshot 2022-09-30 at 6 18 12 PM" src="https://user-images.githubusercontent.com/76843403/193313577-e7d04392-966b-4c87-b556-813eb6deb2ec.png">


## Step 9: Conclusions
So, using Data Exploration and Visualization we found certain features related with the success of the playlists. It's important to mention that we chose a very simple success metrics, while ideally we want to use more advanced feature. ​
We then used some of the feature used before combined with few more to run a preliminary linear regression to test the earlier hypothesis and find what factors make playlist successful. We found that variety in the playlist, specifically, variety in albums and genres make playlist successful.
Image Source: The AuthorBut we need more of these features and we need more advanced success metric​.


## Step 10: Next Steps
These insights are just the tip of the iceberg. This data contains many interesting and important features, and unfortunately, due to time limitation just few of them have been discussed today. ​
The next step in the project should be to run more advanced Linear Regression to detect statistical significance of many other features, on the playlist success, to get a list of features that increase or decrease the success level of the playlist.​
Image Source: The Author​After that, one can use Supervised Machine Learning to personalize search engine or recommender system that supplies playlists to users. Because we want to show the most successful playlists at the top . Then A/B testing can be used to test the new algorithm to the existing recommender systems.
Additionally, Spotify can build a Reranking Model with Impression Discounting to personalize and improve the search engine​. Finally, in all these, one should not forget about new playlists: (Cold Start Problem)​.


