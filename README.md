# Movie-Ratings-Analysis
Introduction to Data Science Course Capstone Project

## Work
The final report can be found in 'Movie Ratings Analysis Report.pdf'.
The code used for analysis can be found in 'movieRatingsAnalysis.py'.

## Goal
The goal of this project was to analyze the impact of different weighted characteristics on the enjoyment of movies.

## Dataset:
The 'movieReplicationSet.csv' dataset contains viewer ratintgs for 400 different movies and their responses to 76 questions about their character. 

The rows represent:
* Row 1: Movie Titles and Questions
* Row 2 - 1098: Participant Responses

The columns represent:
* Column 1 - 400: Movies Ratings (0-4)
* Column 401 - 420: Questions related to Sensation Seeking Behaviors (1-5)
* Column 421 - 464: Questions related to Personality (1-5)
* Column 465 - 474: Questions related to Movie Experiences (1-5)
* Column 475: Gender Identity (1 = female, 2 = male, 3 = self-described)
* Column 476: Only Child (0 = no, 1 = yes, -1 = no response)

## Questions:
1) What is the relationship between sensation seeking and movie experience?
2) Is there evidence of personality types based on the data of these research participants? If so, characterize these types both quantitatively and narratively.
3) Are movies that are more popular rated higher than movies that are less popular?
4) Is enjoyment of ‘Shrek (2001)’ gendered, i.e. do male and female viewers rate it differently?
5) Do people who are only children enjoy ‘The Lion King (1994)’ more than people with siblings?
6) Do people who like to watch movies socially enjoy ‘The Wolf of Wall Street (2013)’ more than those who prefer to watch them alone?
7) There are ratings on movies from several franchises ([‘Star Wars’, ‘Harry Potter’, ‘The Matrix’, ‘Indiana Jones’, ‘Jurassic Park’, ‘Pirates of the Caribbean’, ‘Toy Story’, ‘Batman’]) in this dataset. How many of these are of inconsistent quality, as experienced by viewers?
8) Build a prediction model of your choice (regression or supervised learning) to predict movie ratings (for all 400 movies) from personality factors only. Make sure to use cross-validation methods to avoid overfitting and characterize the accuracy of your model.
9) Build a prediction model of your choice (regression or supervised learning) to predict movie ratings (for all 400 movies) from gender identity, sibship status and social viewing preferences (columns 475-477) only. Make sure to use cross-validation methods to avoid overfitting and characterize the accuracy of your model.
10) Build a prediction model of your choice (regression or supervised learning) to predict movie ratings (for all 400 movies) from all available factors that are not movie ratings (columns 401-477). Make sure to use cross-validation methods to avoid overfitting and characterize the accuracy of your model.
