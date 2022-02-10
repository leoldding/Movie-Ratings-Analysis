# Movie-Ratings-Analysis
Introduction to Data Science Course Capstone Project

This project was the capstone project for the Introduction to Data Science course at NYU taught by Professor Pascal Wallisch in the fall of 2021. 
The final product / analysis is contained within "Capstone Project.pdf" and all of the code is found within "capstone.py".

Dataset:
Contains 400 movies and 1097 individuals.
- Row 1: Movie Titles and Questions
- Row 2 - 1098: Participant Responses

- Column 1 - 400: Movies Ratings (0-4)
- Column 401 - 420: Questions related to Sensation Seeking Behaviors (1-5)
- Column 421 - 464: Questions related to Personality (1-5)
- Column 465 - 474: Questions related to Movie Experiences (1-5)
- Column 475: Gender Identity (1 = female, 2 = male, 3 = self-described)
- Column 476: Only Child (0 = no, 1 = yes, -1 = no response)

Students were asked to respond to the ten questions listed below:

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
