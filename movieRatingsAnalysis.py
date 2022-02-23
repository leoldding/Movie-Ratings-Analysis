#%% Imports
import numpy as np
import pandas as pd
import math
from scipy import stats as scp
from scipy.stats import kstest
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#%% Data Loading

data = np.genfromtxt("data/movieReplicationSet.csv", delimiter = ',', skip_header = 1)
headers = np.genfromtxt("data/movieReplicationSet.csv", delimiter = ',', max_rows = 1, dtype = str)

#%% Question 1 

# load pertinent data
sensationData = data[:, 400:420]
movieExperienceData = data[:, 464:474]

# delete nan rows and zscore
deleteSensationData = np.where(np.isnan(sensationData))[0]
deleteMovieExperienceData = np.where(np.isnan(movieExperienceData))[0]

sensationNanZScoreData = scp.zscore(np.delete(sensationData, deleteSensationData, 0))
movieExperienceNanZScoreData = scp.zscore(np.delete(movieExperienceData, deleteMovieExperienceData, 0)) 

# run PCA
sensationPCA = PCA().fit(sensationNanZScoreData)
sensationPCAEigen = sensationPCA.explained_variance_
sensationPCACovar = sensationPCAEigen / sum(sensationPCAEigen) * 100
sensationPCALoadings = sensationPCA.components_ * -1
sensationPCARotated = sensationPCA.fit_transform(sensationNanZScoreData)

movieExperiencePCA = PCA().fit(movieExperienceNanZScoreData)
movieExperiencePCAEigen = movieExperiencePCA.explained_variance_
movieExperiencePCACovar = movieExperiencePCAEigen / sum(movieExperiencePCAEigen) * 100
movieExperiencePCALoadings = movieExperiencePCA.components_ * -1
movieExperiencePCARotated = movieExperiencePCA.fit_transform(movieExperienceNanZScoreData)

# determine better questions
sensationPCNum = 0
for i in sensationPCAEigen:
    if i > 1:
        sensationPCNum = sensationPCNum + 1

sensationExpVar = 0
for i in range(sensationPCNum):
    sensationExpVar = sensationExpVar + sensationPCACovar[i]
    

counter = 1
for i in range(400,420):
    print(str(counter) + " " + headers[i])
    counter = counter + 1

for i in range(sensationPCNum):
    plt.bar(np.linspace(1,20,20), sensationPCALoadings[i,:])
    plt.xlabel('Question')
    plt.ylabel('Loading')
    plt.title('Sensation Seeking PC' + str(i+1))
    plt.show()
    
"""
SSPC1: -2,-3,-11,-12,-13,-20
    No Exhiliration
    
SSPC2: -3,-13,-20
    No Aerial Activities
    
SSPC3: 10,14
    Horror is Fun

SSPC4: 16,17
    Likes Order and Routine
    
SSPC5: 1,-8,-15,19
    Individual Risky Activities

SSPC6: -18
    Doesn't Ride Motorcycle
"""
SSPCs = np.array(['Lack of Exhiliration', 'No Aerial Related Activities', 'Horror is Fun', 'Preference Towards Order and Routine', 'Risky Activities'])

movieExperiencePCNum = 0
for i in movieExperiencePCAEigen:
    if i > 1:
        movieExperiencePCNum = movieExperiencePCNum + 1
        
movieExperienceExpVar = 0
for i in range(movieExperiencePCNum):
    movieExperienceExpVar = movieExperienceExpVar + movieExperiencePCACovar[i]
    

counter = 1
for i in range(464,474):
    print(str(counter) + " " + headers[i])
    counter = counter + 1

for i in range(movieExperiencePCNum):
    plt.bar(np.linspace(1,10,10), movieExperiencePCALoadings[i,:])
    plt.xlabel('Question')
    plt.ylabel('Loading')
    plt.title('Movie Experience PC' + str(i+1))
    plt.show()
    
"""
MEPC1: -5,-7,-8,-9,-10
    lack of Immersion
    
MEPC2: -2,-3,-6
    Good Storytelling
"""

# run correlations
delAll = np.unique(np.concatenate((deleteSensationData, deleteMovieExperienceData),0))

delAllSensationData = np.delete(sensationData, delAll,0)
delAllMovieExperienceData = np.delete(movieExperienceData, delAll, 0)

sensationRotated = sensationPCA.fit_transform(scp.zscore(delAllSensationData))
movieExperienceRotated = movieExperiencePCA.fit_transform(scp.zscore(delAllMovieExperienceData))


for i in range(movieExperiencePCNum):
    for j in range(sensationPCNum):
        corr, pval = scp.spearmanr(sensationRotated[:, j], movieExperienceRotated[:,i])
        if pval < 0.05:
            plt.scatter(sensationRotated[:,j], movieExperienceRotated[:,i], s = 3, color = 'tab:blue')
            plt.plot(np.unique(sensationRotated[:,j]), np.poly1d(np.polyfit(sensationRotated[:,j], movieExperienceRotated[:,i], 1))(np.unique(sensationRotated[:,j])), linewidth = 4, color = 'tab:orange', label = 'r = ' + str(corr.round(3)))
            plt.xlabel('PCA Rotated ' + SSPCs[j] + ' Data')
            plt.ylabel('PCA Rotated Lack of Immersion Data')
            plt.title('Lack of Immersion vs ' + SSPCs[j])
            plt.legend()
            plt.show()
            print(str(i) + " " + str(j) + " " + str(corr.round(4)) + " " + str(pval.round(4)))


#%% Question 2

personalityData = data[:,420:464]
deletePersonalityData = np.where(np.isnan(personalityData))[0]
personalityNanZScoreData = scp.zscore(np.delete(personalityData, deletePersonalityData, 0))
personalityPCA = PCA().fit(personalityNanZScoreData)
personalityPCAEigen = personalityPCA.explained_variance_
personalityPCACovar = personalityPCAEigen / sum(personalityPCAEigen) * 100
personalityPCALoadings = personalityPCA.components_ * -1
personalityPCARotated = personalityPCA.fit_transform(personalityNanZScoreData)


personalityPCNum = 0
for i in personalityPCAEigen:
    if i > 1:
        personalityPCNum = personalityPCNum + 1

personalityExpVar = 0
for i in range(personalityPCNum):
    personalityExpVar = personalityExpVar + personalityPCACovar[i]
    

counter = 1
for i in range(420,464):
    print(str(counter) + " " + headers[i])
    counter = counter + 1
    
    
for i in range(personalityPCNum):
    plt.bar(np.linspace(1,44,44), personalityPCALoadings[i,:])
    plt.xlabel('Question')
    plt.ylabel('Loading')
    plt.title('Personality PC' + str(i+1))
    plt.show()
    
"""
PPC1: 11,16,36
    Extrovert
        
PPC2: 19,20,29,30
    Emotional

PPC3: -1,6,21   
    Introvert

PPC4: 3,13,-17,-18,28
    Great Worker

PPC5: -5,-9,17,19,22,-25,-27,32,-34,39
    Kindhearted and Energetic

PPC6: 8,23,24,27,41
    Some Interests

PPC7: 30,41
    Artistic Nature

PPC8: 15,17,34,-43    
    Calm and Focused
"""

# Init:
numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
Q = np.empty([numClusters,1])*np.NaN # init container to store sums

# Compute kMeans:
for ii in range(2, 11): # Loop through each cluster (from 2 to 10!)
    kMeans = KMeans(n_clusters = int(ii)).fit(personalityPCARotated[:,0:8]) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(personalityPCARotated[:,0:8],cId) # compute the mean silhouette coefficient of all samples
    Q[ii-2] = sum(s) # take the sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,250)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(Q[ii-2]))) # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot padding

idealK = 2

kMeans = KMeans(n_clusters = idealK).fit(personalityPCARotated[:,0:8]) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_


#%% Question 3 

popularity = np.array([])
means = np.array([])
medians = np.array([])

for i in range(400):
    popularity = np.append(popularity, np.count_nonzero(~np.isnan(data[:,i])))
    means = np.append(means, np.nanmean(data[:,i]))
    medians = np.append(medians, np.nanmedian(data[:,i]))
    
plt.plot(popularity, means, 'o', markersize = 5, label = 'Average Ratings')
plt.plot(popularity, medians, 'o', markersize = 5, label = 'Median Ratings')
plt.xlabel('Popularity (Amount of Ratings)')
plt.ylabel('Ratings')
plt.title('Popularity by Ratings')
plt.legend()
plt.show()

avgPop = np.mean(popularity)

lowPopMed = np.array([])
lowPop = np.array([])
highPopMed = np.array([])
highPop = np.array([])

for i in range(len(popularity)):
    if popularity[i] < avgPop:
        lowPopMed = np.append(lowPopMed, medians[i])
        lowPop = np.append(lowPop, popularity[i])
    else:
        highPopMed = np.append(highPopMed, medians[i])
        highPop = np.append(highPop, popularity[i])
        
popks, popksp = kstest(lowPopMed, highPopMed)

plt.hist(lowPopMed, bins = np.arange(1.85,4.35, 0.5), rwidth = 0.4, label = 'Low Popularity Medians', color = "tab:cyan")
plt.plot(np.full(130,np.mean(lowPopMed)), np.arange(0,130), label = 'Mean Median of Low Popularity', color = 'blue' )
plt.hist(highPopMed, bins = np.arange(1.65,4.15, 0.5), rwidth = 0.4, label = 'High Popularity Medians', color = "tab:orange")
plt.plot(np.full(130,np.mean(highPopMed)), np.arange(0,130), label = 'Mean Median of High Popularity', color = 'orange')
plt.legend()
plt.show()

#popularityu, popularityp = scp.mannwhitneyu(lowPop, highPop)

#%% TTest for 4,5,6
def ttest(movie, criteria, cnum1, cnum2, movieName, criteriaName, criteria1name, criteria2name):
    delAll = np.array([])
    
    for i in range(len(movie)):
        if np.isnan(movie[i]) or (criteria[i] != cnum1 and criteria[i] != cnum2) or np.isnan(criteria[i]):
            delAll = np.append(delAll,i)
    
    movieClean = np.delete(movie,delAll.astype(int))
    criteriaClean = np.delete(criteria,delAll.astype(int))
    
    crit1 = np.where(criteriaClean == cnum1)[0]
    crit2 = np.where(criteriaClean == cnum2)[0]
    
    movieCrit1 = movieClean[crit1]
    movieCrit2 = movieClean[crit2]
    
    ## maybe add u and p labels
    plt.hist(movie, bins = np.arange(-0.25,4.75,0.5), rwidth = 0.8, label = movieName, color = 'tab:cyan')
    plt.hist(movieCrit1, bins = np.arange(-0.22,4.78,0.5), rwidth = 0.5, label = criteria1name, color = 'tab:orange')    
    plt.hist(movieCrit2, bins = np.arange(-0.28,4.72,0.5), rwidth = 0.5, label = criteria2name, color = 'tab:red')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.title(movieName + " based on " + criteriaName)
    plt.legend()
    plt.show()
    
    u,p = scp.mannwhitneyu(movieCrit1,movieCrit2)
    ks,ksp = kstest(movieCrit1, movieCrit2)
    return u,p,ks,ksp
    

#%% Question 4 

index = -1
for i in range(400):
    if headers[i] == "Shrek (2001)":
        index = i
      
shreku, shrekp, shrekks, shrekksp = ttest(data[:,index], data[:,474],1,2, "Shrek", "Gender", "Female", "Male")


#%% Question 5 

index = -1
for i in range(400):
    if headers[i] == "The Lion King (1994)":
        index = i
        
lionKingu, lionKingp, lionKingks, lionKingksp = ttest(data[:,index], data[:,475],0,1, "Lion King", "Only Children", "Has Siblings", "No Siblings")


#%% Question 6 

index = -1
for i in range(400):
    if headers[i] == "The Wolf of Wall Street (2013)":
        index = i
        
wallStreetU, wallStreetp, wallStreetks, wallStreetksp = ttest(data[:,index], data[:,476],1,0, "Wolf of Wall Street", "Social Viewing", "Viewing Together", "Viewing Alone")

#%% Question 7 

starWars = data[:,[21,174,342,273,93,336]]
starWars = np.delete(starWars, np.where(np.isnan(starWars))[0],0)
fSW,pSW = scp.kruskal(starWars[:,0], starWars[:,1], starWars[:,2], starWars[:,3], starWars[:,4], starWars[:,5])
starWarsMeans = np.array([])
starWarsMedians = np.array([])
for i in range(len(starWars[0])):
    starWarsMeans = np.append(starWarsMeans, np.mean(starWars[:,i]))
    starWarsMedians = np.append(starWarsMedians, np.median(starWars[:,i]))
plt.subplot(3,3,1)
plt.plot(np.arange(0,6), starWarsMeans)
plt.plot(np.arange(0,6), starWarsMedians)
plt.title("Star Wars")
plt.ylim(2,4)

harryPotter = data[:,[230,394,387,258]]
harryPotter = np.delete(harryPotter, np.where(np.isnan(harryPotter))[0],0)
fHP,pHP = scp.kruskal(harryPotter[:,0], harryPotter[:,1], harryPotter[:,2], harryPotter[:,3])
harryPotterMeans = np.array([])
harryPotterMedians = np.array([])
for i in range(len(harryPotter[0])):
    harryPotterMeans = np.append(harryPotterMeans, np.mean(harryPotter[:,i]))
    harryPotterMedians = np.append(harryPotterMedians, np.median(harryPotter[:,i]))
plt.subplot(3,3,2)
plt.plot(np.arange(0,4), harryPotterMeans)
plt.plot(np.arange(0,4), harryPotterMedians)
plt.title("Harry Potter")
plt.ylim(2,4)

theMatrix = data[:,[306,172,35]]
theMatrix = np.delete(theMatrix, np.where(np.isnan(theMatrix))[0],0)
fTM,pTM = scp.kruskal(theMatrix[:,0], theMatrix[:,1], theMatrix[:,2])
theMatrixMeans = np.array([])
theMatrixMedians = np.array([])
for i in range(len(theMatrix[0])):
    theMatrixMeans = np.append(theMatrixMeans, np.mean(theMatrix[:,i]))
    theMatrixMedians = np.append(theMatrixMedians, np.median(theMatrix[:,i]))
plt.subplot(3,3,3)
plt.plot(np.arange(0,3), theMatrixMeans)
plt.plot(np.arange(0,3), theMatrixMedians)
plt.title("The Matrix")
plt.ylim(2,4)

indianaJones = data[:,[33,32,4,142]]
indianaJones = np.delete(indianaJones, np.where(np.isnan(indianaJones))[0],0)
fIJ,pIJ = scp.kruskal(indianaJones[:,0], indianaJones[:,1], indianaJones[:,2], indianaJones[:,3])
indianaJonesMeans = np.array([])
indianaJonesMedians = np.array([])
for i in range(len(indianaJones[0])):
    indianaJonesMeans = np.append(indianaJonesMeans, np.mean(indianaJones[:,i]))
    indianaJonesMedians = np.append(indianaJonesMedians, np.median(indianaJones[:,i]))
plt.subplot(3,3,4)
plt.plot(np.arange(0,4), indianaJonesMeans)
plt.plot(np.arange(0,4), indianaJonesMedians)
plt.title("Indiana Jones")
plt.ylim(2,4)

jurassicPark = data[:,[370,37,47]]
jurassicPark = np.delete(jurassicPark, np.where(np.isnan(jurassicPark))[0],0)
fJP,pJP = scp.kruskal(jurassicPark[:,0], jurassicPark[:,1], jurassicPark[:,2])
jurassicParkMeans = np.array([])
jurassicParkMedians = np.array([])
for i in range(len(jurassicPark[0])):
    jurassicParkMeans = np.append(jurassicParkMeans, np.mean(jurassicPark[:,i]))
    jurassicParkMedians = np.append(jurassicParkMedians, np.median(jurassicPark[:,i]))
plt.subplot(3,3,5)
plt.plot(np.arange(0,3), jurassicParkMeans)
plt.plot(np.arange(0,3), jurassicParkMedians)
plt.title("Jurassic Park")
plt.ylim(2,4)

piratesOfTheCaribbean = data[:,[351,75,204]]
piratesOfTheCaribbean = np.delete(piratesOfTheCaribbean, np.where(np.isnan(piratesOfTheCaribbean))[0],0)
fPC,pPC = scp.kruskal(piratesOfTheCaribbean[:,0], piratesOfTheCaribbean[:,1], piratesOfTheCaribbean[:,2])
piratesOfTheCaribbeanMeans = np.array([])
piratesOfTheCaribbeanMedians = np.array([])
for i in range(len(piratesOfTheCaribbean[0])):
    piratesOfTheCaribbeanMeans = np.append(piratesOfTheCaribbeanMeans, np.mean(piratesOfTheCaribbean[:,i]))
    piratesOfTheCaribbeanMedians = np.append(piratesOfTheCaribbeanMedians, np.median(piratesOfTheCaribbean[:,i]))
plt.subplot(3,3,6)
plt.plot(np.arange(0,3), piratesOfTheCaribbeanMeans)
plt.plot(np.arange(0,3), piratesOfTheCaribbeanMedians)
plt.title("Pirates of the Caribbean")
plt.ylim(2,4)

toyStory = data[:,[276,157,171]]
toyStory = np.delete(toyStory, np.where(np.isnan(toyStory))[0],0)
fTS,pTS = scp.kruskal(toyStory[:,0], toyStory[:,1], toyStory[:,2])
toyStoryMeans = np.array([])
toyStoryMedians = np.array([])
for i in range(len(toyStory[0])):
    toyStoryMeans = np.append(toyStoryMeans, np.mean(toyStory[:,i]))
    toyStoryMedians = np.append(toyStoryMedians, np.median(toyStory[:,i]))
plt.subplot(3,3,7)
plt.plot(np.arange(0,3), toyStoryMeans)
plt.plot(np.arange(0,3), toyStoryMedians)
plt.title("Toy Story")
plt.ylim(2,4)

batman = data[:,[181,46,235]]
batman = np.delete(batman, np.where(np.isnan(batman))[0],0)
fB,pB = scp.kruskal(batman[:,0], batman[:,1], batman[:,2])
batmanMeans = np.array([])
batmanMedians = np.array([])
for i in range(len(batman[0])):
    batmanMeans = np.append(batmanMeans, np.mean(batman[:,i]))
    batmanMedians = np.append(batmanMedians, np.median(batman[:,i]))
plt.subplot(3,3,8)
plt.plot(np.arange(0,3), batmanMeans)
plt.plot(np.arange(0,3), batmanMedians)
plt.title("Batman")
plt.ylim(2,4)

plt.subplot(3,3,9)
plt.plot(np.arange(0,2), np.full(2,2), label = 'Means')
plt.plot(np.arange(0,2), np.full(2,1), label = 'Medians')
plt.legend()
plt.title("Legend")


plt.tight_layout()
plt.show()


#%% Multiple Regression for 8 9 10

def delAll(x, y):
    delX = np.where(np.isnan(x))[0]
    delY = np.where(np.isnan(y))[0]
    delete = np.unique(np.concatenate((delX,delY),0))
    return np.delete(x, delete, 0), np.delete(y, delete, 0)

def multreg(x, y, personality = False, personalityCols = ([]), sensation = False, sensationCols = ([]), movieExp = False, movieExpCols = ([])):
    regressions = np.array([])
    rsquared = np.array([])
    rmse = np.array([])
    for i in range(len(y[0])):
        predictor, prediction = delAll(x, y[:,i])
        if personality == True:
            predictor[:,personalityCols] = personalityPCA.fit_transform(predictor[:,personalityCols])
        if sensation == True:
            predictor[:,sensationCols] = sensationPCA.fit_transform(predictor[:,sensationCols])
        if movieExp == True:
            predictor[:,movieExpCols] = movieExperiencePCA.fit_transform(predictor[:,movieExpCols])
        xtrain, xtest, ytrain, ytest = train_test_split(predictor, prediction, test_size = 0.2, random_state = 42)
        lr = LinearRegression()
        lr.fit(xtrain, ytrain)
        yprediction = lr.predict(xtest)
        regressions = np.append(regressions, lr)
        rsquared = np.append(rsquared, r2_score(ytest,yprediction))
        rmse = np.append(rmse, np.sqrt(mean_squared_error(ytest,yprediction)))
    return regressions, rsquared, rmse


#%% Question 8 

regressions1, rsquared1, rmse1 = multreg(data[:,420:464], data[:, :400], personality = True, personalityCols = np.arange(0,44))

small = min(rmse1)
big = max(rmse1)
count = 0
for i in range(len(rmse1)):
    if rmse1[i] < 1:
        count = count + 1
    if rmse1[i] == small:
        print('Smallest RMSE for ' + headers[i] + ' with ' + str(rmse1[i]))
    if rmse1[i] == big:
        print('Biggest RMSE for ' + headers[i] + ' with ' + str(rmse1[i]))
print(count)



#%% Question 9 

regressions2, rsquared2, rmse2 = multreg(data[:, 474:477], data[:, :400])

small = min(rmse2)
big = max(rmse2)
count = 0
for i in range(len(rmse2)):
    if rmse2[i] < 1:
        count = count + 1
    if rmse2[i] == small:
        print('Smallest RMSE for ' + headers[i] + ' with ' + str(rmse2[i]))
    if rmse2[i] == big:
        print('Biggest RMSE for ' + headers[i] + ' with ' + str(rmse2[i]))
print(count)
#%% Question 10

regressions3, rsquared3, rmse3 = multreg(data[:, 400:477], data[:, :400], personality = True, personalityCols = np.arange(20,64), sensation = True, sensationCols = np.arange(0,20), movieExp = True, movieExpCols = np.arange(64,74))

small = min(rmse3)
big = max(rmse3)
count = 0
for i in range(len(rmse3)):
    if rmse3[i] < 1:
        count = count + 1
    if rmse3[i] == small:
        print('Smallest RMSE for ' + headers[i] + ' with ' + str(rmse3[i]))
    if rmse3[i] == big:
        print('Biggest RMSE for ' + headers[i] + ' with ' + str(rmse3[i]))
print(count)