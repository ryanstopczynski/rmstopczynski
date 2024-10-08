# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:09:54 2022

@author: stopc
"""

#Final Project 
#Professor Arindam Ray
#ISM 4641 Python for Business Analytics
#Ryan Stopczynski

#%%
#Dataset

#10,000 Chess Games with Centipawn Loss
#https://www.kaggle.com/datasets/tompaulat/10000-chess-games-with-centipawn-loss?resource=download

#Average centipawn loss is the number of hundredths of a pawn by which a player deviated 
#from the most accurate move calculated by a computer. Average centipawn loss is the number
#of hundredths of a pawn by which a player deviated from the most accurate move calculated
#by a computer. 

#Elo rating is a system that measures the relative strength of a player, each player's Elo rating
#is represented by a number that reflects that person's results in previous rated games.

#%%
#Research Questions

#I want to analyze if centipawn loss has an impact on the result of a high elo chess match.
#If not centipawn loss, I want to anlyze what features have an impact on the result of a high elo chess match.
    #Is the result based on the ratings of the players?
    #Is the result based on the elo of the players
    #Is the result based on the piece color of the players?
    #Is the result based on the moves/rounds of the match?
#I want to analyze the effictiveness of predicting a result of a high elo chess match based on a combination of features.

#%%
#Import Dataset into Python (Laptop)
#Read a comma-separated values (csv) file into DataFrame

#import pandas as pd

#chess = pd.read_csv(r'C:\Users\stopc\OneDrive - University of South Florida\chess.csv')

#%%
#Import Dataset into Python (PC)
#Read a comma-separated values (csv) file into DataFrame

import pandas as pd

chess = pd.read_csv(r'C:\Users\stopc\OneDrive - University of South Florida\Documents\University\OneDrive - University of South Florida\University\Junior 02-03\Fall 2022\ISM 4641\chess.csv')

#%%
#Data Head
#Returns the first n rows for the object based on position

print(chess.head())

#%%
#Data Info
#Prints information about a DataFrame including the index dtype and columns, non-null values and memory usage

print(chess.info())

#%%
#Decribe Data
#Generates descriptive statistics including those that summarize the central tendency, dispersion and shape of a dataset’s
#distribution, excluding NaN values

print(chess.describe().T)

#%%
#Dataset Shape
#Return a tuple representing the dimensionality of the DataFrame

print(chess.shape)

#%%
#Data Types
#Returns a Series with the data type of each column

print(chess.dtypes)

#%%
#Null Values
#Return the sum of a boolean same-sized object indicating if the values are NA

print(chess.isna().sum())

#%%
#Columns
#Indicates variable as a list of the DataFrame columns

cols = list(chess.columns)

print(cols)

#%%
#Unique
#Replaces integers with strings and classifies column as unique

chess['Result'] = chess['Result'].replace('0', 'Black Win').replace('1', 'Draw').replace('2', 'White Win')
chess['Result'].unique()

#%%
#Count
#Return a series containing counts of unique values

print(chess['Result'].value_counts())

#%%
#Impute (Replace) Null Values
#Fill NA/NaN values using the specified method.
#Reurn sum of NA/NaN values

for col in cols:
    if chess[col].dtypes == object:
        chess[col] = chess[col].fillna(chess[col].mode()[0])
    else:
        chess[col] = chess[col].fillna(chess[col].mean())

#If dtype == object then fill values using mode
#The mode of a set of values is the value that appears most often

#If dtype != object then fill values using mean
#The mean of a set of values is the average of all values that appear

print(chess.isna().sum())

#%%
#Convert String to Float

col_list=[]
for i in chess.columns:
    if((chess[i].dtypes=='object') & (i!='Result')):
        col_list.append(i)
        
#If the dtype of column is object and not Result then append to col_list

print(col_list)

#%%
#Preprocessing
#Convert the raw data into a clean data set

#Label Encoder

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()

for i in col_list:
    chess[i]=labelencoder.fit_transform(chess[i])

#Encode target labels with value between 0 and n_classes-1

#%%
#Heatmap
#Show correlation between fetures

import seaborn as sns
import matplotlib.pyplot as plt

corr = chess.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

#Each square shows the correlation between the variables on each axis. Correlation ranges from -1 to +1. 
#Values closer to zero means there is no linear trend between the two variables. 
#The close to 1 the correlation is the more positively correlated they are; that is as one increases so does the other 
#and the closer to 1 the stronger this relationship is. 
#A correlation closer to -1 is similar, but instead of both increasing one variable will decrease as the other increases. 
#The diagonals are all 1/dark green because those squares are correlating each variable to itself 
#(so it's a perfect correlation). 
#For the rest the larger the number and darker the color the higher the correlation between the two variables. 
#The plot is also symmetrical about the diagonal since the same two variables are being paired together in those squares.

#%%
#Labeling dependent and independent

from sklearn.model_selection import train_test_split

x = chess.drop(columns = ['Result'], axis = 1)
print(x.head())
y = chess['Result']
print(y.head())

#Y (dependant)(value we are trying to predict) becomes just the Result column
#X (independant) becomes all columns excpet for Result

print(x.shape, y.shape)

#%%
#Train Test Split
#Split arrays or matrices into random train and test subsets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 123)

#The test size indicates that there is an 80/20 split, 20% of the data will be used in testing

print(x_train.shape, x_test.shape)

#%%
#Decision Tree Classifier 
#Most Accurate Depth

#Supervised Learning (Result is the target variable)
#Muli-class Classification (Target (Result) has 3 possibilities (Black Win, Draw, White Win))

from sklearn import tree 
from sklearn import metrics

#Creates a decision tree classifier

clf = tree.DecisionTreeClassifier(max_depth = 9, random_state=123) #Most Accurate Depth
clf.fit(x_train, y_train)
y_train_pred = clf.predict(x_train)
y_pred = clf.predict(x_test)

print(y_pred)

#%%
#Metrics
#APIs for evaluating the quality of a model’s predictions

print(metrics.confusion_matrix (y_test, y_pred)) #Compute confusion matrix to evaluate the accuracy of a classification
print(metrics.accuracy_score(y_test, y_pred)) #Accuracy classification score
print(metrics.precision_score(y_test, y_pred, average = None)) #Compute the precision
#The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
print(metrics.recall_score(y_test, y_pred, average = None)) #Compute the recall
#The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.

#%%
#Visualizing Confusion Matrix
#Plotting Train and Test True Confusion Matrix

from sklearn.metrics import confusion_matrix

def plot_confusionmatrix(y_train_pred,y_train,dom):
    print(f'{dom} confusion matrix')
    cf = confusion_matrix(y_train_pred,y_train)
    sns.heatmap(cf,annot=True,cmap='Blues',fmt='g')
    plt.tight_layout()
    plt.show() 

plot_confusionmatrix(y_train_pred,y_train,dom='True')
plot_confusionmatrix(y_test,y_test,dom='True')

#Top is Training Data
#Bottom is Testing Data

#%%
#Plotting
#Decision Tree Classifier
#No Max Depth
#Gini impurity is a function that determines how well a decision tree was split

tree.plot_tree(clf, feature_names = cols, class_names = chess['Result'])
plt.savefig('accurate_dtClf.pdf')

#Incomprehensible Model
#%%
#Evaluation
#Shows Precsion, Recall, and F1-Score (Combination of Precision and Recall)

from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test,y_pred))

#%%
#Importance
#Shows importance of each feature on Outcome

for name, score in zip(chess[cols], clf.feature_importances_):
 print(name, score)

#Only Moves and White Av CP Loss were significant

#%%

#However, this is not the final analystical model
#Although most accurate, the number of nodes, branches, and leaves makes it impossible to evaluate
#Also with such a high accuracy score there is bound to be overfitting
#So, the max depth must be quantified to only include the features with the highest importance
#This being Moves and White Av CP Loss

#%%
#Decision Tree Classifier 
#Most Important Depth

clf = tree.DecisionTreeClassifier(max_depth = 2, random_state=123)
clf.fit(x_train, y_train)
y_train_pred = clf.predict(x_train)
y_pred = clf.predict(x_test)

print(y_pred)

#%%
#Metrics

print(metrics.confusion_matrix (y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))
print(metrics.precision_score(y_test, y_pred, average = None))
print(metrics.recall_score(y_test, y_pred, average = None))

#Lower accuracy, precision, and recall scores

#%%
#Plotting

tree.plot_tree(clf, feature_names = cols, class_names = chess['Result'])
plt.savefig('important_dtClf.pdf')

#Readable Model

#%%
#Evaluation

cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test,y_pred))

#Lower Precision, Recall, and F1-Scores

#%%
#Importance

for name, score in zip(chess[cols], clf.feature_importances_):
 print(name, score)
 
 #Only shows Moves and White Av CP Loss
 
#%%
#Conclusion

#Q.
#I want to analyze if centipawn loss has an impact on the result of a high elo chess match.

#A.
#Yes, White Average Centipawn Loss has the second largest impact on the result of a high elo chess match based on the 
#analytical model.

#Q.
#If not centipawn loss, I want to anlyze what features have an impact on the result of a high elo chess match.
    #Is the result based on the ratings of the players?
    #Is the result based on the elo of the players
    #Is the result based on the piece color of the players?
    #Is the result based on the moves/rounds of the match?

#A.
#The impact of most other features is negligible except for moves, which is the biggest contributing factor to the outcome 
#of a high elo chess match based on the analytical model. Even, the third highest contributing factor being Black ELO is low
#realtive to the two highest importances and therfore does not have a significant impact on on the outcome.

#Q.
#I want to analyze the effictiveness of predicting a result of a high elo chess match based on a combination of features.

#A.
#The accuracy of predicting the result of a high elo chess match based on a specified combination of features is around 94%.
#However, when favoring readability over accuracy it falls to 85%.
#An accuracy score that falls between 80%-90% is realistic and denotes an excellent model.
#An accuracy score that falls between 90%-100% is unrealistic and proably has overfitting issues.

#%%
#Rationale

#Chess is a dynamic game that is heavily nuanced. It is impossible to predict the outcome of a match based on extentuating 
#variable such as ELO. However, when taking into account factors that occur within a game, and account for how a player is
#actually playing it becomes possible to predict. The moves in a given match and how these moves impact the centipawn loss
#are direct contributing factors that can be used to predict the result of the match. 