#%%
#Dataset
#https://www.kaggle.com/datasets/priyanka841/heart-disease-prediction-uci/data

#Heart disease refers to several types of heart conditions.
#The most common type of heart disease in the United States is 
#coronoary artery disease (CAD), which affects the blood flow to the heart.

#Symptoms may include a heart attack, arrhythmia, and heart failure.

#Heart disease is the leading cause of death for men, women, and people of 
#most racial and ethnic groups in the United States.
#One person dies on average every 33 seconds in the United States from heart disease.
#In 2021 about 695,000 individuals in the United States died from heart disease,
#that is 1 in every 5 deaths.

#Heart disease cost the United States about $239.9 billion in 2021 due to the 
#cost of health care services, medicines, and lost of productivity.

#Therfore, it is of paramount importance to create tools for early detection for both financial
#gain and as a moral obiligation to protect the citizens of the United States. 

#This machine learning classification will provide me the necessary information
#to create a prediction dashboard in tableau

#%%
#Import Necessary Libraries into Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import seaborn as sns

#numpy, pandas, matplotlib, sklearn, seaborn

#%%
#Import "Heart Diseease Prediction UCI" into Python
heart = pd.read_csv(r'C:\Users\stopc\OneDrive - University of South Florida\University\Senior 03-04\Spring 2024\ISM 4300\CapstoneProject_RyanStopczynski_20240416\Data\heart.csv')

#%%
#Prints information about a DataFrame including the index dtype and columns, non-null values and memory usage
print(heart.info())

#age: The person's age as a number of years
#sex: The person's biological sex (0 = Female, 1 = Male)
#cp: The experienced chess pain (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Anginal Pain, 4 = Asymptomatic)
#trestbps: The person's given resting blood pressure
#chol: The person's given cholesterol measurment in mg/dl
#fbs: Ther person's given fasting blood sugar (> 120 mg/dl, 0 = False, 1 = True)
#restecg: The person's given resting electrocardiographic measurment (0 = Normal, 1 = Having ST-T wave abnormality, 
#2 = Showing probable or definite left ventricular hypertrophy by Estes' criteria)
#thalach: The person's given maximum heart rate
#exang: Exercise induces angina (0 = No, 1 = Yes)
#oldpeak: ST depression induced by exercise relative to rest
#slope: The slope of the peak from the exercise ST segment (0 = Unsloping, 1 = Flat, 2 = Downsloping)
#ca: The number or major blood vessels (0 = 0, 1 = 1, 2 = 2, 3 = 3)
#thal: Indication of blood disorder called thalassemia (3 = Normal, 6 = Ficed Defect, 7 = Reversable Defect)

#target: Indication of Heart Disease (0 = No, 1 = Yes)

#%%
#Data Head
#Returns the first n rows for the object based on position

print(heart.head())

#%%
#Decribe Data
#Generates descriptive statistics including those that summarize the central tendency, dispersion and shape of a dataset’s
#distribution, excluding NaN values

print(heart.describe().T)

#%%
#Dataset Shape
#Return a tuple representing the dimensionality of the DataFrame

print(heart.shape)

#%%
#Data Types
#Returns a Series with the data type of each column

print(heart.dtypes)

#%%
#Null Values
#Return the sum of a boolean same-sized object indicating if the values are NA

print(heart.isna().sum())

#%%
#Columns
#Indicates variable as a list of the DataFrame columns

cols = list(heart.columns)

print(cols)

#%%
#Encode Catagorical Columns

print(np.unique(heart[['thal']].values))
print(np.unique(heart[['ca']].values))
print(np.unique(heart[['ca']].values))

#%%
#StandardScaler
#Removes the mean and scales to unit variance

heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])

#%%
#Heatmap
#Show correlation between fetures

corr = heart.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

#Each square shows the correlation between the variables on each axis. Correlation ranges from -1 to +1. 
#Values closer to zero means there is no linear trend between the two variables. 
#The close to 1 the correlation is the more positively correlated they are; that is as one increases so does the other 
#and the closer to 1 the stronger this relationship is. 
#A correlation closer to -1 is similar, but instead of both increasing one variable will decrease as the other increases. 

#%%
#Save Heartmap as PDF

plt.savefig(r'C:\Users\stopc\OneDrive - University of South Florida\University\Senior 03-04\Spring 2024\ISM 4300\CapstoneProject_RyanStopczynski_20240416\Images\heatmap.png', format='png')

#%%
#Train Test Split
#Split arrays or matrices into random train and test subsets

y = heart['target']
X = heart.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#The test size indicates that there is an 80/20 split, 20% of the data will be used in testing

print(X_train.shape, X_test.shape)

#%%
#First Machine Learning Algorithm
#Logistic Regression

#This algorithm is a supervised learning algorithm used for binary classification
#tasks, where the target variable has two possible outcomes (Yes or No).
#It is a statistical model that predicts the probability of a binary outcome
#based on one or more predictor variables.

logisticgregression_heart_model = LogisticRegression()

# Fit the model on the training data
logisticgregression_heart_model.fit(X_train, y_train)

#%%
# Evaluate the model on the testing data

accuracy = logisticgregression_heart_model.score(X_test, y_test)
print(f'Logistic Regression Accuracy: {accuracy}')

#The regression indicates the accuracy as about 88.52%

#%%
#Second Machine Learning Algorithm
#Decision Tree Classifier

#This algorithm builds a flowchart-like tree structure where each internal node
#denotes a test on an attribute, each branch represents an outcome of the test, 
#and each leaf node holds a class label.

#It is constructed by recursively splitting the training data into subsets based
#on the values of the attributes until a stopping criterion is met, such as
#the maximum depth of the tree or minimum number of samples to split the node.

#The Decision Tree Classifier has two primary hyperparamaters:
#choice of splitting criterion and the maximum number of features.

decisiontree_scores = []
criterion_scores = []
for criterion in ['gini', 'entropy']:
    for i in range(1, len(X.columns) + 1):
        decisiontree_classifier = DecisionTreeClassifier( 
                                               max_features=i, 
                                               random_state=42)
        decisiontree_classifier.fit(X_train, y_train)
        decisiontree_scores.append(decisiontree_classifier.score(X_test, y_test))
    best_max_features = np.argmax(decisiontree_scores) + 1
    print(f'Best Max Features for {criterion}: {best_max_features}')
    criterion_scores.append(decisiontree_scores[np.argmax(decisiontree_scores)])

best_criterion = 'gini' if not np.argmax(criterion_scores) else 'entropy'
print(f'Best Criterion: {best_criterion}')

#Best max features for gini: 14
#Best max features for entropy: 14
#Best criterion: gini

#%%
#Decision Tree Classifier
#Using best max features and criterion

decisiontree_classifier = DecisionTreeClassifier(criterion='gini', max_features = 14, random_state = 42)

decisiontree_classifier.fit(X_train, y_train)
print(decisiontree_classifier.score(X_test, y_test))

#The regression indicates the accuracy as about 85.25%

#%%
#Third Machine Learning Algorithm
#Random Forest

#This algorithm combines the output of multiple decision trees to reach a single result.
#Through the opinions of many "trees" which are each individual models, random forest
#is used to make better predictions through creating a more robust and accurate 
#overall model.

#It can handle datasets containing both continuous and categorical variable, so
#it is more suited for classification and regression tasks

#Similar to the previous algorithms hyperparamaters must be identified, however
#this time through the use of RandomizedSearchCV which tests paramaters along with
#cross-validation on the training set to give the best choice

randomforest_param_grid = {
   'n_estimators': range(1, 100, 10),
    }
randomforest = RandomForestClassifier()

randomforest_random = RandomizedSearchCV(param_distributions=randomforest_param_grid,
                              estimator = randomforest,
                              scoring = 'accuracy',
                              verbose = 0, 
                              n_iter = 10, 
                              cv = 4)

randomforest_random.fit(X_train, y_train)
best_paramaters = randomforest_random.best_params_
print(f'Best parameters: {best_paramaters}')

#Best parameters: n_estimators constantly changes due to a variety of potential factors.

#Randomness within the model, variability, hyperparameter tuning, evaluation metrics,
#size and complexity of the dataset, and random seed or state.

#%%
#Random Forest Classifier
#Using best parameter for n_estimators

best_n_estimators = randomforest_random.best_params_['n_estimators']
randomforest_best = RandomForestClassifier(n_estimators=best_n_estimators)

randomforest_best.fit(X_train, y_train)
print(randomforest_best.score(X_test, y_test))

#The regression indicates an accuracy based on the best_n_estimator paramater

#%%
#Fourth Machine Learning Algorithm
#Support Vector Machines (SVM)

#The algorithm aims to find the best possible line, or decision boundry,
#that seperates the data points of different classes.

#The boundary is called a hyperplane when working in high-dimensional feature spaces.
#The idea is to maximize the margin, which is the ditance between the hyperplane 
#and the closest data points of each category, to distinguish data classes.


#SVC has several hyperparamaters
#This code will identify the primary ones: choice of kernal, regularization, 
#and degree of the polynomial kernel.

def evaluate_supportvectorclassification(X_train, X_test, y_train, y_test, kernel, C, degree=None):
    if kernel == 'poly':
        supportvectorclassification_classifier = SVC(kernel=kernel, C=C, degree=degree)
    else:
        supportvectorclassification_classifier = SVC(kernel=kernel, C=C)

    supportvectorclassification_classifier.fit(X_train, y_train)
    return supportvectorclassification_classifier.score(X_test, y_test)

supportvectorclassification_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for kernel in kernels:
    best_score = -1
    best_C = None
    if kernel == 'poly':
        for C in range(1, 11):
            for degree in range(3, 10):
                score = evaluate_supportvectorclassification(X_train, X_test, y_train, y_test, kernel, C, degree)
                if score > best_score:
                    best_score = score
                    best_C = C
                    best_degree = degree
        print(f'Best polynomial score for C={best_C}: {best_degree}')

    else:
        for C in range(1, 11):
            score = evaluate_supportvectorclassification(X_train, X_test, y_train, y_test, kernel, C)
            if score > best_score:
                best_score = score
                best_C = C
        print(f'Best score for {kernel} and C={best_C}: {best_score}')

    supportvectorclassification_scores.append(best_score)
    print(f'Best choice of C for {kernel}: {best_C}')

best_kernel_index = np.argmax(supportvectorclassification_scores)
best_kernel = kernels[best_kernel_index]
print(f'Best kernel: {best_kernel}')
    
#The best kernel choice is 'RBF,' and the regularization parameter must be 1.0

#%%
#SVC Classifier
#Using best kernel choice and regularization parameter previously identified
supportvectorclassification_classifier = SVC(kernel = 'rbf', C=1)
supportvectorclassification_classifier.fit(X_train, y_train)
print(supportvectorclassification_classifier.score(X_test, y_test))

#The regression indicates the accuracy as about 90.16%

#%%
#Fifth Machine Learning Algorithm
#K-Nearest Neighbors (KNN)

#Relies on the idea that similar points tend to have similar labels or values.
#During training, the KNN algorithm stores the entire training dataset as a reference.
#Then, when making predicitons it calculates the distance between the input 
#data point and all the training examples.
 
knearestneighbor_scores = []
for k in range(1, 40):
    knearestneighbor_classifier = KNeighborsClassifier(n_neighbors=k)
    knearestneighbor_classifier.fit(X_train, y_train)
    knearestneighbor_scores.append(knearestneighbor_classifier.score(X_test, y_test))

best_k = np.argmax(knearestneighbor_scores) + 1
print(f'Best choice of k: {best_k}')
   
#The prediction indicates the best choice of K as 8

#%%
#For regression, it calculates the average or weighted average of the target values
#of the K neighbors to predict the value for the input data point

k=8
knearestneighbor_classifier = KNeighborsClassifier(n_neighbors = k)
knearestneighbor_classifier.fit(X_train, y_train)
y_pred = knearestneighbor_classifier.predict(X_test)
print(f'Accuracy: {np.sum(y_pred==y_test)/len(y_test)}')

#The regression indicates the accuracy as about 91.80%

#%%
#Show the accuricies of all five machine learning algorithms

print(randomforest_best.score(X_test, y_test))
print(logisticgregression_heart_model.score(X_test, y_test))
print(decisiontree_classifier.score(X_test, y_test))
print(supportvectorclassification_classifier.score(X_test, y_test))
print(knearestneighbor_classifier.score(X_test, y_test))

#KNN has the highest accuracy out of all the machine learning algorithms

#%%
#Feature Importance

#Based on KNN, highest accuracy machine learning algorithm, determine the significance
#of each feature towards making decisions.

#Feature importance helps to understand which features are most influential in 
#model's decision making process

best_k = 8

knearestneighbor_classifier = KNeighborsClassifier(n_neighbors=best_k)
knearestneighbor_classifier.fit(X_train, y_train)

base_accuracy = knearestneighbor_classifier.score(X_test, y_test)

feature_importance = []
for feature in X_train.columns:
    X_train_without_feature = X_train.drop(columns=[feature])
    X_test_without_feature = X_test.drop(columns=[feature])
    
    knearestneighbor_classifier_without_feature = KNeighborsClassifier(n_neighbors=best_k)
    knearestneighbor_classifier_without_feature.fit(X_train_without_feature, y_train)
    
    accuracy_without_feature = knearestneighbor_classifier_without_feature.score(X_test_without_feature, y_test)
    
    feature_importance.append(base_accuracy - accuracy_without_feature)

sorted_features = [feature for _, feature in sorted(zip(feature_importance, X_train.columns))]
sorted_importance = sorted(feature_importance)

#%%
#Plot the Feature Importance Model for KNN Classifier

plt.figure(figsize=(10, 7))
plt.barh(sorted_features, np.abs(sorted_importance))
plt.xlabel('Change in Accuracy')
plt.ylabel('Features')
plt.title(f'Feature Importance (Change in Accuracy) for KNN Classifier (k={best_k})')
plt.gca().invert_yaxis()
plt.show()

#%%
#Save Feature Importance Plot as PDF

plt.savefig(r'C:\Users\stopc\OneDrive - University of South Florida\University\Senior 03-04\Spring 2024\ISM 4300\CapstoneProject_RyanStopczynski_20240416\Images\featureimportance.png', format='png')

#The six most important features are thalach, trestbps, exang, cp, ca, and age