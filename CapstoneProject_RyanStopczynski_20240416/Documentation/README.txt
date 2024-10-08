README

PatientPal: The Interactive Heart Disease Prediction Tool


The Heart Disease Prediction Dashboard is a healthcare analytics
tool designed to aid medical professionals in diagnosing heart 
disease during clinical consultations. Leveraging machine learning 
algorithms and patient data, the dashboard predicts the risk of heart 
disease based on key patient characteristics such as thalach, trestbps,
exang, cp, ca, and age. The project involves data collection, preprocessing, 
feature engineering, and the use of predictive models such as logistic regression, 
decision trees, random forests, support vector machines and k-nearest neighbors. The 
final deliverable is an interactive Tableau dashboard that allows medical 
professionals to input patient data and receive real-time predictions, 
enhancing the efficiency and accuracy of clinical diagnosis for heart disease.

Table of Contents
- Installation
- Usage
- Configuration

Installation

Python:

1. Install the latest version of Anaconda (24.3.0)
2. Install the latest version of Spyder (5.5.1)
3. Using the CMD.exe Prompt install the following packages/libraries
	1. numpy==1.26.4
	2. pandas==2.1.4
	3. matplotlib==3.8.0
	4. sikit-learn==1.2.2
	5. seaborn==0.12.4
4. Open the PatientPal_MachineLearningAlgorithms.py file located in the 'Python' folder in Spyder

Tableau:

1. Install the latest version of Tableau (2023.2)
2. Open the PatientPal_InteractivePredictionTool.twbx file located in the 'Tableau' folder in Tableau

Usage

Python:

1. Run the necessary libraries code block
2. Change the Heart Disease Precition UCI import to your file path

heart = pd.read_csv(r'(Your File Path)\heart.csv')

3. After changing, run the data import code block
4. Run each remaining code block one-by-one 
5. If you want to save the created plots as PDF's change the file location

plt.savefig(r'(Your File Path)\featureimportance.png', format='png')
plt.savefig(r'(Your File Path)\heatmap.png', format='png')

Tableau:

1. Go to the dashboard labeled "PatientPal"
2. If this is during a clinical consulation take the patients thalach, trestbps, exang, cp, ca, and age
3. Imput patient data into the corresponding single value dropdown's
4. Watch as the graphs change to reflect the imputed data


Note:

See PatientPal_ProjectReport.docx for information on methodology, findings, and conclusions.