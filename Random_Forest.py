import matplotlib.pyplot as plt 
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, ConfusionMatrixDisplay

from sklearn import tree
#************************************************************************************
from sklearn import preprocessing

#************************************************************************************

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster      import KMeans
from sklearn.tree         import DecisionTreeRegressor
from sklearn.tree         import DecisionTreeClassifier
from sklearn.tree         import plot_tree
from sklearn.ensemble     import RandomForestClassifier
from sklearn.model_selection import train_test_split

#************************************************************************************
#Importing the dataset

data = pd.read_csv('H:\\Programming AI\\مجلد للتطبيق و التجريب\\Random Forest Classifier\\train.csv')

print(data.head())
print(data.info())

print(data['Gender'].value_counts())
print(data['Married'].value_counts())
print(data['Education'].value_counts())
print(data['Dependents'].value_counts())
print(data['Self_Employed'].value_counts())

print(data['Property_Area'].value_counts())

print(data['Loan_Status'].value_counts())

#**************************************************************************************
#Data PreProcessing & Null Values Imputation

data["Gender"] = data["Gender"].map({"Male" : 1 , "Female" : 0})
data['Education'] = data['Education'].map({'Graduate': 1 , 'Not Graduate' : 0})
data['Loan_Status'] = data['Loan_Status'].map({'Y' : 1 , 'N' : 0})
data['Dependents'].replace('3+' ,3 , inplace=True) 
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1 , 'No': 0})
data['Property_Area'] = data['Property_Area'].map({'Semiurban' : 1 , 'Urban': 2 , 'Rural': 3})
data['Married'] = data['Married'].map({"Yes" : 1 , "No" : 0})

print(data)

# Null Values Imputation
print(data.isnull().sum())
rev_null=['Gender','Married','Dependents','Self_Employed','Credit_History','LoanAmount','Loan_Amount_Term']
data[rev_null]=data[rev_null].replace({np.nan:data['Gender'].mode(),
                                   np.nan:data['Married'].mode(),
                                   np.nan:data['Dependents'].mode(),
                                   np.nan:data['Self_Employed'].mode(),
                                   np.nan:data['Credit_History'].mode(),
                                   np.nan:data['LoanAmount'].mean(),
                                   np.nan:data['Loan_Amount_Term'].mean()})

print(data.isnull().sum())

#**************************************************************************************
#Splitting the Dataset into the Training set and Test set
X = data.drop(columns=['Loan_ID','Loan_Status']).values
y = data['Loan_Status'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 0)
#**************************************************************************************
#Random Forest Model

Model = RandomForestClassifier(criterion = 'entropy', random_state = 42)
Model.fit(X_train, y_train)

# Evaluating on Training set
y_pred_train = Model.predict(X_train)
print(classification_report(y_train, y_pred_train))

#Evaluating on Testing set
y_pred_test = Model.predict(X_test)
print(classification_report(y_test, y_pred_test))
#**************************************************************************************
#Visualization

feature_importance = pd.DataFrame({'Model':Model.feature_importances_},index = data.drop(columns = ['Loan_ID','Loan_Status']).columns)
feature_importance.sort_values('Model',ascending = True,inplace = True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize = (20,10))
rfc_feature = ax.barh(index, feature_importance['Model'], color = 'red', label = 'Random Forest')
ax.set(yticks = index , yticklabels = feature_importance.index)
plot_tree(Model.estimators_[0], filled=True, feature_names=[f"Feature {i}" for i in range(20)], class_names=["Class 0", "Class 1"])

ax.legend()
plt.show()

