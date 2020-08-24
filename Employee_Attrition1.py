# Employee Attrition Problem
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Dataset
ex_emp = pd.read_csv("Ex_Employees.csv")
print (ex_emp.head())

exi_emp = pd.read_csv("Employee_existing.csv")
print(exi_emp.head())

comb = ex_emp.append(exi_emp)
print(comb.head())

print(ex_emp.shape)
print(exi_emp.shape)
print(comb.shape)

comb.info()# confirm the datatype
comb.isna().sum() # confirm if there is a nan value in the data

print(comb.drop("Emp ID", 1))
print(ex_emp.drop("Emp ID", 1))
print(exi_emp.drop("Emp ID", 1))

print(ex_emp['dept'].value_counts())
print(ex_emp['salary'].value_counts())

#Visualising our variables
sns.countplot(ex_emp['salary'].value_counts())
plt.savefig('salary.png')

# sns.countplot(ex_emp['dept'].value_counts())
plt.savefig('dept.png')

sns.barplot(y = "satisfacti", x = "salary",  data = comb, hue = "promotio", capsize =.05 )
plt.savefig('sal_promotion_satis.png')

sns.pairplot(comb)
plt.savefig('fig1.png')
# Encoding dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

X = ex_emp.iloc[:, [0,1,2,3,4,5,6,7]].values
print(X.shape)
y = ex_emp.iloc[:, 8].values
print(y.shape) 

# Traing dataset in to test train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0, test_size = 0.25)

# Fitting Decision Tree to Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

#Get the accuracy on the training data
forest.score(X_train, y_train)

# Return the feature importances (the higher, the more important the feature).
importances = pd.DataFrame({'features':ex_emp.iloc[:, [0,1,2,3,4,5,6,7]].columns,'importance':np.round(forest.feature_importances_,3)}) #Note: The target column is at position 0
importances = importances.sort_values('importance',ascending=False).set_index('features')
importances   

importances.plot.bar()
plt.savefig('importance.png')

