# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 step 1:start the program
 
 step 2:import pandas module and import the required data set.
 
 step 3:Find the null values and count them.
 
 step 4:Count number of left values.
 
 step 5:From sklearn import LabelEncoder to convert string values to numerical values.
 
 step 6:From sklearn.model_selection import train_test_split.
 
 step 7:Assign the train dataset and test dataset.
 
 step 8:From sklearn.tree import DecisionTreeClassifier.
 
 step 9:Use criteria as entropy.

 step 10:From sklearn import metrics.
 
 step 11:Find the accuracy of our model and predict the require values.
 
 step 12:End the program

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vikaash P
RegisterNumber:  212223240180
import pandas as pd
data = pd.read_csv("C:/Users/admin/Downloads/Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
### Data:

![image](https://github.com/user-attachments/assets/a2dc3aeb-f63d-47c9-8de6-477b2c6aeb67)

### Accuracy:

![image](https://github.com/user-attachments/assets/c6c02a2a-2585-459c-86a1-65a5fa40685e)


### Predict:

![image](https://github.com/user-attachments/assets/6d28345e-8184-435e-af08-0b1485782197)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
