import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df=pd.read_csv("P1Data.csv")
df=df.drop(columns=['Name'])
stressmap={"Low":0,"Medium":1,"High":2}
df['Stress Level']=df['Stress Level'].map(stressmap)
edumap={'Class 8':0,'Class 9':1,'Class 10':2,'Class 11':3,'Class 12':4,'BA':5,'BSc':6,'BTech':7,'MA':8,'MSc':9,'MTech':10}
df['Education Level']=df['Education Level'].map(edumap)
anxiousmap={'No':0,'Yes':1}
df['Anxious Before Exams']=df['Anxious Before Exams'].map(anxiousmap)
targetmap={'Same': 0, 'Improved': 1, 'Declined': 2}
df['Academic Performance Change']=df['Academic Performance Change'].map(targetmap)
df=pd.get_dummies(df,columns=['Gender'], dtype=int)
# print(df.head())
# print(df.dtypes)
#df['Wellness Score'] = Sleep Duration + Physical Activity
# df['Mental Burden Score'] = Stress Level + Anxious Before Exams + (9 - Sleep Duration)
# df['Sleep Screen Ratio'] = Sleep Duration / Screen Time
# df['Activity Screen Balance'] = Physical Activity - Screen Time
df['Wellness Score'] = (df['Sleep Duration (hrs)'] + df['Physical Activity (hrs/week)'] )
df['Mental Burden Score'] = (df['Stress Level'] + (9-df['Sleep Duration (hrs)']) + df['Anxious Before Exams'])
df['Sleep Screen Ratio'] = df['Sleep Duration (hrs)'] / (df['Screen Time (hrs/day)'])
df['Activity Screen Balance'] = (df['Physical Activity (hrs/week)']) - (df['Screen Time (hrs/day)'])
#print(df[['Wellness Score', 'Mental Burden Score', 'Sleep Screen Ratio', 'Activity Screen Balance']].describe())


X=['Age','Sleep Duration (hrs)','Physical Activity (hrs/week)','Screen Time (hrs/day)','Stress Level','Education Level','Anxious Before Exams','Gender_Female','Gender_Male','Gender_Other','Wellness Score','Mental Burden Score','Sleep Screen Ratio','Activity Screen Balance']
y='Academic Performance Change'
X_train,X_test,y_train,y_test=train_test_split(df[X],df[y],test_size=0.2,random_state=42)
#scale
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
# print(X_train_scaled.shape)
# print(X_test_scaled.shape)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
logistic_model=LogisticRegression(class_weight='balanced',max_iter=1000)
logistic_model.fit(X_train_scaled,y_train)
y_pred=logistic_model.predict(X_test_scaled)
y_prob = logistic_model.predict_proba(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)  
print("Accuracy:", accuracy)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree_model=DecisionTreeClassifier(class_weight='balanced',random_state=42)
decision_tree_model.fit(X_train_scaled,y_train)
y_pred_dt=decision_tree_model.predict(X_test_scaled)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)
classification_rep_dt = classification_report(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
print("Decision Tree Classification Report:\n", classification_rep_dt)
print("Decision Tree Confusion Matrix:\n", conf_matrix_dt)
# decision_tree_model=DecisionTreeClassifier(max_depth=5,random_state=42)
# decision_tree_model.fit(X_train_scaled,y_train)
# y_pred_dt=decision_tree_model.predict(X_test_scaled)
# accuracy_dt = accuracy_score(y_test, y_pred_dt)
# print("Decision Tree Accuracy:", accuracy_dt)
# classification_rep_dt = classification_report(y_test, y_pred_dt)
# conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
# print("Decision Tree Classification Report:\n", classification_rep_dt)
# print("Decision Tree Confusion Matrix:\n", conf_matrix_dt)
# decision_tree_model=DecisionTreeClassifier(max_depth=7,random_state=42)
# decision_tree_model.fit(X_train_scaled,y_train)
# y_pred_dt=decision_tree_model.predict(X_test_scaled)
# accuracy_dt = accuracy_score(y_test, y_pred_dt)
# print("Decision Tree Accuracy:", accuracy_dt)
# classification_rep_dt = classification_report(y_test, y_pred_dt)
# conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
# print("Decision Tree Classification Report:\n", classification_rep_dt)
# print("Decision Tree Confusion Matrix:\n", conf_matrix_dt)
# decision_tree_model=DecisionTreeClassifier(max_depth=3,random_state=42)
# decision_tree_model.fit(X_train_scaled,y_train)
# y_pred_dt=decision_tree_model.predict(X_test_scaled)
# accuracy_dt = accuracy_score(y_test, y_pred_dt)
# print("Decision Tree Accuracy:", accuracy_dt)
# classification_rep_dt = classification_report(y_test, y_pred_dt)
# conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
# print("Decision Tree Classification Report:\n", classification_rep_dt)
# print("Decision Tree Confusion Matrix:\n", conf_matrix_dt)

# #Random Forest
# from sklearn.ensemble import RandomForestClassifier
# random_forest_model=RandomForestClassifier(random_state=42)
# random_forest_model.fit(X_train_scaled,y_train)
# y_pred_rf=random_forest_model.predict(X_test_scaled)
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print("Random Forest Accuracy:", accuracy_rf)
# classification_rep_rf = classification_report(y_test, y_pred_rf)
# conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
# print("Random Forest Classification Report:\n", classification_rep_rf)
# print("Random Forest Confusion Matrix:\n", conf_matrix_rf)