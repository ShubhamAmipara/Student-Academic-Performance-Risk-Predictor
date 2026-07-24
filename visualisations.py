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
df['At Risk'] = np.where(df['Academic Performance Change'] == 2,1,0)
#print(df[['Wellness Score', 'Mental Burden Score', 'Sleep Screen Ratio', 'Activity Screen Balance']].describe())
X=['Age','Sleep Duration (hrs)','Physical Activity (hrs/week)','Screen Time (hrs/day)','Stress Level','Education Level','Anxious Before Exams','Gender_Female','Gender_Male','Gender_Other','Wellness Score','Mental Burden Score','Sleep Screen Ratio','Activity Screen Balance']
y='At Risk'
X_train,X_test,y_train,y_test=train_test_split(df[X],df[y],test_size=0.2,random_state=42)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
#Create 2 pipelines, one for each model, first standard scaler then the model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
logistic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression( class_weight='balanced', max_iter=1000))
])
decision_tree_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('decision_tree', DecisionTreeClassifier(class_weight='balanced', random_state=42))
])
cross_val_scores_logistic = cross_val_score(logistic_pipeline, X_train, y_train, cv=5, scoring='f1')
cross_val_scores_decision_tree = cross_val_score(decision_tree_pipeline, X_train, y_train, cv=5, scoring='f1')
print("Logistic Regression Cross-Validation F1 Scores:", cross_val_scores_logistic)
print("Decision Tree Cross-Validation F1 Scores:", cross_val_scores_decision_tree)
# print("Logistic Regression Cross-Validation F1 Score Mean:", cross_val_scores_logistic.mean())
# print("Logistic Regression Cross-Validation F1 Score Std:", cross_val_scores_logistic.std())
# print("Decision Tree Cross-Validation F1 Score Mean:", cross_val_scores_decision_tree.mean())
# print("Decision Tree Cross-Validation F1 Score Std:", cross_val_scores_decision_tree.std())
from sklearn.model_selection import GridSearchCV
#Define parameter grid
logistic_param_grid = {
    'logistic__C': [0.01, 0.1, 1, 10, 100],
}
param_grid_decision_tree = {
    'decision_tree__max_depth': [None,3, 5,7,10],
    'decision_tree__min_samples_split': [2, 5, 10],
    'decision_tree__min_samples_leaf': [1, 2, 4]
}
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
grid_search_logistic = GridSearchCV(logistic_pipeline, logistic_param_grid, cv=5, scoring='f1')
grid_search_logistic.fit(X_train, y_train)
print("Best Parameters for Logistic Regression:", grid_search_logistic.best_params_)
best_logistic_model = grid_search_logistic.best_estimator_
y_pred_logistic = best_logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_logistic)
print("Tuned Logistic Regression Accuracy:", accuracy)
classification_rep = classification_report(y_test, y_pred_logistic)
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
print("Tuned Logistic Regression Classification Report:\n", classification_rep)
print("Tuned Logistic Regression Confusion Matrix:\n", conf_matrix_logistic)
grid_search_decision_tree = GridSearchCV(decision_tree_pipeline, param_grid_decision_tree, cv=5, scoring='f1')
grid_search_decision_tree.fit(X_train, y_train)
print("Best Parameters for Decision Tree:", grid_search_decision_tree.best_params_)
best_decision_tree_model = grid_search_decision_tree.best_estimator_
y_pred_dt = best_decision_tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_dt)
print("Tuned Decision Tree Accuracy:", accuracy)
classification_rep = classification_report(y_test, y_pred_dt)
conf_matrix_decision_tree = confusion_matrix(y_test, y_pred_dt)
print("Tuned Decision Tree Classification Report:\n", classification_rep)
print("Tuned Decision Tree Confusion Matrix:\n", conf_matrix_decision_tree)
print(grid_search_decision_tree.best_score_)

#Visualisations
sns.heatmap(conf_matrix_logistic, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#set tick labels to class names
plt.xticks(ticks=[0.5, 1.5], labels=['Not At Risk', 'At Risk'])
plt.yticks(ticks=[0.5, 1.5], labels=['Not At Risk', 'At Risk'])
plt.title('Confusion Matrix Heatmap')
plt.savefig('confusion_matrix.png')
#Save this confusion matrix as an image file
plt.show()

print("Best Logistic Regression Coefficients:", best_logistic_model.named_steps['logistic'].coef_[0])
#Create a pandas Series with feature names as index and coefficients as values
feature_importance = pd.Series(best_logistic_model.named_steps['logistic'].coef_[0], index=X)
#Sort the Series by absolute value of coefficients
feature_importance = feature_importance.reindex(feature_importance.sort_values(ascending=False).index)
#Plot the feature importance
feature_importance.plot(kind='barh')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.title('Feature Importance from Logistic Regression Coefficients')
plt.savefig('feature_importance.png')
plt.show()