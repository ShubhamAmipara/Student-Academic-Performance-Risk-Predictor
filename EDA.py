import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("P1Data.csv")
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df['Academic Performance Change'].value_counts())
plt.figure(figsize=(8,6))
plt.bar(df['Academic Performance Change'].value_counts().index,df['Academic Performance Change'].value_counts().values)
plt.xlabel('Academic Performance Change')
plt.ylabel('Count')
plt.title('Distribution of Academic Performance Change')
plt.show()
numeric=['Sleep Duration (hrs)','Physical Activity (hrs/week)','Screen Time (hrs/day)']
categorical=['Gender','Stress Level','Education Level','Anxious Before Exams']
#plot a histogram or KDE plot for each one, colored/grouped by the target variable
for col in numeric:
    plt.figure(figsize=(8,6))
    for category in df['Academic Performance Change'].unique():
        subset = df[df['Academic Performance Change'] == category]
        sns.kdeplot(subset[col], label=category, fill=True)
    plt.xlabel(col)
    plt.title(f'Distribution of {col} by Academic Performance Change')
    plt.legend()
    plt.show()
#plot count plots grouped by the target variable
for col in categorical:
    plt.figure(figsize=(8,6))
    sns.countplot(data=df, x=col, hue='Academic Performance Change')
    plt.xlabel(col)
    plt.title(f'Count of {col} by Academic Performance Change')
    plt.legend(title='Academic Performance Change')
    plt.show()