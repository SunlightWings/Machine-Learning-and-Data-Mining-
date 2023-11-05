
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# replace with your dataset path
dataset_path='./dataset/dataGaia.csv'
df=pd.read_csv(dataset_path)

df.head()
df.describe()

# get the unigue labels
df['SpType-ELS'].unique()

df.drop(columns=['Unnamed: 0'],inplace=True)

# encode the labels
label_encoder = preprocessing.LabelEncoder()
  

df['labels']= label_encoder.fit_transform(df['SpType-ELS'])
  
df['labels'].unique()

data=df.drop(columns=['SpType-ELS'])

# Count the number of NaN values in each column
nan_count_per_column = data.isna().sum()

# Count the total number of NaN values in the entire DataFrame
total_nan_count = data.isna().sum().sum()

print("Number of NaN values in each column:")
print(nan_count_per_column)

print("Total number of NaN values in the DataFrame:", total_nan_count)

data.duplicated().sum()

data.head()

import seaborn as sns
import matplotlib.pyplot as plt
classes=['A', 'B', 'F', 'G', 'K', 'M','O']
# Create the count plot
sns.countplot(x='labels', data=data)

text_box = {
    'boxstyle': 'round',
    'facecolor': 'white',
    'alpha': 0.8
}

plt.text(5.2, 108000, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
# Set plot labels
plt.xlabel('label')
plt.ylabel('Count')
plt.title('Number of Labels')
plt.tight_layout()
plt.savefig('output/Number of Label.png',bbox_inches='tight')
# Show the plot
plt.show()
data.hist(bins=50, figsize=(20, 20),color='steelblue')
# the text isn't plotted
plt.text(1, 1, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=100, bbox=text_box, color='red')
plt.tight_layout()
plt.savefig('output/Histogram of features.png',bbox_inches='tight')
plt.show()

X=data.drop(columns=['labels'])
y=data['labels']

# Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

X.describe()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

df_scaled = pd.DataFrame(data=scaled_data, columns=X.columns)
df_scaled.describe()

df_scaled.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df_scaled,df['labels'],test_size=0.01,random_state=43)

from sklearn.neighbors import KNeighborsClassifier

# Instantiate the KNN model with default parameters
knn = KNeighborsClassifier()

x_train.shape,y_train.shape,x_test.shape,y_test.shape

params=knn.get_params()
print(params)

knn.fit(x_train, y_train)
preds=knn.predict(x_test.values)

preds[:20]
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, preds)
print(accuracy)

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
# Compute the confusion matrix
cm = confusion_matrix(y_test, preds)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
classes=['A', 'B', 'F', 'G', 'K', 'M','O']
# Create a heatmap visualization of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt=".4f",xticklabels=classes, yticklabels=classes, cbar=True)
plt.text(5.8, 0, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('output/Confusion matrix for default KNN.png',bbox_inches='tight')
plt.show()
from sklearn.metrics import classification_report
# Generate the classification report
report = classification_report(y_test, preds)

print(report)
correlation_matrix = data.corr()

# Plot the correlation heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',annot_kws={'fontsize': 8})
plt.title('Correlation Heatmap of Attributes')
plt.text(24.5, -0.2, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.savefig('output/Correlation Heatmap of Attributes.png',bbox_inches='tight')
plt.show()

correlation_with_target = correlation_matrix['labels']
# Plot the correlation values as a barplot
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target[:-1].index, y=correlation_with_target[:-1].values,)
plt.title('Correlation with Target Variable')
plt.xlabel('Attribute')
plt.ylabel('Correlation')
plt.xticks(rotation=90)
plt.text(22, 0.6, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.savefig('output/Correlation with Target Variable.png',bbox_inches='tight')
plt.show()

x_train,x_test,y_train,y_test=train_test_split(df_scaled,df['labels'],test_size=0.002,random_state=43)
weights = np.random.rand(26)
weighted_knn=KNeighborsClassifier(metric_params={'w': weights})
print(weights)

weighted_knn.fit(x_train,y_train)

x_test.values.shape

preds1=weighted_knn.predict(x_test.values)


from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
# Compute the confusion matrix
cm1 = confusion_matrix(y_test, preds1)
cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
classes=['A', 'B', 'F', 'G', 'K', 'M','O']
# Create a heatmap visualization of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, cmap="Blues", fmt=".4f",xticklabels=classes, yticklabels=classes, cbar=True)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix for random weighted KNN")
plt.text(6.1, 0, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.savefig('output/Confusion Matrix for random weighted KNN.png',bbox_inches='tight')
plt.show()


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, preds1)
print(accuracy)


from sklearn.metrics import classification_report
# Generate the classification report
report = classification_report(y_test, preds1)
print(report)

correlation_with_target.index
weights[18]=4
weights[17]=4
weights[16]=3
weights[15]=3

weighted_knn2=KNeighborsClassifier(metric_params={'w': weights})
print(weights)

weighted_knn2.fit(x_train,y_train)
preds2=weighted_knn2.predict(x_test.values)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, preds2)
print(accuracy)
from sklearn.metrics import classification_report
# Generate the classification report
report = classification_report(y_test, preds2)
print(report)
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
# Compute the confusion matrix
cm2 = confusion_matrix(y_test, preds)
cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
classes=['A', 'B', 'F', 'G', 'K', 'M','O']
# Create a heatmap visualization of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm2, annot=True, cmap="Blues", fmt=".4f",xticklabels=classes, yticklabels=classes, cbar=True)
plt.text(6.1, 0, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion matrix for weighted KNN")
plt.savefig('output/Confusion matrix for weighted KNN.png',bbox_inches='tight')
plt.show()
from sklearn.model_selection import cross_val_score
# split the dataset to perform validation as validating on whole dataset takes a lot of time since dataset size is huge
x_train1,x_val,y_train1,y_val=train_test_split(df_scaled,df['labels'],test_size=0.1,random_state=43)
# Define a range of K values to test
k_values = range(1, 30)  # You can adjust the range as needed

# Perform k-fold cross-validation
k_scores = []
for k in k_values:
    # Create a KNN classifier object
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_val.values, y_val.values, cv=10, scoring='accuracy',n_jobs=-1)
    k_scores.append(scores.mean())

# Find the K value with the best performance
best_k = k_values[np.argmax(k_scores)]

# Print the results
print("Mean accuracy scores for different K values:")
for k, score in zip(k_values, k_scores):
    print(f"K = {k}: Mean Accuracy = {score:.4f}")

print(f"\nBest K value: {best_k}")

# plot to see clearly
plt.plot(k_values, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Accuracy vs K value')
plt.text(24, 0.848, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.savefig('output/Accuracy vs K value.png',bbox_inches='tight')
plt.show()

# train KNN for best k value
knn = KNeighborsClassifier(n_neighbors=best_k)
x_train,x_test,y_train,y_test=train_test_split(df_scaled,df['labels'],test_size=0.002,random_state=43)
knn.fit(x_train,y_train)

preds=knn.predict(x_test.values)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, preds)
print(accuracy)

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
# Compute the confusion matrix
cm = confusion_matrix(y_test, preds)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
classes=['A', 'B', 'F', 'G', 'K', 'M','O']
# Create a heatmap visualization of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt=".4f",xticklabels=classes, yticklabels=classes, cbar=True)
plt.text(5.8, 0, 'THA076BCT026\nTHA076BCT027\nTHA076BCT041', fontsize=8, bbox=text_box, color='red')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion matrix for KNN with best K")
plt.savefig('output/Confusion matrix for KNN with best K.png',bbox_inches='tight')
plt.show()

from sklearn.metrics import classification_report
# Generate the classification report
report = classification_report(y_test, preds)
print(report)


