#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Loading the data into a dataframe

import csv
import pandas as pd
import matplotlib.pyplot as plt

#Importing a limited section of the dataset for now for easier processing
df = pd.read_csv("poisonous_mushrooms.csv", nrows=10000)
# Source: https://www.kaggle.com/datasets/davinascimento/poisonous-mushrooms?resource=download
# this assumes that you have the csv downloaded and stored in the same directory as this file

# Iterates through the columns, prints out counts of each data for each column
for column in df:
	df_series = df[column]
	results = df_series.value_counts()
	print("Results for column: ", column)
	print(results)
	print("Missing: ", df_series.isnull().sum(), "\n")

# results will be looked at more carefully in below cells, so don't worry about scrolling


# Focusing first on the counts of different attributes in the categorical cells...
# 
# First is the poisonous or edible attribute (p = poisonous, e = edible)
# 
# This is the target data. As the results below show, about 55% of this subsample (when N = 2000) is poisonous, which is relatively balanced

# In[4]:


print(df["class"].value_counts())


# Cap Data:
# 
# Labels from the Kaggle dataset
# 
# cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s, oval=o
# 
# cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s, l: silky
# 
# cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, black=k
# 
# 
# These seem like they would make good features (at least in terms of data quality)
# 

# In[5]:


for cat in ["cap-shape", "cap-surface", "cap-color"]:
	print(df[cat].value_counts())
	print("Missing: ", df_series.isnull().sum(), "\n")


#Results suggest solid mix of cap shapes and surface types; might need to sample from shape and color data to balance the input sets


# cap-diameter:

# In[6]:


print(df["cap-diameter"].describe())
print("Missing: ", df_series.isnull().sum(), "\n")


# gill data:
# 
# All categories, summaries below. Some are missing attributes:
# 
# gill-attachment: attached=a, descending=d, free=f, notched=n
# 
# gill-spacing: close=c,crowded=w,distant=d
# 
# gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y

# In[7]:


for column in ["gill-attachment", "gill-spacing", "gill-color"]:
	df_series = df[column]
	print(column)
	print("Missing: ", df_series.isnull().sum(), "\n")


# Gill spacing missing enough data that it would probably be simpler to ignore it
# 
# Stem Data:
# 

# In[8]:


for column in ["stem-width", "stem-height"]:
	df_series = df[column]
	print(column)
	print(df_series.describe())
	zeros = df_series.value_counts().get(0.0)
	print("Missing: ", zeros, "\n")


# Other relevant data:
# does-bruise-or-bleed
# has-ring
# 
# Imbalanced features, but not too hard to clean
# 
# 
# Spore color, Veil Color, and Veil Type has more missing entries than filled, so those will likely be ignored

# ### Correlation Analysis
# Several of the attributes are continous values, so we can find the correlation between them

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Map poisionous or not to an integer
mapping = {'p': 1, 'e': 0}
df_r = df["class"].map(mapping)

# Rename target feature to "poisionous" in the dataframe
df_r.rename({"class": "poisonous"})
df["poisonous"] = df_r
df.drop("class", axis=1)
df.drop("id", axis=1)

# Create a heatmap for the correlation matrix
continuousFeatures = ["poisonous", "cap-diameter", "stem-width", "stem-height"]
dfC = df[continuousFeatures]
print(dfC)

# from Lab 0
# Measure the relationships between different features
# Visualize relationships using scatter plots
sns.pairplot(df)
plt.title('Pair Plot of Continous Features')
plt.tight_layout()
plt.show()

# Calculate correlation matrix
correlation_matrix = dfC.corr('spearman')

# Visualize correlation matrix using heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


# These results suggest that all three of these continous variables are not directly correlated with the poisonous attribute. Outliers have yet to be handled, so this might need to be rerun to ensure that this isn't affecting the numbers.
# 
# As for the categorical data, we can simply look at the counts of how many of each label are poisonous or not to try to get a feel for which features matter.
# 
# 
# For example, the following code looks at cap shape.

# In[10]:


def countFeatureAndPoisonousCases(df, feature, printTable=False):
	features = [feature, "poisonous"]
	df_T = df[features]
	df_T = df_T.groupby(feature)["poisonous"].value_counts().sort_index()
	df_T = df_T.to_frame().reset_index()
	if printTable:
		print(df_T)
	sns.barplot(x=df_T[feature], y=df_T["count"], hue=df_T["poisonous"])
	plt.show()

countFeatureAndPoisonousCases(df, "cap-shape", printTable=True)


# The next code blocks do the same thing for a number of other variables...

# In[11]:


for attr in ["cap-shape", "cap-surface", "cap-color", "stem-color", "stem-surface", "gill-attachment","gill-color", "does-bruise-or-bleed", "season"]:
	countFeatureAndPoisonousCases(df, attr)


# Based on the results above, I suspect we should use stem color, stem surface, gill attachment, gill color, cap surface, and cap color. Diameter and height might also be interesting, but don't seem to provide any apparent benefit at the moment.

# ### Delving Deeper into Data

# In[12]:


#from statsmodels.graphics.mosaicplot import mosaic

# Sample code showing how to use the Crosstab feature from pandas
def generateCrosstab(dataframe, ind, col):
	inList = []
	colList = []
	for indCat in ind:
		inList.append(dataframe[indCat])
	for colCat in col:
		colList.append(dataframe[colCat])
	return pd.crosstab(index=inList, columns=colList)

ct = generateCrosstab(df, ["cap-color", "season"], ["poisonous", "does-bruise-or-bleed"])
print(ct)

# Create the mosaic plot
#mosaic(ct.stack(), title='Mosaic Plot Example')

# Show the plot
#plt.show()


# In[13]:


print(generateCrosstab(df, ["gill-color", "gill-attachment"], ["poisonous"]))


# From the gill data, we find the following strong predictors:
# Green gills + attached or descending --> more likely poisonsous, "P" (just realized that gill attachment has some non-listed attributes in the guide)
# 
# Brown gills:
# attached and detached strongly poisonous, same with "s"; "e" & "x" less so
# 
# Pink gills:
# detached more likely good, as with "e" & "s". "x" typically poisonous
# 
# 
# (Probably worth delving into this for later models, but thinking we should keep it simple for now)
# 

# ### Logistic Regresion with Stochastic Gradient Descent
# 

# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


X = df.drop('class', axis = 1)
y = df['class']

label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object': 
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

if y.dtype == 'object':
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .3, random_state=42)

X_train.head()


# In[53]:


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report 
import pickle

# Initialize and train classifier
classifier = SGDClassifier(loss='log_loss', max_iter=100)
model = classifier.fit(X_train, y_train)
pickle.dump(model, open('sgdmodel.pkl', 'wb'))
y_pred = model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred, normalize=True)
class_report = classification_report(y_test, y_pred)

# Print results
print("The accuracy of the model is:", lr_accuracy)
print("Classification Report:\n", class_report)


# In[54]:


from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

confusion_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix for Logistic Regression (SGD Classifier)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression (SGD Classifier)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# ### Deep Neural Network with Stochastic Gradient Descent
# 

# In[42]:


from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

X = df.drop('class', axis=1)
y = df['class']

# Encode features
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object': 
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#deep neural network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),#Hidden layer
    layers.Dense(1, activation='sigmoid') #Output layer
])

# stochastic gradient descent
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
pickle.dump(model, open('dnnmodel.pkl', 'wb'))
# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[43]:


print("The accuracy of the model is:", accuracy)
print("Classification Report:\n", class_report)


# In[46]:


# Plot Loss per Epoch for Deep Neural Network
plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Deep Neural Network with SGD(Loss per Epoch)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

