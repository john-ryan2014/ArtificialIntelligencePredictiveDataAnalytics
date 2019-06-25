
# coding: utf-8

# In[67]:

#Name: John Ryan
#Artificial Intelligence 2 Assignment 2
# For this assignment, I used the Sci-kit with Pandas in order to create a classifier to 
# predict the outcome of a Bank marketing campaign.

import pandas as pd
## Read in the original data and set the column headings

columnHeadings=['id','age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','output']
OriginalData = pd.read_csv('./Data/trainingSet.txt', header=None, names=columnHeadings)


# In[68]:

from pandas import DataFrame
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import numpy as np


# Extract Target Feature
targetLabels = OriginalData['output']


# Extract Numeric Descriptive Features in order to preprocess the categorical features
numeric_features = ['age','balance','day','duration','campaign','pdays','previous']
numeric_dfs = OriginalData[numeric_features]

numeric_dfs.head()


# Extract Categorical Descriptive Features
cat_dfs = OriginalData.drop(numeric_features + ['id'] + ['output'],axis=1)


# Remove any missing values and apply one-hot encoding
cat_dfs.replace('?','NA')
cat_dfs.fillna( 'NA', inplace = True )

#transform data into array of dictionaries of feature:level pairs
cat_dfs = cat_dfs.T.to_dict().values()

#convert to numeric encoding
vectorizer = DictVectorizer( sparse = False )
vec_cat_dfs = vectorizer.fit_transform(cat_dfs) 

encoding_dictionary = vectorizer.vocabulary_
for k in sorted(encoding_dictionary.keys()):
    mapping = k + " : column " + str(encoding_dictionary[k]) + " = 1"
    


# Merge Categorical and Numeric Descriptive Features
train_dfs = np.hstack((numeric_dfs.as_matrix(), vec_cat_dfs ))


# In[69]:

from sklearn import tree
from sklearn import cross_validation

#Create and train the decision tree model using entropy based information gain
decTreeModel = tree.DecisionTreeClassifier(criterion='entropy')

#fit the model using the numeric representations of the training data
decTreeModel.fit(train_dfs, targetLabels)


# In[70]:

#Read in the queries file into a dataframe 
queries = pd.read_csv('./Data/queries.txt', header=None, names=columnHeadings)

#The output column is dropped to get rid of all the question marks
qs = queries.drop( ['output'],axis=1)


# In[71]:

#extract the numeric features
q_num = qs[numeric_features].as_matrix() 

#convert the categorical features
q_cat = qs.drop(numeric_features,axis=1)
q_cat_dfs = q_cat.T.to_dict().values()
q_vec_dfs = vectorizer.transform(q_cat_dfs) 

query = np.hstack((q_num, q_vec_dfs ))


# In[72]:

# Make the actual predictions
predictions = decTreeModel.predict(query)


# In[73]:

# Set the predictions into a dataframe format
preds = pd.DataFrame(predictions)
preds.columns = ['Prediction']


# In[74]:

#Append the predictions to each test id, while also dropping the features which are not 
# going to be outputted in the solutions file
Result = qs.join(preds)
Solutions = Result.drop(['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome'],axis=1)


# In[79]:

# Write the predictions to a text file
Solutions.to_csv('./data/Solutions/c12429652.txt',sep=',',index= False,header=None)


# In[ ]:




# In[ ]:



