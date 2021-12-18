#!/usr/bin/env python
# coding: utf-8

# ### Understand the Problem part

# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.
# 
# Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.

# ### Data Dictionary

# Variable - Description
# 
# Item_Identifier - Unique product ID
# 
# Item_Weight - Weight of product
# 
# Item_Fat_Content - Whether the product is low fat or not
# 
# Item_Visibility - The % of total display area of all products in a store allocated to the particular product
# 
# Item_Type - The category to which the product belongs
# 
# Item_MRP - Maximum Retail Price (list price) of the product
# 
# Outlet_Identifier - Unique store ID
# 
# Outlet_Establishment_Year - The year in which store was established
# 
# Outlet - The size of the store in terms of ground area covered
# 
# Outlet_Location_Type - The type of city in which the store is located
# 
# Outlet_Type - Whether the outlet is just a grocery store or some sort of supermarket
# 
# Item_Outlet_Sales - Sales of the product in the particulat store. This is the outcome variable to be predicted.

# In[1]:


# import all the required libraries


# In[72]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt       #Data Visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import random         #For missing value imputation

from sklearn.preprocessing import StandardScaler       #For Scaling

from sklearn.model_selection import train_test_split        #Testing

from sklearn.linear_model import LinearRegression        #Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Importing Data
train=pd.read_csv("Train_UWu5bXk.csv")
test=pd.read_csv("Test_u94Q5KV.csv")


# In[3]:


#Finding train and test columns
print("Train columns : ", train.columns)
print("\nTest columns : ",test.columns)


# It is clear that Item_Outlet_Sales is the Target Variable.

# ### Exploratory Data Analysis

# We plot histograms for continous data and barplots for categorical variables

# EDA is done only on train data

# In[4]:


train.shape


# In[7]:


# Target Varible
sns.distplot(train["Item_Outlet_Sales"])


# This is highly right skewed

# In[8]:


train.dtypes


# In[9]:


# Working on categorical varibales
# wont work on Item_Identifier...no meaning
plt.subplot(221)
train["Outlet_Location_Type"].value_counts().plot.bar(title="Barplot-Outlet Location Type")
plt.subplot(222)
train["Outlet_Size"].value_counts().plot.bar(title="Barplot-Outlet Size")
plt.subplot(223)
train["Item_Type"].value_counts().plot.bar(title="Barplot-Item Type")
plt.subplot(224)
train["Item_Fat_Content"].value_counts().plot.bar(figsize=(15,8),title="Barplot-Item Fat Content")


# #### Inferences
# 1. There are more tier 3 type outlets as compared to tier 1 type
# 
# 2. Maximum Outlets are medium sized
# 
# 3. Fruits and Vegetables is very highly present where as seafood is very less
# 
# 4. Here may be names are wrong, but there is a particular type which is higly present

# In[10]:


plt.subplot(221)
train["Outlet_Type"].value_counts().plot.bar(figsize=(15,8),title="Barplot-Outlet Type")

plt.subplot(222)
train["Outlet_Identifier"].value_counts().plot.bar(title="Barplot-Outlet Identifier")

plt.subplot(223)
train["Outlet_Establishment_Year"].value_counts().plot.bar(title="Barplot-Outlet Establishment Year")


# #### Inferences
# 1. Supermarket Type 1 is maximum and Type 3 is minimum
# 
# 2. Maximum types of Outlets are in same proportion
# 
# 3. Maximum number of outlets established was in year 1985, after that almost equal numbers are established after 2-3 years

# In[11]:


#Working on Numeric Data
plt.figure(figsize=(15,8))
plt.subplot(221)
sns.distplot(train["Item_Weight"].dropna())

plt.subplot(222)
sns.boxplot(train["Item_Weight"])


# Item Wight is not skewed

# In[12]:


plt.figure(figsize=(15,8))
plt.subplot(221)
sns.distplot(train["Item_Visibility"])

plt.subplot(222)
sns.boxplot(train["Item_Visibility"])


# Item Visibility is highly skewed, and a lot of outliers are there too.

# In[13]:


plt.figure(figsize=(15,8))
plt.subplot(221)
sns.distplot(train["Item_MRP"])

plt.subplot(222)
sns.boxplot(train["Item_MRP"])


# Item MRP is not skewed

# In[14]:


# For categorical varibles
train.boxplot(column="Item_Outlet_Sales",by="Item_Fat_Content")


# In[15]:


train.boxplot(column="Item_Outlet_Sales",by="Item_Type")


# In[16]:


sns.pairplot(train,diag_kind='kde')


# ### Missing Value Treatment

# In[17]:


test['Item_Outlet_Sales']="to_find"


# In[18]:


df=pd.concat([train,test],axis=0)


# In[19]:


df.head()


# In[20]:


df.isnull().sum()


# In[21]:


df.shape


# Since very high proportion of 2 columns are missing, so for their missing value treatment, lets apply prediction model for them too

# In[22]:


item=df[['Item_Identifier','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Item_Weight']]
outlet=df[['Outlet_Establishment_Year','Outlet_Location_Type','Outlet_Type','Outlet_Size']]


# In[23]:


outlet['Outlet_Size'][outlet['Outlet_Size'].isnull()]="check"


# In[24]:


outlet['Outlet_Location_Type'].value_counts()


# In[25]:


outlet.columns


# In[26]:


outlet['Outlet_Establishment_Year']=outlet['Outlet_Establishment_Year'].replace({1985:0,1987:1,1999:2,1997:3,2004:4,2002:5,2009:6,2007:7,1998:8})
outlet['Outlet_Location_Type']=outlet['Outlet_Location_Type'].replace({'Tier 3':0,'Tier 2':1,'Tier 1':2})
outlet['Outlet_Type']=outlet['Outlet_Type'].replace({'Supermarket Type1':0,'Grocery Store':1,'Supermarket Type3':2,'Supermarket Type2':3})
outlet['Outlet_Size']=outlet['Outlet_Size'].replace({'Medium':0,'check':3,'Small':2,'High':1})


# In[27]:


sns.pairplot(outlet)


# on seeing pairplot of outlet size and outlet type, we clearly see that for type 0,2,3, there are fixed sizes

# So we substitute the missing values accordingly

# In[28]:


outlet['Outlet_Size'][outlet['Outlet_Type']==1]=2


# In[29]:


outlet['Outlet_Size'][outlet['Outlet_Location_Type']==1]=2


# In[30]:


outlet['Outlet_Size'].isnull().sum()


# In[31]:


outlet['Outlet_Size']=outlet['Outlet_Size'].replace({0:'Medium',3:'check',2:'Small',1:'High'})


# In[32]:


df['Outlet_Size'][df['Outlet_Size'].isnull()]=outlet['Outlet_Size']


# In[33]:


df.isnull().sum()


# In[34]:


df.shape


# In[35]:


req=pd.get_dummies(item)
req.head()


# In[36]:


#sns.pairplot(req)


# In[39]:


sns.distplot(item['Item_Weight'].dropna(), kde=True, rug=True)


# In[41]:


df['Item_Weight'] = df['Item_Weight'].fillna(random.choice(df['Item_Weight'].values.tolist()))


# In[42]:


df['Item_Weight'].isnull().sum()


# In[43]:


df.isnull().sum()


# In[44]:


df.head()


# ### Scaling data

# Item Visibility vs Item MRP are highly incomparable. to make them equally participate, we scale them

# In[45]:


scalar = StandardScaler()  
scalar.fit(df[['Item_Weight','Item_Visibility','Item_MRP']])
df[['Item_Weight','Item_Visibility','Item_MRP']]=scalar.transform(df[['Item_Weight','Item_Visibility','Item_MRP']])


# In[46]:


df_new = pd.get_dummies(df,columns=['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Establishment_Year',
                                   'Outlet_Size','Outlet_Type','Outlet_Location_Type'])
df_new.head()


# In[47]:


df_new.shape


# In[48]:


train1=df_new[df["Item_Outlet_Sales"]!="to_find"]
print(train1.shape)
train1.head()


# In[50]:


testt1=df_new[df["Item_Outlet_Sales"]=="to_find"]
print(testt1.shape)
testt1.head()


# In[51]:


test1=testt1.drop(columns=["Item_Identifier","Item_Outlet_Sales"])


# #### Splitting Data

# In[52]:


x=train1.drop(columns=["Item_Outlet_Sales","Item_Identifier"])
y=train1["Item_Outlet_Sales"]
#x = (x-np.min(x))/(np.max(x)-np.min(x))


# In[53]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=123)


# ### Designing Models

# #### Linear Regression

# In[54]:


lr=LinearRegression()
model_lr = lr.fit(x_train,y_train)
pred_lr = model_lr.predict(x_test)
RMSE = np.sqrt(np.mean((y_test-pred_lr)**2))
MAPE = np.mean(np.absolute((y_test-pred_lr)/pred_lr))
RMSE,MAPE


# #### Decision Tree

# In[55]:


dtree = DecisionTreeRegressor()
model_dtree = dtree.fit(x_train,y_train)
pred_dtree = model_dtree.predict(x_test)
RMSE = np.sqrt(np.mean((y_test-pred_dtree)**2))
MAPE = np.mean(np.absolute((y_test-pred_dtree)/pred_dtree))
RMSE,MAPE

x=x[['Item_MRP','Item_Visibility','Item_Weight','Outlet_Type_Grocery Store',
     'Outlet_Identifier_OUT027','Outlet_Type_Supermarket Type3','Outlet_Establishment_Year_1985','Outlet_Identifier_OUT019',
     'Outlet_Type_Supermarket Type1','Outlet_Establishment_Year_1998','Outlet_Identifier_OUT010']]
# In[60]:


knn = KNeighborsRegressor()
model_knn = knn.fit(x_train,y_train)
pred_knn = model_knn.predict(x_test)
RMSE = np.sqrt(np.mean((y_test-pred_knn)**2))
MAPE = np.mean(np.absolute((y_test-pred_knn)/pred_knn))
RMSE,MAPE


# #### Cross Validation

# In[63]:


l = []
for model in [lr,dtree,knn]:
    Scores = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
    l.append(Scores.mean())
l


# In[64]:


l = []
for model in [lr,dtree,knn]:
    Scores = cross_val_score(model, x, y, scoring='neg_mean_absolute_error', cv=10)
    l.append(Scores.mean())
l


# From cross validation it is clear that random forest is best model till now, but we still tune these and see further

# #### Decision Tree Tuning
dtree_param = {
    'criterion':['mse','mae'],
    'splitter':['best','random'],
    'max_depth':[2,3,4,5,6,7,8,9,10],
    'min_samples_split':[10,20,30,40,50],
    'max_features':['auto','sqrt','log2'],
    'random_state':[123]
}

grid_search = GridSearchCV(estimator=dtree, param_grid=dtree_param, cv=5)
cv_grid = grid_search.fit(x_train,y_train)
cv_grid.best_params_
# In[65]:


dtree_tuned = DecisionTreeRegressor(criterion='mse',max_depth=5,max_features='auto',
                               min_samples_split=10,random_state=123,splitter='best')
model_dtree_tuned = dtree_tuned.fit(x_train,y_train)
pred_dtree_tuned = model_dtree_tuned.predict(x_test)
RMSE = np.sqrt(np.mean((y_test-pred_dtree_tuned)**2))
MAPE = np.mean(np.absolute((y_test-pred_dtree_tuned)/pred_dtree_tuned))
RMSE,MAPE


# ##### KNN Tuning
knn_param = {
    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size':[20,30,40],
    'n_jobs':[1,-1],
    'n_neighbors':[4,5,6],
    'p':[1,2,3],
    'weights':['uniform','distance']#,
#     'random_state':[123]
}

grid_search = GridSearchCV(estimator=knn, param_grid=knn_param, cv=5)
cv_grid = grid_search.fit(x_train,y_train)
cv_grid.best_params_
# In[76]:


knn_tuned = KNeighborsRegressor(algorithm = 'auto', leaf_size = 20, n_jobs = 1,
                                n_neighbors = 6, p = 3, weights = 'uniform')
model_knn_tuned = knn_tuned.fit(x_train,y_train)
pred_knn_tuned = model_knn_tuned.predict(x_test)
RMSE = np.sqrt(np.mean((y_test-pred_knn_tuned)**2))
MAPE = np.mean(np.absolute((y_test-pred_knn_tuned)/pred_knn_tuned))
RMSE,MAPE


# #### Till now best.....

# In[78]:


req = dtree_tuned.predict(test1)


# In[79]:


sample=test[["Item_Identifier","Outlet_Identifier"]]


# In[80]:


sample["Item_Outlet_Sales"]=req


# In[81]:


sample.head()


# In[129]:


sample.to_csv("sample_sale.csv", index=False)


# In[ ]:




