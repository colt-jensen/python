#!/usr/bin/env python
# coding: utf-8

# In[1]:


print('Hello World')


# In[14]:


pwd


# In[26]:


print('This is my source material - https://www.youtube.com/watch?v=7eh4d6sabA0')
print('importing data resource - https://www.datacamp.com/community/tutorials/pandas-read-csv')


# In[17]:


ls


# In[18]:


cd '/Users/Colts45/Downloads/Python-Jupyter/'


# In[35]:


import pandas as pd
df1 = pd.read_csv('vgsales.csv')
df1


# In[21]:


df.shape


# In[22]:


df.describe


# In[23]:


df.values


# In[29]:


print('Esc then H gives you all keyboard shortcuts - ex. in cmnd mode a adds code line above & b below')


# In[33]:


print('dd to delete, click code and shift+tab and it will tell you what the code does')


# In[34]:


print('control + return runs selected code')


# In[37]:


import pandas as pd

music_data = pd.read_csv('music.csv')
music_data


# In[38]:


print('lets clean the data')


# In[40]:


x = music_data.drop(columns=['genre'])
y = music_data['genre']
y


# In[43]:


music_data


# In[44]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']
music_data

model = DecisionTreeClassifier()
model.fit(x,y)
predictions = model.predict([[21,1],[22,0]])
predictions


# In[45]:


print('model predicts that a 21 yr old male will choose hip-hop and 22 fem will like dance')


# In[46]:


print('70-80% should be used as training data')


# In[51]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
predictions

score = accuracy_score(y_test, predictions)
score


# In[52]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.8)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
predictions

score = accuracy_score(y_test, predictions)
score


# In[53]:


print('this is how you save a model so that you do not need to retrain it everytime')


# In[56]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']

joblib.dump(model, 'music-recommend.joblib')

#predictions = model.predict([[21,1]])


# In[55]:


pip install joblib


# In[58]:


print('use # to comment - control + / will comment any highlighted section of code')


# In[59]:


model = joblib.load('music-recommend.joblib')
predictions = model.predict([[21,1]])
predictions


# In[60]:


print('visualization')


# In[62]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(x,y)

tree.export_graphviz(model, out_file='music_recommender.dot',
                    feature_names = ['age', 'gender'], #show rules for these in each node
                    class_names= sorted(y.unique()),
                    label = 'all', #every node has readable labels
                    rounded = True, #rounded corners
                    filled = True) #each cell filled with color


# In[64]:


print('use visual studio code to visualize by dragging a dropping data in vs code window and clicking the three dots in the upper right-hand corner')


# In[65]:


print('OLS example')


# In[66]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(9876789)


# In[67]:


print('artificial data')


# In[69]:


nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x ** 2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

#need intercept so add column of 1s 
X = sm.add_constant(X)
y = np.dot(X, beta) + e


# In[70]:


print('fit and summary')


# In[71]:


model = sm.OLS(y, X)
results = model.fit()
print(results.summary())


# In[72]:


print("Parameters: ", results.params)
print("R2: ", results.rsquared)


# In[73]:


print('We simulate artificial data with a non-linear relationship between x and y:')


# In[74]:


nsample = 50
sig = 0.5
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, np.sin(x), (x - 5) ** 2, np.ones(nsample)))
beta = [0.5, 0.5, -0.02, 5.0]

y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

res = sm.OLS(y, X).fit()
print(res.summary())


# In[75]:


print("Parameters: ", res.params)
print("Standard errors: ", res.bse)
print("Predicted values: ", res.predict())


# In[77]:


print('Draw a plot to compare the true relationship to OLS predictions. Confidence intervals around the predictions are built using the wls_prediction_std command.')


# In[76]:


pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, y, "o", label="data")
ax.plot(x, y_true, "b-", label="True")
ax.plot(x, res.fittedvalues, "r--.", label="OLS")
ax.plot(x, iv_u, "r--")
ax.plot(x, iv_l, "r--")
ax.legend(loc="best")


# In[78]:


print('let me try this ols thing')


# In[79]:


model = df.OLS(Global_sales, NA_Sales)
results = model.fit()
print(results.summary())


# In[92]:


import pandas as pd
import statsmodels.api as sm
import numpy as np
df1 = pd.read_csv('vgsales.csv')

y = df1.Global_Sales
x = df1.NA_Sales
X = sm.add_constant(X) 


# In[93]:


est = sm.OLS(y, X)

est = est.fit()
est.summary()


# In[94]:


# import formula api as alias smf import statsmodels.formula.api as smf 
# formula: response ~ predictors 

import statsmodels.formula.api as smf

est = smf.ols(formula='Global_Sales ~ NA_Sales', data=df1).fit() 
est.summary()


# In[95]:


est = smf.ols(formula='Global_Sales ~ NA_Sales + JP_Sales', data=df1).fit() 
est.summary()


# In[96]:


print('use for interpretation/guidance - https://www.datarobot.com/blog/ordinary-least-squares-in-python/')

