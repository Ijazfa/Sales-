#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/Ijaz khan/Downloads/advertising.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


fig, axs = plt.subplots(3, figsize = (5,5))
TV_plt = sns.boxplot(df['TV'] ,ax=axs[0])
Radio_plt = sns.boxplot(df['Radio'], ax=axs[1])
Newspaper_plt = sns.boxplot(df['Newspaper'], ax = axs[2])
plt.tight_layout()


# In[9]:


sns.boxplot(df['Sales'])
plt.show()


# In[10]:


sns.pairplot(df,x_vars=['TV','Radio','Newspaper'],y_vars='Sales', height = 4, aspect=1, kind = 'scatter')
plt.show()


# In[11]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[12]:


x = df['TV']
y = df['Sales']


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=5)


# In[15]:


x_test.shape


# In[16]:


y_test.shape


# In[17]:


x_train.head()


# In[18]:


y_train.head()


# In[19]:


from sklearn.linear_model import LinearRegression 
import statsmodels.api as sm


# In[20]:


x_train_sm = sm.add_constant(x_train)


# In[21]:


lr = sm.OLS(y_train,x_train_sm).fit()


# In[22]:


lr.params


# In[23]:


print(lr.summary())


# In[24]:


plt.scatter(x_train, y_train)
plt.plot(x_train, 6.845 + 0.056 * x_train, 'r')
plt.show()


# In[25]:


y_train_pred = lr.predict(x_train_sm)
res = (y_train - y_train_pred)


# In[26]:


fig = plt.figure()
sns.distplot(res,bins=15)
fig.suptitle('Error')
plt.xlabel('y_train - y_train_pred')
plt.show()


# In[27]:


plt.scatter(x_train,res)
plt.show()


# In[28]:


x_test_sm = sm.add_constant(x_test)

y_pred = lr.predict(x_test_sm)


# In[29]:


y_pred.head()


# In[30]:


from sklearn.metrics import mean_squared_error,r2_score


# In[31]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[32]:


r2_sq = r2_score(y_test,y_pred)
r2_sq


# In[33]:


plt.scatter(x_test, y_test)
plt.plot(x_test,6.845 + 0.056 * x_test, 'r')
plt.show()


# In[34]:


from sklearn.metrics import mean_squared_error,r2_score


# In[35]:


mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f'Mean Squared Error : {mse}')
print(f'R2 Score : {r2}')


# In[36]:


plt.scatter(x_test,y_test, color = 'blue', label = 'True values')
plt.plot(x_test,y_pred,color='red',label= 'predicted values')
plt.xlabel('x values')
plt.ylabel('y values')
plt.title('Linear Regression : True vs Predicted values')
plt.legend()
plt.show()


# In[ ]:




