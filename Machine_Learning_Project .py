#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC


# In[10]:


df = pd.read_csv(r'C:\Users\soham\OneDrive\Desktop\ITVedant\ML\Project\loan_data.csv')
df.head()


# In[11]:


df.shape


# In[12]:


df.info()


# In[13]:


df.describe()


# In[15]:


temp = df['Loan_Status'].value_counts()
plt.pie(temp.values,labels=temp.index,autopct='%1.1f%%')
plt.show()


# In[19]:


plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()


# In[22]:


#There are some extreme outlier’s in the data we need to remove them.
df = df[df['ApplicantIncome'] < 25000]
df = df[df['LoanAmount'] < 400000]


# In[23]:


plt.subplots(figsize=(15, 5))
for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
    plt.subplot(1, 2, i+1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()


# In[24]:


df.groupby('Gender').mean()['LoanAmount']


# In[30]:


df.groupby(['Married', 'Gender']).mean()['LoanAmount']


# In[36]:


X=df.iloc[:,4]
Y=df.iloc[:,3]
plt.scatter(X,Y)
plt.show


# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


X=df["ApplicantIncome"]
Y=df["LoanAmount"]


# In[39]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# In[40]:


X_train=np.array(X_train).reshape(-1,1)
X_test=np.array(X_test).reshape(-1,1)


# In[41]:


logistic_model=LogisticRegression()


# In[42]:


logistic_model.fit(X_train,Y_train)


# In[43]:


predictions=logistic_model.predict(X_test)
predictions


# In[44]:


logistic_model.predict([[89]])


# In[45]:


plt.scatter(X_test,Y_test)
plt.scatter(X_test,predictions,color="red")
plt.show()


# In[46]:


from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score


# In[47]:


accuracy=accuracy_score(Y_test,predictions)
accuracy


# In[48]:


error=1-accuracy
error


# In[52]:


from sklearn.tree import DecisionTreeClassifier 
X_train=np.array(X_train)
X_test=np.array(X_test)


# In[53]:


tree_model=DecisionTreeClassifier(max_depth=3)
tree_model


# In[54]:


tree_model.fit(X_train,Y_train)


# In[55]:


from sklearn import tree


# In[56]:


plt.figure(figsize=(10,8))
tree.plot_tree(tree_model,filled=True)


# In[57]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[58]:


km=KMeans(n_clusters=3)
km


# In[59]:


y_predict=km.fit_predict(df[["ApplicantIncome","LoanAmount"]])
y_predict


# In[60]:


df["Cluster"]=y_predict
df.head()


# In[61]:


df1=df[df.Cluster==0]
df2=df[df.Cluster==1]
df3=df[df.Cluster==2]


# In[66]:


plt.scatter(df1.ApplicantIncome, df1["LoanAmount"], color="Red", label="Cluster 1")
plt.scatter(df2.ApplicantIncome, df2["LoanAmount"], color="Green", label="Cluster 2")
plt.scatter(df3.ApplicantIncome, df3["LoanAmount"], color="Black", label="Cluster 3")
plt.xlabel("ApplicantIncome")
plt.ylabel("LoanAmount")
plt.legend()


# In[67]:


scaler=MinMaxScaler()
scaler.fit(df[["LoanAmount"]])


# In[68]:


scaler=MinMaxScaler()


# In[69]:


scaler.fit(df[["LoanAmount"]])


# In[70]:


df["LoanAmount"] = scaler.fit_transform(df[["LoanAmount"]])


# In[71]:


df["LoanAmount"]


# In[72]:


scaler.fit(df[["ApplicantIncome"]])


# In[73]:


df["ApplicantIncome"] = scaler.fit_transform(df[["ApplicantIncome"]])


# In[74]:


df["ApplicantIncome"]


# In[75]:


y_predict=km.fit_predict(df[["ApplicantIncome","LoanAmount"]])
y_predict


# In[76]:


df["revised_cluster"]=y_predict
df.head()


# In[77]:


revised_cluster = km.fit_predict(df[["ApplicantIncome", "LoanAmount"]])

df1 = df[revised_cluster == 0]
df2 = df[revised_cluster == 1]
df3 = df[revised_cluster == 2]


# In[81]:


plt.scatter(df1.ApplicantIncome, df1["LoanAmount"], color="Red", label="Cluster 1")
plt.scatter(df2.ApplicantIncome, df2["LoanAmount"], color="Green", label="Cluster 2")
plt.scatter(df3.ApplicantIncome, df3["LoanAmount"], color="Black", label="Cluster 3")

plt.scatter(km.cluster_centers[:, 0], km.cluster_centers[:, 1], color="Purple", label="centroid")

plt.xlabel("ApplicantIncome")
plt.ylabel("LoanAmount")
plt.legend()
plt.show()


# In[ ]:




