
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns

from sklearn import tree
import graphviz
from graphviz import Source
from sklearn.tree import export_graphviz
 
from sklearn.preprocessing import MinMaxScaler
    
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix


# In[2]:


test_performance = pd.read_csv("StudentsPerformance.csv")
test_performance.head()    


# In[3]:


total_score_filter = test_performance["math score"] + test_performance["reading score"] + test_performance["writing score"]
test_performance["total_score"] = total_score_filter
test_performance

male_filter = test_performance["gender"] == "male"
male = test_performance[male_filter]

female_filter = test_performance["gender"] == "female"
female = test_performance[female_filter]


# In[8]:


male["math score"].hist(bins = 15)
plt.xlabel("Math Scores")
plt.ylabel("Frequency")
plt.title("Math Score Distribution for Male students")


# In[9]:


female["math score"].hist(bins = 15)
plt.xlabel("Math Scores")
plt.ylabel("Frequency")
plt.title("Math Score Distribution for Female Students")


# In[10]:


male["writing score"].hist(bins = 15)
plt.xlabel("Writing Scores")
plt.ylabel("Frequency")
plt.title("Writing Score Distribution for Male Students")


# In[11]:


female["writing score"].hist(bins = 15)
plt.xlabel("Writing Scores")
plt.ylabel("Frequency")
plt.title("Writing Score Distribution for Female Students")


# In[12]:


male["reading score"].hist(bins = 15)
plt.xlabel("Reading Scores")
plt.ylabel("Frequency")
plt.title("Reading Score Distribution for Male Students")


# In[13]:


female["reading score"].hist(bins = 15)
plt.xlabel("Reading Scores")
plt.ylabel("Frequency")
plt.title("Reading Score Distribution for female Students")


# In[14]:


lmTM = smf.ols("Q('total_score') ~ Q('gender') + Q('test preparation course') + Q('math score')", test_performance).fit()
lmTM.summary()


# In[15]:


lmTR = smf.ols("Q('total_score') ~ Q('gender') + Q('test preparation course') + Q('reading score')", test_performance).fit()
lmTR.summary()


# In[16]:


lmTW = smf.ols("Q('total_score') ~ Q('gender') + Q('test preparation course') + Q('writing score')", test_performance).fit()
lmTW.summary()


# In[17]:


plt.scatter(test_performance["total_score"],lmTM.fittedvalues)
plt.title("Total Scores vs Predicted Math Scores")
plt.xlabel("Total Score")
plt.ylabel("Predicted Math Scores")


# In[18]:


plt.scatter(test_performance["total_score"],lmTR.fittedvalues)
plt.title("Total Scores vs Predicted Reading Scores")
plt.xlabel("Total Score")
plt.ylabel("Predicted Reading Score")


# In[19]:


plt.scatter(test_performance["total_score"],lmTW.fittedvalues)
plt.title("Total Scores vs Predicted Writing Scores")
plt.xlabel("Total Score")
plt.ylabel("Predicted Writing Score")


# In[20]:


((test_performance["total_score"] - lmTM.fittedvalues)**2).mean()


# In[21]:


((test_performance["total_score"] - lmTR.fittedvalues)**2).mean()


# In[22]:


((test_performance["total_score"] - lmTW.fittedvalues)**2).mean()


# In[23]:


test_performance = test_performance.drop(["race/ethnicity", "parental level of education"], axis = 1)

test_performance = pd.get_dummies(test_performance, columns = ["gender", "test preparation course", "lunch"], drop_first = True)


# In[24]:


X = test_performance.drop('total_score', 1)
y = test_performance["total_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 


# In[25]:


knn = KNeighborsRegressor(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

((y_test - y_pred)**2).mean()


# In[26]:


plt.scatter(y_test, y_test - y_pred)
plt.title("Total Scores vs Predicted Total Score")
plt.xlabel("Total Score")
plt.ylabel("Predicted Total Score")
plt.axhline()


# In[27]:


knn4 = KNeighborsRegressor(n_neighbors = 4)
knn4.fit(X_train, y_train)
y_pred4 = knn4.predict(X_test)

((y_test - y_pred4)**2).mean()


# In[28]:


knn5 = KNeighborsRegressor(n_neighbors = 5)
knn5.fit(X_train, y_train)
y_pred5 = knn5.predict(X_test)

((y_test - y_pred5)**2).mean()


# In[29]:


scaler = MinMaxScaler(feature_range=(0, 1))


# In[30]:


X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)


# In[31]:


knn_scaled = KNeighborsRegressor(n_neighbors = 3)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)

((y_test - y_pred_scaled)**2).mean()


# In[32]:


plt.scatter(y_test, y_test - y_pred_scaled)
plt.title("Total Scores vs Scaled Predicted Total Score")
plt.xlabel("Total Score")
plt.ylabel("Scaled Predicted Total Score")
plt.axhline()


# In[33]:


mses = []
for k in range(1,15):
    knn_scaled = KNeighborsRegressor(n_neighbors = k)
    knn_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = knn_scaled.predict(X_test_scaled)
    mse = ((y_pred_scaled - y_test)**2).mean()
    mses.append(mse)

plt.plot(mses)


# In[34]:


test_performance4 = pd.read_csv("StudentsPerformance.csv")

test_performance4 = pd.get_dummies(test_performance4, columns = ["gender", "test preparation course", "lunch", "parental level of education", "race/ethnicity"])

test_performance4 = test_performance4.drop(["writing score", "reading score"], axis = 1)


# In[35]:


iX = test_performance4.drop('math score', 1)
iy = test_performance4["math score"]

iX_train, iX_test, iy_train, iy_test = train_test_split(iX, iy, test_size = 0.2)


# In[36]:


ireg = tree.DecisionTreeRegressor(max_depth = 4)
ireg = ireg.fit(iX_train,iy_train)
iy_pred = ireg.predict(iX_test)

((iy_pred - iy_test)**2).mean()


# In[37]:


dot_data = tree.export_graphviz(ireg, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("yetanother_test_perform.dot")


# In[ ]:


with open ("yetanother_test_perform.dot", "r") as fin:
    with open("yetanother_test_perform_fixed.dot","w") as fout:
        for line in fin.readlines():
            line = line.replace("X[0]","math score")
            line = line.replace("X[1]","gender female")
            line = line.replace("X[2]","gender male")
            line = line.replace("X[3]","test preparation course_completed")            
            line = line.replace("X[4]","test preparation course_none")
            line = line.replace("X[5]","lunch_free/reduced")
            line = line.replace("X[6]","lunch_standard")
            line = line.replace("X[7]","parental level of education_associate's degree")
            line = line.replace("X[8]","parental level of education_bachelor's degree")
            line = line.replace("X[9]","parental level of education_high school")
            line = line.replace("X[10]","parental level of education_master's degree")
            line = line.replace("X[11]","parental level of education_some college")
            line = line.replace("X[12]","parental level of education_some high school")
            line = line.replace("X[13]","race/ethnicity_group A")
            line = line.replace("X[14]","race/ethnicity_group B")
            line = line.replace("X[15]","race/ethnicity_group C")
            line = line.replace("X[16]","race/ethnicity_group D")
            line = line.replace("X[17]","race/ethnicity_group E")
            fout.write(line)


# In[ ]:


test_performance5 = pd.read_csv("StudentsPerformance.csv")

test_performance5 = pd.get_dummies(test_performance5, columns = ["gender", "test preparation course", "lunch", "parental level of education", "race/ethnicity"])

test_performance5 = test_performance5.drop(["math score", "reading score"], axis = 1)


# In[ ]:


iiX = test_performance5.drop('writing score', 1)
iiy = test_performance5["writing score"]

iiX_train, iiX_test, iiy_train, iiy_test = train_test_split(iiX, iiy, test_size = 0.2)


# In[ ]:


iireg = tree.DecisionTreeRegressor(max_depth = 4)
iireg = iireg.fit(iiX_train,iiy_train)
iiy_pred = iireg.predict(iiX_test)

((iiy_pred - iiy_test)**2).mean()


# In[ ]:


dot_data = tree.export_graphviz(iireg, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iiyetanother_test_perform.dot")


# In[ ]:


with open ("iiyetanother_test_perform.dot", "r") as fin:
    with open("iiyetanother_test_perform_fixed.dot","w") as fout:
        for line in fin.readlines():
            line = line.replace("X[0]","writing score")
            line = line.replace("X[1]","gender female")
            line = line.replace("X[2]","gender male")
            line = line.replace("X[3]","test preparation course_completed")            
            line = line.replace("X[4]","test preparation course_none")
            line = line.replace("X[5]","lunch_free/reduced")
            line = line.replace("X[6]","lunch_standard")
            line = line.replace("X[7]","parental level of education_associate's degree")
            line = line.replace("X[8]","parental level of education_bachelor's degree")
            line = line.replace("X[9]","parental level of education_high school")
            line = line.replace("X[10]","parental level of education_master's degree")
            line = line.replace("X[11]","parental level of education_some college")
            line = line.replace("X[12]","parental level of education_some high school")
            line = line.replace("X[13]","race/ethnicity_group A")
            line = line.replace("X[14]","race/ethnicity_group B")
            line = line.replace("X[15]","race/ethnicity_group C")
            line = line.replace("X[16]","race/ethnicity_group D")
            line = line.replace("X[17]","race/ethnicity_group E")
            fout.write(line)


# In[ ]:


test_performance6 = pd.read_csv("StudentsPerformance.csv")

test_performance6 = pd.get_dummies(test_performance6, columns = ["gender", "test preparation course", "lunch", "parental level of education", "race/ethnicity"])

test_performance6 = test_performance6.drop(["math score", "writing score"], axis = 1)


# In[ ]:


ivX = test_performance6.drop('reading score', 1)
ivy = test_performance6["reading score"]

ivX_train, ivX_test, ivy_train, ivy_test = train_test_split(ivX, ivy, test_size = 0.2)


# In[ ]:


ivreg = tree.DecisionTreeRegressor(max_depth = 4)
ivreg = ivreg.fit(ivX_train,ivy_train)
ivy_pred = ivreg.predict(ivX_test)

((ivy_pred - ivy_test)**2).mean()


# In[ ]:


dot_data = tree.export_graphviz(ivreg, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("ivyetanother_test_perform.dot")


# In[ ]:


with open ("ivyetanother_test_perform.dot", "r") as fin:
    with open("ivyetanother_test_perform_fixed.dot","w") as fout:
        for line in fin.readlines():
            line = line.replace("X[0]","reading score")
            line = line.replace("X[1]","gender female")
            line = line.replace("X[2]","gender male")
            line = line.replace("X[3]","test preparation course_completed")            
            line = line.replace("X[4]","test preparation course_none")
            line = line.replace("X[5]","lunch_free/reduced")
            line = line.replace("X[6]","lunch_standard")
            line = line.replace("X[7]","parental level of education_associate's degree")
            line = line.replace("X[8]","parental level of education_bachelor's degree")
            line = line.replace("X[9]","parental level of education_high school")
            line = line.replace("X[10]","parental level of education_master's degree")
            line = line.replace("X[11]","parental level of education_some college")
            line = line.replace("X[12]","parental level of education_some high school")
            line = line.replace("X[13]","race/ethnicity_group A")
            line = line.replace("X[14]","race/ethnicity_group B")
            line = line.replace("X[15]","race/ethnicity_group C")
            line = line.replace("X[16]","race/ethnicity_group D")
            line = line.replace("X[17]","race/ethnicity_group E")
            fout.write(line)


# In[ ]:


test_performance7 = pd.read_csv("StudentsPerformance.csv")

total_score_filter = test_performance7["math score"] + test_performance7["reading score"] + test_performance7["writing score"]
test_performance7["total_score"] = total_score_filter

test_performance7 = pd.get_dummies(test_performance7, columns = ["gender", "test preparation course", "lunch", "parental level of education", "race/ethnicity"])

test_performance7 = test_performance7.drop(["math score", "writing score", "reading score"], axis = 1)


# In[ ]:


vX = test_performance7.drop('total_score', 1)
vy = test_performance7["total_score"]

vX_train, vX_test, vy_train, vy_test = train_test_split(vX, vy, test_size = 0.2)


# In[ ]:


vreg = tree.DecisionTreeRegressor(max_depth = 4)
vreg = iireg.fit(vX_train,vy_train)
vy_pred = vreg.predict(vX_test)

((vy_pred - vy_test)**2).mean()


# In[ ]:


dot_data = tree.export_graphviz(vreg, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("vyetanother_test_perform.dot")


# In[ ]:


with open ("vyetanother_test_perform.dot", "r") as fin:
    with open("vyetanother_test_perform_fixed.dot","w") as fout:
        for line in fin.readlines():
            line = line.replace("X[0]","total_score")
            line = line.replace("X[1]","gender female")
            line = line.replace("X[2]","gender male")
            line = line.replace("X[3]","test preparation course_completed")            
            line = line.replace("X[4]","test preparation course_none")
            line = line.replace("X[5]","lunch_free/reduced")
            line = line.replace("X[6]","lunch_standard")
            line = line.replace("X[7]","parental level of education_associate's degree")
            line = line.replace("X[8]","parental level of education_bachelor's degree")
            line = line.replace("X[9]","parental level of education_high school")
            line = line.replace("X[10]","parental level of education_master's degree")
            line = line.replace("X[11]","parental level of education_some college")
            line = line.replace("X[12]","parental level of education_some high school")
            line = line.replace("X[13]","race/ethnicity_group A")
            line = line.replace("X[14]","race/ethnicity_group B")
            line = line.replace("X[15]","race/ethnicity_group C")
            line = line.replace("X[16]","race/ethnicity_group D")
            line = line.replace("X[17]","race/ethnicity_group E")
            fout.write(line)


# In[ ]:


test_performance8 = pd.read_csv("StudentsPerformance.csv")

total_score_filter = test_performance8["math score"] + test_performance8["reading score"] + test_performance8["writing score"]
test_performance8["total_score"] = total_score_filter


# In[ ]:


race_eth_prob = test_performance8["race/ethnicity"].value_counts(normalize = True)


# In[ ]:


score240_filter = test_performance8["total_score"] >= 240
score240 = test_performance8[score240_filter]

hi_score = score240["race/ethnicity"].value_counts(normalize = True)


# In[ ]:


race_df = pd.DataFrame(race_eth_prob)
race_df["high_score"] = hi_score


# In[ ]:


np.abs(race_df["race/ethnicity"] - race_df["high_score"]).sum()/2


# In[ ]:


sample = test_performance8.sample(198)


# In[ ]:


race_df["random"] = sample["race/ethnicity"].value_counts(normalize = True)


# In[ ]:


np.abs(race_df["random"] - race_df["race/ethnicity"]).sum()/2


# In[ ]:


tvds = []
for i in range(1000):
    sample = test_performance8.sample(198)
    sample_counts = sample["race/ethnicity"].value_counts(normalize = True)
    race_df["random"] = sample_counts    
    sample_tvd = np.abs(race_df["random"] - race_df["race/ethnicity"]).sum()/2
    tvds.append(sample_tvd)


# In[ ]:


pd.Series(tvds).hist()
plt.title("Total Variation Distance: Race/Ethnicity & Total Score")
plt.xlabel("Frequency")
plt.ylabel("Total Variation Distance")

