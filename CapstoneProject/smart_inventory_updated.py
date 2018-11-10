
# coding: utf-8

# # Machine Learning Hackathon
# ## Supervised Learning
# ## Project: Smart Inventory

# ## Getting Started
# 
# In the targeted approach the company tries to identify in advance customers who are likely to churn. The company then targets those customers with special programs or incentives. This approach can bring in huge loss for a company, if churn predictions are inaccurate, because then firms are wasting incentive money on customers who would have stayed anyway. There are numerous predictive modeling techniques for predicting customer churn. These vary in terms of statistical technique (e.g., neural nets versus logistic regression versus survival analysis), and variable selection method (e.g., theory versus stepwise selection).
# 

# ----
# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the census data. Note that the last column from this dataset, `'churn'`, will be our target label (where churn or not). All other columns are features about each individual in the census database.

# In[3]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score



# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the Census dataset
tele = pd.read_csv("bigml_telecom.csv")

# Success - Display the first record
display(tele.head(n=5))


# ### Implementation: Data Exploration
# A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals having churn. In the code cell below, you will need to compute the following:
# - The total number of records, `'n_records'`
# - The number of individuals having churn.
# - The number of individuals not having churn.
# -Percentage of individuals having churn
#  

# In[135]:


tele.shape[0]


# In[5]:


print(tele.dtypes)


# In[115]:


#Check Missing values 
tele.info()


# In[114]:


#Check NAN values
tele.isna().sum()


# In[7]:


#print(tele.groupby('churn')['phone number'].count())
n_records = tele.shape[0]
n_churn= tele.groupby('churn')['phone number'].count()
#Since true is the second element taking secode element for percentage of churn ie n_churn[1]
greater_percent = (n_churn[1]/n_records)*100
print("The total number of customers",n_records)
print("The number of customers that didn't churn",n_churn[0])
print("The number of customers that churn",n_churn[1])
print("The percentage of customers churned",greater_percent)


# In[8]:


tele.describe()


# In[10]:


# Split the data into features and target label
churn_data = tele['churn']
features_data = tele.drop('churn', axis = 1)
#features_log_transformed = pd.DataFrame(data = features_data)
hists = features_data.hist(figsize=(16, 16), bins=20,edgecolor='black')
# Visualize skewed continuous features of original data
#vs.distribution(tele)


# In[109]:



features_data.columns=['state','acclen','arcode','phno','intl plan','vc plan','vmailmsgno','daymin','daycall','daychg','evnmin','evncall','evnchg','nghtmin',
                    'nghtcall','nghtchg','intmin','intcall','intchg','custcall']
pd.plotting.scatter_matrix(features_data,figsize=(15,15))


# In[116]:


drp = tele[['state','area code','phone number','international plan','voice mail plan','churn']]
X= tele.drop(drp,1)
y= tele.churn
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)


# In[117]:


X.describe()


# In[118]:


#churn_data = tele['churn']
#features_data = tele.drop('churn', axis = 1)
features_log_transformed = pd.DataFrame(data = X)


# In[137]:


# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['account length', 'total day minutes','total day calls','total eve minutes','total eve calls','total night minutes','total night calls','customer service calls','total intl charge','total intl calls','total intl minutes','total night charge','total eve charge','total day charge','number vmail messages']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))
#features_log_minmax_transform.describe()


# In[120]:


#TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
#features_final = pd.get_dummies(data=features_log_minmax_transform, columns=['international plan', 'voice mail plan','state'])

# TODO: Encode the 'income_raw' data to numerical values
churn_data = pd.get_dummies(data=y)


# Print the number of features after one-hot encoding
encoded = list(churn_data.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
print(encoded)
display(churn_data.head(n = 5))


# In[125]:


ax = churn_data.hist(figsize=(4,4), bins=20,edgecolor='black')
for axe in ax.flatten():
    axe.set_xlabel("Churn")
    axe.set_ylabel("Frequency")


# In[126]:


n_churn = tele.groupby('churn')['phone number'].count()
print(n_churn[1])
print(tele.shape[0])

TP = np.sum(n_churn[1])
print(TP)
FP = tele.shape[0]-TP
print(FP)
TN = 0
FN = 0


# TODO: Calculate accu(racy, precision and recall
accuracy = (TP+TN)/tele.shape[0]
#print(accuracy)
recall = TP/(TP+FN)
print(recall)
precision = TP/(TP+FP)
print(precision)




# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
fscore = (1+0.5**2)*((recall*precision)/(0.5**2*precision+recall))
print(fscore)

# Print the results 
print("Naive Predictor: [F-score: {:.4f}]".format(fscore))


# ### Implementation - Creating a Training and Predicting Pipeline
# To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
# In the code block below, you will need to implement the following:
#  - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
#  - Fit the learner to the sampled training data and record the training time.
#  - Perform predictions on the test data `X_test`, and also on the training `X_train`.
#    - Record the total prediction time.
#  - Calculate the F-score for both the training subset and testing set.
#    - Make sure that you set the `beta` parameter!

# ###  Supervised Learning Models
# **The following supervised learning models are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - Logistic Regression

# -  Linear Regression
# 
# Linear regression is great when the relationship to between covariates and response variable is known to be linear (duh). This is good as it shifts focus from statistical modeling and to data analysis and preprocessing. It is great for learning to play with data without worrying about the intricate details of the model.
# 
# A clear disadvantage is that Linear Regression over simplifies many real world problems. More often than not, covariates and response variables don’t exhibit a linear relationship. Hence fitting a regression line using OLS will give us a line with a high train RSS.
# 
# In summary, Linear Regression is great for learning about the data analysis process. However, it isn’t recommended for most practical applications because it oversimplifies real world problems
# 
# -  Random Forest
#     - Good Real world example is Classification of Urban Remote Sensing Images [https://www.researchgate.net/publication/3204218_Decision_Fusion_for_the_Classification_of_Urban_Remote_Sensing_Images]
#     - Strengths
#         - can handle missing values
#         - little to no tweaking required
#         - can work in parallel
#     - Weakness
#         - bias for multiclass problems towards more recurring classes 
#         - difficulty in interpreting the model
#     - Since, Random Forest are great on almost any machine learning problem, and will not get overfit like decision tree, I think it would be appropriate along with the fact that our dataset has categorical features.
# 
# 
# **Adaboost** is a Boosting type Ensemble Learning Method. In the industry, boosting algorithms have been used for the binary classification problem of face detection where the algorithm has to identify wheter a portion of an image is a face or background (ref: https://en.wikipedia.org/wiki/Boosting_(machine_learning)). One of the main strenghts of Adaboost is that it is a fast algorithm, agnostic to the classifier and less prone to overfitting. During the iterative training, it continuously gives more weight to missclassified labels to allow the classifier to focus on the harder cases which increases the overall model's performance. On the other hand, noisy data and outliers in the data can negatively impact the performance so data pre processing is important. Furthermore, if a complex model is used as the base classifier, this can lead to overfitting to the training data. In my opinion, this model is a good candidate for the problem as our dataset is large yet clean. Therefore we will be able to perform multiple quick trainining iterations to maximize our overall accuracy on the unseen testing data.
# 
# 
# 

# In[131]:


from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score

results_logreg = {}

time_logreg = {}


logreg = LogisticRegression()
start = time()
logreg.fit(X_train, y_train)
end = time()
time_logreg['train_time'] = end-start

start = time()
predictions_test = logreg.predict(X_test)
predictions_train = logreg.predict(X_train)
end = time()
time_logreg['pred_time'] = end-start
#results_logreg['acc_train'] = accuracy_score(y_train[:300],predictions_train)
results_logreg['f_train'] = fbeta_score(y_train,predictions_train,beta=0.5)
#results_logreg['acc_test'] = accuracy_score(y_test,predictions_test)
results_logreg['f_test'] = fbeta_score(y_test,predictions_test,beta=0.5)

#print ("Logistic Regression: [Train Time : {:.4f},Predict Time : {:.4f}, Accuracy Train: {:.4f},F train: {:.4f},Acc Test: {:.4f},F Test: {:.4f}]".format(time_logreg['train_time'],time_logreg['pred_time'],results_logreg['acc_train'],results_logreg['f_train'],results_logreg['acc_test'],results_logreg['f_test']))

#print('Logistic regression  f score =',round(metrics.accuracy_score(y_test, predictions_test),2))
print('Logistic regression  f train score =',results_logreg['f_train'])
print('Logistic regression  f test score =',results_logreg['f_test'])
print('Logistic regression  prediction timing =',time_logreg['pred_time'])
print('Logistic regression  training =',time_logreg['train_time'])


# In[133]:


scores = cross_val_score(logreg, X, y, cv=5, scoring='f1') 
print('Logistic regression of each partition\n',scores)
print('Mean score of all the scores after cross validation =',round(scores.mean(),2)) 
results_logreg['mean_score'] =  round(scores.mean(),2)


# In[71]:


conf = (metrics.confusion_matrix(y_test, predictions_test))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(conf,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt="d",)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[72]:


#results = {}
#results[logreg] = {}
#results[logreg][0] = {}

#results_logreg = {}
#results_logreg['train_time'] = end-start
#results_logreg['acc_test'] = accuracy_score(y_test,predictions_test)
#results_logreg['f_test'] = fbeta_score(y_test,predictions_test,beta=0.5)


# In[73]:


FP = conf[1][0]
FN = conf[0][1]
TP = conf[0][0]
TN = conf[1][1]
print('False Positive ',FP)
print('False Negative ',FN)
print('True Positive ',TP)
print('True Negative ',TN)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print('\nTrue Positive Rate :',round(TPR,2))
# Specificity or true negative rate
TNR = TN/(TN+FP) 
print('\nTrue Negative Rate :',round(TNR,2))
# Precision or positive predictive value
PPV = TP/(TP+FP)
print('\nPositive Predictive Value :',round(PPV,2))
# Negative predictive value
NPV = TN/(TN+FN)
print('\nNegative Predictive Value :',round(NPV,2))
# Fall out or false positive rate
FPR = FP/(FP+TN)
print('\nFalse Positive Rate :',round(FPR,2))
# False negative rate
FNR = FN/(TP+FN)
print('\nFalse Negative Rate :',round(FNR,2))
# False discovery rate
FDR = FP/(TP+FP)
print('\nFalse Discovery Rate :',round(FDR,2))

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
#print('\nOverall accuracy :',round(ACC,2))


# In[134]:


results_rf = {}
timing_rf  = {}
rf_clf = RandomForestClassifier(n_estimators=120, criterion='entropy')
start = time()
rf_clf.fit(X_train, y_train)
end = time()
timing_rf['train_time'] = end-start

start = time()
predictions_test = rf_clf.predict(X_test)
predictions_train = rf_clf.predict(X_train)
end = time()

timing_rf['pred_time'] = end-start
#results_rf['acc_train'] = accuracy_score(y_train[:300],predictions_train)
results_rf['f_train'] = fbeta_score(y_train,predictions_train,beta=0.5)
#results_rf['acc_test'] = accuracy_score(y_test,predictions_test)
results_rf['f_test'] = fbeta_score(y_test,predictions_test,beta=0.5)


print('Random Forest Classifier f train score =',results_rf['f_train'])
print('Random Forest Classifier  f test score =',results_rf['f_test'])
print('Random Forest Classifier training time =',timing_rf['train_time'])
print('Random Forest Classifier  predict =',timing_rf['pred_time'])
#print ("Random Forest: [Train Time : {:.4f},Predict Time : {:.4f}, Accuracy Train: {:.4f},F train: {:.4f},Acc Test: {:.4f},F Test: {:.4f}]".format(timing_rf['train_time'],timing_rf['pred_time'],results_rf['acc_train'],results_rf['f_train'],results_rf['acc_test'],results_rf['f_test']))

#rf_pred_test = rf_clf.predict(X_test)
#print('Accuracy of Random forest :',round(metrics.accuracy_score(y_test, predictions_test),2))


# In[75]:


rf_scores = cross_val_score(rf_clf, X, y, cv=5, scoring='f1')
print('Cross Validation scores using random forest \n',rf_scores)
print('Mean of Cross Validation scores',round(rf_scores.mean(),2)) 
results_rf['Mean_Score'] = round(rf_scores.mean(),2)


# In[76]:


rf_conf = (metrics.confusion_matrix(y_test, predictions_test))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(rf_conf,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt="d",)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[77]:


FP = rf_conf[1][0]
FN = rf_conf[0][1]
TP = rf_conf[0][0]
TN = rf_conf[1][1]
print('False Positive ',FP)
print('False Negative ',FN)
print('True Positive ',TP)
print('True Negative ',TN)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print('\nTrue Positive Rate :',round(TPR,2))
# Specificity or true negative rate
TNR = TN/(TN+FP) 
print('\nTrue Negative Rate :',round(TNR,2))
# Precision or positive predictive value
PPV = TP/(TP+FP)
print('\nPositive Predictive Value :',round(PPV,2))
# Negative predictive value
NPV = TN/(TN+FN)
print('\nNegative Predictive Value :',round(NPV,2))
# Fall out or false positive rate
FPR = FP/(FP+TN)
print('\nFalse Positive Rate :',round(FPR,2))
# False negative rate
FNR = FN/(TP+FN)
print('\nFalse Negative Rate :',round(FNR,2))
# False discovery rate
FDR = FP/(TP+FP)
print('\nFalse Discovery Rate :',round(FDR,2))

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
#print('\nOverall accuracy :',round(ACC,2))


# In[130]:


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


gaussian = GaussianNB()
svc = SVC(random_state=0)
adaboost = AdaBoostClassifier(random_state=0)

results_adb = {}
timing_adb = {}

start = time()
adaboost.fit(X_train, y_train)
end  = time()

timing_adb['train_time'] = end-start

start = time()
predictions_test = adaboost.predict(X_test)
predictions_train = adaboost.predict(X_train)
end = time()

timing_adb['pred_time'] = end-start
#results_adb['acc_train'] = accuracy_score(y_train[:300],predictions_train)
results_adb['f_train'] = fbeta_score(y_train,predictions_train,beta=0.5)
#results_adb['acc_test'] = accuracy_score(y_test,predictions_test)
results_adb['f_test'] = fbeta_score(y_test,predictions_test,beta=0.5)

print('Adaboost Classifier f train score =',results_adb['f_train'])
print('Adaboost Classifier  f test score =',results_adb['f_test'])
print('Adaboost Classifier  prediction time =',timing_adb['pred_time'])
print('Adaboost Classifier  training time =',timing_adb['train_time'])

#adb_pred_test = adaboost.predict(X_test)
#print ("AdaBoost: [Train Time : {:.4f},Predict Time : {:.4f}, Accuracy Train: {:.4f},F train: {:.4f},Acc Test: {:.4f},F Test: {:.4f}]".format(timing_adb['train_time'],timing_adb['pred_time'],results_adb['acc_train'],results_adb['f_train'],results_adb['acc_test'],results_adb['f_test']))

#print('Accuracy of Adaboost :',round(metrics.accuracy_score(y_test, predictions_test),2))


# In[79]:


gaussian.fit(X_train, y_train)
gausn_pred_test = gaussian.predict(X_test)
print('Accuracy of Gaussian :',round(metrics.accuracy_score(y_test, gausn_pred_test),2))


# In[80]:


svc.fit(X_train, y_train)
svc_pred_test = svc.predict(X_test)
print('Accuracy of Adaboost :',round(metrics.accuracy_score(y_test, svc_pred_test),2))


# In[82]:


adb_scores = cross_val_score(adaboost, X, y, cv=5, scoring='f1')
print('Cross Validation scores using adaboost \n',adb_scores)
print('Mean of Cross Validation scores',round(adb_scores.mean(),2)) 
results_adb['Mean_Score'] = round(adb_scores.mean(),2)


# In[83]:


adb_conf = (metrics.confusion_matrix(y_test, predictions_test))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(adb_conf,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt="d",)
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[84]:


FP = adb_conf[1][0]
FN = adb_conf[0][1]
TP = adb_conf[0][0]
TN = adb_conf[1][1]
print('False Positive ',FP)
print('False Negative ',FN)
print('True Positive ',TP)
print('True Negative ',TN)
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print('\nTrue Positive Rate :',round(TPR,2))
# Specificity or true negative rate
TNR = TN/(TN+FP) 
print('\nTrue Negative Rate :',round(TNR,2))
# Precision or positive predictive value
PPV = TP/(TP+FP)
print('\nPositive Predictive Value :',round(PPV,2))
# Negative predictive value
NPV = TN/(TN+FN)
print('\nNegative Predictive Value :',round(NPV,2))
# Fall out or false positive rate
FPR = FP/(FP+TN)
print('\nFalse Positive Rate :',round(FPR,2))
# False negative rate
FNR = FN/(TP+FN)
print('\nFalse Negative Rate :',round(FNR,2))
# False discovery rate
FDR = FP/(TP+FP)
print('\nFalse Discovery Rate :',round(FDR,2))

# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)
#print('\nOverall accuracy :',round(ACC,2))


# #### Results:
# 
# |     Metric       | Linear Regression | Random Forest   |  Adaboost            |
# | :------------:   | :---------------: | :-------------: |  :-------------:     |              
# | F-train          |         0.2680    |   1.0000        |   0.5791             |
# | F-test           |         0.0684    |   0.7650        |   0.4749             |
# | Mean-score       |         0.08      |   0.7454        |   0.4000             |
# | Train-Time       |         0.0312    |   0.6772        |   0.2001             |
# | Prediction-Time  |         0.0       |   0.0534        |   0.0156             |

# In[85]:


#results = {}
#results[logreg.__class__.__name__] = {}

#results[logreg.__class__.__name__][0] = results_logreg
#results[rf_clf.__class__.__name__] = {}
#results[rf_clf.__class__.__name__][0] = results_rf
#results[adaboost.__class__.__name__] = {}
#results[adaboost.__class__.__name__][0] = results_adb

#from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
#import visuals as vs
#vs.evaluate(results, accuracy, fscore)

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

D = {u'Label1':26, u'Label2': 17, u'Label3':30}
print(len(results_logreg))

plt.bar(range(len(results_logreg)), list(results_logreg.values()), align='center')
plt.xticks(range(len(results_logreg)), list(results_logreg.keys()))
plt.title("Accuracy and F-Scores of Logistic Regression")
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x

plt.show()

plt.bar(range(len(results_rf)), list(results_rf.values()), align='center')
plt.xticks(range(len(results_rf)), list(results_rf.keys()))
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x
plt.title("Accuracy and F-Scores of Random Forest")
plt.show()

plt.bar(range(len(results_adb)), list(results_adb.values()), align='center')
plt.xticks(range(len(results_adb)), list(results_adb.keys()))
plt.title("Accuracy and F-Scores of AdaBoost")
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x

plt.show()




plt.bar(range(len(time_logreg)), list(time_logreg.values()), align='center')
plt.xticks(range(len(time_logreg)), list(time_logreg.keys()))
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x
plt.title("Training and Prediction Time of Logistic Regression")
plt.show()

plt.bar(range(len(timing_rf)), list(timing_rf.values()), align='center')
plt.xticks(range(len(timing_rf)), list(timing_rf.keys()))
plt.title("Training and Prediction Time of Random Forest")
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x

plt.show()

plt.bar(range(len(timing_adb)), list(timing_adb.values()), align='center')
plt.xticks(range(len(timing_adb)), list(timing_adb.keys()))
plt.title("Training and Prediction Time of Adaboost")
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x

plt.show()



# In[94]:


# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import AdaBoostClassifier
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

print("Start")
# TODO: Initialize the classifier
clf = RandomForestClassifier(random_state=0)

# TODO: Create the parameters list you wish to tune
parameters = {'n_estimators': [10,100,500,1000,2000], 
              'criterion': ['gini', 'entropy']}

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf,parameters,scoring=scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train,y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
#print ("Unoptimized model\n------")
#print ("F-score on testing data: {:.4f}").format(fbeta_score(y_test, predictions, beta = 0.5))
#print ("\nOptimized Model\n------")
#print ("Final F-score on the testing data: {:.4f}").format(fbeta_score(y_test, best_predictions, beta = 0.5))
# Report the before-and-afterscores
print("Unoptimized model\n------")
#print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
#print("F-score on testing data: {:.4f}").format(fbeta_score(y_test, predictions, beta = 0.5)))
print(fbeta_score(y_test, predictions, beta = 0.5))
print("\nOptimized Model\n------")
#print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test['>50K'], best_predictions)))
#print("Final F-score on the testing data: {:.4f}").format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print(fbeta_score(y_test, best_predictions, beta = 0.5))


# In[95]:


start = time()
best_clf.predict(X_test)
end = time()

print("clf took " + str(end-start))

start = time()
best_clf.predict(X_test)
end = time()

print("best_clf took " + str(end-start))


# ###  Supervised Learning Models
# **The following supervised learning models are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - Logistic Regression

# In[96]:


clf


# ### Question 2 - Model Application
# List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen
# - *Describe one real-world application in industry where the model can be applied.* (You may need to do research for this — give references!)
# - *What are the strengths of the model; when does it perform well?*
# - *What are the weaknesses of the model; when does it perform poorly?*
# - *What makes this model a good candidate for the problem, given what you know about the data?*

# In[97]:


best_clf


# #### Results:
# 
# |     Metric     | Unoptimized Model | Optimized Model |  BenchMark predictor |
# | :------------: | :---------------: | :-------------: |  :-------------:     |              
# | F-score        |       0.6896      |   0.7454        |   0.1748             |
# 

# In[98]:


# TODO: Import a supervised learning model that has 'feature_importances_'
# TODO: Train the supervised model on the training set 

# Since, I already trained a RandomForest with our dataset and it has feature_importances_, 
# I will re use the best_clf
model = best_clf

# TODO: Extract the feature importances
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)


# In[101]:


# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print ("Final Model trained on full data\n------")
#print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print("The prediction on full data",fbeta_score(y_test, best_predictions, beta = 0.5))
print("The prediction on reduced features",fbeta_score(y_test, reduced_predictions, beta = 0.5))
#print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))
#print "\nFinal Model trained on reduced data\n------"
#print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions))
#print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))

