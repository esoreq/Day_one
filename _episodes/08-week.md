---
title: "Feature space partitioned with multidimensional origami"
author: "Dr. Eyal Soreq" 
start: true
date: "27/07/2021"
teaching: 60
exercises: 0
questions:
- What is Supervised Learning?
- What is Regression analysis?
- How does this relate to the simple linear model?
- What is Classification analysis?
- How do we measure perfomance in classification problems? 
- What are the most common classification approches?
objectives:
- Go over the General types of Classification methods
- Learn how to apply different Classification algorithms
- Learn how to approximate perfomance to compare between algorithms
---

# What is Supervised Learning?
- Supervised Learning is all about learning from examples.
- The basic idea is to identify meaningful patterns assoicated with some target label or value.
- Then use these patterns to create a mapping function that is able to map unseen data to the trained value or label.


## Two common procedures

| | Classification |	Regression|
|:--|:--|:--|
| dependent variables | categorical	| continuous|
| Output | predicted Class |	predicted value |
| Performance |	Accuracy |	Deviation| 


# Classification and Regression
- In machine learning, both classification and regression are almost always static models. 
- In static models a distinction exists between two indpendnet stages: 
    1. The learning stage, when learners are trained on a training set 
    2. The performance stage where the model is tested using an independent test set.
- This week we will cover classification models 
- Next week, we will cover regression methods


# What is classification?
- Classification belongs to the category of supervised learning where matching targets (y) and input data (X) are used to train the model.
- The goal of classification algorithms is to uncover a perfect relationship $$f(X)=y$$ 
- However, in reality data is noisy... therefore $$f(X)=y +\epsilon$$
- In Classification we use some computational method to train a model $$\hat f(X) \approx f(X)$$:
    - Which means we try to find a mapping function (f)
    - Based on some input dataset (X)
    - To create discrete output labels (y)
    - That is close as possible to the hypothetical model $$f(X)$$


# Types of classification tasks 
- Binary Classification: Classification task with two possible outcomes. Eg: Young/Old or Male/Female
- Multi-class classification: Classification with more than two classes. Eg: Visual/Auditory/Motor
- Multi-label classification: Classification task where each sample is mapped to a set of target labels (more than one class).  
Eg: Attention/Working memory AND Object/Spatial/Number AND Patient/HC 


# Standard steps involved in building a classification model
- Initialize a classifier. Initialization commonly involves defining initial parameters specific to the type of classifier.
- Train the classifier. All classifiers in scikit-learn are trained using the same fit(X, y) method. That fits the model to the training data (X) and the matching label (y).
- Predict the target: Given an unseen observation X, the predict(X) returns the predicted label y.
- Evaluate the classifier model using different strategies 


# Many Different Families of Classification Methods    
It is out of the scope to cover all of these model families (and even this list is not exuastive) 

- [Distance-Based](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn) e.g. K Neighbors Classifier
- [Statistical](https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn) - e.g naive-bayes
- [linear model](https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python) - e.g. logistic regression
- [Kernel](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python) - e.g. Support vector machine 
- [Tree-Based](https://www.datacamp.com/community/tutorials/decision-tree-classification-python)  e.g. decision tree classifier
- [Ensemble Learners](https://www.datacamp.com/community/tutorials/ensemble-learning-python) - bagging and boosting


# Distance-Based 

The nonparametric algorithms assume that the geometric similarity between observations across the feature space relates to the response vector. These relationships enable us to categorize according to these patterns.

One of the most widely used and easiest nonparametric machine learning algorithms is the k-nearest-neighbour (k-NN) algorithm.


# Statistical

The assumption in these techniques is that the data follows a probability function that needs to be inferred from the dataset. The Gaussian naive Bayes model (Gaussian NB) is normally used with continuous data, where each class' continuous value is assumed to follow a Gaussian distribution.


# Linear model

Basically, a linear classifier makes a classification decision based on a linear combination of features. The use of such classifiers is very useful for most classification issues, and it is often considered the first choice for problems with multiple variables (features), as they reach accuracy levels comparable to non-linear classifiers with a lot less training time and effort.
In its simplest form, logistic regression uses a logistic function to model a binary dependent variable, though there are many more complex extensions available. Binomial logistic models have two possible outcomes, like pass/fail, that's represented by two variables, "0" and "1".

# Kernel models
Kernel methods reach their name by using kernel functions, which enable the user to operate in a high-dimensional, implicit feature space without ever calculating its coordinates. These models perform pattern analysis based on a kernel function, which is a similarity function over pairs of data points.
The support vector machine uses a kernel to identify pairwise distances to establish a support vector group of observations that together construct a locolised margin between classes in a way that maximizes the separation between classes.

# Tree-Based
In these techniques, interpretable decision rules inferred from the training data are used to predict a target variable. Using a decision tree as a predictive model is widely used because it is easy to interpret (i.e., it provides a series of decisions that leads to the final classification result).

# Ensemble Learners

These models combine the predictions of several base estimators built with a given learning algorithm to improve the results and robustness over a single estimator. 

## Random forest
An ensemble of decision trees makes up the random forest method. Several different decision trees are used, each created from a subset of features selected at random. Random forests are hard to interpret since they have multiple weighted decision trees (in some cases, decisions are made by voting on seemingly contradictory facts). They perform well with high-dimensional data, however, and do not require any domain knowledge or complicated parameters.


## AdaBoost 
AdaBoost fits a classifier (which could be any classifier, but is commonly used with decision trees) on a dataset, then fits additional copies of that classifier on the same dataset, with weights adjusted to focus on cases that weren't classified correctly.

# let's get dirty

As a first step, we will create 2D data-sets and apply some models to them so we can get a better grasp of how several methods work. We will then examine some evaluation methods for the same set. Last but not least, we hope to explore some of these using actual data.


# The usual 
~~~python 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.append("../")
from Code import preprocessing as pp
~~~


# The players 
~~~python 
from sklearn.datasets import make_moons,make_blobs,make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
~~~

# Setup
~~~python 
%load_ext autoreload
%autoreload 2
plt.style.use('bmh')
%matplotlib inline
~~~


# We start simple
~~~python 
X, y = make_blobs(n_samples=1000, centers=2, n_features=2,cluster_std=3,random_state=2021)
y[y==0]=-1
data = pd.DataFrame(X)
data['y'] = y
sns.pairplot(data,hue='y',palette='tab10', corner=True)
~~~


# We split the data to separate the learning from the evaluation
~~~python 
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 2021,test_size=0.4)
fig,ax = plt.subplots(1,2,figsize=(10,4))
sns.scatterplot(x=X_train[:,0],y=X_train[:,1],hue=y_train,alpha=0.7,palette='tab10',ax=ax[0])
ax[0].set_title('Train set')
sns.scatterplot(x=X_test[:,0],y=X_test[:,1],hue=y_test,alpha=0.7,palette='tab10',ax=ax[1])
ax[1].set_title('Test set')
~~~

# We construct our model set 
~~~python 
models = {
    "Logistic Regression" :LogisticRegression(),
    "Gaussian naive Bayes": GaussianNB(),
    "K-nearest neighbors" :  KNeighborsClassifier(n_neighbors=20),
    "Linear Support Vector" :SVC(kernel="linear",probability=True,random_state=2021),
    "Radial Basis Support Vector" : SVC(kernel="rbf",probability=True,random_state=2021),
    "Decision Tree": DecisionTreeClassifier(min_samples_leaf=10,random_state=2021),
    "Ada Boost": AdaBoostClassifier(random_state=2021),
    "Random Forest (Bagging+)": RandomForestClassifier(min_samples_leaf=10,random_state=2021),
}
~~~

# Visualise results using a custom plot function
~~~python 
def decision_boundaries_plot(ax,mdl,X_train,X_test,Y_train,Y_test,title=None,cmap='PiYG',step_size=0.05):
    extent = np.array([np.min(X_train,axis=0)*1.1,np.max(X_train,axis=0)*1.1]).T.flatten()
    mdl.fit(X_train,Y_train)
    fs1, fs2 = np.meshgrid(np.arange(extent[0],extent[1], step_size),
                           np.arange(extent[2],extent[3], step_size))
    mesh = np.vstack([fs1.flatten(),fs2.flatten()])
    p = mdl.predict_proba(mesh.T)
    z = np.reshape(p[:,1]-p[:,0],fs1.shape)
    ax.imshow( z, cmap=cmap,alpha=0.5,origin='lower',extent=extent,aspect='auto',vmin=-1,vmax=1)
    ax.scatter(x=X_test[:,0],y=X_test[:,1],c=Y_test,alpha=0.7,cmap=cmap)
    ax.set_title(title)
~~~


# Now we inspect output

~~~python 
fig,ax = plt.subplots(2,4,sharey=True,sharex=True,figsize=(16,5))
for ix,(title,mdl) in enumerate(models.items()):
    i,j = np.unravel_index(ix, (2,4))
    decision_boundaries_plot(ax[i,j],mdl,X_train,X_test,y_train,y_test,title)
~~~


# Let's make the problem harder to solve

~~~python 
X, y = make_blobs(n_samples=1000, centers=2, n_features=2,cluster_std=10,random_state=2021)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 2021)
fig,ax = plt.subplots(1,2,figsize=(10,4))
sns.scatterplot(x=X_train[:,0],y=X_train[:,1],hue=y_train,alpha=0.7,palette='tab10',ax=ax[0])
ax[0].set_title('Train set')
sns.scatterplot(x=X_test[:,0],y=X_test[:,1],hue=y_test,alpha=0.7,palette='tab10',ax=ax[1])
ax[1].set_title('Test set')
~~~

# And inspect the output again

~~~python 
fig,ax = plt.subplots(2,4,sharey=True,sharex=True,figsize=(16,5))
for ix,(title,mdl) in enumerate(models.items()):
    i,j = np.unravel_index(ix, (2,4))
    decision_boundaries_plot(ax[i,j],mdl,X_train,X_test,y_train,y_test,title)
~~~

# What about non-linear datasets?

~~~python 
X, y = make_moons(n_samples=(700,300), shuffle=True, noise=0.2, random_state=2020)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 2021)
fig,ax = plt.subplots(1,2,figsize=(10,4))
sns.scatterplot(x=X_train[:,0],y=X_train[:,1],hue=y_train,alpha=0.7,palette='tab10',ax=ax[0])
ax[0].set_title('Train set')
sns.scatterplot(x=X_test[:,0],y=X_test[:,1],hue=y_test,alpha=0.7,palette='tab10',ax=ax[1])
ax[1].set_title('Test set')
~~~

# And inspect the output one more time

~~~python 
fig,ax = plt.subplots(2,4,sharey=True,sharex=True,figsize=(16,5))
for ix,(title,mdl) in enumerate(models.items()):
    i,j = np.unravel_index(ix, (2,4))
    decision_boundaries_plot(ax[i,j],mdl,X_train,X_test,y_train,y_test,title)
~~~


# Why do we need to approximate classifiers perfomance?
- Estimating classifiers performance is an essential tool for answering some of the following questions:
    - Is there any useful information that can be used to classify events above random chance
    - Compare between information sources. I.e. which subsets of the data are more informative  
    - Identify/rank important sources of information. I.e. are there individual features that contribute more information than other
    - Distinguish between global or local effects
    - Are the natural separations in the data linear, polynomial or non-linear                      


# How do we estimate classifiers perfomance 
- There are several [ways](https://scikit-learn.org/stable/modules/model_evaluation.html#metrics-and-scoring-quantifying-the-quality-of-predictions) 
the simplest is using some metric that evaluates the perfomance of each method
- Among the many different metrics, the easiest to understand is classification accuracy
- This measure simply counts the amount of accuratly predicted labels and divids them by the total number of events 


# The wrong way to measure classification accuracy
- Let's take the naive approach and understand the problems with it?

~~~python 
from sklearn.metrics import accuracy_score
bad_perfomance = {}
for i,(title,mdl) in enumerate(models.items()):
    bad_perfomance[title] = accuracy_score(y,mdl.fit(X,y).predict(X))

df = pd.DataFrame(bad_perfomance.values(),index=bad_perfomance.keys()).rename(columns={0:'perfomance'})
ax = df.plot.barh()
for i,rect in enumerate(ax.patches):
    y = rect.get_y()
    v = df.perfomance.iloc[i]
    ax.text(v*1.01,y,f'{v}%')
ax.legend([])        
~~~


# We were measuring the method tendency to overfit 
- When we are creating a classification model, we aim to identify general/global patterns that describe some real phenomena that differentiate between the two groups  
- Overfitting describes the situation where the model is uncovering local noise patterns that are specific to the data-set but do not reflect a real effect 
- As a result, the model will be extremely accurate in the training stage but will fail when tested on unseen data 
- A model is trained by maximizing its accuracy on the training dataset
- However, performance is determined on its ability to perform well on unseen data.
- In this case, overfitting reflects our model attempt to memorize the training data as opposed to uncovering generalized patterns existing in the training data.
- This example also highlights the sensitivity of non-parametric models (decision trees) to overfit data 

# overfitting causes 
- High dimensional data is susceptible to [overfitting](https://en.wikipedia.org/wiki/Overfitting) 
- The larger the feature space, the more it is affected by local noise patterns 
- This problem of higher dimension is known as the Curse of Dimensionality.
- To avoid overfitting, the needed data will need to grow exponentially as you increase the number of dimensions.
- Also, small datasets are more sensitive to the effects of noise 
- Unfortunatly we don’t usually have the luxury of gathering an extensive database in neuroscience. 
- We need a way to approximate real perfomance 

# [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-evaluating-estimator-performance) as a solution 
- Cross-validation is a set of techniques for assessing how the results of a statistical analysis will generalize to an independent data set.
- The simplest way to cross-validate is to split the dataset into two parts (training and testing) and approximate performance only on the test-set 


# let's create a messy high dimensional dataset

- The data is both high-dimensional and contains noise  
- flip_y controls the fraction of samples whose class is assigned randomly.
- Larger values introduce noise in the labels and make the classification task harder.


~~~python 
X, y = make_classification(n_samples=500,
                           n_classes=2,
                           n_redundant=5,
                           flip_y=0.4,
                           n_informative=6,
                           n_clusters_per_class=2,
                           n_features=20,
                           random_state=2021)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 2021)
                    
f, ax = plt.subplots(1,2,figsize=(10, 4))
ax[0].imshow(np.corrcoef(X))
ax[1].imshow(np.corrcoef(X.T))
~~~


# Compare perfomance on train vs test parts

~~~python 
perfomance = {}
for i,(title,mdl) in enumerate(models.items()):
    mdl.fit(X_train,y_train)
    perfomance[title] = {'Train_score':
                         accuracy_score(y_train,mdl.predict(X_train)),
                         'Test_score':
                         accuracy_score(y_test,mdl.predict(X_test))} 
df = pd.DataFrame(perfomance.values(),index=perfomance.keys())
fig,ax = plt.subplots(1,2,figsize=(15,5))
df.plot.barh(ax=ax[0])
ax[0].invert_yaxis()
ax[0].legend(ncol=2,bbox_to_anchor=(0.75, 1.15))
ax[1].table(cellText=df.round(2).values,colLabels=df.columns
            ,rowLabels=df.index,bbox=(0.4, 0, 0.65, 1.1))
ax[1].axis('off')
~~~                    


# Limitations of the train test split approach 

- Requires big datasets 
- Wasteful (a significant portion of the data is not used)
- Is simple to manipulate (aka cherry-picking a random sample that is not representative of the reality)
- Suffers from knowledge leakage (happens when the method's parameters are optimised to the test set)
- One solution calls for a 3-way splitting instead of the 2-way train/test 
    - Separating data into training, validation and test
    - Using the validation set for Parameter tweaking
    - Reporting performance based on the test set 
- However, this method solves one problem but intensifies the rest.


# K-fold 
- Examining the plot below should persuade you that each fold captures both positive and negative classes 
- However, this is only true for cases when events are unsorted and randomly sampled
- This method will fail for sorted events as can be seen in the bottom panel 

~~~python 
k,n,split = 10,len(y),n//k
tick_args = {"axis" : "both", "which" : "both", "bottom" : False,
       "top" : False,"labelbottom":False, "right":False, "left":False, "labelleft":False}
folds = np.eye(10)[np.repeat(np.arange(0,10), split)].astype(bool)
f = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(5, 1,hspace=1)
ax = (plt.subplot(gs[0, 0]),plt.subplot(gs[1:4, 0]),plt.subplot(gs[4, 0]))
data = {'Response vector':np.eye(2)[y].T,
        '10-folds index matrix':folds.T,
        'Sorted response vector (Negative/Positive)':np.eye(2)[np.sort(y)].T.T }
for i,(title,array) in enumerate(data.items()):
    ax[i].imshow(array, interpolation='nearest',aspect='auto');
    ax[i].set_title(title)
    ax[i].tick_params(**tick_args)
~~~

# K-fold (cont.)
- The solution is simple, instead of troubling ourselves we can just shuffle the fold matrix  

~~~python
rng = np.random.default_rng(2021)
f, ax = plt.subplots(1,1,figsize=(10, 4))
v = np.repeat(np.arange(0,10), split)
rng.shuffle(v)
folds = np.eye(10)[v].astype(bool)
ax.imshow(folds.T, interpolation='nearest',aspect='auto');
ax.tick_params(**tick_args)
ax.set_title('10-folds index matrix')
ax.set_ylabel(f"Folds (K={k})");
~~~


# K-fold (using sklearn)
- The previous examples were intended to explain the concept graphically 
- Instead of using your own function it is better to use sklearn internal cross-validation functions 
- Let's visually confirm that this is the same

~~~python
from sklearn.model_selection import KFold
cv = KFold(n_splits=k,random_state=2020, shuffle=True)
folds = np.zeros((len(y),k)).astype(bool)
for i,(train_index, test_index) in enumerate(cv.split(y)):
    folds[test_index,i] = True   
f, ax = plt.subplots(1,1,figsize=(10, 4)) 
ax.imshow(folds.T, interpolation='nearest',aspect='auto');
ax.tick_params(**tick_args)
ax.set_title('10-folds index matrix')
ax.set_ylabel(f"Folds (K={k})");                    
~~~


# Let's use CV to generate some error bars  
- One of the benefits of using CV is that it is the fastest way to estimate some idea regarding the boundaries of your method's performance 

~~~python
perfomance = {}
for i,(title,mdl) in enumerate(models.items()):
    cv = KFold(n_splits=k,random_state=2021, shuffle=True)
    _perf = []
    for j,(ix_train, ix_test) in enumerate(cv.split(y)):
        mdl.fit(X[ix_train],y[ix_train])
        _perf.append(accuracy_score(y[ix_test],mdl.predict(X[ix_test])))
    perfomance[title] = _perf

df = pd.DataFrame(perfomance.values(),index=perfomance.keys()).stack().reset_index()
df.columns = ['model','k','perfomance']
ax = sns.barplot(y="model", x="perfomance", data=df)
~~~    


# What are we missing? 
- We have some error around the accuracy 
- However, we are oblivious to how the models are performing internally 
- More specifically the accuracy measure we use is not providing us with any per class insight 
- Let's dig deeper and examine (in the next section) the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)


# Confusion matrix 
- The Confusion matrix allows visualization of the performance of classification methods
- To understand this better let us take a step back and return to the train test split 
- The central part of the confusion matrix is composed of counting performance per class 
- In the case of the 2-way classification problem (i.e. binary) it is a 2 x 2 matrix 
- The diagonal represents samples classified correctly and the anti-diagonal those that were misclassified 
![](https://prod-images-static.radiopaedia.org/images/49024440/0f59a975b60e83f5309a5f59075e7f_jumbo.jpeg)


~~~python
from sklearn.metrics import confusion_matrix
perfomance = {}
fig,ax = plt.subplots(2,4,sharey=True,sharex=True,figsize=(16,5))

for ix,(title,mdl) in enumerate(models.items()):
    cv = KFold(n_splits=5,random_state=2021, shuffle=True)
    y_hat = y**0
    for j,(ix_train, ix_test) in enumerate(cv.split(y)):
        mdl.fit(X[ix_train],y[ix_train])
        y_hat[ix_test] = mdl.predict(X[ix_test])
    perfomance[title] = y_hat
    i,j = np.unravel_index(ix, (2,4))
    cfm = confusion_matrix(y,y_hat)
    vmin,vmax = np.min(cfm),np.max(cfm)
    _mask = np.eye(*cfm.shape, dtype=bool)
    sns.heatmap(cfm, annot=True, mask=~_mask, cmap='Blues',fmt="d", vmin=vmin, vmax=vmax,ax=ax[i,j])
    sns.heatmap(cfm, annot=True, mask=_mask, cmap='OrRd',fmt="d", vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]),ax=ax[i,j])
    ax[i,j].set_title(title)
~~~     



# Confusion matrix (cont.) 
- The confusion_matrix function is extracting the core information comparing the true labels and the predicted ones 
- The actual numbers are important (as they immediately give an understanding of how the model performed)
- But using these it is also possible to derive some more sophisticated performance metrics 

# Beyond accuracy 
- [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) 
    - Precision is the fraction of true positives divided by the total number of samples classified as positive (i.e. $TP/(TP+FP)$)
    - Recall (also known as sensitivity) is the fraction of true positives divided by the total number of actual positive samples.  (i.e. $TP/(TP+FN)$)
- [The F1 score](https://en.wikipedia.org/wiki/F1_score) is the harmonic mean of the Precision and recall 
- Matthews correlation coefficient [MCC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn-metrics-matthews-corrcoef) takes into account true and false positives and negatives. It is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. 


# If we open the hood of our models
- By looking at the underlying relationship between observations and classes (by evaluating probability), we can gain some insight into the performance of the models (take this with a grain of salt)

~~~python
fig,ax = plt.subplots(2,4,sharey=True,sharex=True,figsize=(16,5))

for ix,(title,mdl) in enumerate(models.items()):
    cv = KFold(n_splits=5,random_state=2021, shuffle=True)
    y_hat = np.vstack([y*0.1,y*0.1]).T
    for j,(ix_train, ix_test) in enumerate(cv.split(y)):
        mdl.fit(X[ix_train],y[ix_train])
        y_hat[ix_test] = mdl.predict_proba(X[ix_test])
    i,j = np.unravel_index(ix, (2,4))
    jx,jy = rng.normal(0,0.1,size=y.shape),rng.normal(0,0.1,size=y.shape)
    ax[i,j].scatter(y_hat[:,0]+jx,y_hat[:,0]+jy,c=y,s=10,alpha=0.5,cmap='PiYG')
    ax[i,j].set_title(title)
~~~


# Detection error tradeoff and receiver operating characteristic curves
A ROC curve is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:
1. True Positive Rate (TPR) is a synonym for recall :
    $$ TPR = \frac{TP}{TP+TN} $$
1. False Positive Rate (FPR) :
    $$ FPR = \frac{FP}{FP+TN} $$

The detection error tradeoff graph plots the false rejection rate versus the false acceptance rate for binary classification systems.  With non-linear scaling of the x and y axes (or simply by logarithmic transformation), both yield tradeoff curves that are more linear than ROC curves, and which highlight the differences of importance most prominently in the critical operating region.

~~~python
from sklearn.metrics import plot_det_curve,plot_roc_curve

fig,ax = plt.subplots(1,2,figsize=(16,5))
for title,mdl in models.items():
    plot_roc_curve(mdl, X, y, ax=ax[0], name=title)
    plot_det_curve(mdl, X, y, ax=ax[1], name=title)
ax[1].legend(bbox_to_anchor=(0.5, 0.5))
~~~

## Questions?