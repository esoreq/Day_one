---
title: "Getting as near as feasible to the ND snake"
author: "Dr. Eyal Soreq" 
start: true
date: "03/08/2021"
teaching: 60
exercises: 0
questions:
- What is Supervised Learning?
- What is Regression analysis?
- What is a linear model?
- How do we measure perfomance in Regression problems? 
- What are the most common Regression models?
objectives:
- Understand the difference between regression and classification
- Understand when and why we use regression 
- Go over the General types of Regression methods
- Understand the diffrence between parametric and non-parametric models 
- Understand how to aproximate the perfomance of regression algorithems
- Learn how to apply different Regression algorithms
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
- This week we will cover regression methods

# Classification recap 
- Recall that in classification we are trying to uncover some means of differentiating 
between two or more classes based on some theoretical pattern or ruleset that exists in the feature space 
- Let's examine the following simple 2d example: 

```python 
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
X, y = make_blobs(n_samples=1000, centers=2, n_features=2,cluster_std=5,random_state=2020)
fig, ax = plt.subplots(1, 3,figsize=(16,5))
ax[0].hist(X[y!=1,0], density=True, facecolor='k', alpha=0.5);
ax[0].hist(X[y==1,0], density=True, facecolor='y', alpha=0.5);
ax[2].hist(X[y!=1,1], density=True, facecolor='k', alpha=0.5);
ax[2].hist(X[y==1,1], density=True, facecolor='y', alpha=0.5);
ax[1].scatter(X[:,0], X[:,1],c=y);
```

# Classification using Linear Discriminant Analysis (LDA) (the simplest parametric model) 

- In LDA our goal is to estimate 

$$P(Y=k|X=x)$$ 

- In other words, we want to model each classes normal distribution parameters 
(mean and covariance across feature space) and use them to calculate 
the probability of any point in space being associated with a specific class 

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
mdl = LinearDiscriminantAnalysis(solver="lsqr").fit(X,y)
```

# The decision boundary  
- We have per class means $$(\mu_1+\mu_2)$$ (centroids) 
- LDA also assumes that all classes have the same within-class covariance
- The decision boundary passes through the middle-point 
between the two class centroids $$(\mu_1+\mu_2)/2$$ as can be seen in the visualization below 
- As long as the assumption of normality holds LDA is an extremely powerful method 

```python
x1 = np.array([np.min(X[:,0], axis=0), np.max(X[:,0], axis=0)])
b, w1, w2 = mdl.intercept_[0], mdl.coef_[0][0], mdl.coef_[0][1]
y1 = -(b+x1*w1)/w2
fig, ax = plt.subplots(1, 2,figsize=(16,5))
ax[0].scatter(X[:,0], X[:,1],c=y,alpha=0.15);
ax[0].plot(x1,y1,c='k',linewidth=2)
ax[0].plot(mdl.means_[0,0],mdl.means_[0,1],'k',linewidth=2,marker='o', markersize=20,alpha=0.95)
ax[0].plot(mdl.means_[1,0],mdl.means_[1,1],'k',linewidth=2,marker='o', markersize=20,alpha=0.95)
xx, yy = np.meshgrid(np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100),
                     np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100))
Z = mdl.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, zorder=0)
```

# What happens when there is a violation of the assumption of normality? 
- The model will fail to capture the patterns in the data
- This point is vital as most of the information you will work with will 
have much more features than two and will be hard to identify the distribution visually

```python 
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=(700,300), shuffle=True, noise=0.2, random_state=2020)
fig, ax = plt.subplots(1, 1,figsize=(10,10))
ax.scatter(X[:,0], X[:,1],c=y);  
mdl = LinearDiscriminantAnalysis(solver="lsqr").fit(X,y)   
xx, yy = np.meshgrid(np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100),
                     np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100))
Z = mdl.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, zorder=0,cmap='Greys')                     
```

# So how can we adress this problem? 
- There are two solutions: 
    1. Expand the feature space to include non-linear projections of the original variables 
    2. Use non-parametric models 
- Let's explore the first option.    


# Examine Polynomial Features projection

```python 
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import PolynomialFeatures
fig = plt.figure(constrained_layout=True,figsize=(16,8))
gs = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
for ix,degree in enumerate(np.arange(1,10,1)):
    i,j = np.unravel_index(ix, (3,3))
    ax = fig.add_subplot(gs[i,j])
    pf = PolynomialFeatures(degree)
    XP = pf.fit_transform(X)
    mdl = LinearDiscriminantAnalysis(solver="lsqr").fit(XP,y)   
    xx, yy = np.meshgrid(np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100),
                         np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100))
    Z = mdl.predict_proba(pf.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z[:, 1].reshape(xx.shape)
    ax.scatter(X[:,0], X[:,1],c=y);  
    ax.pcolormesh(xx, yy, Z, zorder=0,cmap='Greys')
    ax.set_title(f"Polynom degree={degree} (dim={XP.shape[1]})")
```


# What are the limitation of this method 
- As you saw projecting the data to the 5th polynomial does a 
decent job in capturing the patterns that differentiate between the classes 
- You can also see that the model loses generalization (i.e. overfits) as a function of feature dimensionality 
- Furthermore, creating polynomial feature spaces is only practical in low dimensional (small amount of features)


# What do we mean when we say, nonparametric models?
- Nonparametric methods are also called assumption-free models 
- However, this does not mean that they have no parameters 
- Instead of fitting the features, nonparametric models use the training data to generate the decision space 

# Let's look at the Decision Tree as an example 
- The Decision Tree Classifier predicts the categorical label 
based on a set of local rules that become more specific as the tree grows deeper 
- This heuristic is called recursive partitioning, and the goal is to iteratively 
uncover a set of rules/splits that divide the feature space based on local patterns 
- The recursive divisions will continue until all leaves have been visited or 
some stopping criteria have been reached 

# Visualize the feature space and compare to the parametric alternative  
- As you can see each step is more specific, the first step divides the
the y-axis and splits the data into two parts
- The following steps define the boundary space capturing the spatial 
separation with five rules already

```python 
from sklearn.tree import DecisionTreeClassifier
fig = plt.figure(constrained_layout=True,figsize=(16,8))
gs = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)
for ix,degree in enumerate(np.arange(1,32,2)):
    i,j = np.unravel_index(ix, (4,4))
    ax = fig.add_subplot(gs[i,j])
    mdl = DecisionTreeClassifier(max_depth=degree).fit(X,y)  
    xx, yy = np.meshgrid(np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100),
                         np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100))
    Z = mdl.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.scatter(X[:,0], X[:,1],c=y,alpha=0.2);  
    ax.pcolormesh(xx, yy, Z, zorder=0,cmap='Greys')
    ax.set_title(f"max_depth={degree} (dim={X.shape[1]})")
```

# What are the benefits of tree models? 
- Easy to Understand, Interpret, Visualise
    - Decision Trees work on the feature space and use rules that 
are simple to understand and more importantly explain
- Useful in Data exploration  
    - The recursive division will always start with 
the feature that splits the most significant clusters that might reflect a possible general phenomena 
- almost no data preprocessing required: 
    - As opposed to global methods that are biased by outliers, 
    trees use local relationships to generate rules and therefore are less affected by outliers  
- Trees can handle both numerical and categorical variables.
- Can capture Nonlinear relationships without expanding the feature space


# What is Regression?
- Regression belongs to the category of supervised learning where we model the relationship between the feature space and a continuous target variable (aka the response vector).
- The goal of regression algorithms is to uncover a perfect relationship $$f(X)=y$$ 
- However, in reality data is noisy... therefore $$f(X)=y +\epsilon$$
- In Regression we use some computational method to train a model $$\hat f(X) \approx f(X)$$:
    - Which means we try to find a mapping function (f)
    - Based on some input dataset (X)
    - To create continuous output value (y)
    - That is close as possible to the hypothetical model $$f(X)$$

# Can tree models be used to predict values instead of classes? 
- The idea is simple given some relationship to find a set of rules to predict a value 



# Start with a simple problem

- let's start with a linear problem with one feature and one response vector $$ y = 0.5x+\epsilon $$
- We will use a linear model to fit a line onto the toy observation
- Recall from your statistics courses over the year the simple linear model $$ Y=a+\beta X+\epsilon$$ 
- What we have here is a robust predictive framework 
    - Let's refresh your memory with the intuition behind this formula  
    - a is called the intercept, and it merely means some constant baseline to shift the parametric space 
    - $\beta$ is the weight we are multiplying X with to match Y as best as possible 
    - finally $\epsilon$ is the deviation or error between the product of the 
    function and actual value of Y we are trying to predict  
- The stronger the association between y (the continuous response vector) and X 
the easier it is to fit a line (with only two parameters) that will minimise the error 



# Start with the linear model 
~~~python
f = lambda x,noise: 0.5 * x + noise*rng.uniform(0,1,size=x.shape)
N = 100
x = rng.normal(10,5,size=(N,))
x_range = np.linspace(np.min(y),np.max(y),100)
fig, ax = plt.subplots(1, 5,sharey=True,sharex=True,figsize=(20,5))
for i,sd in enumerate(np.arange(1,10,2)):
    y = f(x,sd)
    ax[i].scatter(x, y,s=5,c='gray',alpha=0.8);
    ax[i].plot(x_range,np.polyval([0.5,0], x_range) ,c='g',label='Ground truth');
    coefs = np.polyfit(x, y, 1)
    ax[i].plot(x_range,np.polyval(coefs, x_range) ,c='b',label='linear model');
    ax[i].set_title(f'$y = {coefs[-1]:0.2f}+{coefs[0]:0.2f}*x +\epsilon$')
    if i==4:ax[i].legend()
~~~


# Compare this to a tree model 

- In contrast, the tree model offers a crude simplified solution to the problem

~~~python
from sklearn.tree import DecisionTreeRegressor,plot_tree
f = lambda x,noise: 0.5 * x + noise*rng.normal(0,1,size=x.shape)
N = 1000
x = rng.normal(10,5,size=(N,1))
x_range = np.linspace(np.min(y),np.max(y),100).reshape(-1, 1)
fig, ax = plt.subplots(1, 5,sharey=True,sharex=True,figsize=(20,5))
for i,sd in enumerate(np.arange(1,10,2)):
    y = f(x,sd)
    mdl = DecisionTreeRegressor(max_depth=2).fit(x,y)
    ax[i].scatter(x, y,s=5,c='gray',alpha=0.8);
    ax[i].plot(x_range,mdl.predict(x_range) ,c='r',label='Ground truth');
    ax[i].plot(x_range,np.polyval([0.5,0], x_range) ,c='g',label='Ground truth');
    if i==4:ax[i].legend()
~~~


# What about non-linear space?

## Let's work with a famous model 
- In this 2D regression example, we will use the Hemodynamic Response Function [(HRF) mixture of gammas](https://www.fil.ion.ucl.ac.uk/~karl/Nonlinear%20event%20related%20responses.pdf) 
as our non-linear regression data generator
- The idea is simple, and you probably covered this in many courses:
    - The canonical hemodynamic response function (HRF)} defines the typical response
     of the neurovascular system (as reflected by the MR signal in a noiseless environment)
     to a brief, intense period of neural stimulation. 
    - With an initial overshoot response followed by a poststimulus undershoot. 
    It is commonly represented as a mixture of gamma probability density functions in the following form: 
- $$H(x) = \frac{l_1^{h_1} x^{h_1 - 1} e^{-l_1 x}}{\Gamma(h_1)}-r\frac{l_2^{h_2} x^{h_2 - 1} e^{-l_2 x}}{\Gamma(h_2)}$$
- where $\Gamma$ is the gamma function with $l, h$ as the scale and shape parameters, respectively, 
and $r$ defines the ratio of response to undershoot. 

# Our hrf function 
- We defined the first peak at 6 seconds and the delay of undershoot has its minima at 16 seconds
- We also defined r as 1/6 

```python 
from scipy.stats import gamma
def sample_hrf(X=500,noise_ratio=0):
    if type(X)==int:
        X = np.random.uniform(0,32,X).reshape(-1, 1)                  
    Y =  gamma.pdf(X , 6)-gamma.pdf(X , 16)/6
    n = X.shape[0]
    if noise_ratio:
        Y = Y+np.random.normal(0,noise_ratio,size=(n,1))
    return Y.reshape(-1,),X  
yn,Xn = sample_hrf(noise_ratio=0.02)   
Y,X = sample_hrf(np.linspace(0,32,32))  
fig, ax = plt.subplots(1, 1,figsize=(8,3))
ax.scatter(Xn, yn,c='lightgray',alpha=0.4);
ax.plot(X, Y,c='k');
ax.plot([0,32], [0,0],'--r');    
```

# How do regression trees work on nonelinear space?
- Regression trees and classification trees work in a similar way 
- They divide the dataspace and create local rules that can be applied to new events 
- How would you describe the HRF textually?
- A simple tree would look like this: 

~~~bash
    |--- if x <= 10
    |   |--- and if x <= 2
    |   |   |--- y = 0
    |   |--- else 
    |   |   |--- y = 0.2
    |--- else >  10
    |   |--- and if x <= 12
    |   |   |--- y: 0.05
    |   |--- else
    |   |   |--- y: -0.01
~~~


# Regression trees overfit 
- Like all instance based modles they are prune to overfitting
- Let's try to visualize the idea of overfitting using the HRF

```python
from sklearn.tree import DecisionTreeRegressor,plot_tree
fig = plt.figure(constrained_layout=True,figsize=(16,6))
gs = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)
x_plot = np.linspace(0, 32, 500).reshape(-1, 1)
for ix,degree in enumerate(np.arange(1,9,1)):
    i,j = np.unravel_index(ix, (2,4))
    ax = fig.add_subplot(gs[i,j])
    mdl = DecisionTreeRegressor(max_depth=degree).fit(Xn,yn)  
    y_hat = mdl.predict(x_plot.reshape(-1, 1))
    ax.scatter(Xn, yn,c='lightgrey',alpha=0.8,label='Data');
    ax.scatter(X, Y,alpha=0.2,s=5);  
    ax.plot(X, Y,c='b',linewidth=4,label='Ground Truth',alpha=0.25);
    ax.plot(x_plot.reshape(-1, ), y_hat,c='r',label='Fitted model',linewidth=2,alpha=0.5);
```


# regression trees are a fast and straightforward way to understand complex relationships 
- We can also visually explore the ruleset of the tree 
- The information in the graph reflects the model's decision space 
- The top node is called the root of the tree 
	- The top value reflects the split rule (in this case a threshold point) As you can see 152 samples are above 9.215 and 348 are below 
	- MSE stands for mean, standard error and is a common way to estimate regression models (however there are many other ways) 
	- Finally, the value represents the predicted value the tree will assign a point 
	- for example, if we take x=6 it will take us in the following path: 
		- left (6<9.8), right (6>2.023),  left (6<7.723), right ( 6>3.25) 
		- our path will end there, and we will be assigned a value of 0.151


```python
mdl = DecisionTreeRegressor(max_depth=4).fit(Xn,yn)
fig, ax = plt.subplots(1, 1,figsize=(20,8))
plot_tree(mdl, filled=True,max_depth=3,fontsize=9,ax=ax)
plt.show()
```

# Can we solve non-linear problems with linear models?

- We can Use the same Polynomial expansion method from the LDA classification step
- As you can see a 9th polynomial is fitted pretty good 
- But up untill now we used our eyes to decide, is there a better way? 

```python
from sklearn.linear_model import LinearRegression
import matplotlib.gridspec as gridspec
fig = plt.figure(constrained_layout=True,figsize=(16,8))
gs = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
x_plot = np.linspace(0, 32, 64).reshape(-1, 1)
for ix,degree in enumerate(np.arange(1,18,2)):
    i,j = np.unravel_index(ix, (3,3))
    ax = fig.add_subplot(gs[i,j])
    pf = PolynomialFeatures(degree)
    xx = pf.fit_transform(Xn)
    mdl = LinearRegression().fit(xx, yn)
    y_hat = mdl.predict(pf.fit_transform(x_plot))
    ax.scatter(Xn, yn,c='lightgray',alpha=0.4,label='Data');
    ax.plot(x_plot, y_hat,c='r',label='Fitted model');
    ax.plot(X, Y,c='b',alpha=0.5,linewidth=4,label='Ground Truth');
    ax.set_title(f"Polynom degree={degree} (dim={xx.shape[1]})")
    ax.legend()
```

# Perfomance measures in regression 
- Recall that in regression our task is to predict a continuous value (the depndent variable)
based a set of independent variables that we assume hold some informative association to the target.
- The various metrics we will use to evaluate regression results are
    - Mean Absolute Error (MAE) $$\text{MAE} = 1/n \sum^n_{i=1} |y_i-\hat y_i|$$    
    - Mean Squared Error(MSE) $$\text{MSE} = 1/n \sum^n_{i=1} (y_i-\hat y_i)^2$$
    - Root-Mean-Squared-Error(RMSE) $$ \text{RMSE} = \sqrt{\frac{\sum^n_{i=1} (y_i-\hat y_i)^2}{n}}$$
    - Median absolute error (MAD)  $$\text{MAD}(y, \hat{y}) = \text{median}(\mid y_1 - \hat{y}_1 \mid, \ldots, \mid y_n - \hat{y}_n \mid).$$
    - $$R^2$$ Error $$ R^2 = 1-\frac{MSE(model)}{MSE(baseline)}$$


# Mean Absolute Error (MAE)  
- MAE is the absolute difference between the target and predicted values.
- It is more robust to outliers compared to MSE. 
- MAE is a linear score which means all the individual differences are weighted equally.
- One benefit of MAE is that it is simple to interpret as no transformation are done to the data.

$$MAE = 1/n \sum^n_{i=1} |y_i-\hat y_i|$$


# Mean Squared Error (MSE) 
- MSE is one the most common metric for fitting regression models.
- It is simply the average of the squared difference between the target predicted values. 
- Squareing the differences means the metric is sensative even to small errors
- This leads to over-estimation of error.

$$MSE = 1/n \sum^n_{i=1} (y_i-\hat y_i)^2$$

# Root Mean Squared Error (RMSE)
- RMSE is the most widely used metric for regression tasks
- It is the square root of the MSE.
- However, the errors are first squared and then avereged
- This poses a high penalty on large errors. 
- And suggests that RMSE is most useful when large errors are undesired.    

$$RMSE = \sqrt{\frac{\sum^n_{i=1} (y_i-\hat y_i)^2}{n}}$$ 

# $$R^2$$ Error
- $$R^2$$ is another metric used for evaluating the performance of a regression model
- The $$R^2$$ metric is used to compare a regression model with some constant baseline (usually the mean)
- The larger the difference, between the model and the baseline the greater the value (and the better the model)
- $$R^2$$ range is $$(-\infty,1)$$; as a result, worse models then the mean constant will be negative 

$$ R^2 = 1-\frac{MSE(model)}{MSE(baseline)}$$

# Let's tie everything with an example based on our real dataset 

- I start by creating a helper function to calculate the different metrics

```python
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,median_absolute_error
def calc_error(y,y_hat,i=None,kind='model'):
    err = {'Kind':kind,
            'MAE':mean_absolute_error(y,y_hat),
            'MSE':mean_squared_error(y,y_hat),
            'MAD':median_absolute_error(y,y_hat),
            'RMSE':mean_squared_error(y,y_hat,squared=False),
            'r2':r2_score(y,y_hat)}
    return pd.DataFrame(err,index=[i])
```

# Is it possible to predict the age of a person from the freesurfer volume approximations?

- Define the feature space and the response vector 
- Make sure to remove nan's i.e. observations with no age 

```python
d_clinical = pp.load('clean')["ADRC_ADRCCLINICALDATA"]
d_volume = pp.load('volume')
age_mapper = d_clinical.set_index('Subject')[['ageAtEntry']].drop_duplicates().to_dict()['ageAtEntry']
age = d_volume.Subject.map(age_mapper)+d_volume.days_since_entry/365
d_volume.insert(1,'Age',age)
d_volume = d_volume.dropna(subset=['Age'])
y = d_volume.Age
X = d_volume.iloc[:,7:]
```

# Calculate the model perfomance using random shuffle (aka bootstrap)

- In ML you need to be sure that your model is valid before you can dig deeper
- If you enouge data it is always a good idea to check your model compared to a baseline model

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
rng = np.random.default_rng(2021)
n_perms = 100
ss = ShuffleSplit(n_splits=n_perms, test_size=0.25, random_state=2021)
mdl = LinearRegression()
score = pd.DataFrame()
for i,(ix_train, ix_test) in enumerate(ss.split(X)):
    X_train,X_test = X.iloc[ix_train,:].values,X.iloc[ix_test,:].values
    y_train,y_test = y.iloc[ix_train].values,y.iloc[ix_test].values
    y_hat = mdl.fit(X_train,y_train).predict(X_test)
    y_null = mdl.fit(rng.permutation(X_train),y_train).predict(X_test)
    score = score.append(calc_error(y_test,y_hat,i,"Model"))
    score = score.append(calc_error(y_test,y_null,i,'Null'))
```


# How did we do? 

```python
sns.scatterplot(data=score,x='r2',y='MAD',hue='Kind')
```

# What about significance ? 

- This will be covered in our last lecture next week 



## Questions?