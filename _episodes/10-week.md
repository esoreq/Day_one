---
title: "How to compare models in supervised learning"
author: "Dr. Eyal Soreq" 
start: true
date: "10/08/2021"
teaching: 60
exercises: 0
questions:
- Why are we resampling?  
- How does resampling work?
- Bootstrap and Other Resampling Methods
- How do we compare models? 
objectives:
- Understand the different resampling methods
- Understand how to compare models in ML 

---
# Resampling

- Sampling with and without replacement
- Cross validation and LOOCV 
- Bootstrap (using sampling with replacement)
- Jackknife 
- Permutation resampling 


# Resampling with replacement

## Preliminaries

Our dataset can reflect either a population or sample (it is a function of context) of size N (i.e. the number of observations). In the univariate case, we define X as our dependent variable and x_i is a specific observation in that feature vector. There are various ways to take a sample, e.g. a simple random sample (SRS) of size n, where n < N.

A **population** is the entire group that you want to draw conclusions about.

A **sample** is the specific subset of the population. The size of the sample is always less than the total size of the population.

## Relevant Statistical Formulas (you should all know by now)

| Name | Symbol | Formula   |<p width=10em>Pyhton</p>   |  <p width=35%>description </p>  |  
|  :---  |  :---  |  :---:  |  :----  |  :----  |  
| Mean | $$\mu$$ | $$\mu = 1/N \sum_{i=1}^N x_i$$ | `mu = np.mean(x)` | The population mean is an average of a group characteristic.|  
| Standard deviation (SD) | $$\sigma$$ | $$\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_{i}-\mu)^{2}}$$ |  `sd = np.std(x)` | The population standard deviation is the amount of variation or dispersion of a set of values.|  
| Proportion | $$\mu_{\hat{p}}$$ | $$p = \frac{\sum x}{N} = \frac{\text {number of 1s}}{N}$$ | `p = np.mean(x)` | The population proportion is an average of a group characteristic for a binary value.|
| Proportion (SD) | $$\sigma_{\hat{p}}$$ | $$\sigma = \sqrt{p(1-p)}$$  |`sd = np.sqrt(p*(1-p))` |  The dispersion of the feature|
| Degree of freedom | $$df$$ | $$df = N - 1 - k$$  |`df = X.shape[0]-1-X.shape[1]` |  The number of "observations" (pieces of information) in the data that are free to vary when estimating statistical parameters|
 

# Resampling
- The process of selecting a subset(a predetermined number of observations) from a larger population.

1. Probability Sampling is when we choose a sample based on the theory of probability.
1. Non-Probability Sampling is when we choose a sample based on non-random criteria i.e. not every member of the population has a chance of being included (less relevant to our case).

# Types of Probability Resampling (not exhaustive)

- Simple Random Sampling: Samples are drawn with a uniform probability from the domain.
- Stratified Sampling: Samples are drawn within pre-specified categories or factors.


# Aspects to consider when performing sampling

- Sample Goal. What is the metric you wish to estimate using the sample.
- Population. Where are observations sampled or simulated from.
- Selection Criteria. How do you select observations in each sample.
- Sample Size. What is the number of observations that will constitute the sample.
- Sample proportion. What multilevel factors are reflected in the sample


# Let's inject some code to the theory

~~~python
import numpy as np 
import pandas as pd 
import seaborn as sns
SEED = 2021
N = 50 #observations
rng = np.random.default_rng(SEED)
~~~

## **Random Sampling**: Each element of the population has an equal chance of being selected.


### **Without replacement**: 
- Each element in the population is equally likely to be selected but only once.

#### One option 
~~~python
X = np.arange(N)
sample_size = (10, 5) # 10 obs over 5 folds 
subsets = rng.choice(X, sample_size, replace=False)
subsets = pd.DataFrame(subsets)
sns.heatmap(subsets, cmap='Spectral')
~~~

#### If data is large 
~~~python
sample_size = (10, 5)
X = np.arange(N)
rng.shuffle(X)
subsets = pd.DataFrame(X.reshape(sample_size))
sns.heatmap(subsets, cmap='Spectral')
~~~


### **With replacement**: 
- All elements of a population have an equal chance of being selected more than once - n can be larger than N in this example. 

~~~python
X = np.arange(N)
sample_size = (20, 5) # 20 obs over 5 folds 
subsets = rng.choice(X, sample_size, replace=True)
subsets = pd.DataFrame(subsets)
sns.heatmap(subsets, cmap='Spectral')
~~~   

## Stratified Sampling
Samples are drawn within pre-specified categories or factors.
1. We group the entire population into subpopulations by some common property.
1. We then randomly sample from those groups individually, such that the groups are still maintained in the same ratio. 

~~~python
X = np.arange(N)
ratio, groups = (0.25, 0.45, 0.3), [0, 1, 2]
Y = np.repeat(groups, (np.array(ratio)*N).astype(int))
folds, sample_size = 5, 20
subset_size = np.array(ratio)*sample_size
dummy = np.eye(len(groups))[Y]
subsets = []
for i, k in enumerate(subset_size):
    _X = X[np.where(dummy[:, i] == 1)[0]]
    subsets.append(pd.DataFrame(rng.choice(_X, (int(k), folds))))
subsets = pd.concat(subsets)
sns.heatmap(subsets, cmap='Spectral')
~~~  

# Why are we resampling? 
- Resampling is intended to approximate the variability of some target metrics given some empirical conditions.
- This, in turn, gives us the ability to conduct an empirical comparison within the limitations of our dataset.
- It also allows us to augment our data in different forms (e.g. to handle imbalanced data or to introduce some heuristics to increase heterogeneity of the dataset)

# Cross-Validation
A crucial part of machine learning is training the model. A model is created based on a set of data called the training set. 
We then demonstrate the accuracy of the model on an unseen subset of the data, which is often referred to as a test or validation set. 
Multiplication of independent subsets of training and, therefore, multiple models are extremely beneficial for many reasons. Each of these splits, however, may limit our model's ability to learn a precise representation of the problem. This is sometimes a required feature of our learning pipeline (e.g. fitting age-band specific models as part of an ensemble learning algorithm). Generally, however, we will make use of all of our data (since it is very expensive to acquire).  
Unless manually constrained, models will grow in complexity in an attempt (often futile) to explain every data point. Usually, this kind of behaviour is called overfitting, and it can often be found in situations (at least in our field) when you have more observations than features (aka the "curse of dimensionality" or large P, small N, (P >> N)).
Generally, this means our model will be very good at predicting known situations, but it will struggle to generalize.
Using an iterative split and reporting mean performance over different iterations is one method of evaluating ML performance with minimal data sacrifice, as demonstrated last week. You may also look at the predictions we can make from unseen data. This process is called cross validation, i.e. learn from the training subset and test on the validation subset.  

# Data Subset

When selecting a model, we distinguish 3 different parts of the data that we have as follows:
- Training set (usualy 75-90% of the data) on which the model is trained
- Validation set (usualy 25-10% of the data) on which the model is assessed (Also called hold-out or development set)
- Testing set - the unseen data on which the perfomance of the model is reported

# Common types of CV
Two types of cross-validation can be distinguished: 
- Exhaustive 
- Non-exhaustive 

# Exhaustive CV
 methods which learn and test on all possible ways to divide the original sample into a training and a validation set.

## Leave-p-out cross-validation
- Creates all the possible training/test sets by removing p samples from the complete set. For n samples, this produces n over p train-test pairs. 
- Let's view this visually 

## For example take 20/80% split 
~~~Python
import matplotlib.pyplot as plt
from sklearn.model_selection import LeavePOut

fig,ax = plt.subplots(figsize=(10,4))
params = {"xticklabels" : 0,
          "yticklabels" : 0,
          "annot": False,
          "cbar": False,
          "linewidths" : 1,
          'cmap': 'Paired'}
Leave_p, N = 0.2, 10
N_test = int(N*Leave_p)
Y = np.ones(N)
lpo=LeavePOut(N_test)
data = []
for i, (train, test) in enumerate(lpo.split(Y)):
    tmp = np.ones(N)
    tmp[test]=0
    data.append(tmp)
sns.heatmap(pd.DataFrame(data).T,  ax=ax, **params)
ax.set_xlabel('Number of splits')
ax.set_ylabel('Number of observations')
~~~

- Due to the high number of iterations which grows combinatorically with the number of samples this cross-validation method can be very costly.

## Leave-one-out cross-validation
Leave-one-out cross-validation (LOOCV) is a particular case of leave-p-out cross-validation with p = 1.

~~~Python
fig, ax = plt.subplots(figsize=(10, 4))
Y = np.ones(N)
lpo = LeavePOut(1)
data = []
for i, (train, test) in enumerate(lpo.split(Y)):
    tmp = np.ones(N)
    tmp[test] = 0
    data.append(tmp)
sns.heatmap(pd.DataFrame(data).T,  ax=ax, **params)
ax.set_xlabel('Number of splits')
ax.set_ylabel('Number of observations')
~~~

# Non-exhaustive CV
Methods which learn and test on a subset (often overlaping) of all possible ways to divide the original sample (we are approximating the performance using some assumptions).

## Holdout method
- We randomly assign data points to two sets d0 and d1, usually referred to as training and testing sets. Usually, the test set is smaller than the training set, but it's not set in stone. Training (building a model) on d0, testing (evaluating it) on d1.
- Cross-validation typically involves averaging the results of several model-testing runs; by contrast, holdout involves only a single run. In fact, if the multiple runs are not averaged, you may achieve highly misinforming results without such averaging. 

~~~Python
from sklearn.model_selection import train_test_split
N = 10
X = rng.uniform(1,2,size=(N, 8))
Y = rng.choice([0, 1], p=(0.35, 0.65), size=(N, 1))
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=2021)

fig, ax = plt.subplots(1, 2, figsize=(
    10, 4), gridspec_kw={'width_ratios': [1, 8]})
params['cmap'] = 'Paired'
sns.heatmap(pd.DataFrame(Y),  ax=ax[0], **params)
params['cmap'] = 'Blues'
sns.heatmap(pd.DataFrame(X+2*X*Y),  ax=ax[1], **params)
~~~

## K-Fold 
- Split the dataset into k groups of samples, called folds (if k=n, this is equivalent to the Leave One Out strategy), of equal sizes (if possible). 
- The prediction function is learned using  folds, and the fold left out is used for test.

~~~Python
from sklearn.model_selection import KFold
fig, ax = plt.subplots(figsize=(10, 4))
Y = np.ones(N)
kf = KFold(5)
data = []
for i, (train, test) in enumerate(kf.split(Y)):
    tmp = np.ones(N)
    tmp[test] = 0
    data.append(tmp)
sns.heatmap(pd.DataFrame(data),  ax=ax, **params)
ax.set_ylabel('Number of folds')
ax.set_xlabel('Number of observations')
~~~

## Stratified k-fold
- Stratified k-fold is a variation of k-fold which returns stratified folds. 
- each set contains approximately the same ratio of samples of each target class as the complete set.

~~~Python
from sklearn.model_selection import StratifiedKFold
N = 100
X = rng.uniform(1, 2, size=(N, 8))
Y = rng.choice([1, 2, 3], p=(0.10, 0.25, 0.65), size=(N, 1))
K = 2
skf = StratifiedKFold(K)
params["linewidths"]=0
data = []
for i, (train, test) in enumerate(skf.split(X, Y)):
    fig, ax = plt.subplots(1, 2, figsize=(
        10, 2), gridspec_kw={'width_ratios': [1, 8]})
    C = np.ones((N, 1))
    C[train] = 0
    params['cmap'] = 'Paired'
    sns.heatmap(pd.DataFrame(Y+Y*C*2),  ax=ax[0], **params)
    params['cmap'] = 'Blues'
    sns.heatmap(pd.DataFrame(X+X*C*2),  ax=ax[1], **params)
    plt.tight_layout()
~~~


# Comparing models 


## let's start by creating a regression dataset 
- BTW feel free to mess with this framework 

~~~python
from sklearn.datasets import make_regression
def make_dataset(N=1000,
                 N_outlier=100,
                 Noise=(5, 3, 0.1),
                 plot=True,
                 SEED=2021):
    outlier = np.repeat([False, True], [N-N_outlier, N_outlier])
    X, y, coef = make_regression(n_samples=N,
                                 n_features=2,
                                 n_informative=2,
                                 noise=3,
                                 coef=True,
                                 random_state=SEED)

    X[outlier] = X[outlier] + \
        rng.normal(loc=Noise[0], scale=Noise[1], size=(N_outlier, X.shape[1]))
    y[outlier] = y[outlier] + \
        rng.normal(loc=np.max(y), scale=np.ptp(y)*Noise[2], size=(N_outlier,))
    df = pd.DataFrame(np.hstack(
        [X, y.reshape(-1, 1), outlier.reshape(-1, 1)]), columns=['x1', 'x2', 'y', 'outlier'])
    if plot:sns.pairplot(df, hue='outlier')
    return X, y, df

make_dataset()

~~~

## Choose two models to compare 
- BayesianRidge
The algorithm estimates a probabilistic model of the regression problem.
- RandomForestRegressor
A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

~~~python
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
models = {'Bayesian': BayesianRidge(),
          'RandomForest': RandomForestRegressor()}
~~~

## Define some metrics to compare

```python
from sklearn.metrics import r2_score,median_absolute_error
def calc_error(y,y_hat,i=None,kind='model'):
    err = {'Kind':kind,
            'MAD':median_absolute_error(y,y_hat),
            'r2':r2_score(y,y_hat)}
    return pd.DataFrame(err,index=[i])
```

## We are ready to compare the models and we need to consider our options

### Ideal scenario - You work with simulated data and can potentially collect unlimited independent datasets
- Good for testing models capabilities 
- Acceptable sample size would have 10 observations per feature (10:1) 
- There are a number of factors to take into consideration in this issue, including the effect size estimation and experimental design

~~~python
import pingouin as pg
score_ideal = pd.DataFrame()
obs = 10
for i in range(obs):
    seed = rng.integers(1e9)
    X, y, _ = make_dataset(plot=False, SEED=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=2021)
    for key, mdl in models.items():
        y_hat = mdl.fit(X_train, y_train).predict(X_test)
        score_ideal = score_ideal.append(calc_error(y_test, y_hat, i, key))
sns.scatterplot(x='r2',y='MAD', hue='Kind', data=score_ideal)        
~~~

### We can measure the significance 
- Of the 'Bayesian' models being different from the 'RandomForest'
    - For MAD  
~~~python        
sns.displot(x='MAD', hue='Kind', data=score_ideal)
ix = score_ideal.Kind == 'Bayesian'
pg.ttest(score_ideal[ix].MAD.values, score_ideal[-ix].MAD.values)
~~~
    - Or r2
~~~python        
sns.displot(x='r2', hue='Kind', data=score_ideal)
ix = score_ideal.Kind == 'Bayesian'
pg.ttest(score_ideal[ix].r2.values, score_ideal[-ix].r2.values)
~~~

## Use a naive 10 x CV approach instead
- It is very rare that we can have access to unlimited datasets 
- More often we will have the following

~~~python
score_naive = pd.DataFrame()
X, y, _ = make_dataset(plot=False, SEED=seed)
kfolds = 10
kf = KFold(n_splits=kfolds, random_state=seed,shuffle=True)
for i, (train, test) in enumerate(kf.split(X)):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    for key, mdl in models.items():
        y_hat = mdl.fit(X_train, y_train).predict(X_test)
        score_naive = score_naive.append(calc_error(y_test, y_hat, i, key))
fig,ax = plt.subplots(1,2,figsize=(12,4),sharex=True,sharey=True)        
sns.scatterplot(x='r2', y='MAD', hue='Kind', data=score_ideal,  ax=ax[0])
sns.scatterplot(x='r2', y='MAD', hue='Kind', data=score_naive,ax=ax[1])
ax[1].legend([])   
ax[1].set_title('Ideal')
ax[0].set_title('Naive 10xCV');
~~~
- This violates the t-test independence assumption

## We can measure the significance 

~~~python        
sns.displot(x='r2', hue='Kind', data=score)
ix = score_naive.Kind == 'Bayesian'
pg.ttest(score_naive[ix].r2.values, score_naive[-ix].r2.values)
~~~
~~~python        
sns.displot(x='MAD', hue='Kind', data=score)
ix = score_naive.Kind == 'Bayesian'
pg.ttest(score_naive[ix].MAD.values, score_naive[-ix].MAD.values)
~~~

## We can also compare the two methods

- In the interest of simplicity, I will use a t-test (even though a factorial analysis of variance would be more appropriate)

~~~python
ix = score_ideal.Kind == 'Bayesian'
display(pg.ttest(score_ideal[ix].MAD.values, score_naive[ix].MAD.values))
ix = score_ideal.Kind == 'RandomForest'
display(pg.ttest(score_ideal[ix].MAD.values, score_naive[ix].MAD.values))
~~~


# Altenativly we can use the Bootstrap approuch
Bootstrapping is any test or metric that uses random sampling with replacement, and falls under the broader class of resampling methods.

If we only have one dataset, and compute a statistic on it, we can't see how variable it is, A bootstrap creates a large number of datasets that you might have seen and computes the statistic on each one. 

keep in mind that bootstrapping does not create new data. Instead, it treats the original sample as a proxy for the real population and then draws random samples from it. Consequently, the central assumption for bootstrapping is that the original sample accurately represents the actual population.


~~~python
from sklearn.model_selection import ShuffleSplit
score_boot = pd.DataFrame()
X, y, _ = make_dataset(plot=False, SEED=seed)
n_perms = 100
ss = ShuffleSplit(n_splits=n_perms, test_size=0.5, random_state=2021)
for i, (train, test) in enumerate(ss.split(X)):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    for key, mdl in models.items():
        y_hat = mdl.fit(X_train, y_train).predict(X_test)
        score_boot = score_boot.append(calc_error(y_test, y_hat, i, key))
fig, ax = plt.subplots(figsize=(12, 4), sharex=True, sharey=True)
sns.scatterplot(x='r2', y='MAD', hue='Kind', data=score_boot,  ax=ax)
ax.set_title('Bootstrap');
~~~

## Use estimated statistics to compare 
- Normaly we would compare the mean score of one distribution to a baseline distribution. 
- We can calculate a p-value, which approximates the probability that the score would be obtained by chance. 
- This is calculated as $(C+1)/(n_permutations + 1)$, Where C is the number of permutations whose score >= the mean score.


~~~python
vm = score[ix].r2.mean()
empirical_p = (np.sum(vm > score[-ix].r2)+1)/(n_perms+1)
fig, ax = plt.subplots(figsize=(12, 4))
sns.scatterplot(x='r2', y='Kind', hue='Kind', data=score_boot,  ax=ax)
ax.plot([vm,vm],[0,1], 'r:*')
~~~


## Questions?