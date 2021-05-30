---
title: "Modules and packages as a focus"
author: "Dr. Eyal Soreq" 
date: "02/06/2021"
teaching: 15
exercises: 0
questions:
- FIXME
objectives:
- FIXME
keypoints:
- FIXME
---


# Modules and Packages

- Python core language is compact. 
- This is an intentional design feature to maintain simplicity. 
- Much of the powerful functionality comes through external modules and packages.


#  Module
- A module is simply a file with the `.py` extension, containing Python definitions, functions, and statements. 
- Modules are imported from other modules using the import command.
- The first time a module is loaded into a running Python script, it is initialized by executing the code in the module once.
- If we change the module, we need to reload it for these changes to take effect

# Modules are useful becuase they:

1. Simplify your code by calling readable functions instead of multiple lines of code
2. Make it simpler to maintain, improve and collaborate 
3. Allow you to organize your code in different files (each a different module)
    

# Why use Modules? 

- Using modules, you can break down large programs into smaller, more manageable pieces. 
- Code can also be reused through modules.
- Instead of copying their definitions between different programs, we can create modules for our most commonly used functions.
- a module is also the simplest way to share code with a collaborator

# Imports best practice 

- Use **import x** for importing packages and modules.
- Use **from x import y** where x is the package prefix, and y is the module name with no prefix.
- Use **from x import y as z** if two modules named y are to be imported or if y is an inconveniently long name.
- Use **import y as z** only when z is a standard abbreviation (e.g., np for NumPy).

#  Using jupyter magic to write a our my_module.py

This file can be generated using magic or by creating a text file and renaming it. The example is not intended to be considered good practice, but rather to illustrate some  functionality in general. In Python's naming convention, single underscores are used for internal object names. By doing this, we make the functions more recognizable visually and contextualize them. 

~~~python
%%writefile my_module.py
_sum_sq = lambda data: sum(x ** 2 for x in data)
_sum = lambda data: sum(x for x in data)
_mean = lambda data: _sum(data)/float(len(data))
_sq_mean = lambda data: _sum_sq(data)/float(len(data)-1)
_var = lambda data: sum((x - _mean(data))**2/float(len(data)-1) for x in data)
~~~

#  Using import to load my_module
- As a result, we can import this Python file and all the lambda functions we defined in it will exist  in our scope under my_module. 
- By adding the `%load_ext` and `%autoreload` magic commands I am making sure that any updates I make to my module will be accessible also in my jupyter notebook

~~~python
import my_module 
%load_ext autoreload
%autoreload 2
~~~

# Lets test our module 


~~~python
print(my_module._var(list(range(3,20))))
~~~

~~~
25.5
~~~
{: .output}


#  Add Sample standard deviation to your_module

- The standard deviation of the sample mean is defined using the following formula 

$$ s = \sqrt{\frac{1}{N-1}\sum^n_{i=1} (x_i-\bar{x})^2} $$

- Try to implement your std function and place it into your module file

> ## solution
> >~~~python
%%writefile -a my_module.py
_std = lambda data: _var(data)**0.5
> > ~~~
{: .solution}

- We sometimes create our own mathematical functions for a variety of reasons
- Testing these with established modules is a good habit to have
- The assert keyword lets you test if a condition in your code returns True, if not, the program will raise an AssertionError.
- You can write a message to be written if the code returns False, check the example below.


~~~python
import statistics 
seq = list(range(3,20))
assert my_module._std(seq) == statistics.stdev(seq)
assert my_module._var(seq) == statistics.stdev(seq) ,"variance is not equal to standard deviation" 
~~~


# We can call a module from within a function

> ## challenge
> Paste the following code in your notebook and run it 
> ~~~
def temporal_hello():
    from datetime import datetime,time
    local_dt = datetime.now()
    if (time(6)<local_dt.time()<time(12)): 
        print("Good Morning")
    elif (time(12)<local_dt.time()<time(18)):    
        print("Good Afternoon")
    else:
        print("Good night") 
> ~~~
>
{: .challenge}

You just created a function that importes a module called datetime from a Python package called  datetime

# let's look inside the module first 


- Using the dir command we can see the different function that compose a module


~~~python
",".join(dir(datetime))
~~~


> ## Output
> > ~~~
'__add__, __class__, __delattr__, __dir__, __doc__, __eq__, __format__, __ge__, __getattribute__, __gt__, __hash__, __init__, __init_subclass__, __le__, __lt__, __ne__, __new__, __radd__, __reduce__, __reduce_ex__, __repr__, __rsub__, __setattr__, __sizeof__, __str__, __sub__, __subclasshook__, astimezone, combine, ctime, date, day, dst, fold, fromisocalendar, fromisoformat, fromordinal, fromtimestamp, hour, isocalendar, isoformat, isoweekday, max, microsecond, min, minute, month, now, replace, resolution, second, strftime, strptime, time, timestamp, timetuple, timetz, today, toordinal, tzinfo, tzname, utcfromtimestamp, utcnow, utcoffset, utctimetuple, weekday, year'
> > ~~~
{: .solution}

As you can see these are all methods and functions associated with dealing with time

# Now compare it with the following 

- Using the same dir command we can see the different modules that compose a package

~~~python
import datetime
",".join(dir(datetime))
~~~


> ## Output
> > ~~~
'MAXYEAR,MINYEAR,__builtins__,__cached__,__doc__,__file__,__loader__,__name__,__package__,__spec__,date,datetime,datetime_CAPI,sys,time,timedelta,timezone,tzinfo'
> > ~~~
{: .solution}

This higher level of abstraction comprises modules or classes that can work independently, but have a common function in terms of time, and are thus part of the datetime  package.

# How to define a package

- Packages are a way of structuring Python’s module namespace by using “dotted module names”
- They are simple directories, containing several python scripts.
- Defining a package in Python requires a uniqe folder name and used to require a special file called `__init__.py`

# Make a directory called my_package

- We will use Jupyter magic to change the cell to `%%bash`

~~~python
%%bash
mkdir -p my_package
mv my_module.py my_package/my_stats.py #rename the file and move it into my_package
~~~


# Now we can import our package 
- Using the from notation we import the new my_stats module to our scope 

~~~python
%%bash
from my_package import my_stats
my_stats._std(list(range(3,20)))
~~~

# How do I install additional packages?

- Using a package manager such as Anaconda makes it easy to install almost any additional package. 
- Moreover, it simplifies collaboration by sending a jupyter notebook with a special file called `.yml` and that's it 


# MCN_2021 

- Here is how you set up your environment for the MCN_2021 course using a `.yml` file 
- Create a new python notebook and name it `setup_MCN_2021.ipynb`
- It is up to you to decide what text to include in your markdown cells to remind yourself why you do what you do


## Step 1. create a folder structure

~~~python
%%bash
mkdir ~/MCN_2021/ && cd ~/MCN_2021/
. ~/.make_dir
~~~

## Step 2. create a .yml file with required dependencies

~~~python
%%writefile ~/MCN_2021/MCN_2021.yml
name: MCN_2021

dependencies:
  - python==3.8
  - nipype
  - nilearn
  - joblib
  - scikit-learn==0.22.0
  - nitime
  - bokeh
  - pip
  - pip:
    - nibabel==2.5.2
    - niwidgets
    - jupyterlab
~~~

## Step 3. use yml file to create a new environment 

- If you would like to confirm that everything worked as expected, you can perform this step in the terminal.

~~~python
%%bash
conda env create --name MCN_2021 --file  ~/MCN_2021/MCN_2021.yml
~~~

## Step 4. register the new environment using  ipykernel

~~~python
%%bash
python -m ipykernel install --user --name MCN_2021 --display-name "Python (MCN_2021)"
~~~


## Step 5. Include FSL environment variables

~~~python
%%writefile ~/.local/share/jupyter/kernels/MCN_2021/kernel.json
{
 "argv": [
  "/opt/conda/bin/python",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python (MCN_2021)",
  "env": {
     "FSLDIR":"/home/jovyan/fsl",
     "PATH":"$PATH:/home/jovyan/fsl/bin",
     "FSLOUTPUTTYPE":"NIFTI_GZ"
},
 "language": "python"
}
~~~

# And what about this course? 

- This is an introductory course so we are much simpler 😊

## Step 1. create a folder structure

~~~python
%%bash
mkdir ~/SYS_2021/ && cd ~/SYS_2021/
. ~/.make_dir
~~~

## Step 2. create a .yml file with required dependencies

~~~python
%%writefile ~/SYS_2021/SYS_2021.yml
name: SYS_2021

dependencies:
  - python==3.8
  - scikit-learn
  - numpy 
  - pandas
  - matplotlib
  - bokeh
  - seaborn
  - plotly
~~~

## Step 3. use yml file to create a new environment 

~~~python
%%bash
conda env create --name SYS_2021 --file  ~/SYS_2021/SYS_2021.yml
~~~

## Step 4. register the new environment using  ipykernel

~~~python
%%bash
python -m ipykernel install --user --name SYS_2021 --display-name "Python (SYS_2021)"
~~~

## Links to expand your understanding 

For those interested in learning more...

- [Modules in python](https://docs.python.org/3/tutorial/modules.html#modules)
- [Packages in python](https://docs.python.org/3/tutorial/modules.html#packages) 


{% include links.md %}