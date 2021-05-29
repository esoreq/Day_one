---
title: "Taking python to the next level"
author: "Dr. Eyal Soreq" 
date: "05/03/2021"
start: true
teaching: 40
exercises: 20
questions:
- What are Functions?
- What are Classes?
- What are Methods?
- What are Modules? 
- What are Packages?
objectives:
- Learn how to write functions 
- Understand what a class is 
- Understand how to create a module 
- Learn how to load a local module 
- Learn how to install a packages
- Create a python_sandbox conda environment with a list of packages
keypoints:
- FIXME
---

# What are Functions?
Python is full of functions: we've already encountered a variety of built-in functions that can be used for many different tasks. It is likely, however, that it will be necessary for you to write your own functions to deal with problems that your data presents.

Functions in Python fall into three types:
- Built-in functions (all the functions we have used so far are considered built-in)
- Lambda or anonymous functions are very similar to the mathematical definition above; they receive input and return it after applying some rule or expression. 
- General purpose functions that are created to do something useful.

# What are Functions good for?

The two most important reasons that we use functions are *reusability* and *abstraction*

 - Reusability: Once defined, a function can be used repeatedly. By reusing functions in your program, you can save time and effort.
- Abstraction. To use a particular function, you have to know its name, its function, the arguments you need to give it, and what results you will receive.
- Writing large programs that actually work is possible because of the ability to break them up into many abstract, reusable parts.

# Creating and Calling a Function

- In Python a function is defined using the def keyword:

~~~python
def my_first_function():
  print("Hello from my first function")
type(my_first_function)  
~~~

~~~
function
~~~
{: .output}

- You call a function, useing it's name followed by parenthesis

~~~python
my_first_function()
~~~

~~~
Hello from my first function
~~~
{: .output}


# Parameters, Arguments and Function scope

- A parameter is a variable name declared when the function is created.
- A variable is only available from inside the region it is created. This is called scope.
- Parameters are specified after the function name, inside the parentheses. 
- Arguments are the data you pass into the method's parameters
- You can add as many parameters as you want, just separate them with a comma.

- Consider the following example: 

~~~python
def my_second_function(first_name):
  print(f"Hello this function is {first_name} second function")
my_second_function('Eyal')  
~~~

~~~
Hello this function is Eyal's second function
~~~
{: .output}


- What will happen if we run our function like this?


~~~python
my_second_function() 
~~~

> ## Output
> > ~~~
TypeError: my_second_function() missing 1 required positional argument: 'first_name'
> > ~~~
> > {: .error}
{: .solution}

- When creating a function you declare what is the minimum amount of data the function requires in order to run
- By adding a default value, we can fix the above problem

~~~python
def my_second_function(first_name='Eyal'):
  print(f"Hello this function is {first_name} second function")
my_second_function()  
~~~

~~~
Hello this function is Eyal's second function
~~~
{: .output}

# Number of Arguments

- What will happen if we add another argument not declared in the function scope

~~~python
my_second_function('Eyal','Soreq') 
~~~

> ## Output
> > ~~~
TypeError: my_second_function() takes from 0 to 1 positional arguments but 2 were given
> > ~~~
> > {: .error}
{: .solution}


# Consider the following example: 

~~~python
last_name = 'Soreq'
def my_third_function(first_name):
    print(f"Hello this function is {first_name} {last_name} third function")
my_third_function('Eyal')  
~~~

> ## What do you think will happen?
> > ~~~
Hello this function is Eyal Soreq third function
> > ~~~
> > {: .output}
{: .solution}

<!-- # Imports best practice 

We will go over this in detail next week, but it should be stated. 

- Use **import x** for importing packages and modules.
- Use **from x import y** where x is the package prefix, and y is the module name with no prefix.
- Use **from x import y as z** if two modules named y are to be imported or if y is an inconveniently long name.
- Use **import y as z** only when z is a standard abbreviation (e.g., np for NumPy). -->

## Links to expand your understanding 

For those interested in learning more...

- [functions-python-tutorial](https://www.datacamp.com/community/tutorials/functions-python-tutorial)

{% include links.md %}