---
title: "Python Overview"
author: "Dr. Eyal Soreq" 
date: "05/03/2021"
teaching: 15
exercises: 0
questions:
- What is Conda?
- Why should I use a package and environment management system as part of my research workflow?
- What is an environment? 
- How do I create my own environment in the gwdg jupyter cloud 
objectives:
- Understand what Conda is and how it can simplify analysis pipelines.
- Learn how to install and configure Conda on any system.
- Be able to create reproducible computing environments for (almost) any task.

keypoints:
- Conda is a Python-based environment and package manager
- It enables us to use 
---

# What are Data structures?

Data structures organize and store data so that it's easy to access and use. The data and the operations on the data are defined by them. In data science and computer science, there are a variety of different data structures that make it easy for the researchers and engineers to concentrate on the basics and not get lost in the details of data description or access.

In fact all the Basic Data Types we just covered are considered Primitive Data Structures
The non-primitive types are the more sophisticated data structures. Instead of just storing one value, they store a whole bunch of them.

# What are Lists?
- Lists are the basic sequence building block in Python.
- They are mutable, therefore lists elements can be changed!
- Lists are constructed with brackets \[\]
- Every element in the list is separated by commas

# Constructing lists

- Lists can hold any basic data type
- They can also hold mixed types

~~~python
empty_list = []
print(empty_list)
some_list = ['frontal', 'parietal', 'temporal', 'occipital']
print(some_list)
another_list = [1, 2, 3, 4]
print(another_list)
mixed_list = ['frontal', 2.1, 0.112e-2, 2-2j,True,'a']
print(mixed_list)
~~~

> ## Output
> > ~~~
[]
['frontal', 'parietal', 'temporal', 'occipital']
[1, 2, 3, 4]
['frontal', 2.1, 0.00112, (2-2j), True, 'a']
> > ~~~
> > {: .output}
{: .solution}


# lists have length

~~~python
print(f"some_list length is \t:{len(some_list)}")
print(f"another_list length is \t:{len(another_list)}")
print(f"mixed_list length is \t:{len(mixed_list)}")  
~~~

> ## Output
> > ~~~
some_list length is 	:4
another_list length is 	:4
mixed_list length is 	:6
> > ~~~
> > {: .output}
{: .solution}


# lists have indices

- Indexing and slicing works just like in strings 

~~~python
print(f"Access start index using [0]\t\t\t= {mixed_list[0]} \n\
Access end index using  [-1] \t\t\t= {mixed_list[-1]} \n\
Use the colon [start:end] to perform slicing \t= {mixed_list[1:3]} \n\
Get everything UPTO [:end] \t\t\t= {mixed_list[:4]} \n\
Get everything FROM [start:] \t\t\t= {mixed_list[3:]} " )
print(f"Get everything [:]\t\t= {mixed_list[:]} \n\
Get every second element [::2] \t= {mixed_list[::2]} \n\
Get list in reverse [::-1]\t= {mixed_list[::-1]}" )
~~~

> ## Output
> > ~~~
Access start index using [0]			        = frontal 
Access end index using  [-1] 			        = a 
Use the colon [start:end] to perform slicing 	= [2.1, 0.00112] 
Get everything UPTO [:end] 			            = ['frontal', 2.1, 0.00112, (2-2j)] 
Get everything FROM [start:] 			        = [(2-2j), True, 'a'] 
Get everything [:]				                = ['frontal', 2.1, 0.00112, (2-2j), True, 'a'] 
Get every second element [::2] 			        = ['frontal', 0.00112, True] 
Get list in reverse [::-1]			             = ['a', True, (2-2j), 0.00112, 2.1, 'frontal']
> > ~~~
> > {: .output}
{: .solution}


# lists can be also be concatenated

- You can combine two lists together 
- And you can multiply lists using integers

~~~python
print(f"some_list + mixed_list length is \t:{len(some_list+mixed_list)}")
print(some_list*2)
~~~

> ## Output
> > ~~~
some_list + mixed_list length is 	:10
['frontal', 'parietal', 'temporal', 'occipital', 'frontal', 'parietal', 'temporal', 'occipital']
> > ~~~
> > {: .output}
{: .solution}

## Links to expand your understanding 

For those interested in learning more...

- [Conda Essentials](https://learn.datacamp.com/courses/conda-essentials)
- [Building and Distributing Packages with Conda](https://learn.datacamp.com/courses/building-and-distributing-packages-with-conda)
- [Some background on ipython and jupyter](https://www.datacamp.com/community/blog/ipython-jupyter)
- [Jupyter Notebook Tutorial: The Definitive Guide](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)


{% include links.md %}

