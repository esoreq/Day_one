---
title: "Jupyter lab revisited"
author: "Dr. Eyal Soreq" 
date: "05/03/2021"
teaching: 15
exercises: 0
questions:
- What are Magic commands
objectives:
- FIXME
keypoints:
- FIXME
---


# Jupyter notebook 

- Magic commands are enhancements created for the interactive Python project which Jupyter notebook can be viewed as a web evolution of 
- Magic commands are designed to simplify everyday tasks in the standard data analysis workflow
- There are two types of Magic commands: 
    - Line magics, which are denoted by a single % prefix and operate on a single line of input, 
    - Cell magics, which are denoted by a double %% prefix and operate on multiple lines of input. 
- For a full list of magic commands press [here](https://ipython.readthedocs.io/en/stable/interactive/magics.html#line-magics)
- Here we will explore some of the most useful ones 


# %time and %%time 

- When developing a pipeline, it is useful to know how much time a specific function needs 
- Time and its variations will do precisely that 
- To measure one line of code, we will use %time
- To measure a cell, we will use %%time


~~~python
age2group = lambda age: 'children' if age<=11 else 'teens' if age<=21 else 'adults' if age<=65 else 'elderly'
is_children = lambda age: age<=11 
%time print(list(map(age2group,filter(is_children, list(range(5,80,1))))))
~~~

~~~
['children', 'children', 'children', 'children', 'children', 'children', 'children']
CPU times: user 137 µs, sys: 62 µs, total: 199 µs
Wall time: 188 µs
~~~
{: .output}



~~~python
age2group = lambda age: 'children' if age<=11 else 'teens' if age<=21 else 'adults' if age<=65 else 'elderly'
anything_but_children = lambda age: age>11 
%time print(list(map(age2group,filter(anything_but_children, list(range(5,80,1))))))
~~~

~~~
['teens', 'teens', 'teens', 'teens', 'teens', 'teens', 'teens', 'teens', 'teens', 'teens', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'adults', 'elderly', 'elderly', 'elderly', 'elderly', 'elderly', 'elderly', 'elderly', 'elderly', 'elderly', 'elderly', 'elderly', 'elderly', 'elderly', 'elderly']
CPU times: user 65 µs, sys: 0 ns, total: 65 µs
Wall time: 67.2 µs
~~~
{: .output}


## Links to expand your understanding 

For those interested in learning more...

- [FIXME](https://learn.datacamp.com/courses/conda-essentials)

{% include links.md %}