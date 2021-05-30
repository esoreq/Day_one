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


# %load_ext autoreload

- This magic command is crucial for almost any actual work using Jupyter (except tutorials :)
- The  autoreload extension tracks any changes in your imports and will *auto* *reload* them to your scope 
- By running this, we allow any imported file to be updated 

~~~python
%load_ext autoreload
%autoreload 2
~~~

#  %system

- If you want access to the shell, this magic command will do it.

~~~python
%system date +%s%N
~~~

- However, using exclamation mark (!) is even more useful  

~~~python
date = !date +%s%N
date
~~~


# %whos

- This magic command plots a list of variables in your environment. 
- Their type and some additional info
- You can pass %whos a type to examine only variables of that type

~~~python
%whos function
~~~

~~~
Variable                Type        Data/Info
---------------------------------------------
age2group               function    <function <lambda> at 0x7fb3b57af820>
age_group               function    <function age_group at 0x7fb3b57af1f0>
anything_but_children   function    <function <lambda> at 0x7fb3b57af0d0>
is_children             function    <function <lambda> at 0x7fb3b57af670>
temporal_hello          function    <function temporal_hello at 0x7fb3b57e99d0>
~~~
{: .output}

# %who_ls 

- This magic command shows you the list of variables in your environment. 
- It also can use the type to subset the output
- Using 'type' will retrive only class objects

~~~python
class test():
  pass
%who_ls type
~~~

# %time and %%time 

- When developing a pipeline, it is useful to know how much time a specific function needs 
- Time and its variations will do precisely that 
- To measure one line of code, we will use %time
- To measure a cell, we will use %%time


# Compare %time

~~~python
age2group = lambda age: 'children' if age<=11 else 'teens' if age<=21 else 'adults' if age<=65 else 'elderly'
is_children = lambda age: age<=11 
%time print(list(map(age2group,filter(is_children, list(range(5,80,1))))))
~~~

> ## Output
> > ~~~
['children', 'children', 'children', 'children', 'children', 'children', 'children']
CPU times: user 137 µs, sys: 62 µs, total: 199 µs
Wall time: 188 µs
> > ~~~
{: .solution}


# With %%time

~~~python
%%time 
age2group = lambda age: 'children' if age<=11 else 'teens' if age<=21 else 'adults' if age<=65 else 'elderly'
is_children = lambda age: age<=11 
print(list(map(age2group,filter(is_children, list(range(5,80,1))))))
~~~


> ## Output
> > ~~~
['children', 'children', 'children', 'children', 'children', 'children', 'children']
CPU times: user 456 µs, sys: 0 ns, total: 456 µs
Wall time: 423 µs
> > ~~~
{: .solution}




#  %timeit  and %%timeit 
- %timeit will measure multiple iterations of the same line and show some stats on them
- %%timeit will do the same for the cell


~~~python
%timeit even_squares = [x**2 for x in range(1,int(1e5)) if x**2%2==0]
~~~


> ## Output
> > ~~~
44.4 ms ± 259 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
> > ~~~
{: .solution}


~~~python
%%timeit
even_squares = [x**2 for x in range(1,int(1e5)) if x**2%2==0]
odd_squares = [x**2 for x in range(1,int(1e5)) if x**2%2!=0]
~~~


> ## Output
> > ~~~
88.5 ms ± 543 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
> > ~~~
{: .solution}

# %reset 

- Use %reset to Clear All Variables in IPython - Requires User Confirmation.
- Use %reset -f to Clear All Variables in IPython - No User Confirmation.

# %xdel

- Delete a variable, clearing only it from memory.

~~~python
even_squares = [x**2 for x in range(1,int(1e5)) if x**2%2==0]
%whos list
%xdel even_squares
%whos list
~~~

> ## Output
> > ~~~
Variable       Type    Data/Info
/--------------------------------
even_squares   list    n=49999
No variables match your requested type.
> > ~~~
{: .solution}


# %%svg is cool 

- render a cell using some external programing languaege in thsi case scalable vector graphics 
- Try this :)

~~~python
%%svg
<svg width="800" height="200">
  <g transform="translate(100,100)"> 
    <text id="TextElement" x="0" y="0" style="font-family:Verdana;font-size:24; visibility:hidden"> It's MAGIC!
      <set attributeName="visibility" attributeType="CSS" to="visible" begin="1s" dur="6s" fill="freeze" repeatCount="indefinite" />
      <animateMotion path="M 0 0 L 100 100" begin="1s" dur="3s" fill="freeze" repeatCount="indefinite" />
      <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="-30" to="0" begin="1s" dur="3s" fill="freeze" repeatCount="indefinite"/> 
      <animateTransform attributeName="transform" attributeType="XML" type="scale" from="1" to="3" additive="sum" begin="1s" dur="3s" fill="freeze" repeatCount="indefinite" /> 
    </text> 
  </g> 
  Sorry, your browser does not support inline SVG.
</svg>
~~~

# But %%bash is useful 



## Links to expand your understanding 

For those interested in learning more...

- [FIXME](https://learn.datacamp.com/courses/conda-essentials)

{% include links.md %}