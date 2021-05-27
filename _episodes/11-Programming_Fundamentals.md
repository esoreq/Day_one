---
title: "Programming Fundamentals using Python"
author: "Dr. Eyal Soreq" 
date: "05/03/2021"
teaching: 35
exercises: 25
questions:
- What are conditional statements? 
- What are iterators? 
- What are functions?  
objectives:
- Learn about different conditional statements
- Learn how to use conditional statements 
keypoints:
- FIXME
---

# Programming Fundamentals

- In this section, we will go over the things that make programming so powerful 
- We will cover conditional statements 
- For and while loops 
- And finally, cover function creation 

# Conditional statements

- A conditional statement allows us to take alternate actions based on a set of rules
- In these types of procedures, `if`, `elif`, and `else` keywords are used.
- Using the "if" and "else" combinations, we can construct a rule tree to perform some logic  

~~~python
if rule1:
    do something
elif rule2:
    do something different
else: # if both are False
    do another thing
~~~

# Simple examples


~~~python
rule1 = True
if rule1:
    print(f'Rule 1 is {rule1}')
~~~

~~~
Rule 1 is True
~~~
{: .output}


> If we change rule1 to *False*, what will happen?
{: .Discussion}


# Add complexity

- Guess what the output will be without running the code

~~~python
rule1 = False    
if not rule1:
    print(f'Only if Rule 1 is {rule1} go here')
else: 
    print(f"Not Rule 1 is {not rule1}")
~~~

> ## Output
> > ~~~
Only if Rule 1 is False go here
> > ~~~
> > {: .output}
{: .solution}


# Multiple `if`, 'elif` and `else` Branches example 

- Guess what the output will be without running the code
- Then try to change values to reach a specific branch

~~~python
rule1,rule2 = True,False    
if not rule1:
    print(f'Only if Rule 1 is {rule1} go here')
elif not rule2:
    print(f'Only if Rule 2 is {rule2} go here')
else: 
    print(f"Not Rule 1 is {not rule1}")
~~~


> ## Output
> > ~~~
Only if Rule 2 is False go here
> > ~~~
> > {: .output}
{: .solution}

# Nested `if` example (see Indentation!!!)

~~~python
var1, var2 = 'CRTX','FPN'

if var1 == 'CRTX':
    if var2 == 'FPN':
        print(f'The {var2} is part of the {var1}')
~~~


> ## Output
> > ~~~
The FPN is part of the CRTX
> > ~~~
> > {: .output}
{: .solution}

# Ternary operator

- Python supports ternary in a nice way 
- If you don't know what ternary operator and want to dive in the rabbit hole check this [link](https://en.wikipedia.org/wiki/%3F:) 
- However, for our cases, it is just a short form of conditional statements that takes the following form: 

```python
one_line = 'yes' if expression==True else 'no'
```

## Here's an example

~~~python
age = 13
teen = True if age>=13 and age <=18 else False
print(f'It is {teen} that you are a teen if you are {age}')
~~~
~~~
It is True that you are a teen if you are 13
~~~
{: .output}

# Conditional Loop `While`

- The idea of conditional statements can be combined to generate a statement that will repeat until the condition is no longer valid.
- The general syntax of a while loop is:

~~~python
while condition:
    do something until condition is False
else:
    do once and exit the loop
~~~

# Simple example

~~~python
counter = 20
while counter>0:
    print(f'{counter}',end='|')
    counter -= 1
~~~

> ## Output
> > ~~~
20|19|18|17|16|15|14|13|12|11|10|9|8|7|6|5|4|3|2|1|
> > ~~~
> > {: .output}
{: .solution}

# While loops are dangerous 
- Think what will happen if you type + instead of minus in the previous example
- Unless we add another condition, this loop will run forever 
- When this happens by mistake, you can always use the stop icon at the top to break the loop

## Lets add conditional fences

~~~python
counter = 30
while counter>0 and counter<50:
    print(f'{counter}',end='|')
    counter += 1
else: 
    print(f'\n\nDone !!')
~~~

> ## Output
> > ~~~
30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|

Done !!
> > ~~~
> ## Try to guess the output here by comparing the following code
> > ~~~
counter = 30
while counter>0 and counter<=50:
    print(f'{counter}',end='|')
    counter += 1
else: 
    print(f'\n\nDone !!')
> > ~~~
{: .solution}


# We can adjust the loop using `break`, `continue` and `pass`

- We can use control statements in our loops to add additional functionality for various cases
- `Break` Breaks out of the current loop.
- `Continue` Goes to the top of current loop
- `Pass` do nothing 

# Why do we need a control statement that does nothing? 

- Python requires that code blocks (after if, except, def, class etc.) will not be empty.
- Empty code blocks are however useful in a variety of different contexts
- One main one is during implementation as a place holder to remember to deal with
- let's test this with some silly function 


~~~python
for num in range(-5,5):
    if not num % 2: # Even numbers are divided by 3
        print(f'f({num})={num/3:.3f}')
    elif num<0: # odd negative numberes become positive
        print(f'f({num})={(num**2)**.5:.3f}')
    else: # Otherwise, I still haven't decided 
        pass
~~~

> ## Output
> > ~~~
30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|

Done !!
> > ~~~
> ## Try to guess the output here by comparing the following code
> > ~~~
counter = 30
while counter>0 and counter<=50:
    print(f'{counter}',end='|')
    counter += 1
else: 
    print(f'\n\nDone !!')
> > ~~~
{: .solution}



> ## What do we need to change to include 50 in the sequence?
> > ~~~
30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|

Done !!
> > ~~~
> > {: .output}
{: .solution}



# Simple example


- A `for` loop goes through items that are in any iterable objects
- Iterable objects include strings, lists, tuples, dictionaries, and sets
- These are called object Iterators
- An iterator is an object that allows us to go over a sequence one element at a time 


~~~python
for item in Iterator:
    do_something 
~~~

# Simple examples

- lets loop over a sequence 
- Let's create a list of strings with zero-padded numbers as a suffix
- For example week_01,...,week_10
- Think on how you would do this using a for loop and a list 

~~~python
rule1 = True
if rule1:
    print(f'Rule 1 is {rule1}')
~~~

~~~
Rule 1 is True
~~~
{: .output}


> If we change rule1 to *False*, what will happen?
{: .Discussion}



> ## Output
> > ~~~
Only if Rule 1 is False go here
> > ~~~
> > {: .output}
{: .solution}


> ## Output
> > ~~~
some_list + mixed_list length is 	:10
['frontal', 'parietal', 'temporal', 'occipital', 'frontal', 'parietal', 'temporal', 'occipital']
> > ~~~
> > {: .output}
{: .solution}



~~~python
rule1 = False    
if not rule1:
    print(f'Only if Rule 1 is {rule1} go here')
else: 
    print(f"Not Rule 1 is {not rule1}")
~~~

~~~
Rule 1 is True
Only if Rule 1 is False go here
~~~
{: .output}


> ## Output
> > ~~~
some_list + mixed_list length is 	:10
['frontal', 'parietal', 'temporal', 'occipital', 'frontal', 'parietal', 'temporal', 'occipital']
> > ~~~
> > {: .output}
{: .solution}


## Links to expand your understanding 

For those interested in learning more...

- [FIXME](https://learn.datacamp.com/courses/conda-essentials)

{% include links.md %}