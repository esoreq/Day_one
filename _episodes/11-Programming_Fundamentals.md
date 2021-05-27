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


# Multiple *if*, *elif* and *else* Branches example 

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

# Nested *if* example (see Indentation!!!)

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

# Conditional Loop *While*

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


# We can adjust the loop internaly using *break*, *continue* and *pass* 

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
num = -5
while num<5:
    if not num % 2: # Even numbers are divided by 3
            print(f'f({num})={num/3:.3f}')
    elif num<0: # odd negative numberes become positive
        print(f'f({num})={(num**2)**.5:.3f}')
    else: # Otherwise, I still haven't decided 
        pass
    num+=1
~~~


> ## Output
> > ~~~
f(-5)=5.000
f(-4)=-1.333
f(-3)=3.000
f(-2)=-0.667
f(-1)=1.000
f(0)=0.000
f(2)=0.667
f(4)=1.333
> > ~~~
{: .solution}


# Challenge: Add a rule that is applied on the odd positive numbers

> ## Here is one possible solution
> > ~~~python
num = -5
while num<5:
    if not num % 2: # Even numbers are divided by 3
            print(f'f({num})={num/3:.3f}')
    elif num<0: # odd negative numberes become positive
        print(f'f({num})={(num**2)**.5:.3f}')
    else: # Otherwise, I still haven't decided 
        print(f'f({num})={(-num**3)**.5:.3f}')
    num+=1
> > ~~~
> ## Output
> > ~~~
f(-5)=5.000
f(-4)=-1.333
f(-3)=3.000
f(-2)=-0.667
f(-1)=1.000
f(0)=0.000
f(1)=0.000+1.000j
f(2)=0.667
f(3)=0.000+5.196j
f(4)=1.333
> > ~~~
> > {: .output}
{: .solution}



# Challenge: Change the code to go from -50 to 50 with steps of 10

> ## Here is one possible solution
> > ~~~python
start_condition = -50
stop_condition = 50
num = start_condition
steps = 10
while num<stop_condition:
    if not num % 20: # Even numbers are divided by 3
            print(f'f({num})={num/3:.3f}')
    elif num<0: # odd negative numberes become positive
        print(f'f({num})={(num**2)**.5:.3f}')
    else: # Otherwise, I still haven't decided 
        print(f'f({num})={(-num**3)**.5:.3f}')
    num+=steps
> > ~~~
> ## Output
> > ~~~
f(-50)=50.000
f(-40)=-13.333
f(-30)=30.000
f(-20)=-6.667
f(-10)=10.000
f(0)=0.000
f(10)=0.000+31.623j
f(20)=6.667
f(30)=0.000+164.317j
f(40)=13.333
> > ~~~
> > {: .output}
{: .solution}

# Use `range()` function to **generate** a sequence 

- The `range()` generates a sequence of numbers and is immutable 
- It takes one to three input arguments (i.e. start, stop and step)
- The stop is not included 
- [Click on this link to learn more about the range object](https://treyhunner.com/2018/02/python-range-is-not-an-iterator/#:~:text=Unlike%20zip%20%2C%20enumerate%20%2C%20or%20generator,range%20objects%20are%20not%20iterators.)


> ## Same concept using range
> > ~~~python
start = -50
stop = 50
step = 10
seq = range(start,stop,step)
index = 0
while index<len(seq):
    print(seq[index],end=',')
    index+=1
> > ~~~
> ## Output
> > ~~~
-50,-40,-30,-20,-10,0,10,20,30,40,
> > ~~~
> > {: .output}
{: .solution}

- But if we know in advance the sequence perhaps we don't need the condition

# Enter `for` loops using iterable objects

- A `for` loop goes through items that are in any iterable objects
- Iter-**able** objects are any data type that can be iterated over
- These include strings, lists, tuples, dictionaries, sets and range


## Same concept using `for` and `range`
~~~python
start,stop,step = -50,50,10
for number in range(start,stop,step):
    print(number,end=',')
~~~

~~~
-50,-40,-30,-20,-10,0,10,20,30,40,
~~~
{: .output}

# Iterators doing the hard work 

- Iter-**ators** are the agents that perform the iteration.
- An iterator is an object that allows us to go over a sequence one element at a time using the `iter()` function and the `next` method
- There is a big difference between an iterable object and an iterator derived from that object: the former has no memory, while the latter is like a stack -- with each use of it you have one fewer element in the stack

# An example will help illustrate this
~~~python
for i in range(3):
    for number in range(-50,50,10):
        print(number,end=',')
    print(f'- #{i} iter')   
~~~

~~~
-50,-40,-30,-20,-10,0,10,20,30,40,- #0 iter
-50,-40,-30,-20,-10,0,10,20,30,40,- #1 iter
-50,-40,-30,-20,-10,0,10,20,30,40,- #2 iter5
~~~
{: .output}

~~~python
range_iterator = iter(range(-50,50,10))
for i in range(3):
    for number in range_iterator:
        print(number,end=',')
    print(f'- #{i} iter')  
~~~

~~~
-50,-40,-30,-20,-10,0,10,20,30,40,- #0 iter
- #1 iter
- #2 iter
~~~
{: .output}

~~~python
for i in range(3):
    range_iterator = iter(range(-50,50,10))
    for number in range_iterator:
        print(number,end=',')
    print(f'- #{i} iter')  
~~~

~~~
-50,-40,-30,-20,-10,0,10,20,30,40,- #0 iter
-50,-40,-30,-20,-10,0,10,20,30,40,- #1 iter
-50,-40,-30,-20,-10,0,10,20,30,40,- #2 iter
~~~
{: .output}

- This weird beahviour is useful for many things outside the scope of this course but is important to know

# Exercises

1. Using steps of 46 and starting from 25 and ending with 163, print out a sequence of strings that begins with the word Week (e.g. Week_025, … , Week_163) and is on the same line seperated by commas.

## Expected output
> ~~~
> Week_25, Week_71, Week_117, Week_163, 
> ~~~
{: .output}

> ## Here is one possible solution
> > ~~~python
for num in range(25,190,46):
    print(f'Week_{num}')
> > ~~~
{: .solution}

1. For all the numbers from 1 to 15 (including) print if they are odd,(O) or even (E).

> ## Expected output
> ~~~
> 1-O,2-E,3-O,4-E,5-O,6-E,7-O,8-E,9-O,10-E,11-O,12-E,13-O,14-E,15-O,
> ~~~
{: .output}

> ## Here is one possible solution
> > ~~~python
for num in range(1,16):
    print(f'{num}-{["E","O"][num%2]}',end=',')
> > ~~~
{: .solution}


1. Using the following sequence `abcdEFGHjd0rG` print out if a single charecter is UPPER or lower case
  
> ## Expected output
> ~~~
> a-l,b-l,c-l,d-l,E-U,F-U,G-U,H-U,j-l,d-l,0-l,r-l,G-U,
> ~~~
{: .output}

1. Using one of the following quotes
    - "People say nothing is impossible, but I do nothing every day."
    - "The best thing about the future is that it comes one day at a time."
    - "The difference between stupidity and genius is that genius has its limits."
- And using a for loop print **on the same line** the same sentence with underscores instead of spaces    

> ## Here is one possible solution
> > ~~~python
for char in "The best thing about the future is that it comes one day at a time.":
    print(f'{[char,"_"][char.isspace()]}',end='')
> > ~~~
{: .solution}

## Links to expand your understanding 

For those interested in learning more...

- [FIXME](https://learn.datacamp.com/courses/conda-essentials)

{% include links.md %}