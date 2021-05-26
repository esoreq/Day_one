---
title: "Python Overview"
author: "Dr. Eyal Soreq" 
date: "05/03/2021"
teaching: 45
exercises: 0
questions:
- How can I store data in programs?
- What types of Data can I store? 
- What is a variable in Python?
objectives:
- Best Practices in Python
- Syntax Essentials in Python
- Understand the different expression types and how to use them
keypoints:
- Conda is a Python-based environment and package manager
- It enables us to use 
---

# Syntax Essentials and Best Practices

One of the trickiest things for Python newcomers to adapt to is the syntax. 
In this opening section, I'll go over some syntax essentials as well as some formatting best practices.
This will help you keep your code consistent and hopefully elegant.


# Syntax Essentials rules

1. Code blocks are defined by indentation (can be either space or tab)
1. One statement per line
1. Python is case sensitive : $vara \neq  varA$
1. Path specification uses forward slashes (regardless of OS): `~/user/home`
1. There is no need to add a command terminator ;
1. You can combine two executable statements using a semicolon `;` 
1. String literals can be defined using single <'> double <"> or even triple <'''> quotes 
1. It is considered good conduct to keep lines of code short 
    1. Backslash \\ can be used to stack lines of code together
    1. Expressions enclosed in brackets i.e. (), [] or {} don't need a backslash
1. Comments in Python begin with a hash mark (#) and whitespace character and continue to the end of the line. 
1. Keywords are protected and should not be used as variables

# One statement per line

- If you put a line break in the wrong place, you will get an error message. 
- To avoid that you should have one statement per line.
- However, as with all rules, there are some exceptions some of which we will cover later in the course


# Explicit line joining 
- Using a backslash `\`, we can break long commands across many lines 

~~~python
print \
('Multi\
 line\
 command')
~~~

~~~
Multi line command
~~~
{: .output}


# Many commands in single line 
- Using the semicolon \; we can achieve the opposite, i.e. to combine multiple commands in one line


~~~python
print('Multi');print('Line');print('Output')
~~~

~~~
Multi
Line
Output
~~~
{: .output}


#  Implicit line joining

- Expressions in parentheses, square brackets or curly braces can be split over more than one physical line without using backslashes. 


~~~python
brain_lobes = ['Frontal',
               'Parietal',
               'Occipital',
               'Temporal']
print(brain_lobes)
~~~

    
~~~
['Frontal', 'Parietal', 'Occipital', 'Temporal']
~~~
{: .output}

# Indentations for structure
- In contrast to the closing statements common in Bash (such as fi) or MATLAB (such as end) Python uses indentations to understand the structure of your code.
- So you should make sure to use indentations correctly and consistently.
- This makes code more straightforward to read and ultimately understand 
- Indentations can be created using either tabs or spaces (usually four spaces) 

# Python Keywords

- Keywords are the reserved words in Python.
- You cannot use a keyword as a variable name, function name or any other identifier. 
- These 35 keywords are used to define the syntax and structure of the Python language.
- To examine them you can run the following code 


~~~python
import keyword
print(keyword.kwlist)
~~~

~~~
['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
~~~
{: .output}


# Case matters

- Python is case sensitive. 
- This means the **HELP** is not equal to **help**
- Most of Python keywords are written with lowercase letters.
- Notable exceptions to this rule are True and False boolean values and the **None** variable that all use a mixture of cases. 


# Use Comments

>"Code is more often read than written."
>> Guido van Rossum

- It is important to add comments to your Python code. 
- To do this use the # character, everything that comes after it won't be executed.

~~~python
print("This will run.")  # This won't run
~~~

>"Code tells you how; Comments tell you why."
>>Jeff Atwood (aka Coding Horror)

- Commenting your code serves multiple purposes, for example:
    - Planning and Reviewing, i.e. outline of the desired functionality of some future code
    - Code Description, i.e. explain the intent of specific sections of code
    - Algorithmic description, i.e. explain how the algorithm works or how it's implemented within your code or even add a link to the source
    - Tagging, i.e. label specific sections of your code where you need to take action, e.g. BUG, FIXME, TODO, UPGRADE.
- Comments should be short, sweet, and to the point.


# Use Multiline Comments 

- Multi-line comments can be either achieved primitively using multiple stacking of hashtags #

~~~python
# This 
# is 
# stacking 
~~~

- Alternatively, docstrings can be used by combining three commas 

~~~python
def some_function(arg1): 
    '''Summary line. some_function will do some thing with arg1
  
    Extended description of the function. 
    some_function will get arg1 and do things with it
    some of the things are complex, and some are simple
    some_function will then return some value.  
    
    Parameters: 
    arg1: the thing we will do stuff to 
  
    Returns: 
    Some value back 
    '"
~~~


# Variable Names

> *Code is read much more often than it is written*
>> Guido van Rossum 2001


1. Variable names should use lowercase.
1. Variable names cannot:
    - Start with a number
    - Contain spaces
    - Use any of these symbols :'",<>/?|\()!@#$%^&*~-+
1. Avoid using the characters:
    - 'l' (lowercase letter el)
    - 'O' (uppercase letter oh)
    - 'I' (uppercase letter eye)
1. short_names vs long_name
1. long_names should be separated using the underscore symbol (_)
1. This makes them readable 
1. Remember that making variable names simple to understand minimizes the time spent on commenting 


# Use blank lines

- Using blank lines is the simplest way to separate code blocks visually 
- Even multiple blank lines to distinguish between different parts of the code. 
- It won't affect the result of your script.


# Use white spaces 

- Python allows white spaces in assignment 
- This makes nicer looking code 


<!-- # Imports best practice 

We will go over this in detail next week, but it should be stated. 

- Use **import x** for importing packages and modules.
- Use **from x import y** where x is the package prefix, and y is the module name with no prefix.
- Use **from x import y as z** if two modules named y are to be imported or if y is an inconveniently long name.
- Use **import y as z** only when z is a standard abbreviation (e.g., np for NumPy). -->

# Variables and Basic Data Types

A variable implies change and is a way of referring to some space in your computer memory that is allocated by your computer program to store a specific type of information. In other words, it is a symbolic name for a physical address in memory that contains static or dynamic values. Python supports many different Data Types, and in contrast to other programming languages where you need to specify the data type of a variable, python will automatically find out the data type at the process of allocation.


- None (aka null object or variable) 
- Boolean Type (True or False)
- Strings (strings in Python are arrays of bytes representing unicode characters)
- Integers $$\pm\mathbb{Z}$$
- Floats $$\pm\mathbb{R}$$
- Complex $$\pm\mathbb{C}$$

~~~python
my_none_variable = None 
my_bool_variable = True 
my_string_variable = 'STRING'
my_int_variable = 1
my_float_variable = 1.1
my_complex_variable = 1.1+1j
~~~

# Variables Naming Styles

1. lowercase/UPPERCASE
    - single letter - b/B
    - single name - var/VAR
    - lower_with_underscores/UPPER_WITH_UNDERSCORES
1. mixed cases
    - CamelCase - capitalize all the starting letters
    - mixedCase - initial lowercase character



# Introspective functions 

- Introspection is the ability to interrogate objects at runtime.
- Everything in python is an object. 
- Every object in Python may have attributes and methods. 
- By using introspection, we can dynamically examine python objects. 

~~~python
type(None) # This function returns the type of an object.
dir(None) # This function return list of methods and attributes associated with that object.
id(None) # This function returns a special id of an object representing a specific location in memory.
help(None) # This function is used to find what other functions do
print(None) # prints the specified message to the screen, or other standard output devices.
~~~


# Variables are just skins to a place in memory

- The id of a variable returns a unique integer representing the identity of an object
- This is also the address of the object in memory
- When you change the variable, you are creating a new object 


~~~python
some_var = None
print(id(some_var))
some_var = 'some different data'
print(id(some_var))
~~~

~~~
4305322280
4397182208
~~~
{: .output}


# Basic Data Types

We can use type to examine the different classes these variables are instances of 

~~~python
print(f'{type(my_none_variable)}')
print(f'{type(my_bool_variable)}')
print(f'{type(my_string_variable)}')
print(f'{type(my_int_variable)}')
print(f'{type(my_float_variable)}')
print(f'{type(my_complex_variable)}')
~~~

~~~
<class 'NoneType'>
<class 'bool'>
<class 'str'>
<class 'int'>
<class 'float'>
<class 'complex'>
~~~
{: .output}

# Immutable vs Mutable Objects

- In Python, there are two types of objects:
    - Immutable objects can't be changed
    - Mutable objects can be changed
    
- All the basic data types are immutable!!!  


# Basic Arithmetic operations on integers (whole numbers)

- As we already saw, Python has various "types" of numbers (numeric literals).
- It also has many different operators. 
- Arithmetic Operators perform various arithmetic calculations on these.
- Run these following examples in your own notebook:

~~~python
x,y = 5,4
print(f"+ Addition :\t{x}+{y}={x+y}") 
print(f"- Substraction :\t {x}-{y}={x-y}")
print(f"* Multiplication :\t {x}*{y}={x*y}")
print(f"/ Division :\t {x}/{y}={x/y}")
print(f"% Modulus :\t {x}%{y}={x%y}")
print(f"** Exponent :\t {x}^{y}={x**y}")
print(f"// Floor Division :\t {x}/{y}={x//y}")
print(f"() Use parentheses to specify order:\t {x}*({x}/{y}-{y})={x*(x/y-y)}")
~~~

# Basic Arithmetic operations on floats 

- Floating point numbers have a decimal point and/or use an exponential (e) to define the number.

~~~python
x,y,z = 5e-3,2e2,0.56e4
print(f"X={x},Y={y},Z={z}")
~~~


> ~~~
<class 'NoneType'>
<class 'bool'>
<class 'str'>
<class 'int'>
<class 'float'>
<class 'complex'>
~~~
{: .solution}

# Basic Arithmetic operations on complex numbers

- Python complex numbers are of type complex.
- Every complex number contains one real part and one imaginary part.

~~~python
x,y = 1+1j, 2-2j
print(f"Real Parts (x={x.real},y={y.real}) | Imaginary Parts = (x={x.imag},y={y.imag})") 
~~~

# Basic Arithmetic operations on strings (SAY WHAT!?)

- In general, you cannot perform mathematical operations on strings, even if the strings look like numbers.
- However, some arithmetic operators do work on strings and open up some nice options. Let's explore this. 

~~~python
x,y = 'cere','bellum'
print(f"+ Addition :                             {x}+{y}={x+y}") 
print(f"* Multiplication :                       {x}*{3}={x*3}") 
print(f"Combinations :                           {x}*{3}+{y}={x*3+y}")
print(f"() Use parentheses to specify order:     {x}+(({x}+{y})*{2})={x+((x+y)*2)}")
~~~


# Basic Numeric Comparison Operators

- Comparison operators are used to comparing two values

~~~python
x,y = 5,4
print(f" isEqual(==)          {x}=={y} is {x==y}")
print(f" notEqual(!=)         {x}!={y} is {x!=y}")
print(f" isGreater(>)         {x}>{y}  is {x>y}")
print(f" isSmaller(<)         {x}<{y}  is {x<y}")
print(f" isGreaterOrEqual(>=) {x}>={y} is {x>=y}")
print(f" isSmallerOrEqual(<=) {x}<={y} is {x<=y}")
~~~




# Basic Numeric Assignment Operators

[realpython - Basic Data Types in Python](https://realpython.com/python-data-types/)

{% include links.md %}

