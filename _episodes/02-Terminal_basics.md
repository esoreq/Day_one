---
title: "Introduction to Linux Terminal"
author: "Dr. Eyal Soreq" 
date: "05/03/2021"
teaching: 15
exercises: 10
questions:
- What is a terminal and why would I use one?
- What is the prompt and what does it indicate? 
objectives:
- Explain how the shell relates to the input and output of your computer.
- Explain the advantages of command-line interfaces over graphical interfaces.
- Explain the read, run, and print steps of the shell.
keypoints:
- 
---

## Introduction to Linux Based Command Line Interactions


### graphical user interface (GUI)
In its most basic form, all computers have some interaction to store, process, create data, take actions, and communicate with other computers and people. There are many ways to interact with the computer. However, the most common is the GUI, which uses a mix of windows, icons, mice, and pointers to perform various tasks. This user interface type makes it easier for users to navigate through and use hundreds of programs, often from within a complex hierarchy.

### Command-line interface (CLI)
Under the hood, all computers have a command-line interface, or CLI, which distinguishes it from the more common GUI, which most people use nowadays. A CLI's heart is a read-evaluate-print loop or REPL: when a user types a command and presses enter, the computer reads the command, runs the command, and prints out the output until the user breaks the cycle. One of the programs in charge of enabling this cycle is called a shell or terminal.

### The Shell/Terminal
The shell is a program that serves as a keyboard-driven interface between the user and the operating system. It includes a command-line interpreter that accepts the user input via the keyboard, evaluates it, starts programs if necessary, and returns the output in the form of text output to the user. Each shell has its programming language, which makes it possible to write scripts that automate complex tasks. Each shell runs in a terminal.

### What would I use a Terminal 
In the old days, independent devices, or so-called "hard copy terminals" (printer or screen plus keyboard), were used. Modern computers no longer have those, replaced by terminal emulators - programs that provide users with a graphical window for interacting with the shell. As scientists, we rely on powerful computer clusters to reveal answers to many questions and to perform many tasks automatically. There are many ways to interact with these computers. However, the fastest and most reliable way is the terminal emulators, which is the main focus of today's presentation.

### What is Bash
One of the differences between a shell and any other program is that a shell runs other programs instead of doing calculations. One of the most popular Unix shells is Bash, named for the Bourne Again Shell (derived from a shell originally written by Stephen Bourne). In most modern Unix implementations and most packages designed for Windows, Bash is the default shell.

> ## Opening a terminal using jupyter cloud
> Go to [jupyter-cloud](https://jupyter-cloud.gwdg.de/), login and press on the Terminal button in the Laucher
{: .challenge}

## Command prompt
Opening the terminal for the very first time, you will be presented with a prompt `$`, indicating that the shell is waiting for your input.

~~~
$
~~~
{: .language-bash}

## Shell Commands structure
You interact with the shell via commands, which can be used to execute CLI programs with names similar to the commands. For each action that you wish to perform using the terminal, you use a program call following this basic scheme:
~~~
Command [options] [arguments] 
~~~
{: .language-bash}

> ## Use `ls` to list your home contents 
> The first command we will use is `ls`, which displays the current directory contents. <br>
> Type `ls` and press the <kbd>Enter</kbd> key to execute it.
{: .challenge}

## Managing Content in the Filesystem
The part of the operating system responsible for managing individual files and directories is called the **file system**. This part of the system categorizes our data into files or directories that contain files. 

> ## Creating a sandbox directory using mkdir
> Make a new directory by typing `mkdir sandbox` in the prompt. <br>
> Here, mkdir is the program name and sandbox is the argument, in this case the name of the directory we are creating.
{: .challenge}

> ## mkdir assumes no such directory exists 
> Type `mkdir sandbox` again. Now the program returns an error.
> ~~~
> $ mkdir sandbox
> mkdir: cannot create directory ‘sandbox’: File exists
> ~~~
> {: .error}
{: .challenge}

> ## mkdir options allow us to issue more advanced commands
> By adding the --help option, prints out the different options you can use to extend the mkdir program.
> Please type the following command in the prompt. <br>
> ~~~
> $ mkdir --help
> ~~~
> {: .language-bash}
{: .challenge}


> ## mkdir options allow us to issue more advanced commands
> Please type the following command in the prompt. <br>
> By adding the -p (aka parent) option, the mkdir program will create a hierarchical folder structure.
> ~~~
> $ mkdir -p sandbox/root/{dir_a,dir_b/{leaf_1,leaf_2},dir_c}
> ~~~
> {: .language-bash}
{: .challenge}


> ## `ls` has options too 
> If we want to confirm our makdir command worked we should use one of the many options the `ls` command has. <br>
> Try to use the `ls --help` to figure out what option is the most sutiable. 
> > ## Solution 
> > ~~~
> > $ ls sandbox/ -R
> > ~~~
> > {: .language-bash}
> > Here we state the root folder to start listing from, and then use the recursive  option to list all subdirectories recursively to the prompt
> > ## Output
> > ~~~
> > sandbox/:
> > root
> > sandbox/root:
> > dir_a  dir_b  dir_c
> > sandbox/root/dir_a:
> > sandbox/root/dir_b:
> > leaf_1  leaf_2
> > sandbox/root/dir_b/leaf_1:
> > sandbox/root/dir_b/leaf_2:
> > sandbox/root/dir_c:
> > ~~~
> > {: .language-bash}
> > 
> {: .solution}
{: .challeng



> ## Removing a folder 
> Please type the following command in the prompt. <br>
> By adding the -p (aka parent) option, the mkdir program will create a hierarchical folder structure.
> ~~~
> $ mkdir -p sandbox/root/{dir_a,dir_b/{leaf_1,leaf_2},dir_c}
> ~~~
> {: .language-bash}
{: .challeng

## Navigating the filesystem
To move to and view directories, files, and content, we need some basic navigation skills. The following commands and excersises will provide you with these capabilities: 


{% include links.md %}

