---
title: "Introduction to Bash Jupyter notebook"
author: "Dr. Eyal Soreq" 
date: "05/03/2021"
teaching: 30
exercises: 10
questions:
- What is Jupyter Notebook?
- What is a Kernel? 
- How to use a notebook to run bash code snippets.
objectives:
- List the components and functionality of Jupyter Notebook.
- Launch and navigate the Jupyter Notebook dashboard.
- Open, create and save Jupyter Notebook files.
- Create, run and delete Markdown and Code cells.  

keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


## What is Jupyter notebook?

- Jupyter notebook is a modern web-based open-source tool specifically designed to support data science projects. 
- You can use Jupyter Notebooks to document workflows and to share code for data processing, analysis and visualization.

## Jupyter Notebook Overview

This course makes extensive use of notebooks, which allow you to document code and thoughts while being able to follow your work. This section explains how to use Jupyter Notebook to facilitate open reproducible science workflows and introduces you to the interface that you will use for running and editing code and Markdown cells. 


## Jupyter Notebook for Open Reproducible Science

- The Jupyter Notebook file format (.ipynb ) is constructed from linearly stacked cells allowing you to construct a single project-oriented file that combines descriptive text, code blocks and code output in a single file. 
- The code cells have output cells associated with them, allowing you to include plots, tables, and textual outputs to communicate your findings, within the notebook file. 
- You can then export the notebook to a .pdf or .html that can then be shared with anyone.


## Key benefits of the Jupyter Notebook
1. **Human readable**: Using Jupyter, you can bridge ideas and concepts with methodology and results, creating a notebook that can be understood by different types of researchers. By adding Markdown text around your code, your project becomes more user-friendly and easier to understand. 
1. **Simple syntax:**  Markdown is simple to learn and use reducing the learning curve needed to produce well-documented Jupyter reports.
1. **Documenting your ideas:** Research is all about creating logical steps based on assumptions followed by tests. However, actual Research is messy, and many things will be left on your digital workbench. Forming the habit of documenting your workflow, making inline references when needed, and explaining the logical workflow, will be priceless to the future you. Just imagine changing or adapting specific parts of a study two years after it was created, without some comments.
1. **Easy to Modify:** analyses contained within a Jupyter Notebook can be easily extended, improved or refined by adding or editing the workflow and rerunning the notebook.
1. **Simple to share:** Sharing your workflow with colleague or supervisor is in the core of the Jupyter DNA. A notebook can be shared using file-sharing services like Dropbox or Google Drive or more sophisticated approaches such as Github. This simplifies the process of validating, replicating, extending, refining and communicating your workflow.
1. **Flexible export formats:** Notebooks can be exported into various formats including HTML, PDF and slideshows.

## Jupyter Notebook types

There are four kinds of notebooks you would want to create:  

- **The lab** - a historical (and dated) record of your analysis
- **The report** - a polished version of some study, intended for collaboration and review 
- **The presentation** - a slideshow designed to communicate ideas with collaborators  
- **The book** - Multi-notebook scientific project presented in book format


##  What are Jupyter kernels 
A kernel is a computer program that runs and inspects the user's code. IPython includes a kernel for Python code. Others have written kernels for various other languages. We will begin by using the bash built-in kernel today. Tomorrow, we will cover how to set up a project-specific environment and start our journy into python.


## So let's open a notebook using the Bash kernel
press <kbd>Shift + CMD + L</kbd> to open a new launcher (or open the file menu and select new Launcher). If you select the bash notebook, a new notebook tab named `Untitled.ipynb` will appear.

## Jupyter Notebook modes 

- There are two modes of action in Jupyter: *Command* and *Edit*.  
- Command mode allows you to edit your notebook, but you cannot type in cells
- to enter command mode you press the  <kbd>Esc</kbd>  button. 
- Pressing <kbd>Enter</kbd> will transfer you to EDIT mode where you can interact with each cell in your notebook. 



# COMMAND mode useful shortcuts

The same goes to COMMAND shortcuts.  
  
| Key | Function | 
| :-- | :-- |
| <kbd>Esc + Y</kbd> | Change cell to code type |
| <kbd>Esc +  M</kbd> | Change cell to markdown type |
| <kbd>Esc + SHIFT + up</kbd> | select cells above |
| <kbd>Esc + SHIFT + down</kbd> | select cells above |
| <kbd>Esc + SHIFT + M</kbd> | merge selected cells |
| <kbd>c</kbd> | copy selected cells |
| <kbd>x</kbd> | cut selected cells |
| <kbd>v</kbd> | paste selected cells |


## EDIT mode useful shortcuts

When in EDIT mode, there are several key commands it is good to know.  
You should practice using these commands until they become second nature. 


| Mac  |  Function | 
| :--  |  :--      |
|<kbd>Tab</kbd>|  code completion/indent |
|<kbd>Shift+Tab</kbd>|  function help |
|<kbd>Cmd + /</kbd>|  comment/uncomment |
|<kbd>Ctrl+Enter</kbd>| run current cell |
|<kbd>Shift+Enter</kbd>|  run current cell + select below|
|<kbd>Alt+Enter</kbd>|  run current cell + insert below|
|<kbd>Ctrl+Shift+-</kbd>|  Split cell at cursor |
|<kbd>Cmd+s</kbd>|   Save notebook |
|<kbd>D+D</kbd>|   Delete selected cells |


## Our first job setting up our profile 

Our last task in the previous episode was to create a file called `.profile`. We will now use this file to set up our environment.
At the top cell in your open notebook copy the follwoing code and run the cell using <kbd>Shift+Enter</kbd>:  

~~~
echo source ~/.bash_aliases >> ~/.profile
source ~/.profile
~~~
{: .language-bash}

~~~
running my .profile
~~~
{: .output}

## What did we just do? 
In sourcing the file, we are executing any commands inside - therefore all the aliases we added to the `.bash_aliases` file will be available in the notebook.
This means that this `.profile` is one stop place to place any settings required by any shell related software, as you will see soon. 

## Different ways of excuting programs 
It is possible to use conditional variables in shell to control how successive commands are treated.
You can then create scripts that continue even if a step fails, issue multiple commands simultaneously, etc.
One of the advantages of bash is that it has the capability to work asynchronously. This means that if we have a set of commands that don't depend on each other, we can execute them without waiting for them to complete. 

- Using the semicolon ';' groups commands without dependency.
- Using a double '&&' groups commands with an `AND` dependency chain .
- Using a double vertical line '||' groups commands with an `OR` dependency chain.
- Using a single and '&' runs the last command in the background.

## Let's create a pipline to setup FSL
In order to illustrate this, let's install and configure FSL, which is a neuroimaging software package which includes many functions useful for preprocessing and analysis of structural and functional fMRI.


## Step 1. 
Start by creating at your home folder two new folders named `fsl` and `temp`

> ## Create `~/temp` and `~/fsl`
> > ~~~
> > mkdir -p ~/{temp,fsl}
> > ~~~
> {: .language-bash}
{: .solution}

## Step 2. 
Goto [FSL home page](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
If you are involved in neuroimaging related projects and have very little or no background in neuroimaging, it would be helpful to bookmark this page.

## Step 3. 
We are going to use a bash software called `wget` that is used to download file from internet links using the command line. 
Start by copying and running the following code in your notebook 

~~~bash
FSL_TAR=fsl-6.0.4-centos7_64.tar.gz
wget -O ~/temp/$FSL_TAR https://fsl.fmrib.ox.ac.uk/fsldownloads/$FSL_TAR
~~~

### What does this mean
- We start by creating a variable that holds the target zip file 
- Then we use the `wget` command with the `-O` option to direct the download to a specific location  


> ## Discussion
> Why is it beneficial to separate the file name from the command?
{: .discussion}


### You should get the following output:
~~~
--2021-05-22 09:29:57--  https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.4-centos7_64.tar.gz
Resolving fsl.fmrib.ox.ac.uk (fsl.fmrib.ox.ac.uk)... 129.67.248.65
Connecting to fsl.fmrib.ox.ac.uk (fsl.fmrib.ox.ac.uk)|129.67.248.65|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 4080459725 (3.8G) [application/x-gzip]
Saving to: ‘/home/jovyan/temp/fsl-6.0.4-centos7_64.tar.gz’
~~~
{: .output}

### This will update after around 80-130s to this 
~~~
--2021-05-22 09:29:57--  https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-6.0.4-centos7_64.tar.gz
Resolving fsl.fmrib.ox.ac.uk (fsl.fmrib.ox.ac.uk)... 129.67.248.65
Connecting to fsl.fmrib.ox.ac.uk (fsl.fmrib.ox.ac.uk)|129.67.248.65|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 4080459725 (3.8G) [application/x-gzip]
Saving to: ‘/home/jovyan/temp/fsl-6.0.4-centos7_64.tar.gz’

/home/jovyan/temp/f 100%[===================>]   3.80G  45.3MB/s    in 85s     

2021-05-22 09:31:22 (46.1 MB/s) - ‘/home/jovyan/temp/fsl-6.0.4-centos7_64.tar.gz’ saved [4080459725/4080459725]
~~~
{: .output}


## Step 4. 

We will use the tar command to unpack the file we just downloaded in our home directory. 
Due to the length of this operation, we will use the run in background option (to continue working while the unpacking is in progress).
Just copy the following and run it using <kbd>Shift+Enter</kbd>


~~~bash
tar -xf ~/temp/$FSL_TAR -C ~/ &
~~~

## Unpack the command 

- Let's unpack the tar command line from right to left

|  |  |
| :-- | :-- |
| `tar` | call the tar program |
| `-` | Add options to the program |
| `x` | extract a archive file |
| `f` | include target archive file |
| `~/temp/` | Where the target file is located |
| `$FSL_TAR` | our targer filename variable |
| `-C` | create extraction in a location  |
| `~/` | the target location is our home folder |
| `&` | Run in the background |
| | |

## What does the output mean ? 

~~~
[1] <some integer>
~~~
{: .output}


In most Unix and Unix-like operating systems, the `ps` program displays the currently-running processes. 
A process has a unique ID, and by using the ps command you can see which ones are currently running.
if we run the following: 

~~~bash
ps a
~~~

We should get an output similar to this: 

~~~
PID TTY      STAT   TIME COMMAND
1543 pts/1    Ss     0:00 /opt/conda/bin/bash --rcfile /opt/conda/lib/python3.8
2012 pts/1    D      0:24 tar -xf /home/jovyan/temp/fsl-6.0.4-centos7_64.tar.gz
2013 pts/1    S      0:38 gzip -d
2017 pts/1    R+     0:00 ps a
~~~
{: .output}

## Let's unpack the output

|  |  |
| :-- | :-- |
| `PID` | Unique process ID |
| `TTY` | The terminal the command is running in |
| `STAT` | Process states that indicate what state the program is in |
| `TIME` | How long this program has been running for (H:MM) |
| `COMMAND` | The actual command  |
| | |

## Configuring your shell environment
FSL requires you to define variables, we want to do this setup once and in the process give you some foundations that will be useful when you wish to set up a similar setup in more complex environments than our jupyter sandbox.

> ## Try to print out the ~.profile contents 
> > ~~~bash
> > cat ~/.profile 
> > ~~~
> > ~~~
> > echo "running my .profile"
> > source /home/jovyan/.bash_aliases
> > ~~~
> {: .output}
{: .solution}

## Add to our profile file some additional variables 
We need to add to our `.profile` file the following lines: 

~~~bash
export FSLDIR=$HOME/fsl
export PATH=$PATH:$FSLDIR/bin
export FSLOUTPUTTYPE=NIFTI_GZ
~~~

> ## Try to print out the ~.profile contents 
> > ~~~bash
> > tee -a ~/.profile << END
> > export FSLDIR=\$HOME/fsl
> > export PATH=\$PATH:\$FSLDIR/bin
> > export FSLOUTPUTTYPE=NIFTI_GZ
> > END
> > ~~~
> > Important - If a variable is included in a script, the $ sign must be "escaped" to tell bash not to interpret it.
> > ## Test if worked 
> > ~~~bash
> > cat  ~/.profile
> > ~~~
> > ~~~
> > echo "running my .profile"
> > source /home/jovyan/.bash_aliases
> > export FSLDIR=$HOME/fsl
> > export PATH=$PATH:$FSLDIR/bin
> > export FSLOUTPUTTYPE=NIFTI_GZ
> > ~~~
> {: .output}
{: .solution}





{% include links.md %}

