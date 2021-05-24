---
title: "ANACONDA 101"
author: "Dr. Eyal Soreq" 
date: "05/03/2021"
teaching: 20
exercises: 20
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

# Conda in Theory?

## What is Conda?
Conta is a package and environment manager based on Python. It assists in the development of reproducible analysis pipelines using crowd-sourced and version-controlled packages. It might be a bit confusing for an individual who is new to this type of approach. In order to make sure we all understand each other, let's establish some quick vocabulary:

## What is an environment?

In a computing environment, people use an assortment of programs, language libraries, etc. to operate a computer.  
Depending on the context, the word environment can have many different meanings, like a generalized term for an ecological ecosystem that can refer to anything from a puddle to a continent.  
As far as we're concerned, an environment is simply a set of tools that are put together to assist us in exploring data-driven questions in a reproducible manner.

## What is an package?
As the name implies, a package is a collection of software, including things like programs  (e.g. Python), programming libraries (e.g. Bash), or other useful tools.  
Using the Conda package management system, you can combine packages and make complex environments.  
It can be summed up like this: Conda creates self-contained modules that contain all of the necessary programs, etc, in order to complete a specific task of computing.


## What is dependency hell
The term dependency hell refers to the problems that users generally face when they rely on many interdependent packages.  
The main source of complications or bugs in dependency hell are changes made to third-party packages that are no longer compatible with one another.


## How does Conda manage dependencies
Conda helps manage dependencies in two primary ways:
Allows the creation of environments that isolate each project, thereby preventing dependency conflicts between projects.  
Provides identification of dependency conflicts at time of package installation, thereby preventing conflicts within projects/environments.

## What is version control 
Version control is what, exactly? Version control tools allow you to keep track of the changes you make to your work over time.  
This feature is a little like "track changes" in Google Docs, but with the difference that you can save changes across several files, not just within one.

## Why is Conda useful?

Using Conda as part of analysis workflows has a lot of advantages:

1. It helps keep your computing environment organized so you're less likely to end up in "dependency hell".
1. Since packages are version-controlled, you can easily switch versions if one doesn't work.
1. Operating systems aren't really an issue (it runs on Mac, Windows, Linux). However, not every package is available for every OS.
1. We can replicate and share environments, so our analysis is more accurate.

## Conda in Practice over the gwdg
1. One thing we ultimately want is having full control over the environment
1. The following is true to your local computer as well as the cluster
1. Anaconda gives us this power exactly


## Our first job extending our profile 

Our *.profile* is the place where all our settings are kept, so it will come as no surprise that we will also use it to configure Anaconda.
We will open terminal instance in the jupyter cloud and append `source /opt/conda/etc/profile.d/conda.sh` to the end of our `.profile` file. 
Remember that we must source the `.profile` file for any changes we make to take  effect.

> # Copy and run the following code 
> ~~~bash
> echo source /opt/conda/etc/profile.d/conda.sh >> ~/.profile
> source ~/.profile
> ~~~
> 
> ~~~
> running my .profile
> ~~~
> {: .output}
{: .challenge}

## Test conda

Run the following code to verify conda is running and configured correctly

~~~bash
conda info
~~~

## You should get an output that resembles this.

~~~
     active environment : base
    active env location : /opt/conda
            shell level : 1
       user config file : /home/jovyan/.condarc
 populated config files : /opt/conda/.condarc
                          /home/jovyan/.condarc
          conda version : 4.10.0
    conda-build version : not installed
         python version : 3.8.8.final.0
       virtual packages : __linux=4.15.0=0
                          __glibc=2.31=0
                          __unix=0=0
                          __archspec=1=x86_64
       base environment : /opt/conda  (writable)
      conda av data dir : /opt/conda/etc/conda
  conda av metadata url : https://repo.anaconda.com/pkgs/main
           channel URLs : https://conda.anaconda.org/conda-forge/linux-64
                          https://conda.anaconda.org/conda-forge/noarch
          package cache : /opt/conda/pkgs
                          /home/jovyan/.conda/pkgs
       envs directories : /home/jovyan/env
                          /opt/conda/envs
                          /home/jovyan/.conda/envs
               platform : linux-64
             user-agent : conda/4.10.0 requests/2.25.1 CPython/3.8.8 Linux/4.15.0-140-generic ubuntu/20.04.2 glibc/2.31
                UID:GID : 1000:100
             netrc file : None
           offline mode : False
~~~
{: .output}           


## Creating and Managing Environments

### Environments 101

Now that we've made sure Conda is working, we're ready to start learning how to use it as an environment-based package manager.   
Environments are an integral part of Conda-based workflows.   
They are customizable, reproducible, and shareable modules that contain the resources for a specific task or set of tasks.   
Environments also help avoid "dependency hell" where required programs are incompatible with previously installed programs or program versions.  


### View installed enviorments 

To start with, let's see what environments we currently have set up.  
This will list all of the environments available for us to use along with their locations.

~~~bash
conda env list
~~~

Should look like this:

~~~
xeus-python              /home/jovyan/env/xeus-python
base                  *  /opt/conda
~~~
{: .output}  

By default, an environment called base is created when installing and intializing Conda.  
`base` contains a number of standard Python packages that may or may not be useful.


### Change .profile to auto load conda base
Because we altered the code in our ~/.bashrc, the base environment isn't loaded automatically when we log into the shell. This can help speed up tasks if you don't need to use anything in the environment but, if we do need to use something in the environment, we'll need to activate the environment first. We'll start with the base environment.

# Sanity check 

You should now see that the word base is in your prompt showing that we've loaded the base environment. Another way to check which environment you have active is to look at the $CONDA_PREFIX shell variable. If you don't have any environment loaded, the output will be blank.

echo $CONDA_PREFIX

## list installed programs 
Now that we have the base environment loaded, let's see which programs it contains for us to use.

`conda list`

The output from this command lists all of the programs, their versions, the build number, and the channel they came from if outside of the defaults channels.

## Check $PATH
We can check to make sure we are using the programs from our environment by using which to print the executable path or by checking our shell $PATH variable. The $PATH variable lists the order of folders from first to last that the shell will look through to find executables. The shell will execute the first binary it finds and ignore the rest so it's important our $PATH is in the correct order.

`echo $PATH`

## Creating Task-Specific Environments

In a typical workflow, it's good practice to create environments for each task or script being executed to keep the environment running as fast as possible, reduce the likelihood of conflicting programs/versions, and assist in debugging when things don't work out. Let's create a new environment for analyzing 16S bacterial sequencing data with mothur.

## Here are some best practices that work for me

1. Use a simple naming convention 
    `[#]_[notebook_description]_[author_initials]_[YYYY-MM-DD].ipynb`
    - `#` The number of notebooks in an analysis can be very large, and clustering notebooks together with an identifying symbol makes it easier to keep track of what you tried.  
    - `notebook_description` is just a short description of what this notebook is about, e.g. `exploratory_analysis`,  `pre_processing`, `FIR_FC_mining` or `final_report`. 
    - `author_initials` Is essential when working together with collaborators using a joint repository or when sending over your work to a supervisor.
    - `Date` Could take any form but is the simplest way of keeping track of what you did 

1. Include a meaningful introduction (H1 header) that describes the notebooks purpose and content.
1. Make your notebook's headings and documentation in markdown cells clear and concise. Tell people about your workflow steps and consider how it might benefit you in the future.
<!-- 1. Refactor and outsource code into modules. Every time you copy and paste a piece of code, you should consider replacing it with a function. This will not only make your notebook more readable, but also make fixing errors easier. -->


## Markdown Syntax guide

- The underlying machinery that supports Jupyter rendering is HTML 
- However, the main idea is not to use it for sophisticated web design but rather to speed up and simplify the processes of analytical investigation. 
- As such Markdown syntax is minimal and sufficient
- If (or when) you feel like you want to create something that isn't supported by the existing syntax you should probably use different software. 
- The current setup supports, creating sophisticated formatting for your text and code with simple to remember the syntax. 


> ## Create a new notebook with a lab structure
> Select the top cell using your mouse 
> Press <kbd>Esc +  M</kbd>
> Write the following in that cell 
> ~~~markdown
> # Lab Template 
> This notebook covers the different ways to use markdown to create an effective data science document.
> ~~~
> Now press <kbd>Shift+Enter</kbd> to render the Markdown cell  
> Use the following shortcut <kbd>Shift+Cmd+s</kbd> or just use the GUI to rename the open notebook  
> Use a name that follows the rules we just covered e.g. `1_Lab_template_ES_YYYY-MM-DD.ipynb`  
{: .challenge}


<!-- 
## Creating a report structure
**The report** is a polished version of your study and is intended for collaboration and review. Making a habit of aggregating useful discoveries into a single report notebook used for communication with colleagues can save you a lot of time!
If you follow the following structure for the final project for this course, you can easily convert your report into an article.

## Report structure 
1. Title   
1. Short introduction that has the following sections:
    - Background (The general context and the specific area of this report)
    - The problem or question you wish to address  
    - The data and methods you are using to answer your questions
    - Summarizes your main result and explain your results in relation to your questions.
    - Discuss how your findings compare to existing knowledge? 
1. Data overview that has the following information
    - Where is the data from 
    - Sample size
    - Group/condition of intrest
    - A full description of the other covariates of intrest
    - What metrics are you using and how were they extracted
1. Methods
    - What are the different methods used in this report 
    - Briefly explain what they do and why they will be suitable for you to answer your questions.
1. Results 
   - There should be a clear title for each result section that guides the reader through the results in context.
   - Start the text with a question
   - Then add a sentence explaining what you did and how
   - Report the findings
   - Tie them to previous research with some references (at least one).
1. Conclusion   
    - State your most important results in the Conclusion section. 
    - Make sure you do not simply summarize your findings - interpret the findings at a higher level of abstraction. 
    - What are the limitations of your study? 
    - Do you believe you did an adequate job of answering the questions stated in the introduction in light of the limitations? 
    - Lastly, what suggestions do you have for future research?
 -->

 ## Cell Headers

- Cell headers come in six sizes 
- They are defined using the pound sign `#` before the text 
- The number of symbols `##` = h2 corresponds with the heading level 

# Here is an example starting from H1 the largest heading 
## H2
### H3
#### H4
##### H5
###### H6 is the smallest heading 

> ## Add a new markdown cell and create all the different heading sizes 
> Try to create this yourself 
> > ## Solution
> > ~~~markdown
> > ## Cell headers come in six sizes 
> > They are defined using the pound sign `#` before the text 
> > # Here is an example starting from H1 the largest heading  
> > ## H2
> > ### H3
> > #### H4
> > ##### H5
> > ###### H6 is the smallest heading 
> > ~~~
> {: .solution}
{: .challenge}

# Styling text 
 
- Bold text is defined by either using two stars `**` **before and after the text** 
- Italic uses one star `*` before and after the *emphasized text*
- Two tildas `~~` before and after text create the ~~strikethrough effect~~
- ***Important text*** can be emphsised using three stars `***` 

# Styling text - *indented quotes*


> Indenting can be initialized using Greater than and space `> ` before the text
> > The text will move two indents with two greater signs
> > > We can continue like this for as long as we want


# HTML tags 

- HTML tags are special words or letters surrounded by angle brackets, < and >. 
- Jupyter relies on HTML to render everything, and as a byproduct, we can use native HTML components  
- Styling text with tags is easy, you declare a region with a `<tag>` and close it like this `</tag>` 
- For example Marked <mark>text</mark> is defined using `<mark>` tag.
- <u>Underline text</u> is almost never used but is defined using `<u>` tag. 
- Adding css attributes such as <mark style="background-color:blue;color:white"> background-color or font color</mark> within the enclosed area is also an option. 
- This is achieved using the inline css style attribute changing the tag to look like this `<mark style="background-color:blue;color:white">`. 
- In a similiar way text can be <small>small</small>, <ins>inserted</ins> using `<small>` and `<ins>`. 
- It can be subscript<sub>text</sub> or Superscript<sup>text</sup> using `<sub>` and `<sup>`
- This opens a whole world of options, that to be honest you will rarely use, but can be fun &#128528;&#128521;. 

# Preformatted Text 

- Sometimes you want the text to show up exactly as you write it.
- Without Markdown tags doing their schtick 
- This can be achieved by indent every line by at least four spaces (or one tab). 
- Alternatively, you can make a code block using 3 or more tildes (~) or backticks (`) on a line before and after the text.


# Code blocks and code highlighting

- When you declare a code block, you can also add the type of language to enable syntax highlighting 

For example `~~~python` 

~~~python 
def f(x):
    #Returns the square root of x
    return x**0.5
~~~

Or `~~~html` 
~~~html 
<hr style="border-top: 2px solid black;">
~~~

# Creating Markdown bullet lists 

- To create a circular bullet point, simply use either `-`, `*` or `*` followed with either one or two spaces.  
    * Each bullet point must be on its line.
        - To construct sub levels just press TAB to before a bullet using one of the methods described here. 

~~~markdown 
- Main bullet point
    * Sub bullet point
        - Subsub bullet point
~~~

# Creating Markdown numbered lists 

1. To create a numbered list, enter `1.` followed by a space
    1. To construct sub levels just press TAB to before a bullet. 
    1. You don't need to number your lists markdown will do that for you  
    Adding text just requires  
    ending each line  
    with 2 spaces `  `
    1. Numbers will continue once you add aline

~~~markdown 
1. Numbered item 1
1. Numbered item 2
    1. Numbered item A
        1. Numbered item a
~~~        


# Finally, I am a sucker for to-do lists that start my notebooks. 

- To start a checklist, use `- [ ]` followed by space, for example:
- Creating checked boxes simply replaces the space with x - like this `- [x]`

~~~markdown 
- [ ]  this is not checked
- [x]  but this is checked
~~~


## Hyperlinks and references 

There are four different ways of adding a hyperlink in a cell. 
The simplest is by declaring the link explicitly
1. https://en.wikipedia.org/wiki/Hyperlink    
Another more subtle approach creates a name link pair
1. [Hyperlink](https://en.wikipedia.org/wiki/Hyperlink)
1. created like this:`[Hyperlink](https://en.wikipedia.org/wiki/Hyperlink)`   

Sometimes you would wish to reference[[1][Hyperlink]] within the text 
1. Creating these references requires a name id pair `[[1][Hyperlink]]`
1. And an invisible id link pair placed at the bottom of the cell
1. `[Hyperlink]: https://en.wikipedia.org/wiki/Hyperlink "Wiki Hyperlink"`  

[Hyperlink]: https://en.wikipedia.org/wiki/Hyperlink "Wiki Hyperlink."

Finally you can always use HTML tags to define a <a href="https://en.wikipedia.org/wiki/Hyperlink">Hyperlink</a>
1. Using this syntax 


```html
<a href="https://en.wikipedia.org/wiki/Hyperlink">Hyperlink</a>
```

## Horizontal rules

Adding some horizontal lines can be as simpple as just writting `---`

----

## Sourcing local video content 

We can run loal video files using relative paths and an html tag

~~~html
<video controls src="../files/some video.mp4">Title</video>
~~~

<video controls src="../files/Reproducibility and Replicability in Science.mp4"></video>


## Images and 

To insert images to your markdown file, use the markup `![Alt text](/path/image.ext)`. The path can either be relative to the website, or a full URL for an external image. The supported formats are .png, .jpg, .gif. You might be able to use some .svg files too, depending on its structure.

Markdown uses an image syntax that is intended to resemble the syntax for links.  
- Start with an exclamation mark: `!`   
- Followed by a set of square brackets, containing the alt attribute text for the image;
- followed by a set of parentheses, containing the URL or path to the image, and an optional title attribute enclosed in double or single quotes.

For clickable images, simply wrap the image markup into a link markup

![](http://www.scottbot.net/HIAL/wp-content/uploads/2012/01/Last-line-of-defense-statistics.gif)

~~~markdown
![Semantic description of image](/images/path/to/folder/image.png)

![Semantic description of image](/images/path/to/folder/image.png)] <br> we can also add some captions

[![Semantic description of image](/images/path/to/folder/image.png)]](path to click)
~~~

# Mathematical equations and LaTeX

- Dealing with mathematical equations can be annoying. 
- However, Jupyter embeds a powerful coding language that was designed specifically for this. 
- With syntax that is as simple as the Markdown and HTML codes, we learned by now. 


# Inline and display equations 
- You can write inline formulas, enclosing the formula with `$` signs. 
- For example, consider the inline form of the Gaussian Normal Distribution

~~~markdown 
$P(x) = \frac{1}{\sigma \sqrt {2\pi}}e^{-(x-\mu)^2/2 \sigma^2}$. 
~~~

- You can also write them in display mode by using two \\$$ signs, for example.

~~~markdown 
$$P(x)=\frac{1}{{\sigma \sqrt {2\pi}}}e^{-(x-\mu)^2/2 \sigma^2}$$. 
~~~

# If you want to learn more  

- Latex is a world on its own, and if you need it, you probably know some of it already. 
- Here is an excellent tutorial designed explicitly for the LaTeX flavour supported by Jupyter [mathjax-basic](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference) 
- And a long list of all the commands available can be found [here](http://www.onemathematicalcat.org/MathJaxDocumentation/TeXSyntax.htm)


# Creating tables 

- While you could use latex to create your tables, Markdown simplifies this. 
- Tables are created using pipes | and hyphens -. 
- Hyphens are used to create each column's header, while pipes separate each column. 
- You must include a blank line before your table for it to correctly render.


# Simple table with heading long lines?

- So we can use all the things we learned up until now to control this table 
- This means we can use inline latex and HTML tags
- This tag created an HTML divider with CSS controlling the cell width
- However the lines are long and hard to read


# Table content can be aligned 

- Content alignment is achieved by including colons :
    - To the right ---: of the hyphens
    - Left :--- of the hyphens
    - Or using colons in both sides :---: will centre the content 

~~~markdown
| latex  | rendering | context |
| ---: | :---: |  :--- |
| `$x_1$`  | $$x_1$$     | Subscripts    |
| `$x^1$`     | $$x^1$$     | Superscripts      |
| `$x_1^y$`     | $$x_1^y$$     | both      |
| `$x_{1^y}$`     | $$x_{1^y}$$     | nested below     |
| `$x^{y_1}$`     | $$x^{y_1}$$    | nested above      |
| `$x^{CRTX}$`     | $$x^{CRTX}$$     | long words      |
~~~

- Which will look like this 

| latex  | rendering | context |
| ---: | :---: |  :--- |
|  x_1  | $$x_1$$ | Subscripts |
|  x^1  | $$x^1$$ | Superscripts |
|  x_1^y  | $$x_1^y$$ | both |
|  x_{1^y}  | $$x_{1^y}$$ | nested below |
|  x^{y_1}  | $$x^{y_1}$$ | nested above |
|  x^{CRTX}  | $$x^{CRTX}$$ | long words |


## Challenge
Try to replicate the following section in your notebook using a combination of what you learned so far.
### Timothy Leary

![](https://cdn-60080014c1ac18031c64f892.closte.com/wp-content/uploads/2017/08/leary-750x420.jpg)


###### American psychologist
**Born**: October 22, 1920, Springfield, Massachusetts, United States  
**Died**: May 31, 1996, Beverly Hills, California, United States

> *“The language of God is not English or Latin; the language of God is cellular and molecular.”*
> ###### ***As quoted in "Leary calls LSD 'sacrament'" in The Tech (8 November 1966), p. 6***
{: .challenge}

> > ## Solution
> > ~~~markdown
> > ### Timothy Leary
> >
> > ![](https://cdn-60080014c1ac18031c64f892.closte.com/wp-content/uploads/2017/08/leary-750x420.jpg)
> >
> > ###### American psychologist
> > **Born**: October 22, 1920, Springfield, Massachusetts, United States  
> > **Died**: May 31, 1996, Beverly Hills, California, United States
> >
> > > *“The language of God is not English or Latin; the language of God is cellular and molecular.”*
> > > ###### ***As quoted in "Leary calls LSD 'sacrament'" in The Tech (8 November 1966), p. 6***
> > ~~~
> {: .solution}

## Links to expand your understanding 

For those interested in learning more...

- [Conda Essentials](https://learn.datacamp.com/courses/conda-essentials)
- [Building and Distributing Packages with Conda](https://learn.datacamp.com/courses/building-and-distributing-packages-with-conda)

{% include links.md %}

