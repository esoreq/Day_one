---
title: "Introduction to Jupyter notebook"
author: "Dr. Eyal Soreq" 
date: "05/03/2021"
teaching: 30
exercises: 10
questions:
- What is Jupyter Notebook?
- What is a Kernel? 
- What is Markdown?
objectives:
- List the components and functionality of Jupyter Notebook.
- Launch and navigate the Jupyter Notebook dashboard.
- Open, create and save Jupyter Notebook files.
- Create, run and delete Markdown and Code cells.  
- Understand the Markdown syntax and render Markdown text.
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

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

## Jupyter Notebook modes 

- There are two modes of action in Jupyter: *Command* and *Edit*.  
- Command mode allows you to edit your notebook, but you cannot type in cells
- to enter command mode you press the  <kbd>Esc</kbd>  button. 
- A useful shortcut is the <kbd>Esc + h</kbd>  which gives you a list of all the shortcuts. 
- Pressing <kbd>Enter</kbd> will transfer you to EDIT mode where you can interact with each cell in your notebook. 


## EDIT mode useful shortcuts

When in EDIT mode, there are key commands that you should force yourself to use until it becomes natural.  


| Mac  | Key | Function | 
| :--  | :-- | :--      |
|<kbd>Tab</kbd>| TAB | code completion/indent |
|<kbd>Shift+Tab</kbd>| SHIFT+TAB | function help |
|<kbd>Cmd + /</kbd>| CMD + / | comment/uncomment |
|<kbd>Ctrl+Enter</kbd>| CTRL + ENTER | run current cell |
|<kbd>Shift+Enter</kbd>| SHIFT + ENTER | run current cell + select below|
|<kbd>Alt+Enter</kbd>| ALT + ENTER | run current cell + insert below|
|<kbd>Ctrl+Shift+-</kbd>| CTRL + SHIFT + MINUS  |  Split cell at cursor |
|<kbd>Cmd+s</kbd>| CMD + s  |  Save notebook |
|<kbd>D+D</kbd>| D + D (press the key twice)  |  Delete selected cells |

{% include links.md %}

