---
title: "End of course assessment"
author: "Dr. Eyal Soreq" 
start: true
date: "10/08/2021"
---
# The assessment

At the end of the course assessment, we attempt to explore a dataset to gain meaningful insights about neurodegeneration. It is intended that this project be completed independently, but groups are encouraged as long as the report is impressive and the contributions of the group members are equal. I am not interested in long projects - I am not looking for a set of methods you have chosen to demonstrate your computational abilities but rather see if you can come up with an engaging and original problem to study. However, the methodology should be related to what we studied in the course (i.e. using t-test to compare two groups is not recommended).

The OASIS-3 compiles MRI and PET data and related clinical information for 1098 participants across multiple ongoing studies in Washington University Knight's Alzheimer Disease Research Center. Participants range in age from 42 to 95 with cognitively normal status, and  cognitively impaired status. OASIS-3 is available as an open access data set to the scientific community to answer questions related to healthy aging and dementia.

I believe the OASIS-3 datasets provided are sufficient to generate a 1st class report and are included as part of the preprocessing module. 
You are not required to download any additional material from the OASIS website (but you are able to do so if you think it is essential).


# What is expected from the assessment?

- You have two weeks to complete the Opt-out assessment.

- The PDF report should be submitted with the notebook used to generate it that can be reproduced all the steps you took.

- The submission should include at least one reproducible Jupyter notebook report using various methods to clean, explore and analyze the datasets and present an interesting finding that leverages data analysis. 

- Please do not try to cover everything in the dataset; it is impossible. Instead, go over the supporting documents and identify an interesting angle that you believe is possible to explore using the summary exports that OASIS-3 contains. 

- In your submission there should be at least one Python script or Jupyter notebook containing supporting functions, and if necessary, another with plot functions. There should be minimal coding in the report (most of it should be in the supporting files). 

## Creating a report structure

**The report** is a polished version of your study and is intended for collaboration and review. Making a habit of aggregating useful discoveries into a single report notebook used for communication with colleagues can save you a lot of time!
If you follow the following structure for the final project for this course, you can easily convert your report into an article.


## Pointers 
- In the first analytic section, you should give a comprehensive description of the dataset subset you chose to focus on. 
- It must demonstrate your proficiency to include all the pertinent information necessary for effective analyses. Afterward, the distributions and relations within your subset should be explained using a descriptive step. This should include an appealing multipanel visualization that supports your text and/or a tabular representation.  
- In addition to this, two analytical sections must be submitted, each supporting an argument in its own right. To illustrate, if you were studying the relationship between grey matter loss and the severity of dementia, you could take several different approaches (e.g. regression analysis, supervised learning etc.). Each section should begin with a short rationale, the method used, the results, a visual representation of the findings and your interpretation.    
- A minimal set of function calls should perform the analysis step. Whenever you use a package that requires multiple steps, wrap them in your own function and include them in the supporting scripts. These should perform the analysis and present the results in tabular or visual form. Accompanied by a concluding markdown cell that summarizes the results and leads to the next step in your analysis. 
- The report should also include a methodological section where you are expected to write a paragraph in markdown explaining the methods you used in the report. 
- Finish the report by including a conclusion Markdown cell that discusses the limitations of your research as well as your suggestions for future studies.
 

# Hints and recommendations 

 
- With this assessment, my objective is not to fail you, but rather to assess how well you can use the tools at your disposal to examine interesting questions concerning cognitive neuroscience, and in this case neurodegenerative ones. 
- Therefore, I would not try to invent the wheel, but rather read through several approaches that have already been applied to this dataset (or the ones that preceded it). Then armed with that information go over the data-files I sent and find the ones that are the most probable. 
- You need to emphasize your abilities in developing cohesive reports using data science, not your potential to find new research directions.
- Begin by creating some exclusion/inclusion criteria that will simplify the data, rather than trying to analyze all the features at once.
- A short and impressive report is better than a long and messy one. 
- Try to write functions that require very little comments to be understood 
- If a function is very long it could and should be simplified   

# Use of packages and modules 
 
- You can use anything that can be installed using either git, pip or conda  
- Please don't use R as part of this assessment involves your Python skills explicitly 

## suggested Report structure 
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
1. References 
    - Should be numerical and included in a bibtex file     


# GOOD LUCK 