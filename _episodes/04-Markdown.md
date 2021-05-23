---
title: "Introduction to Markdown"
author: "Dr. Eyal Soreq" 
date: "05/03/2021"
teaching: 30
exercises: 10
questions:
- What is Markdown?
- How is markdown related to HTML?
- 
objectives:
- Understand the Markdown syntax and render Markdown text.
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script> -->


## More advanced Markdown

- Markdown is the formating codes for creating publishable data science report using Jupyter. 
- In this section, we will go over all the different ways to create a structured document using the Markdown syntax supported by Jupyter Notebook.

## Jupyter Notebook Markdown syntax

- The underlying machinery that supports Jupyter rendering is HTML 
- However, the main idea is not to use it for sophisticated web design but rather to speed up and simplify the processes of analytical investigation. 
- As such Markdown syntax is minimal and sufficient
- If (or when) you feel like you want to create something that isn't supported by the existing syntax you should probably use different software. 
- The current setup supports, creating sophisticated formatting for your text and code with simple to remember the syntax. 
- I recommend that you open your Jupyter notebook and begin experimenting with different options. 


## Cell Headers

- Cell headers come in six sizes form h1-h6
- They are defined using the pound sign # before the text 
- The number of ## = h2 corresponds with the heading level <br />
<br />

~~~
# H1 The largest heading looks like this 
## And this is H2 The second-largest heading
###### Finally the smallest heading is H6
~~~
{: .language-markdown}


# Styling text - **bold** *italic* and ~~strikethrough~~
 
- Bold text is defined by either using two stars /*/* **before and after the text** 
- Or by using the opening HTML tag `<b>` <b>and closeing tag</b> `</b>`
- Italic uses one star \* before and after the *emphasized text*
- Or by using `<i>` _HTML tag_ `</i>`
- Two tildas `~~` before and after text create the ~~strikethrough effect~~
- Or by using `<del>` <del>HTML tag</del>`</del>`


# Styling text - *indented quotes*

Indenting can be initialized using Greater than and space `> ` before the text

> There is no scientific study more vital to man than the study of his brain. Our entire view of the universe depends on it.

>> **Francis Crick (1916-2004)**


# HTML tags 

- HTML tags are special words or letters surrounded by angle brackets, < and >. 
- Jupyter relies on HTML to render everything, and as a byproduct, we can use native HTML components  
- Styling text with tags is easy, you declare a region with a `<tag>` and close it like this `</tag>` 
- For example Marked <mark>text</mark> is defined using `<mark>` tag. 
- Adding css attributes such as <mark style="background-color:blue;color:white"> background-color or font color</mark> within the enclosed area is also an option. 
- This is achieved using the inline css style attribute changing the tag to look like this `<mark style="background-color:blue;color:white">`. 
- In a similiar way text can be <small>small</small>, <ins>inserted</ins> using `<small>` and `<ins>`. 
- It can be subscript<sub>text</sub> or Superscript<sup>text</sup> using `<sub>` and `<sup>`
- This opens a whole world of options, that to be honest you will rarely use <span style="font-size:30px">&#128521;</span>, but can be fun. 


# Preformatted Text 

- Sometimes you want the text to show up exactly as you write it.
- Without Markdown tags doing their schtick 
- This can be achieved by indent every line by at least four spaces (or one tab). 
- Alternatively, you can make a code block using 3 or more tildes (~) or backticks (`) on a line before and after the text.


# code blocks and code highlighting

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

To create a circular bullet point, simply use either `-`, `*` or `*` followed with either one or two spaces. Each bullet point must be on its line. To construct sub levels just press TAB to before a bullet using one of the methods described here. 

~~~markdown 
- Main bullet point
    * Sub bullet point
        - Subsub bullet point
~~~

# Creating Markdown numbered lists 

- To create a numbered list, enter 1. followed by a space, for example:

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

~~~markdown 
| latex | <div style="width:100px">rendering</div> | context | 
| --- | --- | --- | 
| '\sum^n_{i=1}\frac{(w^Tx(i)-y(i))^2}{n}' | $$\frac{1}{n}\sum^n_{i=1}(w^Tx(i)-y(i))^2$$  | Mean Squared Error | 
~~~

| latex | <div style="width:100px">rendering</div> | context |
| --- | --- | --- |
| '\sum^n_{i=1}\frac{(w^Tx(i)-y(i))^2}{n}' | $$\frac{1}{n}\sum^n_{i=1}(w^Tx(i)-y(i))^2$$ | Mean Squared Error |


# use HTML instead

- So we can use all the things we learned up until now to create an HTML table instead of the markdown table


~~~html
<table>
 <tr> <!-- tr = table row  -->
    <th>latex</th> <!-- th = table header  -->
    <th style="width:150px">rendering</th>
    <th>context</th>
</tr>
 <tr>
    <!-- td = table data  -->
    <td style="font-size:2vmin">
        '\sum^n_{i=1}\frac{(w^Tx(i)-y(i))^2}{n}' </td>
    <td>$$\frac{1}{n}\sum^n_{i=1}(w^Tx(i)-y(i))^2$$ </td>
    <td> Mean Squared Error</td>
 </tr>
</table>
~~~


# Simple table with inline formatting 

- markdown formattings will render within the table 
- As will HTML tags 


| language | code | render   | 
| --- | --- | --- |
|  Markdown | `**bold**`  | **bold** |
| ^^ | `*itealic*`  | *itealic* |
| ^^ | `~~deleted~~`  | ~~deleted~~ |
|  HTML | `<b>bold</b>`  | <b>bold</b> |
| ^^ | `<i>itealics</i>`  | <i>itealics</i> |
| ^^ | `<del>deleted</del>`  | <del>deleted</del> |



# multi line within table cell

| Format | Tag example |
| -------- | ----------- |
| Headings | =heading1=<br>==heading2==<br>===heading3=== |
| New paragraph | A blank line starts a new paragraph |
| Source code block | // all on one line<br> {{{ if (foo) bar else   baz }}} |


# Table content can be aligned 

- Content alignment is achieved by including colons :
    - To the right ---: of the hyphens
    - Left :--- of the hyphens
    - Or using colons in both sides :---: will centre the content 

```
| latex  | rendering | context |
| ---: | :---: |  :--- |
| `$x_1$`  | $$x_1$$     | Subscripts    |
| `$x^1$`     | $$x^1$$     | Superscripts      |
| `$x_1^y$`     | $$x_1^y$$     | both      |
| `$x_{1^y}$`     | $$x_{1^y}$$     | nested below     |
| `$x^{y_1}$`     | $$x^{y_1}$$    | nested above      |
| `$x^{CRTX}$`     | $$x^{CRTX}$$     | long words      |
```


| latex  | rendering | context |
| ---: | :---: |  :--- |
|  x_1  | $$x_1$$ | Subscripts |
|  x^1  | $$x^1$$ | Superscripts |
|  x_1^y  | $$x_1^y$$ | both |
|  x_{1^y}  | $$x_{1^y}$$ | nested below |
|  x^{y_1}  | $$x^{y_1}$$ | nested above |
|  x^{CRTX}  | $$x^{CRTX}$$ | long words |


{% include links.md %}

