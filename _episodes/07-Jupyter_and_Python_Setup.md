---
title: "Python Overview"
author: "Dr. Eyal Soreq" 
start: true
date: "05/03/2021"
teaching: 15
exercises: 0
questions:
- What is Python?
- Why Python?
- Alternatives to Python? 
objectives:
- Understand what are the benefits of using Python as a central tool in your PhD.
- Understand Pythons position in the crowded data analytics ecosystem
keypoints:
- Python is an easy and versatile programming language
- It has the largest open-source community
- Python is a dynamic programming language
- Python in conjunction with Jupyter Notebook provides an optimal solution for almost all  research projects
---

# What is Python?
- Python is a popular high-level, object-oriented, interpreted programming language created by a developer called Guido Van Rossum.


# Why Python?

- It is easy to learn and master, and its syntax is straightforward to both learn and read. 
- In the context of Data Science in general and specifically in neuroscience it is quickly the new lingua franca of the field. 


<div class="bk-root" id="07247788-bda6-4511-aae8-5a0b38731232" data-root-id="1182"></div>








<div style="display: table;"><div style="display: table-row;"><div style="display: table-cell;"><b title="bokeh.plotting.figure.Figure">Figure</b>(</div><div style="display: table-cell;">id&nbsp;=&nbsp;'1182', <span id="1295" style="cursor: pointer;">&hellip;)</span></div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">above&nbsp;=&nbsp;[],</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">align&nbsp;=&nbsp;'start',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">aspect_ratio&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">aspect_scale&nbsp;=&nbsp;1,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background_fill_alpha&nbsp;=&nbsp;1.0,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">background_fill_color&nbsp;=&nbsp;'#ffffff',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">below&nbsp;=&nbsp;[LinearAxis(id='1193', ...)],</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">border_fill_alpha&nbsp;=&nbsp;1.0,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">border_fill_color&nbsp;=&nbsp;'#ffffff',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">center&nbsp;=&nbsp;[Grid(id='1196', ...), Grid(id='1200', ...), Legend(id='1232', ...)],</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">css_classes&nbsp;=&nbsp;[],</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">disabled&nbsp;=&nbsp;False,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">extra_x_ranges&nbsp;=&nbsp;{},</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">extra_y_ranges&nbsp;=&nbsp;{},</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">frame_height&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">frame_width&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">height&nbsp;=&nbsp;400,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">height_policy&nbsp;=&nbsp;'auto',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hidpi&nbsp;=&nbsp;True,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">inner_height&nbsp;=&nbsp;0,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">inner_width&nbsp;=&nbsp;0,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_event_callbacks&nbsp;=&nbsp;{},</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_property_callbacks&nbsp;=&nbsp;{},</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">left&nbsp;=&nbsp;[LinearAxis(id='1197', ...)],</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_factor&nbsp;=&nbsp;10,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_interval&nbsp;=&nbsp;300,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_threshold&nbsp;=&nbsp;2000,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">lod_timeout&nbsp;=&nbsp;500,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">margin&nbsp;=&nbsp;(0, 0, 0, 0),</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">match_aspect&nbsp;=&nbsp;False,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">max_height&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">max_width&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border&nbsp;=&nbsp;5,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_bottom&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_left&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_right&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_border_top&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_height&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">min_width&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">name&nbsp;=&nbsp;None,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outer_height&nbsp;=&nbsp;0,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outer_width&nbsp;=&nbsp;0,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_alpha&nbsp;=&nbsp;1.0,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_cap&nbsp;=&nbsp;'butt',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_color&nbsp;=&nbsp;'#e5e5e5',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_dash&nbsp;=&nbsp;[],</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_dash_offset&nbsp;=&nbsp;0,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_join&nbsp;=&nbsp;'bevel',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">outline_line_width&nbsp;=&nbsp;1,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">output_backend&nbsp;=&nbsp;'webgl',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">renderers&nbsp;=&nbsp;[GlyphRenderer(id='1221', ...)],</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">reset_policy&nbsp;=&nbsp;'standard',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">right&nbsp;=&nbsp;[],</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">sizing_mode&nbsp;=&nbsp;'fixed',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">subscribed_events&nbsp;=&nbsp;[],</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">syncable&nbsp;=&nbsp;True,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">tags&nbsp;=&nbsp;[],</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">title&nbsp;=&nbsp;Title(id='1183', ...),</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">title_location&nbsp;=&nbsp;'above',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar&nbsp;=&nbsp;Toolbar(id='1208', ...),</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar_location&nbsp;=&nbsp;'right',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">toolbar_sticky&nbsp;=&nbsp;True,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">visible&nbsp;=&nbsp;True,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">width&nbsp;=&nbsp;600,</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">width_policy&nbsp;=&nbsp;'auto',</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_range&nbsp;=&nbsp;DataRange1d(id='1185', ...),</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_scale&nbsp;=&nbsp;LinearScale(id='1189', ...),</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_range&nbsp;=&nbsp;DataRange1d(id='1187', ...),</div></div><div class="1294" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_scale&nbsp;=&nbsp;LinearScale(id='1191', ...))</div></div></div>
<script>
(function() {
  var expanded = false;
  var ellipsis = document.getElementById("1295");
  ellipsis.addEventListener("click", function() {
    var rows = document.getElementsByClassName("1294");
    for (var i = 0; i < rows.length; i++) {
      var el = rows[i];
      el.style.display = expanded ? "none" : "table-row";
    }
    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";
    expanded = !expanded;
  });
})();
</script>

# Some basic recommendations

- Python projects can require complex development or implementation; in those cases, it is recommended to use some integrated development environment (IDE), for example, PyCharm. 
- While the use of IDE is not discouraged, I feel that it isn't necessary for applied Data Science projects (as opposed to algorithm development, for example). 
- Moreover, to strengthen the encoding of the language syntax, I recommend at the beginning to force yourself to use Jupyter lab or notebook interface only.
- However, my current favourite tool is visual studio code, which has a nice Python and Jupyter built-in support.


# Module Recommendation

- This course's main objective is to familiarize you with the use of this framework  and Python, making the writing process intuitive.
- All slides you will see were created using Jupyter, and all code snippets can be copied and pasted into your notebook.
I urge you, however, to avoid doing so.
Instead, write down the commands in your notebook and try to add your own comments.
- Keep in mind that you are doing this for future you, who in six months will want to know how to do something trivial, and having it in one (or more) notebooks that you have created will be worthwhile 

# What this module will cover
- This module is composed of several sections going over Python essentials such as: 
    1. Python Basics (e.g. Syntax Essentials, Keywords, Variables, Data Types, Comments and Operators)
    1. Python Data Structures (e.g. Lists, Tuples, Dictionaries and Sets)
    1. Python Programming Fundamentals (Conditions, Loops and Functions)
- The goal is to quickly give you a cheat sheet to start using Python as a data scientist.

# Data science tools eco-system 

## Links to expand your understanding 

For those interested in learning more...

- [Conda Essentials](https://learn.datacamp.com/courses/conda-essentials)
- [Building and Distributing Packages with Conda](https://learn.datacamp.com/courses/building-and-distributing-packages-with-conda)
- [Some background on ipython and jupyter](https://www.datacamp.com/community/blog/ipython-jupyter)
- [Jupyter Notebook Tutorial: The Definitive Guide](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)


{% include links.md %}

