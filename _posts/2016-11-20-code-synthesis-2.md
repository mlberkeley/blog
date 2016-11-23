---
layout: post
title:  "Code Synthesis Update 2"
date:   2016-11-20
author: Max Johansen
comments: true
published: true
---

Imagine being able to just *tell* a computer what you want it to do, rather than programming it and having to deal with annoying syntax, semicolons and debugging. For the last few months, the we, Code Synthesis team, have been working on just that. This is similar to the process of [automated theorem proving](https://en.wikipedia.org/wiki/Automated_theorem_proving), the process of using computers to solve proofs, which has recently experienced some [significant breakthroughs](https://github.com/tensorflow/deepmath) through the use of artificial neural networks. Automated theorem proving uses a description of the proof to derive a way to reach the desired end-product, This idea can be applied to our project where we use a description of a program to actually generate the code for that program.

<!-- break -->

To catch ourselves up with the state-of-the-art, we read the [MIT Prophet paper](http://groups.csail.mit.edu/pac/patchgen/). Prophet is a machine learning-based software package that suggests edits to programs based on bugfixes to similar errors in other software projects. The researchers used many standard machine learning techniques such as [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation), which mathematically approximates the explanation of an observation.

We’ve also been researching the use of [LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/), neural networks that essentially use "memories" of past experiences to solve problems, to use surrounding sentences to predict the intention. In addition, we’ve been investigating [thought vectors](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf), a way to turn natural language descriptions of snippets of code into mathematical objects. 

We’ve been using a dataset drawn from one of Berkeley's most popular introductory computer science courses, CS 61A, to analyze the mapping between assignment descriptions and student submissions. (We had to write a web [scraper](https://github.com/macsj200/scraper61a) to obtain the assignment descriptions). Luckily, the course stores whether each student’s code submission passed the requirements of the problem so that we are able to identify changes that students made to their code to solve the problem. This theoretically allows us to implement Prophet-like code suggestion techniques in concert with natural language description of bugfixes.

![Clusters]({{ site.baseurl }}/assets/2016-11-20-code-synthesis-2/image_0.png)

Learned natural language manifolds, colored using various clustering techniques.

![Thought vector table]({{ site.baseurl }}/assets/2016-11-20-code-synthesis-2/image_1.png)

We plan to adopt extra heuristic analysis methods, or observations of the code execution process, as well as add thought vector processing to augment Prophet’s approach. We believe that much insight can be obtained by analyzing the mapping between natural language descriptions of the code and the actual code itself.

