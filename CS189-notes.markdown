---
layout: page 
title:  CS 189 Notes
date:   2017-8-26
author: Daniel
type: course-notes
comments: true
published: true 
hidden: true
permalink: /cs189-notes/
description: Notes for Berkeley's CS 189 - Introduction to Artificial Intelligence
---

These are Machine Learning at Berkeley's notes for the Fall 2017 session of Berkeley's CS 189 class, taught by Professor Anant Sahai and Stella Yu. These notes are _unaffiliated_ with the class, but as there (as of now) will not be any webcasts or class notes we thought it might be helpful if these existed.

These notes are fairly rigorous and assume a lot of background in mathematics, optimization, and probability. **If you've stumbled upon these notes and are looking for a gentler introduction to machine learning we suggest you check out our [crash course series](https://ml.berkeley.edu/blog/tutorials).**

All that being said, we are also just students with very busy lives. We'll try to keep this as updated and devoid of errors as possible. If you spot any mistakes please feel free to email us at [contact@ml.berkeley.edu](mailto:contact@ml.berkeley.edu).

<!-- break -->

## Contents

- [Lecture 1]({{ site.url }}{{ site.baseurl }}/cs189-notes/#lecture1)
- [Discussion 1]({{ site.url }}{{ site.baseurl }}/cs189-notes/#discussion1)
- [Lecture 2]({{ site.url }}{{ site.baseurl }}/cs189-notes/#lecture2)
- [Lecture 3]({{ site.url }}{{ site.baseurl }}/cs189-notes/#lecture3)

<div id="lecture1"></div>
## Lecture 1 (08-24-17)

### Administrative Stuff

* Office Hours are after lecture, 400 Cory right now but is subject to change

* Dicussions are on Friday _only_, all day, first come first serve. Check the [website](http://www.eecs189.org/calendar/) for full list of discussion times and locations

* There is no required book for the classes. However, _Elements of Statistical Learning_ is suggested if you're into that kind of stuff (textbooks that is...)

* The grading can be found [here](http://www.eecs189.org/syllabus/#grading)

* The midterm is Oct. 13, 7 pm

* The final is Dec. 14, 3 pm

### About the Course

* "This is an 'Advanced Upper Division Course'" quoth Sahai

* That means you should probably have taken (mastered) EE16A, EE16B, CS70, and math 53 for sure

* CS170, CS126, and CS127 would be nice to have as well, and your "maturity" (again quoth Sahai) should certainly be at the level of those classes

* This is not a programming class, although it is assumed you have knowledge of material taught in the CS61 series

* Python will be used in this course

### Course structure

* The structure of the course has shifted to be more conceptual and focuses a bit more on advanced neural network topics

    * First 8 lectures
        * Key ML Ideas in the context of (mostly linear) regression
    * Next 3 lectures
        * Introduce non-linear topics, including neural networks (!)
    * Next 3 lectures
        * Transition into classification
    * Next 8 lectures
        * More advanced classification topics (SVMs, the kernel trick, decision trees, ensemble methods, boosting)
    * Last 5-6 lectures
        * Unsupervised learning + Advanced Neural Networks

    * The class will unfortunately not have enough time to teach RL :(

### Levels of Abstraction in ML

Almost all ML problems are solved through these (increasingly detailed) levels of abstraction:

1. Application + Data: What you're trying to do and what the data looks like
    - ex. Have $$ (x_i, y_i) $$ observations of asteroid. Want to determine it's orbit

2. The model: What kind of pattern do we want to find
    - ex. We want to fit an ellipse to our observations of asteroid locations

3. Optimization problem: construct an optimization problem to find parameters for your model
    - ex. Turn "fitting an ellipse" into "minimize some error"

4. Optimization algorithms: how to actually optimize
    - ex. From least squares, we want to minimize 

    $$ \text{min}_{\vec{x}} \|A\vec{x}-\vec{b}\|^2 $$
    
    - To do this we solve the linear system $$ A^T Ax = A^Tb $$

### Ordinary Least Squares

Let's say we have a set of $$ n $$ data points $$ (\vec{a}_i, \vec{b}_i) $$, where $$ \vec{b}_i $$ is $$ m \times 1 $$ and $$ \vec{a}_i $$ is $$ l \times 1 $$. We believe $$ \vec{b}_i \approx X\vec{a}_i $$, where $$ X $$ is $$ m \times l $$. This is our model.

To turn this into an optimization problem we would like to have something of the form

$$ \text{min}_{\vec{x}} \|A\vec{x}-\vec{b}\|^2 $$

To do this we let $$ A $$ be the following (monstrosity of a) matrix

$$ A = \begin{bmatrix} \vec{a}_1^T & 0 & \cdots & \cdots & 0 \\ 0 & \vec{a}_1^T & 0 & \cdots & 0 \\ 0 &  & \ddots & & 0 \\ 0 &  & & 0 & \vec{a}_1^T \\ \vec{a}_2^T & 0 & \cdots & \cdots & 0 \\ 0 & \vec{a}_2^T & 0 & \cdots & 0 \\ 0 &  & \ddots & & 0 \\ 0 &  & & 0 & \vec{a}_2^T \\ \vdots & \vdots & \vdots & \vdots & \vdots \\ \vec{a}_n^T & 0 & \cdots & \cdots & 0 \\ 0 & \vec{a}_n^T & 0 & \cdots & 0 \\ 0 &  & \ddots & & 0 \\ 0 &  & & 0 & \vec{a}_n^T\end{bmatrix} $$

Note that the $$ \vec{a}_i^T $$'s are of $$ l \times 1 $$ dimensional and not just scalars. 

We let $$ \vec{x} $$ be 

$$ \vec{x} = \begin{bmatrix} x_{11} \\ x_{12} \\ \vdots \\ x_{1l} \\ x_{21} \\ \vdots \\ x_{ml} \end{bmatrix} $$

(pardon the pun)

So $$ \vec{x} $$ must be of dimension $$ ml \times 1 $$, and $$ A $$ is of dimension $$ mn \times ml $$ (pardon the pun again...)

Finally let $$ \vec{b} $$ be

$$ \vec{b} = \begin{bmatrix} \vec{b}_1 \\ \vec{b}_2 \\ \vdots \\ \vec{b}_n \end{bmatrix} $$

which is also $$ mn \times 1 $$ dimensional (note this is also a block matrix)

Now if we minimize $$ A \vec{x} - \vec{b} $$ we'll be actually be minimizing the sum of the squared errors for each data point (do the math!).

There are two approaches to minimizing this quantity. One involves vector calculus (which is the topic of discussion), and one involves projections. Let's look at projections.

In order to minimize the above, we note that $$ A\vec{x} - \vec{b} $$ must be perpendicular to the column space of $$ A $$. Thus we can write 

$$ A^T(\vec{b} - A \vec{x} ) = 0 $$

Then, rearranging we arrive at the normal equations

$$ A^TA\vec{x} = A^T\vec{b} $$

<div id="discussion1"></div>
## Discussion 1 (08-25-17)

* Discussion was on math review
* Make sure you know [vector](http://gwthomas.github.io/docs/math4ml.pdf) [calculus](https://d1b10bmlvqabco.cloudfront.net/attach/j2ji31rkkl2og/hzd1mb5ztzo3uu/j6wfutg2r97q/note0.pdf) (Thanks Garret Thomas and Jonathan Xia!)

<div id="lecture2"></div>
## Lecture 2 (08-29-17)

* Goal for today: Understand features
* Motivating example: Kepler's Laws

Gauss wanted to fit an ellipse to the orbit of Ceres and developed the least squares method to do this (the actual [story](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss#Astronomy) is a bit more complicated). 

You'll notice though that an ellipse is obviously not a line (which is usually what least squares predicts). To fit an ellipse, of the form

$$ a_0 + a_1x^2 + a_2y^2 + a_3xy + a_4x + a_5y = 1 $$

To do this we will cast this as an optimization problem

$$ \min || X \vec{a} - \vec{b} ||^2 $$

We let $$ X $$ be

$$ 
X =
\begin{bmatrix} 1 & x_1^2 & y_1^2 & x_1y_1 & x_1 & y_1 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
1 & x_n^2 & y_n^2 & x_ny_n & x_n & y_n \\
\end{bmatrix} $$

and $$ b $$ be

$$ 
b =
\begin{bmatrix} 1 \\
\vdots \\
1 \\
\end{bmatrix} $$

Solving this system of linear equations will give us the best coefficients for the ellipse equation. We call each column of the $$ X $$ matrix a **feature**.

> "A feature is anything we think of as a building block for the model. Something we believe would be useful to model"
> -- Sahai

The formal definition of a linear model of features is

$$ y = \sum_{i=1}^p \alpha_i \phi_i(\vec{x}) $$

where $$ \alpha_i $$ are the weights for our linear regression and $$ \phi_i $$ are our features. For example, in the ellipse example our features were

$$ \begin{align*}
\phi_0(\vec{x}) &= 1 \\
\phi_1(\vec{x}) &= x^2 \\
\phi_2(\vec{x}) &= y^2 \\
\phi_3(\vec{x}) &= xy \\
\phi_4(\vec{x}) &= x \\
\phi_5(\vec{x}) &= y \\
\end{align*} $$

Many fields have domain specific features that are commonly used. For instance, in computer vision [kernels](https://en.wikipedia.org/wiki/Kernel_(image_processing) are very popular for things like edge detection. But in general, monomials are very good features to use (as in the ellipse example). This is because any function can be [approximated](https://en.wikipedia.org/wiki/Taylor_series) as a sum of monomials (aka, a polynomial). This idea also extends into functions of multiple variables. 

### Curse of dimensionality

The number of monomials of at least degree $$ D $$ in an $$ l $$ dimensional space grows very quickly (for an in depth explanation check out Prof Sahai's [post](https://piazza.com/class/j2ji31rkkl2og?cid=85)). For this reason, as our model increases in degree or the dimensionality of the space that our data comes from increases, we need more and more data to create a good model. This is called the curse of dimensionality and is a problem that has plagued Machine Learners throughout space and time.

### Training vs Test Error

* So how do we determine which degree polynomal to use for a given dataset?
* We should compare training error against test error!

The training error is the error (the thing we're trying to minimize) on the data that we're learning on. This error should always decrease as the degree of the polynomial we're using increases. However, when we calculate the error on a test set (some data that wasn't used to train the model), the error should go down with increasing comlexity at first, and then go up. 

Why does test error go down then up with increasing polynomial degree? Well, at first the model's degree is too low to model anything very accurately. As we approach the correct degree polynomial to use the test error should go down. But as we use larger and larger degree polynomials our model will fit the _noise_ of the training data too much. It will overfit the training data and will do poorly on the test data. 


### Visualization

_In this visualization we fit a model (the blue line) to some noisy sine wave data (the red crosses). This model is actaully locally-weghted linear regression, not polynomial regression, but the idea is the same. By moving the slider you can manually adjust the complexity of the model and watch as it overfits and underfits._

_The bottom graph shows the training error and the test error in red and blue respectively. Notice how training error always decreases with increasing complexity, but test error will hit a minimum at some point, and then increase after that. Notice that just minimizing the training error does not work! You can tell whether you are underfitting or overfitting by looking at the training and test errors._

<center>
<div id="visual" style="position: relative;">
<div id="predBox" style="position: relative; overflow: hidden;">
  <img src="{{ site.baseurl }}/assets/tutorials/4/figures.png" id="predPic" style="max-width: none; position: absolute;">
</div>

<div align="center">
<input id="image_id" type="range" style="width: 300px" oninput="showVal(this.value)" max="96" min="1" start="50" />
</div>

<p id="test" style="text-align: center;">Complexity: .5</p>
<div id="errBox" style="position: relative; overflow: hidden;">
  <img src="{{ site.baseurl }}/assets/tutorials/4/errors.png" id="errPic" style="max-width: none; position: absolute;">
</div>
</div>
<p style="font-size: 16px;">
The top image shows models of different complexities fitting to the data. The bottom image shows the the error of the model against a training set (<span style="color: red">red</span>) and a test set (<span style="color: blue">blue</span>). 
</p>
<p style="font-size: 16px;">
We are training a locally-weighted linear regression model (LOWESS) on a sine wave with added gaussian noise. The smoothing parameter is used as a proxy for complexity, and the error is the average L1 distance. Credits to <a target="_blank" href="https://gist.github.com/agramfort/850437">agramfort</a> for the LOWESS implementation. You can find the code that generated this visualization <a target="_blank" href="https://github.com/dangeng/bias-variance-tradeoff">here</a>. This visualization was originally featured on our <a target="_blank" href="https://ml.berkeley.edu/blog/2017/07/13/tutorial-4/">crash course series</a>
</p>
</center>

### Almost Singular Matrices

Under certain conditions $$ A^TA $$ may be almost singular. What does this mean? Well... according to Sahai:

> "When in doubt, look at the eigenstructure" --Sahai

The SVD of an almost singular matrix will have entries in it's diagonal matrix with very very small numbers (a singular matrix will have 0's in it's diagonal). So when we take the inverse of an almost singular matrix we have a matrix whose values blow up (because the inverse of a diagonal matrix is the diagonal matrix with it's diagonal entries inverted).

To prevent things from blowing up we can modify the optimization problem that we're trying to solve. Instead of optimizing

$$ \min_{\vec{x}} \|A\vec{x}-\vec{b}\|^2 $$

we should optimize the error 

$$ \min_{\vec{x}} \|A\vec{x}-\vec{b}\|^2 + \lambda^2 \|\vec{x}\|^2 $$

The intuition behind this is that if $$ \vec{x} $$ gets too large, then the thing we're trying to minimize will get very large as well. By adding the $$ \lambda^2 \|\vec{x}\|^2 $$ penalty term we can essentially force $$ \vec{x} $$ to be small. The larger $$ \lambda $$ (called the "penalty") is, the smaller $$ \vec{x} $$ will be. 

This is called **ridge regression**.

It turns out we can actually find a closed form solution to the ridge regression optimization problem. We can find this through a little vector calculus. We first expand out our cost function and call it $$ C $$

$$
\begin{align*}
    ||\vec{y} - X\vec{w}||^2_2 + \lambda ||\vec{w}||^2_2 &= (\vec{y} - X\vec{w})^T(\vec{y} - X\vec{w}) + \lambda \vec{w}^T\vec{w} \\
    &= y^Ty - 2y^TXw + w^TX^TXw + \lambda w^Tw \\
    &= C \\
\end{align*}
$$

And then take the derivative and set it to zero

$$
\begin{align*}
    \frac{\partial C}{\partial \vec{w}} &= -2y^TX + 2w^TX^TX + 2\lambda w^T \\
    &= 2w^T(X^TX + \lambda I) - 2y^TX \\
    &= 0
\end{align*}
$$

_(Note that the derivative is a [row vector](https://en.wikipedia.org/wiki/Linear_form)!)_

Solving for $$ w $$, we get

$$
\begin{equation*}
    w^T = y^TX(X^TX + \lambda I)^{-1}
\end{equation*}
$$

If we take the transpose we get the standard closed form solution

$$
\begin{equation*}
    w = (X^TX + \lambda I)^{-1}X^Ty
\end{equation*}
$$

<div id="lecture3"></div>
## Lecture 3

### Comments about the Course (from Prof Sahai)

* 189 wants you to know four things
    * Intuition behind core concepts
    * Understand math behind core concepts
    * How to use these concepts to solve problems
    * How to implement these concepts




<script src="{{ site.baseurl }}/assets/tutorials/4/visual.js"></script>
