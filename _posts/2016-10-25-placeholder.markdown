---
layout: post
title:  "Tutorial Post #1"
date:   2016-10-25 12:30:17 -0700
comments: true
---

Machine learning (ML) has received a lot of hype recently, and not without good reason. It has already revolutionized fields from image recognition to healthcare. Yet to many, the subject seems incredibly complicated.

This post, the first in a series of ML tutorials, aims to make machine learning accessible to anyone willing to learn. We’ve designed it to give you a solid understanding of how ML algorithms work as well as provide you the tools to harness it in your projects. 

### What is Machine Learning?

At it’s core, machine learning is not a difficult concept to grasp. In fact, the vast majority of machine learning algorithms are concerned with just one simple task: drawing lines. In particular, machine learning is all about drawing lines through data. What does that mean? Let’s look at an example:

### Classification

Let’s say you’re a computer with a collection of apple and orange images. From each image, you can infer the color and size of a fruit, and for some reason you want to classify the fruit as either an apple or an orange. The first step in many machine learning algorithms is to obtain labeled training data. This means getting information about the properties of the fruit - such as the size and color of the fruit -  and a label - whether the fruit is an apple or an orange. For example, our labeled training data might look like this:

<center>
	<img src="/assets/classification_no_line.png" width="500">
</center>

The red x’s are labeled  apples and the orange x’s are labeled oranges. As you’ll probably notice, apples seems to congregate on the left side of the graph, and oranges the right. This is because for the most part, apples are red and oranges are orange.

Now we get to draw lines! For this particular machine learning problem, our goal is to draw a line between the two labeled groups, called a decision boundary. The simplest decision boundary for our data might look something like this:

<center><div id="container"><canvas id="canvas1" style="border: 1px solid black;" width="300" height="300"></canvas></div></center>

<center><p id="cost">Cost: 100</p></center>

## Stuff for blogging

Use #,##,...,###### like html's <h1>,<h2>,... tags

> Comments/Quotes(?) sections

For **bold**, *italic*, and _**both**_, or ~~strikethrough~~

1. some
2. ordered
  * and
  * sub
3. lists

[Links](http://ml.berkeley.edu)

## Code snippets: ##

Ruby:
{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('David')
#=> prints 'Hi, David' to STDOUT.
{% endhighlight %}

Python:
{% highlight python %}
class cs61a:
  is_silly = True

  def __init__(self):
    self.is_silly = False

cs61a.is_silly
>>True

class.is_silly
>>>False

{% endhighlight %}

js:
{% highlight js %}
var adder = new Function("a", "b", "return a + b");

adder(2, 6);
// > 8
{% endhighlight %}

## Math

You can use TeX and LaTeX notation, MathML notation, AsciiMath notation, or a combination of all three within the same page.

### In-line mathematics

in line $$ax^2 + bx + c = 0$$ math

### Displayed mathematics

$$\sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6}$$

### Sample TeX From my ee16a hw8.tex:

$$\begin{equation*}
  h_{H_2O} = \frac{C_{known}(\frac{V_{in}}{V_o} - 1 \pm \sqrt{1 - 2\frac{V_o}{V_{in}}}) - \epsilon h_{tot}}{80\epsilon}
\end{equation*}$$
