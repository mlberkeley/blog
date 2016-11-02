---
layout: post
title:  "Tutorial Post #1"
date:   2016-10-25 12:30:17 -0700
comments: true
---

Machine learning (ML) has received a lot of hype recently, and not without good reason. It has already revolutionized fields from image recognition to healthcare. Yet to many, the subject seems incredibly complicated.

<!-- break -->

This post, the first in a series of ML tutorials, aims to make machine learning accessible to anyone willing to learn. We’ve designed it to give you a solid understanding of how ML algorithms work as well as provide you the tools to harness it in your projects. 

### What is Machine Learning?

At it’s core, machine learning is not a difficult concept to grasp. In fact, the vast majority of machine learning algorithms are concerned with just one simple task: drawing lines. In particular, machine learning is all about drawing lines through data. What does that mean? Let’s look at an example:

### Classification

Let’s say you’re a computer with a collection of apple and orange images. From each image, you can infer the color and size of a fruit, and for some reason you want to classify the fruit as either an apple or an orange. The first step in many machine learning algorithms is to obtain labeled training data. This means getting information about the properties of the fruit - such as the size and color of the fruit -  and a label - whether the fruit is an apple or an orange. For example, our labeled training data might look like this:

<center>
	<img src="{{ site.baseurl }}/assets/classification_no_line.png" width="500">
</center>

The red x’s are labeled  apples and the orange x’s are labeled oranges. As you’ll probably notice, apples seems to congregate on the left side of the graph, and oranges the right. This is because for the most part, apples are red and oranges are orange.

Now we get to draw lines! For this particular machine learning problem, our goal is to draw a line between the two labeled groups, called a decision boundary. The simplest decision boundary for our data might look something like this:

<center>
	<img src="{{ site.baseurl }}/assets/classification_straight.png" width="500">
</center>

Just a straight line between the two groups. However, much more complicated machine learning algorithms may end up drawing much more complicated decision boundaries such as this:

<center>
	<img src="{{ site.baseurl }}/assets/classification_wavy.png" width="500">
</center>

So we’ve drawn a line that separates apples and oranges in the training data. Our assumption now is that the same line we’ve drawn will be able to distinguish an apple from an orange for any kind of apple/orange image we give it. So if we were given an image of a fruit, represented by the blue X below, we could classify it based on the decision boundary we drew:

<center>
	<img src="{{ site.baseurl }}/assets/classification_testing.png" width="500">
</center>

Now this is the power of machine learning. We take some training data (labeled apple and orange images), run a machine learning algorithm which draws a decision boundary on the data, and then extrapolate what we’ve learned to completely new pieces of data. 


Of course, distinguishing between apples and oranges is quite a mundane task. However, we can apply this strategy to much more exciting problems, such as classifying tumors as malignant or benign, marking emails as spam or not spam, and analyzing fingerprints for a fingerprint scanner.

### Regression

What we just saw was one particular field of machine learning, called classification, which draws lines to separate data. We can also use draw lines that describe data to predict things, called regression.

Say we have some data. In particular, let's say we have the price of various houses versus their square footage. If we visualize the information as a graph, it looks something like this:

<center>
	<img src="{{ site.baseurl }}/assets/house_price_data.png" width="500">
</center>

Each of the X’s represents a different house with some price and some square footage. Notice that although there is variation in the data, there is also evidently a pattern there: as houses get bigger, they also become more expensive. So if someone were to give you a square footage, you could probably guess how much the house is worth. For example, given the size of a house, you could make a rough guess at what its price would be.

<center>
	<img src="{{ site.baseurl }}/assets/house_price_predictor.png" width="500">
</center>

Now we can generalize and ask, for any given square footage, how much will a house be worth? Of course, it would be very hard to get an exact answer. However, an approximate answer is much easier to get. And this is where we start drawing lines! Our goal is to draw a line, called a predictor, through the data so it’s as close as possible to each data point so that it predicts the price of a house from it’s square footage.

<center>
	<img src="{{ site.baseurl }}/assets/human_predictor_in_action.png" width="500">
</center>

In a sense, we can say that the predictor represents a bunch of “typical” houses. What we just did (drawing lines through a dataset), is called regression and is at the heart of machine learning. The only thing that changes between different machine learning techniques, is the way we draw the lines! As a great man once said: “machine learning is just glorified regression!”


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
