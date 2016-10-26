---
layout: post
title:  "Placeholder/Examples for you guys to go off of"
date:   2016-10-25 12:30:17 -0700
comments: true
---
Daniel and Shannon, this is a placeholder post.

We're rolling w Jekyll for the blog cuz I don't have time to build css for this Tuesday night deadline.

Here are some examples so u guys can ref:

## Stuff for blogging

Use #,##,...,###### like html's <h1>,<h2>,... tags

> Comments/Quotes(?) sections

For **bold**, *italic*, and _**both**_, or ~~strikethrough~~

1. some
2. ordered
  * and
  * sub
3. lists

[Links](ml.berkeley.edu)

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

$$\sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6}$$
