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
