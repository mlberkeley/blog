---
layout: post
title:  "Tricking Neural Networks"
date:   2017-10-29
author: Daniel Geng and Rishi Veerapaneni
type: tutorial
comments: true
published: true
description: Simple Adversarial Examples for MNIST
---

<style>
  .img {
  width: 600px;
  }
  
</style>

Assassination by neural network. Sound crazy? Well, it might happen someday, and not in the way you may think. Of course neural networks could be trained to pilot drones or operate other weapons of mass destruction, but even an innocuous (and presently available) network trained to drive a car could be turned to act against its owner. This is because neural networks are extremely susceptible to something called **adversarial examples**.

<!-- break -->

Adversarial examples are inputs to a neural network that result in an incorrect output from the network. It’s probably best to show an example. You can start with an image of a panda on the left which the network thinks with 57.7% confidence is a “panda.” Since the panda category has the highest confidence out of all the categories, it concludes that the object in the image is a panda. Then by adding a very small amount of noise you can get an image that looks exactly the same to a human, but that the network thinks with 99.3% confidence is a “gibbon.”

<center>
  <img src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/goodfellow.png" class="img" style="">
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
From <a target="_blank" href="https://arxiv.org/abs/1412.6572">Explaining and Harnessing Adversarial Examples</a> by Goodfellow et al.
   </p>
  </div>
</center>

So just how would assassination by adversarial example work? Imagine replacing a stop sign with an adversarial example of it--that is, a sign that a human would recognize instantly but a neural network would not even register. Now imagine placing that adversarial stop sign at a busy intersection that you happen to know your assassination target will drive past. As the self driving car approaches the intersection the on-board neural network would fail to see the stop sign and continue right into on-coming traffic.

Now this might just be one convoluted and (more than) slightly sensationalized instance of how people could use adversarial examples for harm, but there are many more. For example, the iPhone X’s “Face ID” feature relies on neural nets to unlock the phone when it recognizes your face, and is therefore susceptible to adversarial attacks. People could construct images to bypass the Face ID security features. Other biometric security systems would be at risk and illegal or improper content could bypass neural-network-based content filters. The existence of adversarial examples means that systems that incorporate deep learning models actually have a very high security risk.


<center>
  <img src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/optical_illusion.jpg" class="img" style="width: 800px">
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
You can understand adversarial examples by thinking of them as optical illusions for neural networks. In the same way optical illusions can trick the human brain, adversarial examples can trick neural networks.
   </p>
  </div>
</center>

The above adversarial example with the panda is a **targeted** example. A small amount of noise was added to an image that caused a neural network to misclassify the image, despite the image looking exactly the same to a human. There are also **non-targeted** examples, which simply try to find _any_ input that tricks the neural network. This input will probably look like white noise to a human, but because we aren’t constrained to find an input that resembles something to a human the problem is a lot easier.

We can find adversarial examples for just about any neural network out there, even state-of-the-art models that have so-called “superhuman” abilities, which is slightly troubling. In fact, it is so easy to create adversarial examples that we will show you how to do it in this post. All the code and dependencies you need to start generating your own adversarial examples can be found in [this](https://github.com/dangeng/Simple_Adversarial_Examples) GitHub repo.

<center>
  <img src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/meme.jpg" class="img" style="">
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
    A meme, extoling the effectivness of adversarial examples
   </p>
  </div>
</center>

## Adversarial Examples on MNIST

_The code for this part can be found in the following GitHub repo (but downloading the code isn't necessary to understand this post):_

<div align="center" style="margin-bottom: 2em">
<a href="https://github.com/dangeng/Simple_Adversarial_Examples" class="button" target="_blank" style="text-align:center;">GitHub Repo</a>
</div>

Before we do anything we should first import the libraries we'll need.

```python
import network as network
import pickle
import mnist_loader
import matplotlib.pyplot as plt
import numpy as np
```

We will be trying to trick a vanilla feedforward neural network that was trained on the MNIST dataset. MNIST is a dataset of $$ 28 \times 28 $$ pixel images of handwritten digits. They look something like this:

<center>
  <img src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/mnist6.png" class="img" style="">
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
   6 MNIST images side-by-side
   </p>
  </div>
</center>

There are 50000 training images and 10000 test images. We first load up the pretrained neural network (which is stolen from [this](http://neuralnetworksanddeeplearning.com/) amazing website/book):

```python
with open('trained_network.pkl', 'rb') as f:  
    net = pickle.load(f)  
    
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
```

For those of you unfamiliar with pickle, it’s a way for python to serialize data (i.e. write to disk) in essence saving classes and objects. Using `pickle.load()` just opens up the saved version of the network.

To show that the neural network is actually trained we can write a quick little function:

```python
def predict(n):
    # Get the data from the test set
    x = test_data[n][0]

    # Print the prediction of the network
    print 'Network output: \n' + str(np.round(net.feedforward(x), 2)) + '\n'
    print 'Network prediction: ' + str(np.argmax(net.feedforward(x))) + '\n'
    print 'Actual image: '
    
    # Draw the image
    plt.imshow(x.reshape((28,28)), cmap='Greys')
```

This method chooses the $$ n^{th} $$ sample from the test set, displays it, and then runs it through the neural network using the `net.feedforward(x)` method. Here’s the output of a few images:

<div>

<img class="pixelated feedforward" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/feedforward/combined_0.png">
<img class="pixelated feedforward" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/feedforward/combined_1.png">
<img class="pixelated feedforward" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/feedforward/combined_2.png">
<img class="pixelated feedforward" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/feedforward/combined_3.png">
<img class="pixelated feedforward" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/feedforward/combined_4.png">
<img class="pixelated feedforward" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/feedforward/combined_5.png">
<img class="pixelated feedforward" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/feedforward/combined_6.png">
<img class="pixelated feedforward" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/feedforward/combined_7.png">
<img class="pixelated feedforward" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/feedforward/combined_8.png">
<img class="pixelated feedforward" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/feedforward/combined_9.png">

    <center>
        <button class="slideshow-button" onclick="plusDivs(-1, 'feedforward')">&#10094;</button>
        <button class="slideshow-button" onclick="plusDivs(+1, 'feedforward')">&#10095;</button>
    </center>
</div>

<center>
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
   The left side is the MNIST image. The right side plots the 10 outputs of the neural network, called activations. The larger the activation at an output the more the neural network thinks the image is that number.
   </p>
  </div>
</center>

So a bit about this pretrained neural network. It has 784 input neurons ($$ 28 \times 28 $$ pixels), one layer of 30 hidden neurons, and 10 output neurons (one for each digit). All it’s activations are sigmoidal, it’s output is a one-hot vector indicating the network’s prediction, and it was trained by minimizing the mean squared error loss.

Alright, so we have a trained network, but how are we going to trick it? We'll first start with a simple non-targeted approach and then once we get that down we'll modify it to work as a targeted approach. 

### Non-Targeted Attack

The idea is to generate some image that is designed to make the neural network have a certain output. For instance, say our goal label/output is 

$$ y_{goal} = 
\begin{bmatrix}
    0 \\
    0 \\
    0 \\
    0 \\
    0 \\
    1 \\
    0 \\
    0 \\
    0 \\
    0 \\
\end{bmatrix}
$$

That is, we want to come up with an image such that the neural network’s output is the above vector. In other words, find an image such that the neural network thinks the image is a 5 (remember, we're zero indexing). It turns out we can formulate this as an optimization problem in much the same way we train a network. Let’s call the image we want to make $$ \vec x $$. We’ll define a cost function as:

$$ C = \frac{1}{2} \| y_{goal} - \hat y (\vec x) \|^2_2 $$

where $$ \| \cdot \|^2_2 $$ is the square of the L2 norm. The $$ y_{goal} $$ is our goal label from above. The output of the neural network given our image is $$ \hat y (\vec x) $$. You can see that if the output of the network given our generated image $$ \vec x $$ is very close to our goal label, $$ y_{goal} $$, then the corresponding cost is low. If the output of the network is very far from our goal then the cost is high. Therefore, finding a vector $$ \vec x $$ that minimizes the cost $$ C $$ results in an image that the neural network predicts as our goal label. Our problem now is to find this vector $$ \vec x $$. 

Notice that this problem is incredibly similar to how we train a neural network, where we define a cost function and then choose weights and biases (a.k.a. parameters) that minimize the cost function. In the case of adversarial example generation, instead of choosing weights and biases that minimize the cost, we hold the weights and biases constant (in essence hold the entire network constant) and choose an $$ \vec x $$ input that minimizes the cost. 

To do this, we’ll take the exact same approach used in training a neural network. That is, we’ll use gradient descent! We can find the derivatives of the cost function with respect to the input, $$ \nabla_x C $$, using backpropagation, and then use the gradient descent update to find the best $$ \vec x $$ that minimizes the cost. 

Backpropagation is of course usually used to find the gradients of the weights and biases with respect to the cost, but in full generality backpropagation is just an algorithm that efficiently calculates gradients on a computational graph (which is what a neural network is). Thus it can also be used to calculate the gradients of the cost function with respect to the inputs of the neural network.

Alright, let’s look at the code that actually generates adversarial examples:

```python
def adversarial(net, n, steps, eta):
    """
    net : network object
        neural network instance to use
    n : integer
        our goal label (just an int, the function transforms it into a one-hot vector)
    steps : integer
        number of steps for gradient descent
    eta : integer
        step size for gradient descent
    """
    # Set the goal output
    goal = np.zeros((10, 1))
    goal[n] = 1

    # Create a random image to initialize gradient descent with
    x = np.random.normal(.5, .3, (784, 1))

    # Gradient descent on the input
    for i in range(steps):
        # Calculate the derivative
        d = input_derivative(net,x,goal)
        
        # The GD update on x
        x -= eta * d

    return x
```

First we create our $$ y_{goal} $$, called `goal` in the code. Next we initialize our $$ \vec x $$ as a random 784-dimensional vector. With this vector we can now start gradient descent, which is really only two lines of code. The first line `d = input_derivative(net,x,goal)` calculates $$ \nabla_x C $$ using backpropagation (the full code for this is in the notebook for the curious, but we’ll skip describing it here as it’s really just a ton of math. If you want a very good description of what backprop is (which is what `input_derivative` is doing) check out [this website](http://neuralnetworksanddeeplearning.com/chap2.html) (incidentally, the same place we got the neural network implementation from)). The second and final line of the gradient descent loop, `x -= eta * d` is just the update. We move in the direction opposite the gradient with step size `eta`.

Here are non-targeted adversarial examples for each class along with the neural network's predictions:

<div>

<h3 class="nontargetedLabel" style="text-align: center">Non-Targeted "0"</h3>

<img class="pixelated nontargeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/non_targeted/combined_0.png">
<img class="pixelated nontargeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/non_targeted/combined_1.png">
<img class="pixelated nontargeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/non_targeted/combined_2.png">
<img class="pixelated nontargeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/non_targeted/combined_3.png">
<img class="pixelated nontargeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/non_targeted/combined_4.png">
<img class="pixelated nontargeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/non_targeted/combined_5.png">
<img class="pixelated nontargeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/non_targeted/combined_6.png">
<img class="pixelated nontargeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/non_targeted/combined_7.png">
<img class="pixelated nontargeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/non_targeted/combined_8.png">
<img class="pixelated nontargeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/non_targeted/combined_9.png">

<center>
<button class="slideshow-button" onclick="plusDivs(-1, 'nontargeted'); document.getElementsByClassName('nontargetedLabel')[0].innerHTML = 'Non-Targeted &quot;' + (slideIndexDict['nontargeted'] - 1 + '&quot;');">&#10094;</button>
<button class="slideshow-button" onclick="plusDivs(+1, 'nontargeted'); document.getElementsByClassName('nontargetedLabel')[0].innerHTML = 'Non-Targeted &quot;' + (slideIndexDict['nontargeted'] - 1 + '&quot;');">&#10095;</button>
</center>

</div>

<center>
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
   The left side is the non-targeted adversarial exampele (a 28 X 28 pixel image). The right side plots the activations of the network when given the image.
   </p>
  </div>
</center>

Incredibly the neural network thinks that some of the images are actually numbers with a very high confidence. The "3" and "5" are pretty good examples of this. For most of the other numbers the neural network just has very low activations for every number indicating that it is very confused.

There might be something bugging you at this point. If we want to make an adversarial example corresponding to a five, then we want to find a $$ \vec x $$ that when fed into the neural network gives an output as close as possible to the one-hot vector representing "5". However, why doesn’t gradient descent just find an image of a "5"? After all, the neural network would almost certainly believe that an image of a "5" was actually a "5" (because it _is_ actually a "5"). I’ve thought about this for a bit and I have an idea (though somebody much smarter than I might have a better reason).

The space of all possible $$ 28 \times 28 $$ images is utterly massive. There are $$ 256^{28 \times 28} \approx 10^{1881} $$ possible different $$ 28 \times 28 $$ pixel black and white images. For comparison, a common estimate for the number of atoms in the observable universe is $$ 10^{80} $$. Taking the Buddhist view, if each atom in the universe contained another universe then we would have $$ 10^{160} $$ atoms. If each atom contained another universe whose atoms contained another universe and so on for about 23 times, then we would almost have reached $$ 10^{1881} $$ atoms. Basically, the number of possible images is mind-bogglingly huge.

And out of all these photos only an essentially insignificant fraction actually look like numbers to the human eye. Whereas given that there are so many images, a good amount of them would look like numbers to a neural network (part of the problem is that our neural network was never trained on images that _don't_ look like numbers, so given an image that doesn't look like a number the neural network's outputs are essentially random). So when we set off to find something that looks like a number to a neural network, we’re much more likely to find an image that looks like noise or static than to find an image that actually looks like a number to a human just by sheer probability.

### Targeted Attack

These adversarial examples are cool and all, but to humans they just look like noise. Wouldn’t it be cool if we could have adversarial examples that actually looked like something? Maybe an image of a ‘2’ that a neural network thought was a 5? It turns out that’s possible! And moreover, with just a very small modification to our original code. What we can do is add a term to the cost function that we’re minimizing. Our new cost function will be:

$$ C = \frac{1}{2} \| y_{goal} - \hat y (\vec x) \|^2_2 + \lambda \| \vec x - x_{target} \|^2_2 $$

Where $$ x_{target} $$ is what we want our adversarial example to look like. So what we’re doing now is we’re simultaneously minimizing two terms. The left term $$ \| y_{goal} - \hat y (\vec x) \|^2_2 $$ we’ve already seen. Minimizing this will make the neural network output $$ y_{goal} $$ when given $$ \vec x $$. Minimizing the second term $$ \lambda \| \vec x - x_{target} \|^2_2 $$ will try to force our adversarial image $$ x $$ to be as close as possible to $$ x_{target} $$ as possible (because the norm is smaller when the two vectors are closer), which is what we want! The extra $$ \lambda $$ out front is a hyperparameter that dictates which of the terms is more important. As with most hyperparameters we find after a lot of trial and error that .05 is a good number to set $$ \lambda $$ to. 

If you know about ridge regularization you might find the cost function above very very familiar. In fact, we can interpret the above cost function as placing a prior on our model for our adversarial examples. 

If you don't know anything about regularization, feel free to click on the blue box to find out more:

{% capture regularization %}

Regularization is a way to use information you already have (called a **prior**) to influence the results of your model. For an example, it is often cited that the most common name in the world is "Muhammad" (and all its variations). It is also often joked that if you had to guess a person's name then you should always guess "Muhammad" because you have the best chances of being right with that guess--probabilistically speaking. 

The joke is that you often have a lot more information to go off of. For instance, if you happen to realize the person is a girl then the probability that her name is "Muhammad" drops to basically zero. If you happen to notice the person has blue eyes and blond hair then the probability drops considerably as well. And if you happen to notice that the person is wearing a sticker saying "Hello I am Ben" then the probability of the person being named "Muhammad" is also basically zero.

The "prior" is the information you have before seeing the person. That is, your knowledge that "Muhammad" is the most common name in the world. But then when you actually see the person you have to update your guess. But you also don't completely forget about your prior. For example, if you could only see the person from across the room and couldn't get a good look at them then you would have to go off of your prior and probably guess "Muhammad." But if you came face to face and realized the person looked exactly like Anna Kendrick then you would probably guess her name was Anna Kendrick. Thus, we say that your observation _washed out_ the _effect_ of the prior. As a side note, your estimate of the probabilities using both the prior and the observations is called the "posterior" probabilities.

In machine learning, one common prior is the ridge prior. The ridge prior says that we expect our weights to have a small L2 norm. Mathematically, we can write a cost function that looks like

$$ C = \|Xw - y\|^2_2 + \lambda\|w\|^2_2 $$

notice that if $$ w $$ gets too large the second term in the cost function will also get very large. The $$ \lambda $$ let's us control the strength of our prior. If $$ \lambda $$ were very high then the effect of a large $$ w $$ would be amplified so optimization would look for smaller $$ w $$'s. This essentially increases the effect of the prior.If $$ \lambda $$ were small then the effect of the prior would be small.

Now check out our cost function for a targeted adversarial attack:

$$ C = \frac{1}{2} \| y_{goal} - \hat y (\vec x) \|^2_2 + \lambda \| \vec x - x_{target} \|^2_2 $$

Look familiar?

The second term on the right hand side acts as a prior. We're saying our prior assumption is that we want our image to look like $$ x_{target} $$, and then we want to find an image that will trick the neural network given this assumption. We can then tune $$ \lambda $$ to find the best value to set it to.

{% endcapture %}
{% include collapsible.html content=regularization title="Regularization"%}

The code to implement minimizing the new cost function is almost identical to the original code (we called the function `sneaky_adversarial()` because we're being sneaky by using a targeted attack. Naming is always the hardest part of programming...):

```python
def sneaky_adversarial(net, n, x_target, steps, eta, lam=.05):
    """
    net : network object
        neural network instance to use
    n : integer
        our goal label (just an int, the function transforms it into a one-hot vector)
    x_target : numpy vector
        our goal image for the adversarial example
    steps : integer
        number of steps for gradient descent
    eta : integer
        step size for gradient descent
    lam : float
        lambda, our regularization parameter. Default is .05
    """
    
    # Set the goal output
    goal = np.zeros((10, 1))
    goal[n] = 1

    # Create a random image to initialize gradient descent with
    x = np.random.normal(.5, .3, (784, 1))

    # Gradient descent on the input
    for i in range(steps):
        # Calculate the derivative
        d = input_derivative(net,x,goal)
        
        # The GD update on x, with an added penalty 
        # to the cost function
        # ONLY CHANGE IS RIGHT HERE!!!
        x -= eta * (d + lam * (x - x_target))

    return x
```

The only thing we’ve changed is the gradient descent update: `x -= eta * (d + lam * (x - x_target))`. The extra term accounts for the new term in our cost function. Let’s take a look at the result of this new generation method:

<div>

<h3 class="targetedLabel" style="text-align: center">Targeted "0" [x_target=6]</h3>

<img class="pixelated targeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/targeted/combined_0.png">
<img class="pixelated targeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/targeted/combined_1.png">
<img class="pixelated targeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/targeted/combined_2.png">
<img class="pixelated targeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/targeted/combined_3.png">
<img class="pixelated targeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/targeted/combined_4.png">
<img class="pixelated targeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/targeted/combined_5.png">
<img class="pixelated targeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/targeted/combined_6.png">
<img class="pixelated targeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/targeted/combined_7.png">
<img class="pixelated targeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/targeted/combined_8.png">
<img class="pixelated targeted" height="250px"  src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/targeted/combined_9.png">

<center>
<button class="slideshow-button" onclick="plusDivs(-1, 'targeted'); document.getElementsByClassName('targetedLabel')[0].innerHTML = 'Targeted &quot;' + (slideIndexDict['targeted'] - 1) + '&quot; [x_target=' + (slideIndexDict['targeted'] + 5) % 10 + ']';">&#10094;</button>
<button class="slideshow-button" onclick="plusDivs(+1, 'targeted'); document.getElementsByClassName('targetedLabel')[0].innerHTML = 'Targeted &quot;' + (slideIndexDict['targeted'] - 1) + '&quot; [x_target=' + (slideIndexDict['targeted'] + 5) % 10 + ']';">&#10095;</button>
</center>

</div>

<center>
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
   The left side is the targeted adversarial exampele (a 28 X 28 pixel image). The right side plots the activations of the network when given the image.
   </p>
  </div>
</center>

Notice that as with the non-targeted attack there are two behaviors. Either the neural network is completely tricked and the activation for the number we want is very high (for example the "targeted 5" image) or the network is just confused and all the activations are low (for example the "targeted 7" image). What’s interesting though is that many more images are in the former category now, completely tricking the neural network as opposed to just confusing it. It seems that making adversarial examples that have been regularized to be more “number-like” tends to make convergence better during gradient descent.

## Protecting Against Adversarial Attacks

Awesome! We’ve just created images that trick neural networks. The next question we could ask is whether or not we could protect against these kinds of attacks. If you look closely at the original images and the adversarial examples you’ll see that the adversarial examples have some sort of grey tinged background. 

<h3 class="compareLabel" style="text-align: center">Targeted Image</h3>

<div>
<img class="toggle pixelated diff" height="400px" onclick="plusDivs(-1, 'diff'); document.getElementsByClassName('compareLabel')[0].innerHTML = 'Original Image'; " src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/diff/original.png">
<img class="toggle pixelated diff" height="400px" onclick="plusDivs(-1, 'diff'); document.getElementsByClassName('compareLabel')[0].innerHTML = 'Targeted Image'; " src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/diff/targeted.png">
</div>

<center>
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
     An adversarial example with noise in the background. <b>Click on the image</b> to toggle between the original and the adversarial example.
   </p>
  </div>
</center>

One naive thing we could try is to use binary thresholding to completely white out the background:

```python
def binary_thresholding(n, m):
    """
    n: int 0-9, the target number to match
    m: index of example image to use (from the test set)
    """
    
    # Generate adversarial example
    x = sneaky_generate(n, m)

    # Binarize image
    x = (x > .5).astype(float)
    
    print "With binary thresholding: "
    
    plt.imshow(x.reshape(28,28), cmap="Greys")
    plt.show()
    
    print "Prediction with binary thresholding: " + 
        str(np.argmax(np.round(net.feedforward(x)))) + '\n'
    
    print "Network output: "
    print np.round(net.feedforward(x), 2)
```

Here's the result:

<h3 class="binaryLabel" style="text-align: center">Adversarial Image</h3>

<div>
<img class="toggle pixelated binary" height="300px" onclick="plusDivs(-1, 'binary'); document.getElementsByClassName('binaryLabel')[0].innerHTML = 'Binarized Image'; " src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/binary/adversarial.png">
<img class="toggle pixelated binary" height="300px" onclick="plusDivs(-1, 'binary'); document.getElementsByClassName('binaryLabel')[0].innerHTML = 'Adversarial Image'; " src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/binary/binary.png">
</div>

<center>
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
     The effect of binary thresholding on adversarial images on MNIST. The left side is the image and the right side is the output of the neural network. <b> Click on the image </b> to toggle between binarized and adversarial.
   </p>
  </div>
</center>

Turns out binary thresholding works! But this way of protecting against adversarial attacks is not very good. Not all images will always have an all white background. For example look at the image of the panda at the very beginning of this post. Doing binary thresholding on that image might remove the noise, but not without disturbing the image of the panda a ton. Probably to the point where the network (and humans) can’t even tell it’s a panda. 

<center>
  <img src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/binary_combined.png" class="img" style="">
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
        Doing binary thresholding on the panda results in a blobby image
   </p>
  </div>
</center>

Another more general thing we could try to do is to train a new neural network on correctly labeled adversarial examples as well as the original training test set. The code to do this is in the ipython notebook (be aware it takes around 15 minutes to run). Doing this gives an accuracy of about 94% on a test set of all adversarial images which is pretty good. However, this method has it's limitations. Primarily in real life you are very unlikely to know how your attacker is generating adversarial examples. 

There are many other ways to protect against adversarial attacks that we won't wade into in this introductory post, but the question is still an open research topic and if you're interested there are many great papers on the subject.

## Black Box Attacks

An interesting and important observation of adversarial examples is that they generally are not model or architecture specific. Adversarial examples generated for one neural network architecture will transfer very well to another architecture. In other words, if I want to trick your model I can create my own model and adversarial examples based off of it and then these same adversarial examples will most probably trick your model as well. 

This has huge implications as it means that it is possible to create adversarial examples for a completely black box model where we have no prior knowledge of the internal mechanics. In fact, a team at Berkeley managed to launch a [succesful attack](https://arxiv.org/pdf/1611.02770.pdf) on a commercial AI classification system using this method.

## Conclusion

As we move toward a future that incorporates more and more neural networks and deep learning algorithms in our daily lives we have to be careful to remember that these models can be fooled very easily. Despite the fact that neural networks are to some extent biologically inspired and have near (or super) human capabilities in a wide variety of tasks, adversarial examples teach us that their method of operation is nothing like how real biological creatures work. As we've seen neural networks can fail quite easily and catastrophically, in ways that are completely alien to us humans. 

Human understanding is centered on causality. We demand a cause for every effect, and not without reason. Evolutionarily it is useful to know what actions result in what outcomes. Striking flint on steel produces sparks that starts a fire? Useful information for our ancestors struggling to survive on the African savannah. We are so inherently intent on imposing causality on the world that almost every culture has created gods that explain away the unexplainable phenomena of the world. Why does it rain and thunder? Because Zeus is mad. Why does the Nile flood every year? Because _Hapi_, the Egyptian god of the Nile, made yearly pilgrimages to Egypt.

We do not completely understand neural networks and to attribute "reason" to them would be akin to the Greeks claiming the sun was Apollo riding his firey chariot across the sky every day. Adversarial examples should humble us. They show that although we have made great leaps and bounds there is still much that we do not know.



<script>
var slideIndexDict = {
    'feedforward': 1,
    'nontargeted': 1,
    'targeted': 1,
    'diff': 1,
    'binary': 1
};
showDivs(slideIndexDict["feedforward"], "feedforward");
showDivs(slideIndexDict["nontargeted"], "nontargeted");
showDivs(slideIndexDict["targeted"], "targeted");
showDivs(slideIndexDict["diff"], "diff");
showDivs(slideIndexDict["binary"], "binary");

function plusDivs(n, cls) {
    showDivs(slideIndexDict[cls] += n, cls);
}

function showDivs(n, cls) {
    var i;
    var x = document.getElementsByClassName(cls);
    if (n > x.length) {slideIndexDict[cls] = 1} 
    if (n < 1) {slideIndexDict[cls] = x.length} ;
    for (i = 0; i < x.length; i++) {
        x[i].style.display = "none"; 
    }
    x[slideIndexDict[cls]-1].style.display = "block"; 
}
</script>

<style>
img.pixelated {
    image-rendering: optimizeSpeed;
    image-rendering: -moz-crisp-edges;
    image-rendering: -o-crisp-edges;
    image-rendering: -webkit-optimize-contrast;
    image-rendering: pixelated;
    image-rendering: optimize-contrast;
    -ms-interpolation-mode: nearest-neighbor;

    display: block;
    margin: 0 auto;
    border-style: solid;
    border-width: 2px;
    margin-bottom: 1em;
}

img.toggle {
    -webkit-transition: opacity .01s; /* Safari */
    transition: opacity .01s;
    cursor: pointer;
}

img.toggle:hover {
    opacity: .8;
    cursor: pointer;
}

button.slideshow-button{
    border:none;
    display:inline-block;
    outline:0;
    padding:8px 16px;
    vertical-align:middle;
    overflow:hidden;
    text-decoration:none;
    color:inherit;
    background-color:#e5e5e5;
    text-align:center;
    cursor:pointer;
    white-space:nowrap;
    margin-bottom: 1em;
    border-radius: 2px;

    -webkit-transition: opacity .2s; /* Safari */
    transition: opacity .2s;
}

button.slideshow-button:hover {
    opacity: .6;
    cursor: pointer;
}

</style>
