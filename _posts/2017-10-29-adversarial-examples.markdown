---
layout: post
title:  "Tricking Neural Networks"
date:   2017-10-29
author: Daniel Geng
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

Neural networks are incredibly versatile and have solved many previously intractable problems. But it turns out they are also very easy to fool. Given a machine learning model it is relatively simple to engineer inputs, called *adversarial examples* that “trick” the model. In fact it is so easy that we’ll show you how to create your own adversarial examples for a neural network trained on handwritten digits. We also have a Jupyter Notebook with all the code you need to start generating your own adversarial examples.

<!-- break -->

Here’s an example of an adversarial example. We start with an image of a panda on the left, which the network thinks with 57.7% confidence is a panda. We then add a very small amount of noise, and get an image that looks exactly the same to a human but that the network thinks with 99.3% confidence is a “gibbon.”

<center>
  <img src="{{ site.baseurl }}/assets/2017-10-31-adversarial-examples/goodfellow.png" class="img" style="">
  <div style="max-width: 70%;">
   <p style="font-size: 16px;">
From <a target="_blank" href="https://arxiv.org/abs/1412.6572">Explaining and Harnessing Adversarial Examples</a> by Goodfellow et al.
   </p>
  </div>
</center>

Adversarial examples are an incredibly important and active area of research right now As we move toward a future that incorporates more and more neural networks and deep learning algorithms in our daily lives, we have to be careful to remember that these models can be fooled very easily. Despite the fact that neural networks are to some extent biologically inspired and have near human capabilities in a wide variety of tasks, their method of operation is nothing like what a real biological creature would do. Neural networks can fail quite easily and catastrophically as we can see in the image above and as we’ll see in the rest of this post.

The existence of adversarial examples means that systems that incorporate deep learning models actually have a fairly high security risk. It’s not inconceivable to think up all sorts of scenarios involving adversarial examples. For instance, assassination by feeding adversarial examples to a self-driving car is a genuine possibility.

## Setup

This blog post complements a Jupyter Notebook that we wrote. You can clone the repo here. Instructions for how to install Jupyter can be found here. 
	||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\\

## Adversarial Examples on MNIST

We will be trying to trick a vanilla (no bells and whistles) feedforward neural network that was trained on the MNIST dataset. MNIST is a dataset of $$ 28 \times 28 $$ pixel images of handwritten digits. They look something like this:

MNIST IMAGE 

There are 50000 training images and 10000 test images. We first load up the pretrained neural network (by the way, the code blocks in this post is only a subset of the code in the notebook. We omit the boring stuff):

```python
with open('trained_network.pkl', 'rb') as f:  
    net = pickle.load(f)  
    
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
```



