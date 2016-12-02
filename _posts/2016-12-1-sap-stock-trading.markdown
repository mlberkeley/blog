---
layout: post
title:  "SAP Hana Vora Stock Trading"
date:   2016-12-1
author: Raul Puric
comments: true
published: true
---

One crucial thing that all statistical learning algorithms need is the ability to handle a tremendous amount of data very quickly. Throughout the years, people have used different frameworks for querying, or fetching, data. Among these include Hadoop’s [MapReduce](https://en.wikipedia.org/wiki/MapReduce) framework and the Apache Spark framework. SAP Hana Vora’s (HV) unique in-memory hadoop query engine for MapReduce frameworks is a promising new tool for big data and performing analysis in a distributed fashion on large databases of information. We demonstrate HV’s potential as a powerful resource for ML by examining its performance on tasks such as stock prediction on market data. To this end we also contribute some additional functionality to the SAP HV library.


<!-- break -->

## Backtesting and Data set

#### Every algorithm needs testing...

We started off implementing our stock predictor in HV by focusing on what we’d be using to test and train our stock predictor. For testing this predictor we used quantopian’s backtesting platform for trading algorithms. Quantopian’s platform uses the quandl test dataset for testing, which contains trading history from prior years. An example of their basic dataset we used:

`{"timestamp":{"s":"1478808720"},"open":{"s":"90.5"},"symbol":{"s":"PM"},"volume":{"s":"32582"},"high":{"s":"90.5"},"low":{"s":"90.4499969482"},"close":{"s":"90.4499969482"}}`

#### It also needs a large amount of training data...

Quantopian additionally provides quandl’s training and validation datasets as a public resource for training. We’ve pulled the training dataset and stored it appropriately in a SQL Database. However, in order to augment our dataset for training purposes and ensure our stock predictor generalizes well to numerous situations, we set up a data scraper on Amazon Web Services (AWS) to continuously pull additional data from bloomberg’s services and expand our existing SQL DataBase of training data.

## The Stock Trading Algorithm

#### The technical details...

Recent approaches to creating stock predictor algorithms have leveraged [Deep Q Learning](https://deepmind.com/research/dqn/) to learn a function to approximate the profit (rewards) received from executing buy and sell operations given input data about stock history. We follow this approach; however, we relax the problem by learning a simpler linear function for these predictions (as opposed to a deep neural network). As input data we take current stock information and the output from the last time step; thus allowing us to leverage recurrence in the problem and build a simple linear model that also has an attention component.

{% capture algorithm %}
To this end we leverage HV’s map reduce capabilities to compute all our buy/sell actions in a training batch by computing a linear operation on all the input data (minus the recurrent information about the last action). After this we use a prefix sum scan (described in <link to Our contribution to Hana Vora md section>) to then perform the linear operation on the recurrent component of our input data. We then compute the resulting rewards and use this information to optimize our learned reward approximation function. Given large training batch sizes this allows us to have a training system that is able to efficiently distribute large amounts of computation. 
{% endcapture %}
{% include collapsible.html content=algorithm title="the algorithm" %}

## Our Contribution to Hana Vora

#### The Prefix Sum...

The prefix sum is an operation on a list of number defined by:

$$y_0 = x_0$$<br>
$$y_1 = x_0 + x_1$$<br>
$$y_2 = x_0 + x_1 + x_2$$<br>
$$...$$

For example:

$$PrefixSum([0,1,2,3,4]) = [0,1,3,6,10]$$

#### Parallelizing the prefix sum with HV

The prefix sum is an inherently serial algorithm so this can produce bottlenecks in performance even if all other computations are done in parallel with MapReduce. To this end we can leverage MapReduce with a graph-like computation structure (visualized below) to efficiently parallelize this inherently serial function. We implemented this algorithm with modularity in mind and it can be generalized to run on any kernel that can describe the reduce steps in the below algorithm (eg. it can work on prefix multiplication by swapping a x+y kernel with a x*y kernel). The parallelization of this algorithm is a relevant contribution to hana vora as it opens up the possibility for other functions such as an exponential moving average, markov models, etc.

## Performance Benchmarks

## Future Considerations

#### More ML functionality for HV...

As a whole HV performed well for parallelizing our analysis of stock data and training our stock predictor. Given the foundation that we have layed in this semester and more time we would like to leverage the timeseries capability with HV to produce more complex models for stock prediction utilizing techniques such as hidden markov models and markov chains. Additionally we’d like to try and implement resilient distributed dataset support (RDD) for HV so it can perform better on tasks such as matrix computation. Giving it a further edge as a tool for ML, and opening up more areas of exploration and use for HV.

