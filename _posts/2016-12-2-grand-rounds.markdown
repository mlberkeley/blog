---
layout: post
title:  "Grand Rounds Medical Data Classification"
date:   2016-12-1
author: Humza Iqbal
comments: true
published: true
---

Since our last demo day, the Grand Rounds team has been firing on multiple cylinders. Our team was provided a database of anonymous medicare information, including information on doctors visits. Associated with each visit were claim numbers, the total charge, the complication code (essentially the reason for the visit), and the NPI number (which indicates which physician was doing the procedure), among other information.

<!-- break -->

Given the total number of doctors visits in the US tops one billion a year, our dataset was massive. Simply querying it every time we wanted to do anything was just too slow. In order to facilitate easy access, we wrote scripts which would transfer all the features we needed to a PostgreSQL database which would be far more efficient to query. 

We’ve organized our goals into two projects. Our first project involved figuring out how to distinguish routine from non-routine doctor’s visits in the dataset. Grand Rounds initially wanted us to classify visits into three groups—”routine”, “unexpected”, and ”readmission”—but we decided to instead go with just “routine” and “non-routine.” We are currently working on implementing a classifier based on time difference between visits as well as using the cost of each visit. 

Our second project involved finding complications for surgeries based on our dataset. That project proved to be more difficult because we had a hard time figuring out exactly what our definition of a complication was, so instead we opted to simply group surgeries together based on how similar their data was to each other. The average of each group became a “centroids,” and surgeries that were far from all centroids were considered complications. Of course this raised problems itself when it came to the notion of “distance” because our dataset consisted of many different codes and procedures which can’t be easily quantified. For example, what exactly is the distance between one surgeon doing a surgery compared to another surgeon doing the same procedure? We considered one hot encoding but that became impractical when considering the large numbers of features. After checking in with Grand Rounds they recommended we use a time series based approach to see how surgeries vary over time, which we are currently implementing.



