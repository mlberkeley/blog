---
layout: post
title:  "Grand Rounds Medical Data Classification"
date:   2016-12-7
author: Humza Iqbal
comments: true
published: true
---

Since our last demo day, the Grand Rounds team has been busy working on many aspects of the project. Grand Rounds provided our team a database of anonymous medicare information, including information about doctors visits. Associated with each visit were claim numbers, the total charge, the complication code (essentially the reason for the visit), and the NPI number (which indicates which physician was doing the procedure), as well as other information.

<!-- break -->

<center>
    <figure>
        <img src="{{ site.baseurl }}/assets/GR/GR-team.jpg" width="100%">
        <figcaption>The GR team: <br>
        (from left) Kunal Singh, Pranav Sekhar, Tatevik Stepanyan, Shrinu Sivakumar, Jaume Vives, Alvin Zhang, Humza Iqbal</figcaption>
    </figure>
</center>


Given the total number of doctors visits in the US tops one billion a year, our dataset was massive. Just accessing the data took too much time, so we wrote scripts which would transfer all the information we needed to a PostgreSQL database, making it faster and easier to access the information.

We’ve organized our goals into two projects. Our first project involved figuring out how to distinguish routine from non-routine doctor’s visits in the dataset. Grand Rounds initially wanted us to classify visits into three groups—”routine”, “unexpected”, and ”readmission”—but we decided to instead go with just “routine” (such as yearly checkups) and “non-routine" (such as emergency treatment for heart failure). We are currently working on implementing a classifier based on the amount of time between visits as well as using the cost of each visit. 

Our second project involved finding complications for surgeries from our dataset. That project proved to be more difficult because we had a hard time figuring out exactly what our definition of a complication was, so instead we opted to simply group surgeries together based on how similar their data was to each other. The average of each group became “centroids,” and surgeries that were far from all centroids were considered complications because they were anomalies. Of course, this raised problems itself when it came to the notion of “distance," or determining how different each visit was from each other, because our dataset consisted of many different codes and procedures which couldn't be quantified easily. For example, what exactly is the distance between one surgeon doing a surgery compared to another surgeon doing the same procedure? We considered [one hot encoding](https://en.wikipedia.org/wiki/One-hot), but that became impractical when considering the large number of features involved in each visit. Grand Rounds recommended that we use a time series-based approach to see how surgeries vary over time, which we are currently implementing.



