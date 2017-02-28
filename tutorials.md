---
layout: page
title: Crash Course Series
---

Our crash course series is a series of tutorials that introduce both beginning and advanced readers to the fundamentals of machine learning and provide insight into how machine learning algorithms actually learn. 

<style>
li ul {
list-style-type: none;
}
</style>
{% for post in site.posts reversed %}
  {% if post.type == "tutorial" %}
  * {{ post.date | date_to_string }} &raquo; [ {{ post.title }} ]({{ site.baseurl }}{{ post.url }})
    * {{ post.description }}
  {% endif %}
{% endfor %}