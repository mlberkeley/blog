---
layout: post
title: Exploring the Variables Created by Jekyll for Liquid
tags: [github-pages, jekyll, liquid]
---

Jekyll creates variables for Liquid from three locations:

1. Your site's `_config.yml` file ([read more](http://jekyllrb.com/docs/configuration/))
1. Other `yml` files in the `_data` folder ([read more](http://jekyllrb.com/docs/structure/))
1. YAML Front Matter in any other file ([read more](http://jekyllrb.com/docs/frontmatter/))

Which means in addition to the [standard variables](http://jekyllrb.com/docs/variables/), you can specify your own.

## **Site** Variables

The `site` variable is the root object for the site, represented as a Ruby hash. This is a very large object since it contains the entire site, its meta data, and other build-time artifacts, but here are some interesting variables from this site:

<table>
<tr>
  <th>Variable</th>
  <th>Contents</th>
  <th>Origin</th>
</tr>
<tr>
  <td>site.url</td>
  <td>{{ site.url }}</td>
  <td>User</td>
</tr>
<tr>
  <td>site.baseurl</td>
  <td>{{ site.baseurl }}</td>
  <td>User</td>
</tr>
<tr>
  <td>site.title</td>
  <td>{{ site.title }}</td>
  <td>User</td>
</tr>
<tr>
  <td>site.permalink</td>
  <td>{{ site.permalink }}</td>
  <td>User</td>
</tr>
<tr>
  <td>site.time</td>
  <td>{{ site.time }}</td>
  <td>Jekyll</td>
</tr>
</table>

## **Data** Variables

The `data` variable is the union of all of the `yml` files in the `_data` folder and is accessible from the `site` variable. Each `yml` file adds another object with the same name as its source file. If there are no data files, then this object is empty.

On this site, I use [ghpaghes-ghcomments](https://github.com/wireddown/ghpages-ghcomments) and [tags](https://github.com/wireddown/wireddown.github.io/tree/feature_tags), and each has its own file: `gpgc.yml` and `tags.yml`. Here are some interesting variables from them:

<table>
<tr>
  <th>Variable</th>
  <th>Contents</th>
  <th>Origin</th>
</tr>
<tr>
  <td>site.data.gpgc.repo_owner</td>
  <td>{{ site.data.gpgc.repo_owner }}</td>
  <td>User</td>
</tr>
<tr>
  <td>site.data.gpgc.repo_name</td>
  <td>{{ site.data.gpgc.repo_name }}</td>
  <td>User</td>
</tr>
<tr>
  <td>site.data.tags[0]</td>
  <td>{{ site.data.tags[0] }}</td>
  <td>User</td>
</tr>
</table>

## **Page** Variables

The `page` variable is the object for a page. Often, the variable is renamed to `post` by layout or include files. Here are interesting variables from this page:

<table>
<tr>
  <th>Variable</th>
  <th>Contents</th>
  <th>Origin</th>
</tr>
<tr>
  <td>page.layout</td>
  <td>{{ page.layout }}</td>
  <td>User</td>
</tr>
<tr>
  <td>page.title</td>
  <td>{{ page.title }}</td>
  <td>User</td>
</tr>
<tr>
  <td>page.date</td>
  <td>{{ page.date }}</td>
  <td>Jekyll</td>
</tr>
<tr>
  <td>page.path</td>
  <td>{{ page.path }}</td>
  <td>Jekyll</td>
</tr>
<tr>
  <td>page.id</td>
  <td>{{ page.id }}</td>
  <td>Jekyll</td>
</tr>
<tr>
  <td>page.url</td>
  <td>{{ page.url }}</td>
  <td>Jekyll</td>
</tr>
<tr>
  <td>page.tags</td>
  <td>{{ page.tags | join: ' ' }}</td>
  <td>User</td>
</tr>
</table>
