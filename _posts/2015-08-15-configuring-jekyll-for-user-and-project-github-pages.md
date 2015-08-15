---
layout: post
title: Configuring Jekyll for User and Project GitHub Pages
tags: [github-pages, jekyll]
---

You've chosen a [Jekyll theme](http://drjekyllthemes.github.io/) and set it up for your [GitHub **user** page](https://help.github.com/articles/user-organization-and-project-pages/#user--organization-pages). It looks great and works well.

That theme is your favorite and you want to reuse the tweaks you added, so you apply it to your [GitHub **project** pages](https://help.github.com/articles/user-organization-and-project-pages/#project-pages), except there's a small problem: you're seeing a double-slash in the URL after the repository name, like http://you.github.io/your-project//a-post.html.

Or maybe there's a much larger problem: many of the in-site links are *broken* and yield a frustrating and confusing 404.

Jekyll was supposed to be fast and simple. What's going on? How can you fix it?

Usually, the defect comes from one or both of these:

1. Your settings -- incorrect data in `_config.yml`
1. The theme -- incorrect usage of Jekyll variables and Liquid tags

Once you tidy your settings and theme, your user and project pages will have normalized URLs and working links. Even better: they will Just Work™  if you use Jekyll locally to preview your sites.

## Configuring your site correctly

There is only one rule to follow for `_config.yml`:

### Rule 0a -- For a User site, `baseurl` must be empty

<table>
<tr>
  <td>Good</td>
  <td><code>baseurl:  </code></td>
</tr>
<tr>
  <td>Bad</td>
  <td><code>baseurl: /</code></td>
</tr>
</table>

### Rule 0b -- For a Project site, `baseurl` must begin with a slash

<table>
<tr>
  <td>Good</td>
  <td><code>baseurl: /your-repository</code></td>
</tr>
<tr>
  <td>Bad</td>
  <td><code>baseurl: your-repository</code></td>
</tr>
<tr />
<tr>
  <td>Bad</td>
  <td><code>baseurl: your-repository/</code></td>
</tr>
</table>

## Using Jekyll and Liquid correctly

There are thee rules to follow:

### Rule 1 -- Always concatenate Jekyll and Liquid tags

<table>
<tr>
  <td>Good</td>
  <td><code>{% raw %}href="{{ site.baseurl }}{{ post.url }}"{% endraw %}</code></td>
</tr>
<tr>
  <td>Bad</td>
  <td><code>{% raw %}href="{{ site.baseurl }}/{{ post.url }}"{% endraw %}</code></td>
</tr>
</table>

This removes the double-slash from your site's URLs.

### Rule 2 -- *(Almost)* Always start links with `{% raw %}{{ site.baseurl }}{% endraw %}`

<table>
<tr>
  <td>Good</td>
  <td><code>{% raw %}href="{{ site.baseurl }}{{ post.url }}"{% endraw %}</code></td>
</tr>
<tr>
  <td>Bad</td>
  <td><code>{% raw %}href="{{ post.url }}"{% endraw %}</code></td>
</tr>
</table>

This fixes almost all of the in-site links. The next rule covers the remainder.

#### *Exception*: Start hyperlinks with `{% raw %}{{ site.url }}{{ site.baseurl }}{% endraw %}` in feed pages, like atom.xml.

Otherwise, feed readers and other aggregators, which rely on absolute URLs, won't be able to send subscribers to your pages.

### Rule 3 -- Always use a trailing slash after `{% raw %}{{ site.baseurl }}{% endraw %}`

<table>
<tr>
  <td>Good</td>
  <td><code>{% raw %}href="{{ site.baseurl }}/" title="Home"{% endraw %}</code></td>
</tr>
<tr>
  <td>Bad</td>
  <td><code>{% raw %}href="{{ site.baseurl }}" title="Home"{% endraw %}</code></td>
</tr>
<tr />
<tr><td colspan="2"/></tr>
<tr>
  <td>Good</td>
  <td><code>{% raw %}href="{{ site.baseurl }}/public/favicon.ico"{% endraw %}</code></td>
</tr>
<tr>
  <td>Bad</td>
  <td><code>{% raw %}href="{{ site.baseurl }}public/favicon.ico"{% endraw %}</code></td>
</tr></table>

This fixes links to resources.

## Checking your site for link correctness

A few simple `grep` searches from your repository's root will show you what you need to fix, and where.

### Find links that don't concatenate Jekyll tags

{% raw %}<pre>grep -r href=\" . | grep 'href=\"{{ *site\.b*a*s*e*url *}}/{' | grep -vE _posts\|_site</pre>{% endraw %}

If there are any hits, apply Rule 1 to your theme.

### Find links that don't use a Jekyll tag in their href

{% raw %}<pre>grep -r href=\" . | grep -v 'href=\"{{ *site\.b*a*s*e*url *}}' | grep -vE _posts\|_site</pre>{% endraw %}

The only hits should be to **external** resources and pages. If there are any that point *into* your site, you need to apply Rule 2 to your theme.

### Find links that don't use a trailing slash after `{% raw %}{{ site.baseurl }}{% endraw %}`

{% raw %}<pre>grep -r href=\" . | grep 'href=\"{{ *site\.baseurl *}}[^/{]' | grep -vE _posts\|_site</pre>{% endraw %}

If there are any hits, apply Rule 3 to your theme.

## Why are these the rules to follow?

Two reasons:

1. It's the way HTTP URLs work
1. It's the way Jekyll works

### HTTP URL structure

HTTP supports two kinds of links: absolute and relative. Absolute links have the full and complete URL, while relative links have a [variety of flavors](https://tools.ietf.org/html/rfc1808#section-5). The one that matters for us is the one that starts with a leading slash. When a browser is on a web page, it can assume that the target of a link with a leading slash is on the same host.

For example, assume a page at `http://host.com/folder/page.html` wants to link to `http://host.com/folder/other.html`, then these are equivalent href links:

* `<a href="http://host.com/folder/other.html">Go</a>`
* `<a href="/folder/other.html">Go</a>`

The browser will navigate to the *same* page when you click on *either* link.

### Jekyll's rendering behavior

Jekyll adopted this style of in-site linking, and so it uses leading-slash relative links whenever it populates variables like `page.url` (for more information, see my [previous post about exploring Jekyll variables]({{ site.baseurl }}{{ page.previous.url }})). So, when providing your own URL data, like `site.baseurl`, it's important that you use the same pattern. See [Jekyll's documentation for GitHub Pages](http://jekyllrb.com/docs/github-pages/#project-page-url-structure) if you want a second (and identical) opinion.

While there is much more behind [HTTP uniform resource locators](https://tools.ietf.org/html/rfc1738#section-3.3) than the simplification below, here are the basic building blocks, with a slightly different delineation so that they match Jekyll variable names.

#### User Page

<tt><strong><span style="color: #fff"><span style="background-color: #859900">&nbsp;http://you.github.io</span><span style="background-color: #6c71c4">/a-post.html&nbsp;</span></span></strong></tt>

* <tt><strong><span style="background-color: #859900; color: #fff">&nbsp;site.url&nbsp;</span></strong></tt>
* <tt><strong><span style="background-color: #6c71c4; color: #fff">&nbsp;page.url&nbsp;</span></strong></tt>

<table>
<tr>
  <th>Violation</th>
  <th>Result</th>
</tr>
<tr>
  <td>Rule 0<br /><em>bare slash</em></td>
  <td>http://you.github.io//a-post.html</td>
</tr>
<tr>
  <td>Rule 1<br /><em>slash between tags</em></td>
  <td>http://you.github.io//a-post.html</td>
</tr>
<tr>
  <td>Rule 2<br /><em>missing site.baseurl</em></td>
  <td><code>href="/a-post.html"</code> <em>(!)</em></td>
</tr>
<tr>
  <td>Rule 3<br /><em>no slash after tag</em></td>
  <td><code>href="" title="Home"</code> <em>(x)</em></td>
</tr>
<tr>
  <td>Rule 3<br /><em>no slash after tag</em></td>
  <td><code>href="/public/favicon.ico"</code> <em>(!)</em></td>
</tr>
</table>

Notes:

* **!** -- This result is still correct, but only for a User page. This is a *false positive* and many theme authors incorrectly assume that this "correct" result is also correct for Project pages.
* **x** -- This is an abnormal relative URL, and browsers will navigate to the [current page](https://tools.ietf.org/html/rfc1808#section-5.2).

#### Project Page

<tt><strong><span style="color: #fff"><span style="background-color: #859900">&nbsp;http://you.github.io</span><span style="background-color: #2aa198">/your-project</span><span style="background-color: #6c71c4">/a-post.html&nbsp;</span></span></strong></tt>

* <tt><strong><span style="background-color: #859900; color: #fff">&nbsp;site.url&nbsp;</span></strong></tt>
* <tt><strong><span style="background-color: #2aa198; color: #fff">&nbsp;site.baseurl&nbsp;</span></strong></tt>
* <tt><strong><span style="background-color: #6c71c4; color: #fff">&nbsp;page.url&nbsp;</span></strong></tt>

<table>
<tr>
  <th>Violation</th>
  <th>Result</th>
</tr>
<tr>
  <td>Rule 0<br /><em>no leading-slash</em></td>
  <td>http://you.github.ioyour-project/a-post.html <em>(404)</em></td>
</tr>
<tr>
  <td>Rule 0<br /><em>trailing-slash</em></td>
  <td>http://you.github.ioyour-project//a-post.html <em>(404)</em></td>
</tr>
<tr>
  <td>Rule 1<br /><em>slash between tags</em></td>
  <td>http://you.github.io/your-project//a-post.html</td>
</tr>
<tr>
  <td>Rule 2<br /><em>missing site.baseurl</em></td>
  <td><code>href="/a-post.html"</code> <em>(404)</em></td>
</tr>
<tr>
  <td>Rule 3<br /><em>no slash after tag</em></td>
  <td><code>href="/your-project" title="Home"</code> <em>(404)</em></td>
</tr>
<tr>
  <td>Rule 3<br /><em>no slash after tag</em></td>
  <td><code>href="/your-projectpublic/favicon.ico"</code> <em>(404)</em></td>
</tr>
</table>
