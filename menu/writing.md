---
layout: page
title: Writing
permalink: /writing
---

# All Posts

{% for post in site.posts %}
<div class="post-preview">
  <h2><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h2>
  <p class="post-meta">
    {{ post.date | date: "%B %-d, %Y" }}
    {% if post.author %} · {{ post.author }}{% endif %}
  </p>
  <p>{{ post.excerpt | strip_html | truncatewords: 50 }}</p>
</div>
{% endfor %}
