---
title: "Pretext Turns Paragraphs Into Data"
date: 2026-03-30
slug: pretext-is-a-layout-primitive
description: "I saw a few Pretext demos on X, installed it, and now I'm sending words to places they don't normally go."
tags: [engineering, typography, ui, javascript, blogging, pretext]
---

# Pretext Turns Paragraphs Into Data

Every now and then a library shows up and immediately makes makes you wonder why this wasn't invented sooner.

That was Pretext for me. At first I thought this was just another goofy frontend ui library, but then I realized **the paragraph is not text anymore, it's data.** 

Browsers are happy to render text.
They get weirdly cagey the second you ask follow-up questions.

How tall is this paragraph going to be?
How many lines did it wrap into?
What are the actual line breaks?
Can I flow this around something without doing CSS yoga at 1am?

Normally that’s where you end up measuring DOM nodes, forcing layout, and doing the kind of work that makes frontend feel like a low-level haunting.

Pretext’s pitch is much better.
You do a prep pass once, and then layout becomes something you can work with instead of something that only happened *to* you.

At the simple end:

```ts
import { prepare, layout } from '@chenglou/pretext'

const prepared = prepare(text, '16px Inter')
const { height, lineCount } = layout(prepared, width, 24)
```

That's already so sick.
No DOM poking. No “let me render this invisibly offscreen real quick.”

The more fun part is this:

```ts
import { prepareWithSegments, layoutNextLine } from '@chenglou/pretext'

const prepared = prepareWithSegments(text, '600 18px Inter')
```

Once you have that, you can lay text out a line at a time, with different widths as you go.
That’s the part that made the whole thing click for me.

That’s why there’s a weird little demo in the middle of this post.
Click the button and the words drop.
Click it again and they snap back into place.
Drag the portrait around and the paragraph re-forms around it.

We tokenize each word into a <span> and suddenly words become interactive.

Here’s the demo:

<div class="pretext-demo" data-pretext-demo="compare">
  <script type="application/json">
    {
      "text": "Pretext turns paragraph layout into reusable data. That makes it useful for message bubbles, draggable editorial layouts, and interfaces where text needs to react to things on the page instead of just sitting there.",
      "imageUrl": "/assets/profilepicture.jpg",
      "imageWidth": 172,
      "imageHeight": 228,
      "stageWidth": 680,
      "stageMinHeight": 460
    }
  </script>
</div>

The part I like most is that I didn’t have to turn my blog into a science project to do this.

The post still lives in plain markdown in [`jeftekhari/blog`](https://github.com/jeftekhari/blog).
My site still fetches that markdown remotely.
This article just drops in one embed marker and lets the site hydrate it.


A lot of frontend tools are interesting right up until the moment they ask you to reorganize your whole app around them.
Then suddenly you’re doing a “small experiment” that somehow requires three build steps, two wrappers, and a spiritual commitment.

Instead it's:
- `npm install @chenglou/pretext`
- add one little client-side hook
- put one weird toy in one blog post
- call it a day

Small enough to ship.
Useful enough to steal later.
Weird enough that I’ll remember it.

Also, separate from the toy factor, I think the actually-practical use cases are pretty obvious:
- shrink-wrapped message bubbles
- virtualization without dumb paragraph-height guesses
- scroll anchoring when text loads late
- editorial layouts that need text to route around an object
- catching overflow and wrapping problems before they hit prod

Anyway, I saw a few posts on X, installed the package, and now this post has gravity.
That feels like a solid use of an afternoon.

If you want to mess with it yourself:

```sh
npm install @chenglou/pretext
```

Start with `prepare()` and `layout()` if you just want measurements.
Jump to `prepareWithSegments()` and `layoutNextLine()` if you want to start doing the cursed/fun stuff.

Thats it, thats the blog post. Here's some cool projects people are working on that made me realize how cool the tool is.
