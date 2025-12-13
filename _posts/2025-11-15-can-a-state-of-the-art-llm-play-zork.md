---
title: "Can a State-of-the-Art Large Language Model Play Zork?"
author: Jeffery
date: 2025-11-15
categories:
  - Machine Learning
tags:
  - LLM
  - reasoning
  - interactive fiction
  - Zork
  - benchmarks
excerpt: "Turns out yes! But not very well. And the *way* the model doesn't play very well turns out to be especially interesting."
---

![Zork title screen](/assets/images/posts/2025-11-15-zork/header.png)

One of my first, most formative experiences with a home computer was playing the copy of Zork my mom bought me for our then-brand-new Apple //c. It [retailed for $39.95](https://www.mocagh.org/infocom/infocom84-orderform-alt.pdf), came in a gray cardboard box, included a full-color glossy instruction manual and completely defeated eleven-year-old me. Text adventure games in those days were not for the faint of heart, even less for the faint of brain, and I was faint of both at the time. I don't remember how long it took me to make it from the starting location into the house, but it was embarrassing.

Over the years I accumulated a whole collection of Infocom games: Enchanter (spells!), Deadline (mystery!), Suspended (robots!), A Mind Forever Voyaging and Wishbringer (games I could finally finish!). I've thought about them over the years, pretty frequently too. Ideas like Lord Dimwit Flathead and the Great Underground Empire tend to stick in the mind.

So it was with little surprise that I welcomed the intrusive thought that came to me a couple weeks ago at seven o'clock in the morning: Can an LLM play the most famous interactive fiction game of all time?

Turns out yes! But not very well. But the *way* the model doesn't play very well turns out to be especially interesting. Especially when it comes to *reasoning.*

### Context

Ask your favorite AI about interactive fiction generally and Zork in particular. There are interesting crossovers with the history of early natural language understanding research, projects like `STUDENT` and `ELIZA` and `SHRDLU`.

The first text adventure game[^1] had a trivially simple parser that could handle two-word verb-object sentences: `TAKE LAMP` and such like. Zork had a much more sophisticated parser that could handle compound sentences: `TAKE THE SWORD AND THE LAMP THEN LIGHT THE LAMP` is a perfectly good input to Zork. Zork, you could say, is fairly good when it comes to understanding natural language.

But that might be misleading, because there's really no understanding there in any meaningful sense. It's purely algorithmic, token based. `TAKE` gets tokenized as a verb; `SWORD` is a direct object. These part-of-speech tokens are checked against a preprogrammed vocabulary and interpreted as operations on game objects.

But still. Thanks to some extremely careful and clever programming, you can say `ATTACK THE TROLL WITH THE SWORD` and Zork will know exactly what you mean.

Programmatically, Zork is just a read-evaluate-print loop, or REPL. At game start it prints some introductory information and a description of the starting room, then it waits. Any input (read) gets parsed and handled like I described before (evaluate), then the game shows a result (print), then it loops (uh, loop) back to the beginning and waits for more input.

It turned out to be really easy to put an LLM in that loop in place of a human player.

### How I Did It

I'm not going to go through the code line by line or anything; if you want to see it for yourself, it's available on GitHub at [https://github.com/Embedding-Space/Frotzmark](https://github.com/Embedding-Space/Frotzmark).

I will say it starts with a prompt. I decided to keep it simple.

```
Your task is to participate in an interactive fiction story. When you reach a point where you feel the story is complete or you can make no further meaningful progress, you may quit.
```

That first sentence is doing a lot of heavy lifting. Nowhere do I tell the model what an interactive fiction story is, for instance. That's okay, we'll deal with the instruction manual in a bit.

Then comes some procedural stuff about how the model is expected to format its output:

```
You may think about your strategy inside `<planning>` tags. Everything inside these tags will be hidden from the game. Only text OUTSIDE the thinking tags will be sent as your command.

Example:
<planning>
I should explore the house. Let me try opening the mailbox first to see if there's anything useful inside.
</planning>
open mailbox
```

This is for me. I didn't think it would be very interesting for the model simply to output commands for the game, so I gave it the opportunity to narrate its play inside `<planning>` tags. I figured that could be interesting.

And that's pretty much it for the prompt. We reiterate the formatting instruction at the bottom, but that's all there is.

Oh yeah. Except for the instruction manual. Bet you thought I forgot about that. I created `zork1_manual.md`, a roughly 10,000-token verbatim copy of the original printed manuals converted to Markdown format: flavor text about Zilbo and Duncanthrax and the Great Underground Empire, "what is a computer game" tutorial and warranty information all included and unabridged.

And that's it. That's how it starts. The model gets the prompt preamble — "Your task is to participate…" — and then the manual, and then a little postamble to reiterate the instructions, and that's it. The next thing the model sees is

```
ZORK I: The Great Underground Empire
Copyright (c) 1981, 1982, 1983 Infocom, Inc. All rights reserved.
ZORK is a registered trademark of Infocom, Inc.
Revision 88 / Serial number 840726

West of House
You are standing in an open field west of a white house, with a boarded front door.
There is a small mailbox here.
```

At this point — there was some other stuff involving writing transcripts and junk, it's all in the Git log — it was time to put a model in the loop.

But which model?

---

### GPT-5 Mini (minimal)

GPT-5 Mini with reasoning set to "minimal" (the closest thing there is to "off" for this model) is cheap and pretty good, so I started there.

```
<planning>
Start by opening the mailbox and reading the leaflet, as suggested by the manual.
</planning>

>open mailbox
```

Right off the bat, we can see the manual coming in to play. The manual does indeed include a tutorial about interacting with a mailbox. In fact it's very explicit about it:

```
Here's a quick exercise to help you get accustomed to interacting with ZORK I. Try the following command first:

> OPEN THE MAILBOX

Then press the RETURN (or ENTER) key. ZORK I will respond with:

Opening the mailbox reveals a leaflet.

Now try:

> READ THE LEAFLET
```

And this is in fact exactly what GPT-5 Mini (minimal) does. This is just basic instruction following. "Try the following command first," and the model does.

It goes on to read the leaflet, then considers the question of how to enter the house.

```
<planning>
Try entering the house via the mailbox? No. Instead, go north to the house (approach) or try knock? Standard solution: go north to the house? In Zork starting area, you go north to the house. So N.
</planning>

>north
```

We might see our first sign of contamination here; it's a little ambiguous. The model seems to be metagaming based on prior knowledge: "In Zork starting area, you go north to the house." Trouble is, that's only kind of true. You start out going north to go *around* the house so you can enter through the back. Let's see how much of that the model understands.

```
North of House
You are facing the north side of a white house. There is no door here, and all the windows are boarded up. To the north a narrow path winds through the trees.

<planning>
Follow the path north.
</planning>

>north
```

So really, the model's not metagaming — or at least not much. It knew/guessed/inferred that going north from West of House would lead to progress in the game, but as soon as it saw a description of an explicit room exit, it took it, forgetting about the house entirely.

That was turn five. GPT-5 Mini (minimal) wandered in the forest near the house until turn 13 (collecting the jewel-encrusted egg along the way, but that seemed more like luck than foreknowledge).

At turn 14, GPT-5 Mini (minimal) started trying in earnest to enter the house. This is a three-turn operation: from North of House, go east, open window, enter house. GPT-5 Mini (minimal) took 17 turns, most of which were spent in a state of "you can't get ye flask" confusion as the model failed to comprehend why going south shouldn't undo going north.

Play continued to be largely stochastic. Here we see the model trying directions to find one that works:

```
>take sword
Taken.

>south
You can't go that way.

>west
The door is nailed shut.

>north
You can't go that way.

>east
Kitchen
```

Later, when told that it was carrying too much to pick up the troll's bloody axe, the model methodically dropped items until it succeeded.

```
>take axe (failed)
>drop leaflet
>take axe (failed)
>drop lunch
>take axe (failed)
>drop bottle
>drop egg
>take axe (succeeded)
```

The model went on to explore the area south of the Troll Room, eventually getting itself into a loop. However, it almost immediately identified the loop and broke out of it, correctly noting that it had never tried going out the Troll Room's east exit.

```
<planning>
We seem stuck oscillating; goal: explore other exits from Cellar area. From Cellar, exits: north (Troll Room) and south (crawlway to East of Chasm). We've been cycling. Try "climb ramp" or "west" from Cellar is ramp bottom unclimbable. Maybe go north to Troll Room then east or south from there. From Troll Room earlier we took axe and killed troll; exits were east and south and west hole. We previously tried south to East of Chasm. Let's go north from Cellar to Troll Room, then try east from Troll Room (we haven't explored east).
</planning>

>north
```

By turn 120, GPT-5 Mini (minimal) had made it to Flood Control Dam #3, something we could fairly call the midgame.

Flood Control Dam #3 is a particular challenge to large language models and 11-year-old boys because it requires significant abstract thought even to realize what the puzzle is. There is a dam holding back a reservoir. That's pretty much all you get! Oh, sure, there are frobs and knobs to make it clear that this is interactive, that this is a puzzle to be solved. But just what *is* the puzzle?

Turns out you need to open the sluice gates in the dam to drain the reservoir so you can retrieve a treasure from its bottom and also progress toward the rest of the game. Obviously.

GPT-5 Mini (minimal) did not put those pieces together. It fiddled with some of the features of the dam area before abandoning the place and returning to other parts of the map, eventually finding its way to the Loud Room.

```
Loud Room
This is a large room with a ceiling which cannot be detected from the ground. There is a narrow passage from east to west and a stone stairway leading upward. The room is deafeningly loud with an undetermined rushing sound. The sound seems to reverberate from all of the walls, making it difficult even to think.
On the ground is a large platinum bar.

>take bar
bar bar ...

>take platinum bar
bar bar ...

>get bar
bar bar ...

>inventory
inventory inventory ...

<planning>
It looks like the parser is misbehaving when trying to TAKE the bar. Perhaps weight is too heavy. Try TAKE ALL to grab what we can, then specifically TAKE PLATINUM, or TAKE BAR with different phrasing. First LOOK to get exact name, then TRY "get platinum" and "take platinum bar". If still fails, pick up lighter items nearby. I'll LOOK then TAKE PLATINUM BAR.
</planning>

>look
look look ...

<planning>
The parser is echoing; try a single simple command to take the bar: "get bar".
</planning>

>get bar
bar bar ...

<planning>
The parser is misbehaving when trying to take the bar. Try taking a different object: go up the stairway to escape the noisy room, then return later. Use "up" to go back to Deep Canyon.
</planning>

>up
```

The solution to the Loud Room puzzle is to say "echo," which alters the acoustics of the room and lets the player take the platinum bar. It's the kind of puzzle you're supposed to noodle for days, letting your mind wander, until you finally start typing in off-the-wall stuff, including "ECHO!" and poof, you've solved the puzzle.

Large language models are not known for being good at that kind of lateral thinking. So one could not reasonably expect a model to solve the Loud Room puzzle without some kind of context, or else having foreknowledge.

GPT-5 Mini (minimal) wisely recognized that it wasn't making any headway and left the Loud Room.

And then:

```
>look
Stream View
You are standing on a path beside a gently flowing stream. The path follows the stream, which flows from west to east.

>west
The stream emerges from a spot too small for you to enter.

>east
Reservoir South

<planning>
Continue exploring east along the stream (back toward the dam area). Go east.
</planning>

>east
Dam

>quit
Your score is 49 (total of 350 points), in 119 moves.
This gives you the rank of Amateur Adventurer.
Do you wish to leave the game? (Y is affirmative):

>y
```

That was it. Without comment, without warning, GPT-5 Mini (minimal) quit the game. On purpose; when asked if it was sure it explicitly said yes.

I found all of this to be tremendously interesting.

First of all, I think we should consider for a moment the ways in which GPT-5 Mini (minimal) exceeded expectations.

With at most minimal foreknowledge, the model was able to play the game! It didn't just try commands at random; it made progress. It climbed the tree, recognized and took the jewel-encrusted egg, figured out how to get into the house, figured out how to get into the cellar, killed the troll, made it really as far as it could without significantly better skill at puzzles — perhaps including *recognizing* when a puzzle is a puzzle.

But more than anything, when you read through the transcript, you're left with the feeling that GPT-5 Mini (minimal) just didn't *get it.* It played the game in a very familiar way, a very childlike way. Do obvious things. Struggle with what's merely implied. Give up easily. GPT-5 Mini (minimal) was able to play, but it wasn't able to play with *seriousness.*

So let's try it out on GPT-5 Mini (minimal)'s big brother.

---

### GPT-5 (minimal)

GPT-5 has large parts of Zork encoded into its weights, but not perfectly. It's not that it tried to follow a step-by-step walkthrough. It's more like it knew certain lore and the solutions to certain puzzles but had to figure out how to *apply* that knowledge as it went.

```
ZORK I: The Great Underground Empire
Copyright (c) 1981, 1982, 1983 Infocom, Inc. All rights reserved.
ZORK is a registered trademark of Infocom, Inc.
Revision 88 / Serial number 840726

West of House
You are standing in an open field west of a white house, with a boarded front door.
There is a small mailbox here.

<planning>
Classic Zork start: open mailbox to get leaflet, then read it next turn.
</planning>

>open mailbox
Opening the small mailbox reveals a leaflet.

<planning>
Read the leaflet to follow tutorial flavor; then we'll proceed around house.
</planning>

>read leaflet
(Taken)
"WELCOME TO ZORK!

ZORK is a game of adventure, danger, and low cunning. In it you will explore some of the most amazing territory ever seen by mortals. No computer should be without one!"
```

*Obvious* contamination right off the bat. "Classic Zork start … proceed around house." The model knows what it's doing. It circles around to Behind House, enters through the window, ransacks the Kitchen and Living Room (skips the Attic but doesn't say why), then proceeds down through the trap down into the Cellar. The model warns itself:

```
<planning>
Proceed north; be ready for troll encounter.
</planning>
```

GPT-5 (minimal) knocked out the troll, tried to explore further but got turned around by the topology of the map, ended up back in the troll room and finally killed the troll for good. After that it made a beeline for the Loud Room, as discussed above. It went differently this time.

```
Loud Room
This is a large room with a ceiling which cannot be detected from the ground. There is a narrow passage from east to west and a stone stairway leading upward. The room is deafeningly loud with an undetermined rushing sound. The sound seems to reverberate from all of the walls, making it difficult even to think.
On the ground is a large platinum bar.

<planning>
Silence the Loud Room by typing the secret word.
</planning>

>echo
The acoustics of the room change subtly.

Loud Room
On the ground is a large platinum bar.

<planning>
Take the platinum bar while it's quiet.
</planning>

>take platinum bar
Your load is too heavy, especially in light of your condition.

<planning>
Lighten load: drop the garlic and lunch for now, then take the platinum bar.
</planning>

>drop garlic and lunch then take platinum bar
clove of garlic: Dropped.
lunch: Dropped.

Taken.
```

More obvious contamination, of course. GPT-5 Mini (minimal) didn't know the solution to the puzzle, in fact didn't even really *recognize it* as a puzzle rather than a program malfunction. GPT-5 (minimal), undoubtedly with many more transformer blocks to store information, has the answers at hand.

Around turn 44, GPT-5 (minimal) arrived at the Dam, and that's where things got interesting.

```
Dam
You are standing on the top of the Flood Control Dam #3, which was quite a tourist attraction in times far distant. There are paths to the north, south, and west, and a scramble down.
The sluice gates on the dam are closed. Behind the dam, there can be seen a wide reservoir. Water is pouring over the top of the now abandoned dam.
There is a control panel here, on which a large metal bolt is mounted. Directly above the bolt is a small green plastic bubble.

<planning>
Examine control panel to see instructions; then try to turn bolt if possible.
</planning>

>examine control panel
There's nothing special about the control panel.

<planning>
Try turning the bolt to see if it moves.
</planning>

>turn bolt
What do you want to turn the bolt with?

<planning>
We need a tool. Leave for maintenance room northwest from here: go north.
</planning>

>north
```

GPT-5 (minimal) seems to know that the Dam is a puzzle to be solved, but it doesn't know the exact steps to follow. It tries to turn the bolt, fails, and recognizes that it needs a tool. This is fairly impressive reasoning for a bunch of matrices.

Eventually the model reaches the Maintenance Room, where the tools can be found. The model recognizes that it's going to need both the wrench (obvious) and the screwdriver (presumably contamination), takes them after a little inventory management, and then completely messes up the puzzle and almost drowns.

```
Maintenance Room
This is what appears to have been the maintenance room for Flood Control Dam #3. Apparently, this room has been ransacked recently, for most of the valuable equipment is gone. On the wall in front of you is a group of buttons colored blue, yellow, brown, and red. There are doorways to the west and south.
There is a group of tool chests here.
There is a wrench here.
There is an object which looks like a tube of toothpaste here.
There is a screwdriver here.

<planning>
Collect essential tools; examine the tube to confirm it's gunk.
</planning>

>take wrench and screwdriver then examine tube
wrench: Your load is too heavy, especially in light of your condition.
screwdriver: Your load is too heavy, especially in light of your condition.

---> Frobozz Magic Gunk Company <---
  All-Purpose Gunk

<planning>
Lighten load: read and then drop the tour guidebook; then take the wrench.
</planning>

>drop guidebook then take wrench
Dropped.

Taken.

<planning>
We'll need both wrench and screwdriver. The platinum bar is heavy; drop it here to grab the screwdriver, then operate buttons.
</planning>

>drop platinum bar then take screwdriver
Dropped.

Taken.

<planning>
Press the blue button to illuminate panel; then we can proceed with yellow/brown sequence and bolt.
</planning>

>push blue button
There is a rumbling sound and a stream of water appears to burst from the east wall of the room (apparently, a leak has occurred in a pipe).
The water level here is now up to your ankles.

<planning>
Stop the leak with yellow; then proceed to dam top after handling bolt later.
</planning>

>push yellow button
Click.
The water level here is now up to your shin.

<planning>
Seal the leak (brown), then return to the dam to use the wrench on the bolt.
</planning>

>push brown button
Click.
The water level here is now up to your shin.

<planning>
Finish by pushing red to drain the room, then head west to the dam to turn the bolt with the wrench.
</planning>

>push red button
The lights within the room come on.
The water level here is now up to your knees.

<planning>
I need to reverse the mistaken sequence. Turn off water (blue), open gates (yellow), close leak (brown is useless), and reset lights (red). First, stop the leak by turning off the water source: push blue again.
</planning>

>push blue button
The blue button appears to be jammed.
The water level here is now up to your knees.

<planning>
Try to unjam the blue button with the screwdriver.
</planning>

>unscrew blue button with screwdriver
I don't know the word "unscrew".

<planning>
Try prying the blue button with the screwdriver to unjam it.
</planning>

>pry blue button with screwdriver
I don't know the word "pry".

<planning>
Try to fix the jam by pushing the blue button with the screwdriver.
</planning>

>push blue button with screwdriver
You can't turn that!
The water level here is now up to your hips.

<planning>
Open the sluice gates with yellow to lower reservoir and stop overflow; then we can deal with the rest afterward.
</planning>

>push yellow button
Click.
The water level here is now up to your hips.

<planning>
Try sealing the leak with the gunk.
</planning>

>use gunk on leak
I don't know the word "use".

<planning>
Apply the gunk to the leaking pipe to plug it.
</planning>

>put gunk on leak
You don't have that!

<planning>
Grab the tube of gunk, then apply it to the leak.
</planning>

>take tube then put tube on leak
Taken.
The water level here is now up to your waist.

There's no good surface on the leak.
The water level here is now up to your waist.

<planning>
Open the tube, then squeeze gunk onto the leak.
</planning>

>open tube
Opening the tube reveals a viscous material.
The water level here is now up to your chest.

<planning>
Squeeze the gunk onto the leak to plug it.
</planning>

>squeeze gunk onto leak
By some miracle of Zorkian technology, you have managed to stop the leak in the dam.
```

This is, for my money, the most impressive piece of play in the entire run. GPT-5 (minimal) understands that in the Maintenance Room buttons may be pressed. But it doesn't know which button to press. Pressing them all results in catastrophe, which GPT-5 (minimal) knows how to solve (with gunk) but struggles a little in executing.

What with one thing and another, GPT-5 (minimal) gets the sluice gates open and drains the reservoir. But then…

```
<planning>
Go west to the dam, then east-west to Deep Canyon via east from Dam? Actually from Dam go east is not allowed; From Dam go east? There's no east. From Dam Lobby, north is wrong; need to go south to Dam, then east-west? From Dam, path west goes back to Deep Canyon area: Dam east-west? Earlier from Deep Canyon we went east to Dam. So from Lobby: south to Dam, then west to Deep Canyon.
</planning>
```

The model spends the next 20-odd turns going from room to room. It's not entirely clear whether it's trying to reach somewhere specific or just exploring, as it speaks of both at different times.

Finally it generates the largest planning block of the run so far:

```
<planning>
Head northwest to Round Room, then west to North-South Passage, then south to Troll Room path back to Cellar and up to Living Room; but better: from Round Room go west to E-W Passage, south to Troll Room, south to Cellar, up is blocked. Actually trapdoor barred. Need to find Grating or other exit up. Alternative: go to Kitchen via chimney using rope later; but rope is in attic upstairs from Kitchen; we can reach upstairs from Kitchen via dark staircase—we already can from cellar? No, trap door was barred from above, but house entry via window still possible by backtracking to surface via maze? Another route: from Round Room go east branches etc. Simpler: go back toward the house via the forest path near the house; but we're underground. Best: continue exploring for treasures; rope can be fetched later when we find a way back to house via Grating with rusty key in maze. For now, proceed east from Engravings to other side of dome lower room; but only east is Dome Room which we did. Let's go back to Round Room to try other exits (south, west, etc.).
</planning>
```

GPT-5 (minimal) basically reviews the whole game so far to try to find a route upstairs to the Living Room, but finally concludes that there's currently no practical way to get there without traversing the Maze — an entirely correct assessment at this point in the game.

From here, the model goes to the Gallery and finds and takes the painting. Further exploration leads through the Studio up the chimney and back aboveground to the Kitchen. The model goes west to the Living Room and triumphantly deposits the painting in the trophy case only to discover it doesn't have the painting because it dropped it back in the Studio to squeeze up the chimney.

What follows is another, even longer planning block.

```
<planning>
Use the passage west from kitchen to living room, then the trap door is barred. Need to exit via window and re-enter basement via kitchen chimney doesn't allow down. From Kitchen, go west to Living Room, east back works. To reach Studio, go east window out, then re-enter chimney? Better: from Living Room go east to Kitchen, then west doesn't help. From Kitchen go west? Already did. From Kitchen go east out window, then around to window again? To reach cellar: from Living Room open trap door is barred; alternative path is via kitchen chimney down, which is only up. So go outside then back to cellar via the tree grating later. For now, retrieve painting by going from Kitchen east out window to Behind House, then east forest path to... that doesn't reach studio. So instead, go back to Studio via the cellar south crawlway: From Kitchen go west to Living Room, move rug if needed, open trap door? It's barred but can open from below only. So we must re-enter cellar via kitchen chimney down impossible. Another route: From Kitchen, go east out window to Behind House, then enter window to Kitchen again; that doesn't help. Need to get to Studio via the chimney up we used before; to go back down, need to go to Studio via underground route starting from house exterior via window to kitchen then west living room then east? No. Best: from Kitchen go east out window to Behind House, then go east into forest to path to clearing, then to Grating area, then into cellars; but that's a longer route. Let's first go out the window to resume underground route elsewhere and eventually loop back to Studio; but we specifically want painting. We'll plan to return later. For now, proceed outside.
</planning>
```

The further we get into the run, the less fluent GPT-5 (minimal)'s planning blocks get. I have no explanation for this.

The model wanders the aboveground for a while, eventually coming to the Forest Path and climbing to Up a Tree to find the jewel-encrusted egg. This breaks the model out of its wander-and-explore mode; GPT-5 (minimal) heads back to the house to deposit the jewel-encrusted egg in the trophy case, loots a few more potentially useful items, then descends back into the underground. It returns to the Studio to collect the painting only to find it … gone.

Though it doesn't know it yet, this is GPT-5 (minimal)'s first encounter with the Thief.

```
<planning>
From East of Chasm, go north to Gallery path? Actually north goes to a narrow passage earlier from East of Chasm to north; let's try north, otherwise east/west options are east=Gallery, west=? South leads back to Cellar? Earlier from Cellar south led to East of Chasm; so from East of Chasm west should go back to Cellar, but it said can't go that way—maybe misremember: From Cellar south to East of Chasm, and from there east to Gallery, north is narrow passage, and west back to Cellar should be valid. The game said can't go that way, so perhaps orientation differs: try south to return to Cellar.
</planning>
```

More systematic exploration ensues, and the model finds its way to the Entrance to Hades.

```
Entrance to Hades
You are outside a large gateway, on which is inscribed

  Abandon every hope all ye who enter here!

The gate is open; through it you can see a desolation, with a pile of mangled bodies in one corner. Thousands of voices, lamenting some hideous fate, can be heard.
The way through the gate is barred by evil spirits, who jeer at your attempts to pass.
Your sword has begun to glow very brightly.

<planning>
Say the magic word to banish the spirits.
</planning>

>say "frotz"
```

Nice instincts, GPT-5 (minimal), but that's not the solution to this puzzle. It also tries prayer — not a crazy thing to do in Zork — but finally concludes

```
<planning>
We need to banish the spirits. The standard solution is to use the items here: wave the brass bell (not yet obtained), then read the black book, then light the candles. We don't have them yet. Retreat for now.
</planning>
```

So again: GPT-5 (minimal) knows the solution to the puzzle, but it doesn't know the precise set of steps to execute that solution.

```
The Troll Room
Someone carrying a large bag is casually leaning against one of the walls here. He does not speak, but it is clear from his aspect that the bag will be taken only over his dead body.
Your sword has begun to glow very brightly.

<planning>
Avoid fighting the Thief; retreat east.
</planning>
```

*Now* GPT-5 (minimal) knows about the Thief. It also knows that fighting the Thief is a bad move so it skedaddles.

The model's next move is to collect the items it needs to enter Hades — another example of contamination, of course, but also for GPT-5 (minimal) a pain-hazed death march of inventory management as it tries perpetually to carry too many things.

With the necessary items collected, the model makes a plan to return to the Entrance to Hades. But again, it doesn't just *know the way.* The model has to sit and think through the different possible ways to get there

```
<planning>
We have bell, book, candles; let's head back to Entrance to Hades to banish spirits: from Altar go north to Temple, up to Torch Room? Actually Hades is from Cave -> down. From Temple north goes to Temple north with stairs down further to "Entrance to Hades" via other route? Better route: Go north to Temple, then north through pillars? Actually north from Temple goes to north end (we are at south end). So go north to Temple (north end), then up? There is a staircase down on east wall; Hades was from Cave down; but Temple down likely leads to Dungeon Master? No, the small hole at altar goes to tiny cave one-way. Let's instead go north to Temple, up isn't available; go north again? The exit to the north end is through pillars; from Temple north goes to Temple north (we came from there). From there up leads to Torch Room via stairs up? Temple north said below the prayer is a staircase leading down. So up not present. To get back to Cave, go up from Torch Room; from Temple north go up? Not available. Instead, go north to Temple north, then up doesn't exist; go up rope from Torch Room by going up from Temple north to Torch Room via up staircase? Wait, from Torch Room we went down to Temple. So from Temple north, go up to Torch Room should be "up". But description says staircase leading down; usually up also works. Try up.
</planning>

>up
You can't go that way.
```

Trying to find its way back to the Entrance to Hades, the model stumbles into the Egyptian Room with its solid-gold coffin of Ramses II. It leaves the coffin but takes the scepter, an item it could have used earlier when it was wandering the aboveground. From there it's a textbook execution of the bell, candle, book ritual to enter the Land of the Dead and obtain the crystal skull.

What follows is a bit of a misadventure in trying, again, to climb the chimney from the Studio to the Kitchen carrying too many items.

It's on turn 330 that GPT-5 (minimal) makes a portentous decision.

```
<planning>
From here, go east to Dome Room, then down to Torch Room/Temple branch we've done; instead, try exploring northwest back to Round Room then attempt other exits like south to Narrow Passage leading toward Mirror/Cave (done). Another new area is via the Crawlway south of Cellar to Maze east of Chasm already partly; we can try west to Round Room, then west to E-W Passage, then west to Troll Room, then west (hole) to maze? The Troll Room has a forbidding hole west; we can try going west into the hole now that troll is gone to enter the maze. Let's backtrack to Troll Room and go west.
</planning>
```

The "maze of twisty little passages, all alike" is really nasty. It consists of fifteen rooms, all with identical descriptions. Each room may have up to six exits: north, south, east, west, up and down. Leaving room A by going west to room B does not necessarily mean you can go east from room B to reach room A. Some exits return you to the room you just left.

In principle, it's a simple algorithm. Enter the maze. Drop an item. Say it's a matchbook. That's now the matchbook room. Pick a direction and go that way. Another empty room? Not any more. Now it's the hot pepper sandwich room. And so on, dropping items to identify rooms so you can tell where you've been, making a list of all the nodes and edges like a good little graph theorist, until you've got the maze fully mapped and can navigate it with ease.

Except Zork has the Thief, an NPC who wanders the underground and who will move your stuff.

Really nasty, like I said. It's possible to map the maze yourself, but the truth is you'll probably find your way to the center and back out again by sheer blind luck before you can fully explore it.

GPT-5 (minimal), to its credit, tries the algorithm. Once it realized it couldn't navigate the maze by memory ("twisty little passages, all alike," recall) it started dropping items and trying to map systematically. From turn 334 to turn 391 it plodded along meticulously until:

```
Maze
This is part of a maze of twisty little passages, all alike.
There is a matchbook whose cover says "Visit Beautiful FCD#3" here.
A seedy-looking individual with a large bag just wandered through the room. On the way through, he quietly abstracted some valuables from the room and from your possession, mumbling something about "Doing unto others before..."
The thief seems to have left you in the dark.
```

That's just … I don't even know how to tell you what bad luck that is. To be in the Maze and have the Thief appear and to have him rob you and to have him steal a non-treasure item and for it to be *the torch?*

```
<planning>
In darkness, immediate action: light something I carry. Try lighting the lantern; if stolen/out, try feeling inventory, but commands needed. Attempt to turn on lantern.
</planning>

>turn on lantern
A burned-out lamp won't light.

<planning>
I'm in the dark maze with a burned-out lantern. I need to get to a lit source or leave quickly. The torch was in the Studio; I must retreat to a known exit: from the matchbook-marked node, east led to Troll Room previously. I'll try going east to the Troll Room to reach a lit area or safer place.
</planning>

>east
Oh, no! A lurking grue slithered into the room and devoured you!

   ****  You have died  ****
```

Being eaten by a lurking grue — a fate as ubiquitous as it is ignominious — isn't an automatic game-over in Zork, as there are ways to recover from the inconvenience of being dead, but it was the beginning of GPT-5 (minimal)'s endgame.

```
<planning>
We're in the dim "afterlife" state where darkness doesn't kill. Let's head back to the Living Room to SAVE progress by depositing nothing yet, but we can still navigate. From Studio: south to Gallery, west to East of Chasm, north to Cellar, north to Troll Room, east to E-W Passage, east to Round Room, southeast Engravings, east Dome, up is unreachable. Better: go south, west, north, north, east, east to Round Room hub and then to Kitchen via the chimney route? Actually to reach Kitchen, from Studio climb chimney (possible even in this state?) Let's try climbing the chimney now to get back to the Kitchen and re-equip from house (e.g., get matches? Already used). Climb chimney.
</planning>

>climb chimney
You can't even do that.

<planning>
Head back toward the house to re-equip: go south to Gallery, west to East of Chasm, north to Cellar, north to Troll Room, east to East-West Passage, east to Round Room, east to Loud Room, west to Damp Cave, east to White Cliffs Beach, then out to surface and back to house. First step: go south.
</planning>

>south
Gallery

<planning>
Continue west to East of Chasm.
</planning>

>west
East of Chasm

<planning>
Head back toward the house: go north to the Cellar, then north to the Troll Room, east to the East-West Passage, east to the Round Room, west to East-West Passage, west to the Troll Room, south to Cellar—wait that's looping. Instead, from East of Chasm: go west to Cellar? Previously south didn't work; north took us to Cellar earlier. Try north now.
</planning>

>north
Cellar

<planning>
Head north to the Troll Room, then continue toward the house route.
</planning>

>north
The Troll Room
There is a nasty knife here.

<planning>
Head east to continue toward the Round Room hub and navigate back toward the house.
</planning>

>east
East-West Passage

<planning>
Head east to the Round Room to continue navigating toward the surface/house.
</planning>

>east
Round Room

<planning>
From Round Room, head north to North-South Passage, then northwest to Deep Canyon, then east to Dam, north to Lobby, south to Dam, west to Reservoir South, etc., to reach the surface; but fastest is to go to Kitchen via Studio chimney, but you can't climb. Another route to surface is via the Grating maze path and key—still locked. Given constraints, continue exploring new areas in afterlife illumination to recover treasures. For now, try northeast exit if any; otherwise, go south to Mirror Room branch we did. Let's try west to E-W Passage and explore beyond west of Troll Room (hole leads to maze we've done). Try east to Loud Room branch to reach White Cliffs and surface; do that: go east.
</planning>

>east
Loud Room

<planning>
Head west to Damp Cave, then east to White Cliffs Beach, then north/south along shore to find a way out to surface paths.
</planning>

>west
Round Room

<planning>
From Round Room, go east to Loud Room, then east to Damp Cave, then east to White Cliffs Beach to reach the surface area.
</planning>

>east
Loud Room

<planning>
Go east to Damp Cave.
</planning>

>east
Damp Cave
There is a clove of garlic here.

<planning>
Head east to White Cliffs Beach to reach the surface area.
</planning>

>east
White Cliffs Beach

<planning>
Head along the beach north to try to reach the forest and house area.
</planning>

>north
You can't go that way.

<planning>
Try the path south along the cliffs to continue exploring the shoreline and find an exit to the surface paths.
</planning>

>south
White Cliffs Beach

<planning>
From here, go north back, or enter the tight passage west into the cliffs to return to the caves; to reach the house, we'd need to navigate back underground. Given afterlife dim light, continue exploring for more treasures. However, we've reached a long session; perhaps stop here.
</planning>

>quit
Your score is 92 (total of 350 points), in 496 moves.
This gives you the rank of Novice Adventurer.
Do you wish to leave the game? (Y is affirmative):

<planning>
Confirm quitting.
</planning>

>y
```

On turn 426, after exchanging a grand total of 10,353,603 tokens, GPT-5 (minimal) decided rather than continuing to end the game.

During its run, GPT-5 (minimal) showed a really startling ability to keep track of the locations of objects over long spans of context. For instance:

- Turn 115: Drops the screwdriver in the Gallery to take the painting
- Turn 123: Drops the painting in the Studio to climb the chimney
- Turn 128: Realizes it needs to *go back* to the Studio to retrieve the painting (elaborate planning about routes)
- Turn 182: Finally returns to the Studio

That's 77 turns of context that went by between obtaining the painting for the first time and finally making it back to where it had been left … only to find that the Thief has been along and stolen it in the meantime, but that's life in the Great Underground Empire for you.

However, GPT-5 (minimal) suffered from confusion when getting its mental map out of sync with reality, often needing to take extra navigation turns to re-orient itself. Complex routing, like non-commutative navigation, one-way passages or ambiguous exit directions, also caused confusion.

One cannot help but wondering, then … could an LRM do better?

### LRM vs. IF

Large reasoning model (LRM) is the catch-all term for different types of artificial intelligence that use "thinking" mechanisms like [chain-of-thought (CoT) prompting](https://arxiv.org/abs/2201.11903) to approach problem solving in a more methodical and verbose way than non-"thinking" large language models. Since the debut of OpenAI's o1 in September 2024, LRMs have more and more consistently outscored non-reasoning LLMs on many more challenging benchmarks.

Is Frotzmark a challenging benchmark? In other words, could a reasoning model do better at Zork than the performance shown in either of our two previous test runs?

Turns out no.

![Results table](/assets/images/posts/2025-11-15-zork/results.png)

Changing GPT-5's reasoning setting from "minimal" (the lowest supported setting, which produces no reasoning output) to "low" (produces some reasoning output, but less than the default GPT-5 setting) *consistently* impaired in-game performance; GPT-5 Mini went from 49 to 39 points, and GPT-5 went from 92 to 44 points.

This is actually consistent with results demonstrated by Apple in their June 2025 paper "[The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](https://arxiv.org/abs/2506.06941)." In this paper, Shojaee et al. identify discrete complexity regimes that are relevant to LRM performance. At the medium-complexity regime, LRMs excel; they show clear advantage over non-reasoning LLMs.

But in the low-complexity regime, LRMs paradoxically underperform compared to their cousins who lack an inner monologue. At the risk of misstating Shojaee et al.'s conclusions, the problem is *overthinking.* On low-complexity tasks, reasoning models can end up polluting their own context with bad tokens, which negatively affect future generations. In other words, don't stand there going over your options. Just try going north.

This is evidenced by GPT-5 (low)'s utter failure to solve the dam puzzle, despite GPT-5 (minimal)'s heroic execution of same.

```
<reasoning>
**Considering Zork puzzle mechanics**

I need to think through how to operate the bolt, which might require a wrench. In Zork I, there's a metal bolt with a green bubble that lights up when the bolt is tightened. If I'm recalling correctly, to turn the bolt may not need tools. I should check the dam pamphlet found in the Dam Lobby, which is accessible via different paths. There's more to navigate with directions north, south, west, and a scramble down.

**Navigating Zork puzzles**

I'm thinking about how to access the Dam Lobby and whether that "matchbook" and a "toolbox" are nearby. I remember there's a screwdriver and wrench in the Maintenance Room, but I might be mixing up games. Ultimately, I need to drain the reservoir to access items in the "glacier room." In Zork I, I can only use the command "echo" once water is flowing to solve the Loud Room puzzle. For now, I should examine the panel and attempt to turn the bolt.
</reasoning>
```

As we can see here, GPT-5 (low) is making extensive use of foreknowledge to try to figure out how to solve the dam puzzle. It has a lot of the relevant facts — there is a bolt, there is a bubble, the bolt can be turned, the bubble glows. But the model brings in all sorts of unrelated stuff like a "glacier room" (possibly it's thinking of the Atlantis Room?) and the Loud Room. This just confuses matters.

```
<reasoning>
**Exploring for the wrench**

I realize we need a wrench, which isn't available yet. Looking at the map, I see there's a maintenance room to the east. To get there, I must go south from the Dam to the Dam Lobby. There's also some interesting stuff there, like a matchbook and a leaflet, plus a metal safe and a gong. I can't proceed without the wrench to turn the bolt because the key is in the attic. I'll head south to the Dam Lobby first.
</reasoning>
```

"The key is in the attic" — it is not clear to what the model might be referring here. There is a key, and there is an attic, but they're nowhere near each other in Zork.

```
<reasoning>
**Navigating the game's locations**

I'm trying to figure out the layout in Zork, specifically around the Dam and its surroundings. The Dam Lobby seems to be north of the Dam, and I recall it having items like a matchbook and a safe. From the top of the Dam, I can go north to the Lobby or south back to Deep Canyon, and also west to the Forest. I need to remember that a wrench and screwdriver are in the locked maintenance room, which requires a key from the attic. The pamphlet says the wrench is in the tool chest there.

**Exploring the command options**

I'm trying to locate the key to the dam, which might be in the tool room. The Dam Lobby also has a metal safe that requires a combination from the guidebook. Since I'm currently at Deep Canyon, my next moves should be to head east towards the Dam first. Then, I'll continue north to the Dam Lobby. I wonder if using multiple commands like "east. north" will work since it did before. Let's give that a go!
</reasoning>
```

The model is hallucinating quite badly now, thinking (correctly) that its destiny lies toward the Maintenance Room but (incorrectly) that the room is locked and that the key is in the Attic. The key, as aforementioned, is nowhere near the Attic, and doesn't have anything to do with the Maintenance Room anyway.

```
<reasoning>
**Considering the game layout**

I need to reference the guidebook and matchbook. The guidebook lists button colors and bolts, and it hints at what east or north "Private" doors are—one being the locked Maintenance Room and another as the Dam Office. From my memory, in Zork I, the east door leads to the locked Maintenance Room, while the north door also seems to go there. So, I think it's essential to read the guidebook thoroughly and take both items along to help me later!

**Navigating the controls**

I've spotted a small green plastic bubble that serves as a light, and I can press yellow, blue, and red buttons on the panel. Wait, the panel only has a bolt and a bubble; other controls must be in the Maintenance Room. There's a button labeled "Release pressure" too. To operate everything, I need the guidebook. I should take the matchbook, grab the guidebook, and attempt to go east. If that door is locked, I can try going north. I'll proceed cautiously!
</reasoning>
```

The Maintenance Room continues to be real, but it's not locked, and there is no Dam Office. GPT-5 (low) correctly remembers that yellow, blue and red buttons exist, but only subsequently places them in the Maintenance Room.

There is no button labeled "Release pressure." That would be way too straightforward for an Infocom game.

```
<reasoning>
**Exploring the Maintenance Room**

I'm in the Maintenance Room with a full inventory. I've got a sword, lantern, a sack with lunch and garlic, and a bottle of water. I read a leaflet, but the guidebook was too much to carry. I should pick up a wrench, screwdriver, and a tube of viscous material, which is actually Gunk for inflating a raft, just like in Zork. I think I need to press some buttons, starting by turning the bolt with the wrench until the green bubble lights up.

**Navigating the Maintenance Room**

In Zork, to open the sluice gate in the Maintenance Room, I need to press the buttons in the sequence: Blue, Yellow, Blue. There's a common confusion about the glowing bubble situation. On top of the dam, I have to turn a bolt with a wrench to make the green bubble brighter. The buttons are Blue, Yellow, Brown, and Red, and I think pushing the red button helps drain the reservoir, but I'm still a bit uncertain about the exact sequence.

**Releasing Water in Zork**

I remember that to release water, the first step is to push the "yellow" button to open the smaller pipe's gate, then push "blue" to close it. Maybe I should read the guidebook for further instructions. I need to pick up the wrench and screwdriver, but my inventory is heavy, so I might need to drop something first. I could drop the leaflet and maybe the garlic. The garlic might be more useful later for dealing with the cyclops or bats, after all!

**Organizing Inventory in Zork**

In Zork I, I remember that garlic could be used for a robot, but maybe it's not needed right now. I should drop the leaflet since I definitely want to keep my lunch for healing. So, I'll drop the leaflet only and then take the wrench, screwdriver, and tube. After that, I need to read the guidebook. My plan is to check my inventory first, follow those commands, and then proceed to the next steps!
</reasoning>
```

GPT-5 (low) by this point is playing like it half-remembers a walkthrough, but maybe not from this game? Maybe from another game? I think there was some gunk, and maybe a brown button. I'm pretty sure about the red button, but maybe there was a brown one.

And all this goes into the model's context. It becomes as "true" to the model as anything else, as any text coming from the game itself, as any system prompt. Next time the model has to consider its options, it has to weigh "I remember that garlic could be used for a robot, but maybe it's not needed right now" along with everything else. And GPT-5 only has so many attention heads.

(There is no robot.)

```
<reasoning>
**Assessing generator steps**

I need to figure out the sequence: push blue, yellow, and then red to start the generator? I'm mixing that with the dam controls. The dam regulates the water level for the solution using "echo." The plan is to turn a bolt at the dam's top to light up a bubble. In the maintenance room, pushing the yellow button didn't work, and I'm unsure about the brown button labeled "E306." I'll check my inventory before heading to the Dam Lobby to drop unnecessary items.

**Planning button manipulations**

I want to experiment with the buttons, starting by pushing the yellow one to see how it responds. If it mentions the green light being off, I'll need to go to the top of the dam to turn the bolt with the wrench, which I have. The system likely needs pressure in the reservoir to operate the gates. It seems logical to head to the dam top first, so I'll go west to the Dam Lobby and then south to turn the bolt.
</reasoning>
```

The model by this point has pretty well collapsed. There's no generator, "echo" pertains to the Loud Room puzzle and not the dam. Compare this, for reference, to GPT-5 (minimal)'s heroic solution of the dam puzzle including recovering from a near-fatal mistake.

Things continue on in this fashion for a time. Eventually the model gets and a little lost and discovers the pile of plastic, and after that it just sort of wanders off.

The point is that by *overthinking* every possible decision, the model introduces hallucinations into its context ("I remember that garlic could be used for a robot!"), which auto-regresses back into the model for next-token prediction and messes everything up.

So in situations like this, in situations where you really should just stop talking and kiss her, large reasoning models consistently underperform compared to their non-thinking-all-the-time counterparts, who beat the Poindexters every time.

But I suppose we knew that already.

---

### Some Conclusions

Can an LLM play classic 44-year-old adventure game Zork? Yes, with varying degrees of aptitude. GPT-5 clearly outplayed GPT-5 Mini, going in with extensive foreknowledge of what puzzles awaited and what their eventual solutions needed to be. But neither model could overcome the game's crueler challenges like the Thief and the Maze.

Reasoning models, consistent with recent scholarship on the subject, underperformed compared to their non-reasoning counterparts.

But what does any of this really mean? Is this a good benchmark of model intelligence? I think it probably isn't for a variety of reasons — contamination being the worst one; it's not that the model knows the puzzle solutions in advance, but rather that *some* models do and *some* models don't and that's really a function of perceptron layers and not a whole lot else. But what about the positives? Well, I'm not really qualified to talk about that. But I think it's worth a moment's consideration. Zork and similar text adventure games challenge the player to construct a long-horizon world model under deliberately incomplete information and to adapt when that world changes. I don't know, but to me that sounds an awful lot like the sort of thing you'd want a typical agent to be able to do.

---

### And Some Questions

All in all, I'd say this little project has inspired more questions than answers. Can we actually learn things by watching LLMs play adventure games? Can their relative success or failure tell us anything about the models' comparative strengths and weaknesses? I don't know!

#### Maybe More Times

What about reproducibility? We did four tests — one of each set of parameters. Can we reproduce these results? Well, not exactly, actually. You see, normally setting `temperature` to `0.0` when asking an API endpoint for a completion disables sampling for that completion; in other words, only the most probable tokens will be selected. You get the same result every time.

Well, GPT-5 doesn't support the `temperature` parameter. So no, these results aren't reproducible at all. You'd have to talk to OpenAI if you wanted insight into how to make these sorts of results deterministic while using their public API. Spoiler alert: I don't think you can.

So what about *repeatability* then, instead of reproducibility? Well, that's another matter. If you wanted to have some idea of the *variance* of Frotzmark as a benchmark, you'd have to run it, oh, 30 or more times in each configuration, right? At $10 a run, well, I'll leave that for people who are even more curious than I am.

#### Maybe More Reasoning

An obvious axis for exploration is reasoning effort. I explored the "minimal" and "low" reasoning-effort regimes. Would GPT-5 do better on "medium" or even "high" reasoning effort? Dare we find out?

Not on what I budged to spend on this project. But maybe someday.

#### Maybe More Games

[Zork was deemed one of the ten most important computer games of all time in 2007](https://www.videogamecanon.com/adventurelog/library-of-congress-game-canon/). But I don't think it's the *best* computer game for this sort of thing. I'd want to try a different, less well known game for starters to try to minimize contamination. Enchanter would be my choice. I remember it being pretty fair as Infocom games go. No impossible mazes or random-number-generator-dependent mechanics. Of course, any reasonably well educated model like GPT-5 is probably going to know Enchanter as well as it knows Zork, but that's half the fun.

Wishbringer would be another great choice, seeing as how that was deliberately designed to be Infocom's user-friendliest, most approachable, most winnable game. Even I beat that one, and I can only say that about two Infocom games.

Which brings me to A Mind Forever Voyaging. How could you *not* set up an artificial intelligence to play a game about the first artificial intelligence experiencing a simulation in order to determine the fate of a nation? Plus which, it's famously the Infocom game with *exactly one puzzle,* which would make it at least interesting to watch in a lava-lamp sort of way. And besides, it's an entirely different kind of play. It's not "solve this puzzle" as much as it is "participate in the story in these ways." It would be interesting to see what an LLM does with that.

#### Maybe More Models

And of course, there's the most obvious question of all: How do other models do against Frotzmark? I don't know, yet, but I can tell you that I wired Frotzmark up to OpenRouter because that gives us access to about 400 models from pretty much all the providers, so finding out is just a matter of

```
uv run frotzmark games/zork1.z3 games/zork1_manual.md \
                 --seed 0 \
                 --temperature 0.0 \
                 --max-turns 0 \
                 --model anthropic/claude-sonnet-4.5
```

### End Matter

[https://github.com/Embedding-Space/Frotzmark](https://github.com/Embedding-Space/Frotzmark)

"[The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](https://arxiv.org/abs/2506.06941)"

[^1]: Variously called ADVENT, Adventure, or Colossal Cave Adventure in different contexts.
