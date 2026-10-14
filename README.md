# Assignment 1: Train an image classifier on your own images

MAS1004 Data & AI, 2026 Fall

Due: Friday 16 October, 22:00, on Cyber Campus (LMS).

## What you do

You choose 3 to 5 categories of image, collect images of them, and train a
classifier that tells them apart. The model is ResNet18, a network that was
already trained on ImageNet, 1.2 million photographs of 1,000 kinds of thing.
You replace its last layer so that it answers with your categories, and train
it further on your images. Then you put it on a web page that anyone can open,
and find out what it actually looks at when it decides.

Two things matter most:

- Categories that come from your own interest: something you care about or
  know well, such as a hobby, your major, or things you see every day and can
  tell apart yourself. Please avoid categories chosen only because they are
  easy to separate, such as cats and cars.
- Finding out which visual features your model uses, by changing images on
  purpose and watching its answers (Problem 7). You do this part yourself.

3 to 5 is the number of categories in one classifier. One model that tells
espresso cups, wine glasses and paper coffee cups apart has 3 categories. You
do not run three separate projects on different topics.

## What you hand in, on Cyber Campus (LMS)

1. `links.txt`, a text file with the address of your web demo and the address
   of your code repository, in this form:

   ```
   demo: https://<your name>.github.io/<your repository>/
   code: https://github.com/<your name>/<your repository>
   ```

   The model runs inside the web page, so the demo works without a server, and
   nobody's photos are sent anywhere.
2. The short report as a PDF: one page of text written by you, and up to
   three more pages of figures if you want them, from the template in
   `short_report/`.
3. The long report as a PDF: written with your agent, from the template in
   `long_report/`.

Both templates are LaTeX files that you compile on Overleaf
(https://www.overleaf.com), with a free account. The sections "The short
report" and "The long report" at the end of this file say how.

## Getting started

You need two things of your own: VS Code with Cline, as set up in class, and a
GitHub account. Make the account at https://github.com/signup if you do not
have one. Your agent does the rest.

In VS Code, open Cline and give it this request:

```
Set up MAS1004 Assignment 1 for me. Follow the section "First setup" in
https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/AGENTS.md
```

It makes your own copy of this repository on your GitHub account, downloads it
to your computer, and installs what the assignment needs. On the way it asks
you to log in to GitHub in your browser. At the end it runs the tests on the
code you were given, six of them pass, and it tells you to open the new folder
in VS Code. Work in that folder, with Cline, from then on.

If you already made your copy in class, open its folder in VS Code and give
Cline this request instead. Use the same request whenever the given files
change. It does not touch the functions you wrote or your images.

```
Update the given files in this repository from the template. Follow the
section "Updating the given files" in
https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/AGENTS.md
```

Some computers cannot run this assignment, because PyTorch has no packages for
them: Intel Macs, Windows laptops with an ARM processor (Snapdragon), and Macs
older than macOS 14. Your agent tells you if yours is one of them. Then you
train on Colab.

### Google Colab

Colab gives you a GPU for free, and training ResNet18 is much faster on a GPU.
Open the notebook directly in Colab with this link:
https://colab.research.google.com/github/jdasam/mas1004-assignment1/blob/main/assignment1_colab.ipynb

Use Colab only to run things: downloading, training, cleaning and checking.
Write your code on your own computer with your agent, and push it. The
notebook takes your code from your repository on GitHub, so push before you
run it, and do not edit code in Colab: the notebook replaces it with what is on
GitHub every time it updates. When the training is done, you download the
model files from Colab and publish the page from your own computer
(Problem 6).

## How you work

`PLAN.md` is the plan for the whole assignment, step by step, with a check for
each step. Read it, then give it to your coding agent one step at a time.
`AGENTS.md` tells your agent how to run things in this repository. Most agents
read it on their own.

You write the bodies of these functions. Each has a docstring that says what
goes in and what comes out, and a test that checks it.

| file | functions | problem |
|---|---|---|
| `src/data.py` | `prepare_image`, `load_folder`, `split_train_test` | 2 |
| `src/train.py` | `build_model`, `train` | 3 |
| `src/evaluate.py` | `predict_logits`, `accuracy`, `confusion_matrix` | 3 |
| `src/evaluate.py` | `worst_examples` | 5 |

Everything else is given: the image downloader, the cleaning tool, the
training script, the export script, the web page and the checker.

You are responsible for everything you hand in, including what your agent
wrote.

- Every number in your reports has to come from output that you saw printed
  by code you ran. Agents sometimes report numbers they never measured.
- When the agent says it fixed something, have it run the test, and look at
  the result yourself.
- You make the guesses and choose the image changes in Problem 7, and you
  write the short report. The agent makes only the image changes you ask for.

---

## Problem 1. Choose your categories and collect the images

Choose 3 to 5 categories, as described at the top. The downloader searches the
web for each category and saves the images, and you work on a copy of them.

Three things to think about when you choose.

- Your categories should not be ImageNet categories. `imagenet_classes.txt`
  lists all 1,000 of them. ResNet18 has already learned to tell those apart, so
  training it on them adds almost nothing. Things that ImageNet calls by one
  name are a good place to look: ImageNet has a category for pizza, but not
  for Neapolitan, New York and Chicago pizza. The cup categories used as
  examples in this README are not a good choice for this reason: ImageNet
  already has espresso, cup, coffee mug, goblet and red wine. They are there
  only to show how things work.
- In Problem 5 you test the model on images from a source that cannot overlap
  with your downloads, such as your own photos, so choose things you can find
  such images of.
- The model can only use what is visible in the picture: an expensive wine
  glass and a cheap one that look the same cannot be told apart by any model.

Write this down:
- Why these categories? Why do you care?
- What do you think the model will actually use to tell them apart: shape,
  colour, texture, background, or something else? Write it before you train.
  Problem 7 tests it.
- How many images did you get for each? Show the counts.

## Problem 2. Turn your folder into numbers

Write `prepare_image`, `load_folder` and `split_train_test`. `prepare_image`
prepares a photo exactly the way every ImageNet photo was prepared when
ResNet18 was trained: the shorter side resized to 256, the 224 by 224 square
in the middle cut out, and each colour channel scaled. The network can only
read your photos if they are prepared the same way.

No image may be in both the training set and the test set. If one is, your
test accuracy measures what the model remembers, not what it learned.

Write this down:
- One of your images next to what `prepare_image` made of it. What was cut
  off? Find an image where the crop cut off part of the thing you care about.

## Problem 3. Train it and report the first result

Write `build_model`, `train`, `predict_logits`, `accuracy` and
`confusion_matrix`. Then train at least four times, before you clean anything:
starting from the ImageNet weights, starting from random numbers, training
only the new last layer, and one setting of your own. Fill in this table.

| tag | start | trained | epochs | lr | trainable parameters | train accuracy | test accuracy |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

Write this down:
- The loss curve. Did the loss go down? Did it keep going down, or did it
  flatten out?
- Your train accuracy is higher than your test accuracy. By how much? What does
  that gap mean?
- The confusion matrix. Which two categories does it mix up most? Does that
  surprise you?
- The same images, the same code, and the same number of epochs, starting from
  ImageNet and starting from random numbers. How far apart are the two test
  accuracies? Why do you think that is?
- Training only the last layer changes about 0.01% of the parameters. How
  close does it get to training all of them?

## Problem 4. Clean your data and train again

Every downloaded image got its label from a search engine. You typed "wine
glass", and whatever came back is now called `wine_glass`: drawings, product
catalogues, charts, photos of a bar, the same picture four times with
different watermarks. The model learns all of them as true examples, and your
test set contains them too.

Only you know what you meant by each category, so you write a rule first and
then decide each image by it. "I removed pictures where the object was not the
main thing in the frame" is a rule. "I removed the ones that looked wrong" is
not. Look at every image yourself. The cleaning tool also lists the images that
look least like the rest of their class, and pairs of near copies, but a
suspect is not automatically junk: decide it by your rule.

Cleaning changes your test set too, so compare the models before and after
cleaning on your new images from Problem 5, which do not change.

Write this down:
- Your rule, in one or two sentences, as you wrote it before you started.
- Three kinds of junk you found, with one picture of each.
- The table of how many images you removed from each class, and why.
- How many near copies it found, and what you did with them.
- Of the first 20 suspects in each class, how many did you remove? Did the
  suspects include junk you had missed when you looked yourself? Did you find
  junk that it did not list?
- The test accuracy before and after, and the accuracy on your new images
  before and after. If a number went down, say so and think about why. That
  happens, and an honest explanation is worth more than a good number.

## Problem 5. Test it on images from a new source

Your test set came from the same search as your training set, so it shares its
habits: the same kind of product photo, the same white backgrounds. Now test
the model on images from somewhere else.

Collect at least 5 images per category from a source that you can be sure is
not in your downloads: photographs you take yourself, photographs a friend
took, frames from a video you recorded, or photos from your phone's gallery.
Images from the web are fine too, as long as you are sure the search that made
your downloads could not have found them. The overlap check compares your new
images with your downloads and shows any near copies.

Save them as JPEG or PNG. An iPhone saves photos as HEIC unless you set
Settings, Camera, Formats to Most Compatible, and the code here cannot read
HEIC. Your agent can convert ones you already took.

Your accuracy here will probably be worse than your test accuracy from
Problem 3. That is not a mistake you made. Explaining it is the assignment.

Write this down:
- Where your new images came from, and why you are sure none of them are among
  your downloads. The output of the overlap check.
- The accuracy on your downloaded test set and the accuracy on your new images,
  side by side.
- Five mistakes it was confident about, with the pictures. For each one, what
  in that image do you think pushed it the wrong way?
- What is different about your new images? Background, lighting, angle,
  distance, what else is in the frame?

## Problem 6. Put the demo on the web

Ask your agent to export your cleaned model to the page and open the page on
your own computer. If you trained on Colab, bring the three model files from
Colab first (the last section of the notebook). The badge at the top of the
page has to be green. A red badge means the page prepares images differently
from your `prepare_image`, and every answer on it is wrong.

Then ask your agent to push and to turn on GitHub Pages. Your demo appears at
`https://<your name>.github.io/<your repository>/` after a minute or two.
Open it on your phone to check that it works away from your computer.

Write this down:
- Your GitHub Pages address.
- One sentence on what happens when you show it something that is none of your
  categories.

## Problem 7. Find out what your model looks at

This is the most important problem in the assignment.

In Problem 1 you wrote down what you thought the model would use to tell your
categories apart. Now you have a trained model and you have seen its mistakes.
In this problem you change images on purpose and see how its answers change,
to find out what it actually uses.

Do this part yourself, not by handing it to your agent. You look at the images
and the mistakes, you make the guess, and you choose the change that tests it.
The agent only makes the changes you ask for.

1. Write down a guess that is specific enough for an image change to prove it
   wrong. "It uses the shape" is too vague. "It tells a wine glass from an
   espresso cup by the long stem" can be tested.
2. Choose the change that would show whether the guess is right, and the images
   to change. Some kinds of change, depending on the guess:
   - one part of the object: cover that part with a plain grey box, or crop it
     out
   - the background: put the same object on a different background
   - colour: turn the image grayscale, or shift its colours
   - fine texture: blur the image
3. Ask your agent for exactly that change, on images you name. For example:
   "Make a copy of each image in data/new_images/wine_glass with the bottom
   third covered by a grey rectangle, and save the copies in
   data/changed/wine_glass_no_stem."
4. Upload the original and the changed images to your demo page and compare
   its answers. The page also shows the square the model actually sees. Check
   that the part you changed is inside it.
5. Use several images for each change, not one. A single image can change its
   answer for reasons that have nothing to do with your guess.

A result that goes against your guess tells you as much as one that supports
it. Write it down, and make a new guess from it.

Write this down:
- Each guess, and what in your images or your model's mistakes made you think
  of it.
- For each change: the images you used, what you changed, why that change
  tests the guess, and the model's answers before and after, with the original
  and the changed images side by side.
- The exact requests you gave your agent in this problem.
- What you now think the model uses to tell your categories apart, and how sure
  you are.

---

## The short report

One page of text. Not one and a bit. One. After the text you may add up to
three pages of figures.

Write it yourself. No language model, at any stage, including for tidying it up
afterwards. If Korean is your first language, write it in Korean. This is the
one piece of work in this course where neither your English nor your polish
counts for anything, and what you actually think counts for everything.

Write freely about what you learned from this assignment:

- which topic you chose, and why
- how the assignment went
- what you learned

Write it on Overleaf, from the template. Log in to Overleaf, then open one of
these links. Each makes a new project of your own with the template in it:

- [English template](https://www.overleaf.com/docs?engine=xelatex&main_document=short_report_en.tex&snip_uri%5B%5D=https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/short_report/short_report_en.tex&snip_name%5B%5D=short_report_en.tex&snip_uri%5B%5D=https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/short_report/mas1004.sty&snip_name%5B%5D=mas1004.sty)
- [Korean template](https://www.overleaf.com/docs?engine=xelatex&main_document=short_report_ko.tex&snip_uri%5B%5D=https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/short_report/short_report_ko.tex&snip_name%5B%5D=short_report_ko.tex&snip_uri%5B%5D=https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/short_report/mas1004.sty&snip_name%5B%5D=mas1004.sty)

If a link does not work, ask your agent to compress the `short_report` folder
into a zip, and on Overleaf choose New Project, Upload Project. Either way, the compiler has to
be XeLaTeX (Menu, Compiler). Do not use Overleaf's AI tools on this report.

The first page of the template is a short guide to the LaTeX you need. Read it,
then delete it and write your report in its place. If the text runs past one
page, or the figures take more than three pages, a red note appears in the
PDF.

## The long report

As long as you like, written with your agent. It holds everything the "Write
this down" boxes ask for, in order, with the pictures and the complete output
of the checker. Every number in it has to be one you saw printed by code you
ran.

The template is `long_report/long_report.tex`. Your agent fills it in, in your
repository: it copies the pictures into `long_report/figures/` and saves the
output of each command into `long_report/outputs/`, from where the report
prints it exactly as it was. Red text in the PDF is what is still missing.

To see the empty template on Overleaf, log in to Overleaf and open this link.
It makes a new project of your own with the template in it:

- [Long report template](https://www.overleaf.com/docs?engine=xelatex&main_document=long_report.tex&snip_uri%5B%5D=https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/long_report/long_report.tex&snip_name%5B%5D=long_report.tex&snip_uri%5B%5D=https://raw.githubusercontent.com/jdasam/mas1004-assignment1/main/long_report/mas1004.sty&snip_name%5B%5D=mas1004.sty)

This project has no pictures or outputs in it. To make the PDF of your filled-in
report, ask your agent to compress the `long_report` folder into a
zip, and on Overleaf choose New Project, Upload Project. Set the compiler to XeLaTeX (Menu,
Compiler) and press Recompile.

## How this is graded

There are no points attached to the problems. Four things are looked at:

- Problem 7: whether your guesses come from your own images and mistakes,
  whether each image change actually tests its guess, and what you concluded
  from the results
- Whether the demo works, at the address you gave, on someone else's computer
- Whether the numbers in your long report are the ones `check.py` actually
  prints
- The short report

A model that scores 95% with no explanation is worth less than one that scores
55% whose owner can tell you exactly which pictures it fails on and why.

## When you are stuck

- Ask your agent to run `src/check.py`. It works out the answers for itself
  from the files on disk, so when it disagrees with your own code, one of the
  two is wrong.
- Ask in class. Both Tuesday and Thursday have time for this.
