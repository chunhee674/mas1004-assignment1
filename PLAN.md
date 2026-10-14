# The plan

This is the plan I would write before starting this assignment. Give it to your
coding agent one step at a time, not all at once. An agent handed twenty steps
will do a bad job of all twenty.

Read it first. Part of what this course is teaching is how to write one of
these yourself, and you cannot learn that by pasting it without looking.

Each step says what to do, and how you know it worked. The second part matters
more. A step is not done because the agent says it is done. It is done when the
check passes.

---

## Step 0. Make sure the room is empty before you move in

Run `uv run pytest tests/test_export.py`. Six tests should pass. They cover
code you were given, so if they fail, nothing you write afterwards will work
either.

Check: six passed.

## Step 1. Choose your categories

Not a coding step. Write down 3 to 5 categories and one sentence each on why
you chose them. They are the categories of one classifier, not separate
projects. Keep the sentences. They go in your report.

Also write down what you think the model will use to tell them apart. Step 18
tests it.

Then have your agent check them against `imagenet_classes.txt`, the 1,000
categories ResNet18 already tells apart. If one of yours is in it, the model
already knows it, and training on it adds almost nothing.

Check: you can say out loud what visible difference the model is supposed to
find between them, and you know where you will get images of them that cannot
be among your downloads (Step 4).

## Step 2. Download the images

```
uv run python src/collect.py --classes "your,categories,here" --n 150
cp -r data/raw data/clean
```

Check: `data/clean/` has one folder per category and each has at least 50 files
in it. If one category came back nearly empty, the search phrase is wrong.
Change the phrase, not the category.

## Step 3. Look at what you downloaded

```
uv run python src/clean.py look
```

Open every sheet in `results/cleaning/look/` and look at every picture. Do not
remove anything yet. Your first training run in Step 9 is on the data as it
came, so that Problem 4 has something to compare against.

Check: you can name three kinds of junk that came back.

## Step 4. Collect images from a new source

At least five per category, into `data/new_images/<category>/`, with the same
folder names as `data/clean`. They must come from a source you can be sure is
not in your downloads: your own photos, a friend's photos, frames from a video
you recorded, or images from the web that you are sure your download could not
have found. Save them as JPEG or PNG, not the HEIC files an iPhone makes by
default (README, Problem 5). If you take photos
yourself, take them in different places and different light. If you take them
all on the same desk in one evening, Problems 4 and 5 have nothing to say.

```
uv run python src/clean.py overlap | tee results/overlap.txt
```

The `| tee results/...` at the end shows the output and also saves it in
`results/`, for the long report. Run it again after you take out any near
copies, so that the saved output is the final one.

Check: every category has a folder in `data/new_images/` with at least five
images in it, `clean.py overlap` finds no near copies, and you can say in one
sentence why none of them can be among your downloads.

## Step 5. Write `prepare_image`

Give the agent the docstring of `prepare_image` in `src/data.py` and ask it to
write the body. Then run:

```
uv run pytest tests/test_data.py -k prepare
```

The tests will fail in specific ways. Give the agent the failure text as it is,
not your summary of it.

Check: every prepare test is green, and you can say what happens to a photo
that is not square.

## Step 6. Write `load_folder` and `split_train_test`

Same again, then `uv run pytest tests/test_data.py`.

Before you move on, read the test called
`test_split_has_no_image_on_both_sides` and make sure you understand why it
exists. If you do not understand why an image being in both halves is a
problem, ask the agent to explain it, and do not accept an answer you cannot
repeat.

Check: all of `tests/test_data.py` is green.

## Step 7. Write `build_model` and `train`

```
uv run pytest tests/test_train.py
```

The first run downloads the ImageNet weights, about 45 MB. The last test trains
on three easy blobs and expects over 90%. If your training loop has a bug, that
is where it shows up. Common ones: forgetting `optimiser.step()`, forgetting
`optimiser.zero_grad()`, training on the test set, measuring accuracy as a
percentage instead of a share, and moving the model to the GPU but not the
batch ("Expected all tensors to be on the same device").

Check: all of `tests/test_train.py` is green.

## Step 8. Write `predict_logits`, `accuracy` and `confusion_matrix`

```
uv run pytest tests/test_evaluate.py -k "not worst"
```

Check: green.

## Step 9. Run the whole thing for the first time

```
uv run python src/run.py
```

On a laptop without a GPU this takes several minutes. On Colab with the GPU
turned on it is much faster.

To train on Colab, commit and push your code first, and run this step and the
ones up to Step 15 in the notebook, which takes your code from GitHub. When a
step asks you to write code, write it here, push it, and bring it over with
section 9 of the notebook. Come back here for Step 16.

Look at `results/run_curves.png` and `results/run_confusion.png`. Everything
`run.py` prints is also saved in `results/run_output.txt`, for the long report.

Check: it finished, it printed a train accuracy and a test accuracy, and the
loss curve goes down. If the two accuracies are identical, look at your split
again.

## Step 10. Run it three more times

```
uv run python src/run.py --scratch --tag scratch
uv run python src/run.py --freeze --lr 1e-3 --tag frozen
uv run python src/run.py <a setting of your own> --tag <a name for it>
```

Write each printed row into the table in your report as you go. Do not wait
until the end and try to remember.

Check: four rows in the table, and you can say how much starting from ImageNet
changed the test accuracy.

## Step 11. Write your cleaning rule

Not a coding step. Using what you saw in Step 3, write down in one or two
sentences what does not belong in each category. Do it before Step 12, so that
the suspect list does not decide the rule for you.

Check: someone else could apply your rule to your images and remove the same
ones you would.

## Step 12. Go through the suspects and the near copies

```
uv run python src/clean.py suspects
```

Open `results/cleaning/suspects/`. For each class sheet, go through the
suspects in order and decide each one by your rule. Then go through
`copies.png` and keep one of each pair. Remove with:

```
uv run python src/clean.py remove <class>/<file> <class>/<file> --reason "<which part of your rule>"
```

Then go back through the sheets from Step 3 for anything the suspect list did
not catch. When you are done:

```
uv run python src/clean.py count | tee results/clean_count.txt
```

Check: `clean.py count` prints how many you removed from
each class and why, and you know how many of the first 20 suspects in each
class you removed.

## Step 13. Train on the clean data and compare fairly

```
uv run python src/run.py --tag clean
uv run python src/check.py --compare run clean | tee results/compare.txt
```

Check: you have the test accuracy before and after, and the accuracy on your
new images before and after, and you can say why the second pair is the fairer
comparison.

## Step 14. Write `worst_examples` and look at the mistakes

```
uv run pytest tests/test_evaluate.py
uv run python src/run.py --tag clean
```

Check: `results/clean_worst.png` shows ten pictures with what the model said
and what they really are.

## Step 15. Put the clean model on your page and check it

```
uv run python src/export_web.py --tag clean
uv run python src/check.py | tee results/check.txt
```

Check: `check.py` says the exported model agrees with Python, and reports an
accuracy on your new images.

## Step 16. Open the demo on your own machine

If you trained on Colab, first copy `model.onnx`, `model.json` and
`selftest.json` from `docs/` on Colab into `docs/` here. Section 15 of the
notebook packs them for you.

```
uv run python -m http.server -d docs 8000
```

Open http://localhost:8000.

Check: the badge at the top is green. If it is red, stop and fix it. A red badge
means the page is not preparing images the way your `prepare_image` prepared
them for training, so everything it says is wrong. Read what the badge says,
fix `prepare_image`, and then train and export again.

## Step 17. Publish it

Ask your agent to commit and push, and to turn on GitHub Pages (AGENTS.md,
Problem 6). It gives you the address.

`docs/model.onnx` is about 45 MB, so this push takes a while. Do it once, with
the model you want to hand in, not after every experiment.

Check: you opened the address on your phone, away from your own wifi, and it
worked. If the page says it could not load the model, have your agent check
whether `docs/model.onnx` and `docs/model.json` are really in your repository
on GitHub.

## Step 18. Find out what your model looks at

Do not give this step to your agent. You make the guesses and choose the tests
yourself, and you ask the agent only to make the image changes you decided on
(README, Problem 7).

1. Look at `results/clean_worst.png`, the mistakes `check.py` listed for your
   new images, and what you wrote in Step 1 about what the model will use.
   Write down one guess about what it uses, specific enough that changing an
   image can prove it wrong.
2. Decide which change to the image would test that guess, and on which
   images.
3. Ask your agent for that change, naming the images and where to save the
   copies. For example: "Make a copy of each image in
   data/new_images/wine_glass with the bottom third covered by a grey
   rectangle, and save the copies in data/changed/wine_glass_no_stem." Keep
   your request word for word for the report.
4. Upload the originals and the changed images to your demo page and write
   down its answers.
5. Make the next guess from what you saw, and repeat.

Check: for each change, you can say what you expected the model to do, what it
did, and what that tells you about your guess.

## Step 19. Write the long report

Go back through the "Write this down" boxes in README.md in order, with your
agent. You should already have every number and every picture you need, from
the checks above.

Your agent fills in `long_report/long_report.tex` (AGENTS.md says how). It
copies the pictures it needs from `results/` into `long_report/figures/`, and
the saved outputs, `results/*.txt`, into `long_report/outputs/`. To make the
PDF, compress the `long_report` folder into a zip, and on Overleaf choose New
Project, Upload Project, set the compiler to XeLaTeX, and Recompile.

Check: no red text is left in the PDF, and every number in it is one you saw
printed by code you ran.

## Step 20. Write the short report

Close the agent. One page of text, by yourself, in Korean if that is your
first language, and up to three more pages of figures if you want them. Open
the English or the Korean template on Overleaf with the links under "The short
report" in README.md, and write freely about what you learned: which topic you
chose and why, how the assignment went, and what you learned.

This is the last step because you can only write it once you have been
through all of the others.

Check: the text fits on one page and the figures on at most three more, with
no red note in the PDF, and you wrote all of it.
