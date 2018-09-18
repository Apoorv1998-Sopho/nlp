# For TAs:
Please check the file `Assignment_2` for evaluations and `helper.py` for custom functions that have been implemented..

## Assumptions
* I have replaced the new lines that were initially contained in the `alice_adventure.txt` with spaces, as the sentences that were being tokenized by the `nltk` `sent_tokenize()` was containing `\n` characters.
  * For performing the above command we used `re.sub(r"\n(\n*)", r"\1", raw)` in python. 

# Else Everyone:
This folder contains some Language Modelling stuff that had been covered in our course NLP.

Basic things covered include:

    Most Likely-hood Estimate
    Language Modelling
    Smoothing
    Sentence Generation

Take away points:

1. Need `smoothing` as there can be cases when we train we don't see a combination of words, but in the `test` we can have such combinations that are never seen before in `train`.
