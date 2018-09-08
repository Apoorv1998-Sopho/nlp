# For TAs:
Please check the file `Assignment_2` for evaluations.

## Assumptions
* I have replaced the new lines that were initially contained in the `alice_adventure.txt` with spaces, as the sentences that were being tokenized by the `nltk` `sent_tokenize()` was containing `\n` characters.
  * For performing the above command we ran the following script on a linux shell. `sed -e ':a' -e 'N' -e '$!ba' -e 's/\n/ /g' alice_adventure.txt > alice_adventure_new_lines_reduced`
* 

# Else Everyone:
This folder contains some basic text proccessing stuff that has been covered in the first week of our course NLP. Will try to run som e notebooks to get comfy. 

Basic things covered include:


    Tokenization
    Stemming
    Spell Correction

Take away points:

1. Some languages work really good if we use greedy algorithms (like Chinese) as they usually have many words that are formed using conjuction of 2 or more words. Whereas in the case of English these 'greedy' algorithms don't seem to provide that much help.
2. Punctuations are also tockenized to be declared as a token in itself, seems that they should not be.
3. What happens when we have something like Out and out, do we put them is same type or not. I think not.
4. 
