# For TAs:
Please check the file Assignment-1 for evaluations.

## Assumptions
* I have removed any word that contained any sort of non aphanumeric charater whil emaking the `clean_tokens`. See the function `rmPunctLowerCase` for details.
* I made all the words in the corpus lower. I know that to do this I have to make a tradeoff for some Proper nouns to be possibily being used same as a common noun.

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
