import nltk
import re
nltk.download('punkt')

# some perticulat functions
from nltk.tokenize import sent_tokenize

# get tokenized setences present in "filename"
def getSents(filename):
    # reading the file and using word_tokenize
    f = open(filename, 'r', encoding="utf8")
    raw = f.read()
    f.close()
    
    """
    remove 1 new line as the dataset has useless
    new lines for no reason, well, for better
    readibility. Anyways.

    will replace a single \n with a space.
    """
    line = re.sub(r"\n(\n*)", "\1", raw)
    return sent_tokenize(raw)

