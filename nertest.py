import nltk

doc = '''Krow Network / Co-founder and Chief Technology Officer August 2018 - PRESENT, NEWARK, NJ Leading multi-platform development to build a useable product for job seekers in the United States. Oversee all development of user facing and back-end services, and fully operate all artificial intelligence programs.'''

# tokenize doc
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
 
sentence = "Mark and John are working at Google."
 
iob_tagged = tree2conlltags(doc)
pprint(iob_tagged)