#!/usr/bin/env python

"""
A feature extractor for chunking.
Copyright 2010,2011 Naoaki Okazaki.
"""

import crfutils
import sys
from sklearn.feature_extraction.text import HashingVectorizer,FeatureHasher

# Separator of field values.
separator = ' '

# Field names of the input data.
fields = 'w pos y'

# Attribute templates.
templates = (
    # (('w', -2), ),
    # (('w', -1), ),
    (('w',  0), ),
    # (('w',  1), ),
    # (('w',  2), ),
    # (('w', -1), ('w',  0)),
    # (('w',  0), ('w',  1)),
    # (('pos', -2), ),
    # (('pos', -1), ),
    # (('pos',  0), ),
    # (('pos',  1), ),
    # (('pos',  2), ),
    # (('pos', -2), ('pos', -1)),
    # (('pos', -1), ('pos',  0)),
    # (('pos',  0), ('pos',  1)),
    # (('pos',  1), ('pos',  2)),
    # (('pos', -2), ('pos', -1), ('pos',  0)),
    # (('pos', -1), ('pos',  0), ('pos',  1)),
    # (('pos',  0), ('pos',  1), ('pos',  2)),
    )

def feature_extractor(X):
    # Apply attribute templates to obtain features (in fact, attributes)
    crfutils.apply_templates(X, templates)
 #    if X:
	# # Append BOS and EOS features manually
 #        X[0]['F'].append('__BOS__')     # BOS feature
 #        X[-1]['F'].append('__EOS__')    # EOS feature

# Open the file and featurize it:
def featurize_file(filename):
    # Create the file handler:
    fi_handler = open(filename, 'r')

    # # Apply the hashing trick
    # hf = FeatureHasher(input_type='string',non_negative=True)

    X = crfutils.get_features(feature_extractor, fields=fields, sep=' ', fi=fi_handler)
    # X = crfutils.get_features(feature_extractor, fields=fields, sep=separator)
    # doc = []
    Y = set()
    for x in X:
        for entry in x:
            Y.add(entry['y'])
            # doc.append(entry['F'])
    # return X, hf.transform(doc)
    return X, list(Y)


if __name__ == '__main__':
    # pass
    # crfutils.main(feature_extractor, fields=fields, sep=separator)
    X = crfutils.get_features(feature_extractor, fields=fields, sep=separator)

    # Apply the hashing trick
    hf = FeatureHasher(input_type='string',non_negative=True)
    # # List of dictionaries:
    # x_set = set()
    # # Iterate over each of the tokens features:

    doc = []
    for x in X:
        # sg_tv.transform(x)
        for entry in x:
            # print entry['F']
            doc+=entry['F']
        # vec = sg_tv.transform(doc)
        print hf.transform(doc)
