#!/bin/bash
# Luis Mojica - June. 2015
# Information Extraction - Opinion Mining - Graphical Model
# The University of Texas at Dallas
# Linear Chain CRF with open gm

import sys
import getopt
# from ..template import featurize
import template.featurize
# from ..template

def main(argv):
    print argv
    options, remainder = getopt.getopt(argv, 'o:v', ['output=',
        'verbose','version=',])


if __name__ == '__main__':
    main(sys.argv[1:])


