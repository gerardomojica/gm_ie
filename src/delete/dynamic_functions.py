#!/bin/bash
# Luis Mojica - June. 2015
# Information Extraction - Opinion Mining - Graphical Model
# The University of Texas at Dallas
# Tests to create dynamic functions

# Global dict:
my_dic = {'uno':1, 'dos':2}

def function_creator(name):
	def foo(label):
		return my_dic[label]
	foo.__name__ = name
	return foo


def main():
	jonas_foo = function_creator('jonas')
	print jonas_foo('dos')



if __name__ == '__main__':
	main()
