"""
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

from submission import *
from util import *


def main():
	with open('weights', 'r') as f:
		weights = {}
		for line in f:
			tokens = line.split()
			weights[tokens[0]] = float(tokens[1])
	interactivePrompt(extractWordFeatures, weights)


if __name__ == '__main__':
	main()
