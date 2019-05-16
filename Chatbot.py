# Import the packages
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as f
import pandas
import csv
import numpy
import random
import re
import os
import unicodedata
import codecs
import itertools

# Import the data files
lines_filepath = os.path.join("data/cornell movie-dialogs corpus", "movie_lines.txt")
conv_filepath = os.path.join("data/cornell movie-dialogs corpus", "movie_conversations.txt")

# Load the data files
with open(lines_filepath, 'r') as file:
	lis = file.readlines()
with open(conv_filepath, 'r') as file:
	conv = file.readlines()

# For the dialogues by each charecter
line_fields = ["lineID", "charecterID", "movieID", "charecter", "text"]
lines = {}

for line in lis:
	line = line.split(" +++$+++ ")
	# Each line need to be in ("lineID", "charecterID", "movieID", "charecter", "text") format
	# Should be indexed by lineID
	linesObj = {}
	for i, fields in enumerate(line_fields):
		linesObj[fields] = line[i].strip()
	lines[line[0]] = linesObj

# Grouping the dialogues of the same movies
conv_fields = ["charecter1ID", "charecter2ID", "movieID", "utteranceIDs"]
conversations = []

for line in conv:
	line = line.split(" +++$+++ ")
	# Each line need to be in ("charecter1ID", "charecter2ID", "movieID", "utteranceIDs") format
	convObj = {}
	for i, fields in enumerate(conv_fields):
		convObj[fields] = line[i].strip()
	lineIDs = eval(convObj["utteranceIDs"])
	convObj['lines'] = []
	# Grouping the lines of utteranceID's
	for lineID in lineIDs:
		convObj['lines'].append(lines[lineID])
	conversations.append(convObj)

print(conversations[0])