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

lines_filepath = os.path.join("data/cornell movie-dialogs corpus", "movie_lines.txt")
conv_filepath = os.path.join("data/cornell movie-dialogs corpus", "movie_conversations.txt")

with open(lines_filepath, 'r') as file:
	lines = file.readlines()

print(lines[0])