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

# Set of pair of Questions and Answers
qa_pairs = []
for conversation in conversations:
	for i in range(len(conversation['lines'])-1):
		inputline = conversation['lines'][i]['text'].strip()
		outputline = conversation['lines'][i+1]['text'].strip()
		if inputline and outputline:
			qa_pairs.append([inputline, outputline])

# Save data in text file
datafile = os.path.join("data/cornell movie-dialogs corpus", "formatted_movie_lines.txt")
delimiter = '\t'

# For normalizing the data
delimiter = str(codecs.decode(delimiter, "unicode_escape"))
with open(datafile, 'w', encoding="utf-8") as outputfile:
	writer = csv.writer(outputfile, delimiter=delimiter)
	for pair in qa_pairs:
		writer.writerow(pair)

# Processing the data
PAD_token = 0 # Used for padding short sentences
SOS_token = 1 # Start-of-sentence token <START>
EOS_token = 2 # End-of-sentence token <END>

class Vocabulary:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
		self.num_words = 3

	def addSentence(self, sentence):
		for word in sentence.split():
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.num_words += 1
		else:
			self.word2count[word] += 1

	def trim(self, min_count):
		keep_words = []
		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)
		
		self.word2index = {}
		self.word2count = {}
		self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
		self.num_words = 3

		for word in keep_words:
			self.addWord(word)

# Removing the special alphabet from words
def unicodeToAscii(s):
	return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(s) != 'Mn')

# Normalizing the words
def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	s = re.sub(r"\s+", r" ", s).strip()
	return s

datafile = os.path.join("data/cornell movie-dialogs corpus", "formatted_movie_lines.txt")
lines = open(datafile, encoding='utf-8').read().strip().split('\n')
pairs = [[normalizeString(s) for s in pair.split('\t')] for pain in lines]

voc = Vocabulary("cornell movie-dialogs corpus")

MAX_LENGTH = 10

def filterPair(p):
	return len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH

def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]

pairs = [pair for pair in pairs if len(pair)>1]
pairs = filterPairs(pairs)

for pair in pairs:
	voc.addSentence(pair[0])
	voc.addSentence(pair[1])

MIN_COUNT = 3

def trimRareWords(voc, pairs, MIN_COUNT):
	voc.trim(MIN_COUNT)
	keep_pairs = []
	for pair in pairs:
		input_sentence = pair[0]
		output_sentence = pair[1]
		keep_input = True
		keep_output = True
		for word in input_sentence.split():
			if word not in voc.word2index:
				keep_input = False
				break
		for word in output_sentence.split():
			if word not in voc.word2index:
				keep_output = False
				break

		if keep_input and keep_output:
			keep_pairs.append(pair)

	return keep_pairs

pairs = trimRareWords(voc, pairs, MIN_COUNT)

def indexesFromSentence(voc, sentence):
	return [voc.word2index[word] for word in sentence.split()] + [EOS_token]

def zeroPadding(l, fillvalue = 0):
	return list(itertools.zip_longest(*l, fillvalue = fillvalue))

def binaryMatrix(l, value = 0):
	m = []
	for i, seq in enumerate(l):
		m.append([])
		for token in seq:
			if token == PAD_token:
				m[i].append(0)
			else:
				m[i].append(1)
	return m

# 3.10.00
def inputVar(l, voc):
	indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
	lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
	padList = zeroPadding(indexes_batch)
	padVar = torch.LongTensor(padList)
	return padVar, lengths

def outputVar(l, voc):
	indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
	max_target_len = max([len(indexes) for indexes in indexes_batch])
	padList = zeroPadding(indexes_batch)
	mask = binaryMatrix(padList)
	mask = torch.ByteTensor(mask)
	padVar = torch.LongTensor(padList)
	return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
	pair_batch.sort(key = lambda x: len(x[0].split()), reverse = True)
	input_batch, output_batch = [], []
	for pair in pair_batch:
		input_batch.append(pair[0])
		output_batch.append(pair[1])
	inp, lengths = inputVar(input_batch, voc)
	output, mask, max_target_len = outputVar(output_batch, voc)
	return inp, lengths, output, mask, max_target_len

small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

# Model
class EncodeRNN(nn.Module):
	def __init__(self, hidden_size, embedding, n_layers = 1, dropout = 0):
		super(EncodeRNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.embedding = embedding
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, 
			dropout = (0 if n_layers == 1 else dropout), bidirectional = True)

	def forward(self, input_seq, input_lengths, hidden = None):
		embedded = self.embedding(input_seq)
		packed = torch.nn.utils.rnn.pack_padding_sequence(embedded, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		outputs, _ = torch.nn.utils.rnn.pad_padding_sequence(outputs)
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size]
		return outputs, hidden

class Attn(torch.nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		self.hidden_size = hidden_size

	def dot_score(self, hidden, encoder_output):
		return torch.sum(hidden * encoder_output, dim = 2)

	def forward(self, hidden, encoder_outputs):
		attn_energies = self.dot_score(hidden, encoder_outputs)
		attn_energies = attn_energies.t()
		return F.softMax(attn_energies, dim = 1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers = 1, dropout = 0.1):
		super(LuongAttnDecoderRNN, self).__init__()
		self.attn_model = attn_model
		self.embedding = embedding
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, 
			dropout = (0 if n_layers == 1 else dropout))
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
		self.attn = Attn(attn_model, hidden_size)

	def forward(self, input_step, last_hidden, encoder_outputs):
		embedded = self.embedding(input_step)
		embedded = self.embedding_dropout(embedded)
		rnn_output, hidden = self.gru(embedded, last_hidden)
		attn_weights = self.attn(rnn_output, encoder_outputs)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
		rnn_output = rnn_output.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))
		output = self.out(concat_output)
		output = F.softmax(output, dim = 1)
		return output, hidden

def maskNLLoss(decoder_output, target, mask):
	nTotal = mask.sum()
	target = target.view(-1, 1)
	gathered_tensor = tensor.gather(decoder_out, 1, target)
	crossEntropy = -torch.log(gathered_tensor)
	loss = crossEntropy.masked_select(mask)
	loss = loss.mean()
	return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder,
	embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length = MAX_LENGTH):
	encoder.optimizer.zero_grad()
	decoder.optimizer.zero_grad()
	loss = 0
	print_losses = []
	n_totals = 0
	encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
	decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
	decoder_hidden = encoder_hidden[:decoder.n_layers]
	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	if use_teacher_forcing:
		for t in range(max_target_len):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
			decoder_input = target_variable[t].view(-1, 1)
			mask_loss, nTotal = maskNLLoss(decoder_output, target_variable[t], mask[t])
			loss += mask_loss
			print_losses.append(mask_loss.item() * nTotal)
			n_totals += nTotal
	else:
		for t in range(max_target_len):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
			_, topi = decoder_output.topk(1)
			decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
			mask_loss, nTotal = maskNLLoss(decoder_output, target_variable[t], mask[t])
			loss += mask_loss
			print_losses.append(mask_loss.item() * nTotal)
			n_totals += nTotal

	loss.backward()
	_ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
	_ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
	encoder_optimizer.step()
	decoder_optimizer.step()
	return sum(print_losses) / n_totals