import subprocess
import logging
import time
from datetime import timedelta
import numpy as np

class LogFormatter():
	def __init__(self):
		self.start_time = time.time()

	def format(self, record):
		elapsed_seconds = round(record.created - self.start_time)

		prefix = "%s - %s - %s" % (
			record.levelname,
			time.strftime('%x %X'),
			timedelta(seconds=elapsed_seconds)
		)
		message = record.getMessage()
		message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
		return "%s - %s" % (prefix, message)


def create_logger(filepath, logger_name, vb=2,mode="w+"):
	"""
	    Create a logger.
	    """
	# create log formatter
	log_formatter = LogFormatter()

	# create file handler and set level to debug
	file_handler = logging.FileHandler(filepath, mode)
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(log_formatter)

	# create console handler and set level to info
	log_level = logging.DEBUG if vb == 2 else logging.INFO if vb == 1 else logging.WARNING
	console_handler = logging.StreamHandler()
	console_handler.setLevel(log_level)
	console_handler.setFormatter(log_formatter)

	# create logger and set level to debug
	logger = logging.getLogger(logger_name)
	logger.handlers = []
	logger.setLevel(logging.DEBUG)
	logger.propagate = False
	logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# reset logger elapsed time
	def reset_time():
		log_formatter.start_time = time.time()

	logger.reset_time = reset_time

	return logger

def get_gpu_memory_map():
	"""Get the current gpu free memory.

	    Returns
	    -------
	    usage: dict
	        Keys are device ids as integers.
	        Values are memory usage as integers in MB.
	    """
	result = subprocess.check_output(
		[
			'nvidia-smi', '--query-gpu=memory.free',
			'--format=csv,nounits,noheader'
		])
	# Convert lines into a dictionary
	result=result.decode('utf-8')
	print(result)
	gpu_memory = [int(x) for x in result.strip().split('\n')]
	gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
	return gpu_memory_map

class Vocab:
	def __init__(self, word2idx):
		self.word2idx = word2idx
		self.idx2word = {idx: word for word, idx in word2idx.items()}
		self.PAD = 0
		self.UNK = 1
		self.PAD_WORD = "<PAD>"
		self.UNK_WORD = "<UNK>"

	def getIdx(self, word):
		return self.word2idx.get(word, self.UNK)

	def getWord(self, idx):
		return self.idx2word.get(idx, self.UNK_WORD)

	def convert2idx(self, words):
		vec = [self.getIdx(word) for word in words]
		#         return torch.LongTensor(vec)
		return vec

	def convert2word(self, idxs):
		vec = [self.getWord(idx) for idx in idxs]
		return vec

	def size(self):
		return len(self.idx2word)
def build_vocab(build_data, pad_idx=0, unk_idx=1):
	print("Constructing vocabularies...")
	tag_vocab = {"<PAD>": pad_idx, "<UNK>": unk_idx}
	word_vocab = {"<PAD>": pad_idx, "<UNK>": unk_idx}
	label_vocab = {"<PAD>": pad_idx, "<UNK>": unk_idx}
	word_label_vocab = {"<PAD>": pad_idx, "<UNK>": unk_idx}
	char_word_vocab = {"<PAD>": pad_idx, "<UNK>": 1, "<CHAR_START>":2,"<CHAR_END>":3}
	char_tag_vocab = {"<PAD>": pad_idx, "<UNK>": 1, "<CHAR_START>":2,"<CHAR_END>":3}
	count_tag = 2
	count_word = 2
	count_label = 2
	count_word_label = 2
	count_char_word = 4
	count_char_tag = 4
	for x in build_data:
		tags_words = x[0]
		phrases_labels = x[1]
		word_labels = x[2]
		for string in tags_words:
			tag, word = string.replace('(', '').replace(')', '').split(' ')
			if not (word in ['<PAD>','<UNK>']):
				char_word = [char for char in word]
			else:
				char_word=[word]
			if not (tag in ['<PAD>','<UNK>','DUMMY_NODE_LABEL']):
				char_tag = [char.lower() for char in tag]
			else:
				char_tag=[tag]

			if not (tag in tag_vocab):
				tag_vocab[tag] = count_tag
				count_tag += 1
			if not (word in word_vocab):
				word_vocab[word] = count_word
				count_word += 1
			for char_t in char_tag:
				if not (char_t in char_tag_vocab):
					char_tag_vocab[char_t] = count_char_tag
					count_char_tag += 1
			for char_w in char_word:
				if not (char_w in char_word_vocab):
					char_word_vocab[char_w] = count_char_word
					count_char_word += 1
		for string in phrases_labels:
			label = string[1]
			if not (label in label_vocab):
				label_vocab[label] = count_label
				count_label += 1
		for word_label in word_labels:
			if not (word_label in word_label_vocab):
				word_label_vocab[word_label] = count_word_label
				count_word_label += 1
	word_vocabulary = Vocab(word_vocab)
	tag_vocabulary = Vocab(tag_vocab)
	label_vocabulary = Vocab(label_vocab)
	word_label_vocabulary = Vocab(word_label_vocab)
	char_word_vocabulary = Vocab(char_word_vocab)
	char_tag_vocabulary = Vocab(char_tag_vocab)
	return {'word_vocab':word_vocabulary,'tag_vocab':tag_vocabulary,'phrase_label_vocab':label_vocabulary,
	        'word_label_vocab':word_label_vocabulary, 'char_word_vocab':char_word_vocabulary,
	        'char_tag_vocab':char_tag_vocabulary}

def build_vocab_postag(build_data, pad_idx=0, unk_idx=1):
	print("Constructing vocabularies...")
	tag_vocab = {"<PAD>": pad_idx, "<UNK>": unk_idx}
	word_vocab = {"<PAD>": pad_idx, "<UNK>": unk_idx}
	# label_vocab = {"<PAD>": pad_idx, "<UNK>": unk_idx}
	# word_label_vocab = {"<PAD>": pad_idx, "<UNK>": unk_idx}
	char_word_vocab = {"<PAD>": pad_idx, "<UNK>": 1, "<CHAR_START>":2,"<CHAR_END>":3}
	# char_tag_vocab = {"<PAD>": pad_idx, "<UNK>": 1, "<CHAR_START>":2,"<CHAR_END>":3}
	count_tag = 2
	count_word = 2
	# count_label = 2
	# count_word_label = 2
	count_char_word = 4
	# count_char_tag = 4
	for x in build_data:
		tags_words = x[0]
		# phrases_labels = x[1]
		# word_labels = x[2]
		for string in tags_words:
			tag, word = string.replace('(', '').replace(')', '').split(' ')
			if not (word in ['<PAD>','<UNK>']):
				char_word = [char.lower() for char in word]
			else:
				char_word=[word]
			# if not (tag in ['<PAD>','<UNK>','DUMMY_NODE_LABEL']):
			# 	char_tag = [char.lower() for char in tag]
			# else:
			# 	char_tag=[tag]

			if not (tag in tag_vocab):
				tag_vocab[tag] = count_tag
				count_tag += 1
			if not (word in word_vocab):
				word_vocab[word] = count_word
				count_word += 1
			# for char_t in char_tag:
			# 	if not (char_t in char_tag_vocab):
			# 		char_tag_vocab[char_t] = count_char_tag
			# 		count_char_tag += 1
			for char_w in char_word:
				if not (char_w in char_word_vocab):
					char_word_vocab[char_w] = count_char_word
					count_char_word += 1
		# for string in phrases_labels:
		# 	label = string[1]
		# 	if not (label in label_vocab):
		# 		label_vocab[label] = count_label
		# 		count_label += 1
		# for word_label in word_labels:
		# 	if not (word_label in word_label_vocab):
		# 		word_label_vocab[word_label] = count_word_label
		# 		count_word_label += 1
	word_vocabulary = Vocab(word_vocab)
	tag_vocabulary = Vocab(tag_vocab)
	# label_vocabulary = Vocab(label_vocab)
	# word_label_vocabulary = Vocab(word_label_vocab)
	char_word_vocabulary = Vocab(char_word_vocab)
	# char_tag_vocabulary = Vocab(char_tag_vocab)
	return {'word_vocab':word_vocabulary,'tag_vocab':tag_vocabulary,'char_word_vocab':char_word_vocabulary}

def padding_batch(batch_seq, pad_idx=0):
	seq_len = [len(seq) for seq in batch_seq]
	longest_seq = max(seq_len)
	batch_size = len(batch_seq)
	padded_seq = np.ones((batch_size, longest_seq)) * pad_idx
	for i, x_len in enumerate(seq_len):
		sequence = batch_seq[i]
		padded_seq[i, 0:x_len] = sequence[:x_len]
	return padded_seq

def padding_char_batch(batch_char_seq,pad_idx=0,start_char_idx=2):
	max_len_seq = 0
	max_len_string = 0
	seq_len = [len(seq) for seq in batch_char_seq]
	for seq in batch_char_seq:
		max_len_seq = max(len(seq), max_len_seq)
		for word in seq:
			max_len_string = max(len(word), max_len_string)
	batch_size = len(batch_char_seq)
	padded_seq = np.ones((batch_size, max_len_seq, max_len_string)) * pad_idx
	padded_seq[:,:,0]=start_char_idx
	for i, x_len in enumerate(seq_len):
		sequence = batch_char_seq[i]
		word_len = [len(word) for word in sequence]
		for j, word_len in enumerate(word_len):
			padded_seq[i, j, 0:word_len] = sequence[j][:word_len]
	return padded_seq

def padding_previous_span_batch(batch_seq, pad_idx=-1):
	seq_len = [len(seq) for seq in batch_seq]
	longest_seq = max(seq_len)
	batch_size = len(batch_seq)
	padded_seq = np.ones((batch_size, longest_seq,2)) * pad_idx
	for i, x_len in enumerate(seq_len):
		sequence = batch_seq[i]
		padded_seq[i, 0:x_len] = sequence[:x_len]
	return padded_seq

def precess_arc(label):
	labels = label.split('-')
	new_arc = []
	for l in labels:
		if l == 'ADVP':
			l = 'PRT'
		# if len(new_arc) > 0 and l == new_arc[-1]:
		#     continue
		new_arc.append(l)
	label = '-'.join(new_arc)
	return label