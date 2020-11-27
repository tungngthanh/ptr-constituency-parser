import nltk
from nltk.tree import Tree
from functools import reduce
from nltk import tree, treetransforms
from copy import deepcopy
from collections import deque
import queue
import argparse
import pickle
from datetime import datetime
from utils import build_vocab,precess_arc
DATA_DIR='../Data/WSJ_parsing_clean/'
TRAIN_PATH=DATA_DIR+'02-21.10way.clean'
DEV_PATH=DATA_DIR+'22.auto.clean'
TEST_PATH=DATA_DIR+'23.auto.clean'

def finding_right_most_sibling(tree, binary_format=True):
	if not binary_format:
		print("so far we only have for the binary format")
	assert binary_format == True
	index_list = []
	for i in range(len(tree.leaves())):
		leaf_pos = tree.leaf_treeposition(i)
		non_zero_list = [j for j in range(len(leaf_pos)) if leaf_pos[j] != 0]
		if len(non_zero_list) == 0:
			index_list.append([[i, len(tree.leaves()) - 1], tree[0].label()])
		elif non_zero_list[-1] == len(leaf_pos) - 2:
			zero_list = [k for k in range(len(leaf_pos)) if leaf_pos[k] == 0]
			subtree = tree[leaf_pos[:zero_list[-2] + 1]]
			if '|' in subtree.label():
				index_list.append([[i, i], 'DUMMY_NODE_LABEL'])
			else:
				index_list.append([[i, i], subtree.label()])

		else:
			subtree = tree[leaf_pos[:non_zero_list[-1] + 1]]
			if '|' in subtree.label():
				index_list.append([[i, i + len(subtree.leaves()) - 1], 'DUMMY_NODE_LABEL'])
			else:
				index_list.append([[i, i + len(subtree.leaves()) - 1], subtree.label()])
	#             subtree.pretty_print()
	return index_list


def finding_biggest_phrase_sibling(tree, binary_format=True):
	if not binary_format:
		print("so far we only have for the binary format")
	assert binary_format == True
	index_list = []
	for i in range(len(tree.leaves())):
		leaf_pos = tree.leaf_treeposition(i)
		non_zero_list = [j for j in range(len(leaf_pos)) if leaf_pos[j] != 0]
		#         last_child_list=[len(tree[leaf_pos[:j]])-1 for j in range(len(leaf_pos))]
		#         non_last_child_list=[j for j in range(len(leaf_pos)) if leaf_pos[j] != last_child_list[j]]
		if len(non_zero_list) == 0:
			index_list.append([[i, len(tree.leaves()) - 1], precess_arc(tree[0].label())])
		else:
			subtree_pos = leaf_pos[:non_zero_list[-1] + 1]
			subtree = tree[subtree_pos]
			if len(subtree) > 1:
				if '|' in subtree.label():
					index_list.append([[i, i + len(subtree.leaves()) - 1], 'DUMMY_NODE_LABEL'])
				else:
					index_list.append([[i, i + len(subtree.leaves()) - 1], precess_arc(subtree.label())])
			else:
				last_child_subtree_list = [len(tree[subtree_pos[:j]]) - 1 for j in range(len(subtree_pos))]
				non_last_child_subtree = [j for j in range(len(subtree_pos)) if
				                          subtree_pos[j] != last_child_subtree_list[j]]
				if len(non_last_child_subtree) == 0:
					index_list.append([[i, 0], precess_arc(tree[0].label())])
				else:
					subtree = tree[subtree_pos[:non_last_child_subtree[-1] + 1]]
					if '|' in subtree.label():
						index_list.append([[i, i - len(subtree.leaves()) + 1], 'DUMMY_NODE_LABEL'])
					else:
						index_list.append([[i, i - len(subtree.leaves()) + 1], precess_arc(subtree.label())])
	#             subtree.pretty_print()
	return index_list

def finding_special_splitting_point(tree, binary_format=True, padding_item=-1):
	index_list = finding_biggest_phrase_sibling(tree, binary_format)
	special_list = [padding_item] * len(index_list)
	for i in range(len(index_list) - 1):
		if index_list[i][0][1] == index_list[i+1][0][1]:
			special_list[i] = i
		elif index_list[index_list[i+1][0][1]][0][1] == i:
			special_list[i] = i
		elif index_list[index_list[i][0][1]][0][1] == i+1:
			special_list[i] = i
		else:
			special_list[i] = padding_item
	return special_list

class timeit():

	def __enter__(self):
		self.tic = self.datetime.now()

	def __exit__(self, *args, **kwargs):
		print('runtime: {}'.format(self.datetime.now() - self.tic))

def prepare_data(train_path=TRAIN_PATH,dev_path=DEV_PATH,test_path=TEST_PATH, map_method='biggest_phrase'):
	assert map_method in ['right_most', 'biggest_phrase'],'We do not support that map method here'
	print("Convert data to binary tree")
	print("Here we use chomsky normal by splitting on right")
	with open(train_path) as f:
		train_raw=f.readlines()
		train_raw=[x.strip() for x in train_raw]
	with open(dev_path) as f:
		dev_raw=f.readlines()
		dev_raw=[x.strip() for x in dev_raw]
	with open(test_path) as f:
		test_raw=f.readlines()
		test_raw=[x.strip() for x in test_raw]
	check=datetime.now()
	train_data = []
	for i in range(len(train_raw)):
		sent = train_raw[i]
		sent_tree = nltk.Tree.fromstring(sent)
		sent_tree.collapse_unary(sent_tree, joinChar="-")
		sent_tree.chomsky_normal_form()
		sent_pos = [sent_tree[sent_tree.leaf_treeposition(j)[:-1]] for j in range(len(sent_tree.leaves()))]
		sent_pos = [' '.join(str(x).split()) for x in sent_pos]
		sent_word_label = ['DUMMY_WORD_LABEL'] * len(sent_tree.leaves())
		for j in range(len(sent_tree.leaves())):
			if len(sent_tree.leaf_treeposition(j)) > 2:
				if len(sent_tree[sent_tree.leaf_treeposition(j)[:-2]]) == 1:
					sent_word_label[j] = sent_tree[sent_tree.leaf_treeposition(j)[:-2]].label()
		#     sent_tree.pretty_print()
		if map_method=='biggest_phrase':
			sent_output = finding_biggest_phrase_sibling(sent_tree)
		elif map_method=='right_most':
			sent_output = finding_right_most_sibling(sent_tree)
		else:
			raise NotImplementedError
		sent_special_point = finding_special_splitting_point(sent_tree)
		sent_word_label = [precess_arc(x) for x in sent_word_label]
		train_data.append([sent_pos, sent_output, sent_word_label, sent_special_point])
	dev_data = []
	for i in range(len(dev_raw)):
		sent = dev_raw[i]
		sent_tree = nltk.Tree.fromstring(sent)
		sent_tree.collapse_unary(sent_tree, joinChar="-")
		sent_tree.chomsky_normal_form()
		sent_pos = [sent_tree[sent_tree.leaf_treeposition(j)[:-1]] for j in range(len(sent_tree.leaves()))]
		sent_pos = [' '.join(str(x).split()) for x in sent_pos]
		sent_word_label = ['DUMMY_WORD_LABEL'] * len(sent_tree.leaves())
		for j in range(len(sent_tree.leaves())):
			if len(sent_tree.leaf_treeposition(j)) > 2:
				if len(sent_tree[sent_tree.leaf_treeposition(j)[:-2]]) == 1:
					sent_word_label[j] = sent_tree[sent_tree.leaf_treeposition(j)[:-2]].label()
		#     sent_tree.pretty_print()
		if map_method == 'biggest_phrase':
			sent_output = finding_biggest_phrase_sibling(sent_tree)
		elif map_method == 'right_most':
			sent_output = finding_right_most_sibling(sent_tree)
		else:
			raise NotImplementedError
		sent_special_point = finding_special_splitting_point(sent_tree)
		sent_word_label = [precess_arc(x) for x in sent_word_label]
		dev_data.append([sent_pos, sent_output, sent_word_label, sent_special_point])
	test_data = []
	for i in range(len(test_raw)):
		sent = test_raw[i]
		sent_tree = nltk.Tree.fromstring(sent)
		sent_tree.collapse_unary(sent_tree, joinChar="-")
		sent_tree.chomsky_normal_form()
		sent_pos = [sent_tree[sent_tree.leaf_treeposition(j)[:-1]] for j in range(len(sent_tree.leaves()))]
		sent_pos = [' '.join(str(x).split()) for x in sent_pos]
		sent_word_label = ['DUMMY_WORD_LABEL'] * len(sent_tree.leaves())
		for j in range(len(sent_tree.leaves())):
			if len(sent_tree.leaf_treeposition(j)) > 2:
				if len(sent_tree[sent_tree.leaf_treeposition(j)[:-2]]) == 1:
					sent_word_label[j] = sent_tree[sent_tree.leaf_treeposition(j)[:-2]].label()
		#     sent_tree.pretty_print()
		if map_method == 'biggest_phrase':
			sent_output = finding_biggest_phrase_sibling(sent_tree)
		elif map_method == 'right_most':
			sent_output = finding_right_most_sibling(sent_tree)
		else:
			raise NotImplementedError
		sent_special_point = finding_special_splitting_point(sent_tree)
		sent_word_label=[precess_arc(x) for x in sent_word_label]
		test_data.append([sent_pos, sent_output, sent_word_label, sent_special_point])
	print(datetime.now() - check)
	return train_data, dev_data, test_data



#%%


#%% md


#%%
def convert_data(data_raw, vocab_dict):
	indices_data = {'sents': [], 'tags': [], 'pointing': [], 'labels': [], 'word_labels': [],'special_splitting':[],'sent_char':[],'tag_char':[]}
	tag_vocabulary=vocab_dict['tag_vocab']
	word_vocabulary = vocab_dict['word_vocab']
	label_vocabulary = vocab_dict['phrase_label_vocab']
	word_label_vocabulary = vocab_dict['word_label_vocab']
	char_word_vocabulary = vocab_dict['char_word_vocab']
	char_tag_vocabulary=vocab_dict['char_tag_vocab']
	for x in data_raw:
		tags_words = x[0]
		tags = []
		words = []
		# char_sent = []
		# char_tags = []
		for string in tags_words:
			tag, word = string.replace('(', '').replace(')', '').split(' ')
			# if not (word in ['<PAD>','<UNK>']):
			# 	char_word = [char.lower() for char in word]
			# 	char_word.insert(0,"<CHAR_START>")
			# else:
			# 	char_word=[word]
			# 	char_word.insert(0, "<CHAR_START>")
			# if not (tag in ['<PAD>','<UNK>','DUMMY_NODE_LABEL','DUMMY_WORD_LABEL']):
			# 	char_tag = [char.lower() for char in tag]
			# 	char_tag.insert(0, "<CHAR_START>")
			# 	char_tag.append("<CHAR_END>")
			# else:
			# 	char_tag=[tag]
			# 	char_tag.insert(0, "<CHAR_START>")
			# 	char_tag.append("<CHAR_END>")
			tags.append(tag)
			words.append(word)
			# char_sent.append(char_word)
			# char_tags.append(char_tag)
		# t = tag_vocabulary.convert2idx(tags)
		# s = word_vocabulary.convert2idx(words)
		# t_char = [char_tag_vocabulary.convert2idx(tags_split) for tags_split in char_tags]
		# s_char = [char_word_vocabulary.convert2idx(words_split) for words_split in char_sent]
		indices_data['sents'].append(words)
		indices_data['tags'].append(tags)
		phrases_labels = x[1]
		indices_data['pointing'].append([string[0][1] for string in phrases_labels])
		indices_data['labels'].append(label_vocabulary.convert2idx([string[1] for string in phrases_labels]))
		word_labels = x[2]
		# indices_data['sent_char'].append(s_char)
		# indices_data['tag_char'].append(t_char)
		indices_data['word_labels'].append(word_label_vocabulary.convert2idx([string for string in word_labels]))
		indices_data['special_splitting'].append(x[3])
	assert (len(indices_data['pointing']) == len(indices_data['labels'])) and (
			len(indices_data['pointing']) == len(indices_data['sents'])) and (
			len(indices_data['pointing']) == len(indices_data['tags'])) and (
			len(indices_data['pointing']) == len(indices_data['special_splitting']))
	return indices_data
# transform_train=convert_data(train_data)
# transform_dev=convert_data(dev_data)
# transform_test=convert_data(test_data)
#
# #%%
#
# print([transform_test[key][0] for key in transform_test])
# print(word_vocabulary.convert2word(transform_test['sents'][0]))
# print(tag_vocabulary.convert2word(transform_test['tags'][0]))
# print(label_vocabulary.convert2word(transform_test['labels'][0]))
# print(word_label_vocabulary.convert2word(transform_test['word_labels'][0]))


