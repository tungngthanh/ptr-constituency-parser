import torch
import os
import subprocess
import re
from sklearn import metrics
import math
from nltk import Tree
import numpy as np
def from_numpy(ndarray):
	return torch.from_numpy(ndarray).pin_memory().cuda(async=True)

class BatchIndices:
	"""
	    Batch indices container class (used to implement packed batches)
	    """

	def __init__(self, batch_idxs_np):
		self.batch_idxs_np = batch_idxs_np
		# Note that the torch copy will be on GPU if use_cuda is set
		self.batch_idxs_torch = from_numpy(batch_idxs_np)

		self.batch_size = int(1 + np.max(batch_idxs_np))

		batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
		self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != batch_idxs_np_extra[:-1])[0]
		self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
		assert len(self.seq_lens_np) == self.batch_size
		self.max_len = int(np.max(self.boundaries_np[1:] - self.boundaries_np[:-1]))
def evaluate_label_tree(model,eval_data, eval_metrics=['accuracy', 'macro-f1', 'precision', 'recall']):
	actuals_pointing = []
	preds_pointing = []
	actuals_label = []
	preds_label = []
	device=model.device()
	for i in range(len(eval_data['sents'])):
		model.hidden = model.init_hidden()
		batch_sentence = [eval_data['sents'][i]]
		batch_tag = [eval_data['tags'][i]]
		batch_index = [eval_data['pointing'][i]]
		batch_target = [eval_data['labels'][i]]

		#         batch_sentence_length=[len(seq) for seq in batch_sentence]
		#         batch_sentence = padding_batch(batch_sentence)
		#         batch_tag = padding_batch(batch_tag)
		#         batch_index = padding_batch(batch_sentence)
		batch_sentence = torch.as_tensor(batch_sentence, dtype=torch.long)
		batch_tag = torch.as_tensor(batch_tag, dtype=torch.long)
		batch_sentence = batch_sentence.to(device)
		batch_tag = batch_tag.to(device)
		#         batch_target = padding_batch(batch_target)
		batch_target = torch.as_tensor(batch_target, dtype=torch.long)
		batch_target = batch_target.to(device)
		batch_label_scores, batch_pointing_scores, _ = model(batch_sentence, batch_tag, [len(eval_data['sents'][i])])
		#         print(batch_tag_scores.size())
		#         print(torch.max(batch_tag_scores,2))
		#         pred=torch.max(batch_tag_scores,2).indices.tolist()[0]
		pred_pointing = torch.max(batch_pointing_scores, 2).indices.tolist()[0]
		#         pred_pointing=convert_code_to_phrase(check_pointing_code(pred_pointing))
		#         gold=eval_data['labels'][i]
		gold_pointing = eval_data['pointing'][i]
		#         gold_pointing=convert_code_to_phrase(check_pointing_code(gold_pointing))
		pred_label = torch.max(batch_label_scores, 2).indices.tolist()[0]
		#         gold=eval_data['labels'][i]
		gold_label = eval_data['labels'][i]
		#         print(gold_pointing)
		#         print(pred_pointing)
		assert len(pred_label) == len(gold_label)
		assert len(pred_pointing) == len(gold_pointing)
		preds_pointing += pred_pointing
		actuals_pointing += gold_pointing
		preds_label += pred_label
		actuals_label += gold_label
	result = {}
	for eval_method in eval_metrics:
		if eval_method == 'accuracy':
			result['accuracy'] = metrics.accuracy_score(actuals_label, preds_label), metrics.accuracy_score(
				actuals_pointing, preds_pointing)
		if eval_method == 'macro-f1':
			result['macro-f1'] = metrics.f1_score(actuals_label, preds_label, average='macro'), metrics.f1_score(
				actuals_pointing, preds_pointing, average='macro')
		if eval_method == 'precision':
			result['precision'] = metrics.precision_score(actuals_label, preds_label,
			                                              average='micro'), metrics.precision_score(actuals_pointing,
			                                                                                        preds_pointing,
			                                                                                        average='micro')
		if eval_method == 'recall':
			result['recall'] = metrics.recall_score(actuals_label, preds_label, average='micro'), metrics.recall_score(
				actuals_pointing, preds_pointing, average='micro')
	return result


class FScore(object):
	def __init__(self, recall, precision, fscore, complete_match, tagging_accuracy=100):
		self.recall = recall
		self.precision = precision
		self.fscore = fscore
		self.complete_match = complete_match
		self.tagging_accuracy = tagging_accuracy

	def __str__(self):
		if self.tagging_accuracy < 100:
			return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f}, CompleteMatch={:.2f}, TaggingAccuracy={:.2f})".format(
				self.recall, self.precision, self.fscore, self.complete_match, self.tagging_accuracy)
		else:
			return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f}, CompleteMatch={:.2f})".format(
				self.recall, self.precision, self.fscore, self.complete_match)


def evalb(evalb_dir, gold_path, predicted_path, output_path, ref_gold_path=None):
	assert os.path.exists(evalb_dir)
	evalb_program_path = os.path.join(evalb_dir, "evalb")
	evalb_spmrl_program_path = os.path.join(evalb_dir, "evalb_spmrl")
	assert os.path.exists(evalb_program_path) or os.path.exists(evalb_spmrl_program_path)
	#     output_path="output_check.txt"

	if os.path.exists(evalb_program_path):
		evalb_param_path = os.path.join(evalb_dir, "nk.prm")
	else:
		evalb_program_path = evalb_spmrl_program_path
		evalb_param_path = os.path.join(evalb_dir, "spmrl.prm")

	assert os.path.exists(evalb_program_path)
	assert os.path.exists(evalb_param_path)

	command = "{} -p {} {} {} > {}".format(
		evalb_program_path,
		evalb_param_path,
		gold_path,
		predicted_path,
		output_path,
	)
	subprocess.run(command, shell=True)

	fscore = FScore(math.nan, math.nan, math.nan, math.nan)
	with open(output_path) as infile:
		for line in infile:
			match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
			if match:
				fscore.recall = float(match.group(1))
			match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
			if match:
				fscore.precision = float(match.group(1))
			match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
			if match:
				fscore.fscore = float(match.group(1))
			match = re.match(r"Complete match\s+=\s+(\d+\.\d+)", line)
			if match:
				fscore.complete_match = float(match.group(1))
			match = re.match(r"Tagging accuracy\s+=\s+(\d+\.\d+)", line)
			if match:
				fscore.tagging_accuracy = float(match.group(1))
				break

	return fscore
def get_tree_from_score_with_special_splitting(tree_score_matrix,tree_special_score):
	# assert len(tree_score_matrix.shape) == 2 and tree_score_matrix.shape[0] == tree_score_matrix.shape[1]
	tree_matrix_size = tree_score_matrix.shape[0]
	list_index = [[0, tree_matrix_size - 1]]
	list_phrase = [[0, tree_matrix_size - 1]]

	count = 0
	while len(list_index) > 0:
		count += 1
		consider_index = list_index.pop()
		start_phrase = consider_index[0]
		end_phrase = consider_index[1]
		if end_phrase - start_phrase < 2:
			continue
		max_score = -100000
		phrase_break = 0
		for i in range(start_phrase, end_phrase):
			if i == start_phrase:
				phrase_score = tree_special_score[start_phrase, start_phrase] + tree_score_matrix[i + 1, end_phrase]
			elif i == end_phrase - 1:
				phrase_score = tree_special_score[end_phrase-1, end_phrase-1]+ tree_score_matrix[i, start_phrase]
			else:
				phrase_score = tree_score_matrix[i, start_phrase] + tree_score_matrix[i + 1, end_phrase]
			if phrase_score > max_score:
				max_score = phrase_score
				phrase_break = i
		if phrase_break - start_phrase >= 1:
			list_index.append([start_phrase, phrase_break])
			list_phrase.append([phrase_break, start_phrase])
		if end_phrase - (phrase_break + 1) >= 1:
			list_index.append([phrase_break + 1, end_phrase])
			list_phrase.append([phrase_break + 1, end_phrase])
	#     print(list_phrase)
	order_dict = {phrase_item[0]: phrase_item[1] for phrase_item in list_phrase}
	order_dict[tree_matrix_size - 1] = 0
	order_list = [order_dict[i] for i in range(tree_matrix_size)]
	return order_list

def get_pointing_special_splitting_from_score(tree_score_matrix,tree_special_score,padding_idx=-1):
	# assert len(tree_score_matrix.shape) == 2 and tree_score_matrix.shape[0] == tree_score_matrix.shape[1]
	tree_matrix_size = tree_score_matrix.shape[0]
	list_index = [[0, tree_matrix_size - 1]]
	list_phrase = [[0, tree_matrix_size - 1]]
	list_special_splitting=[padding_idx]*tree_matrix_size

	count = 0
	while len(list_index) > 0:
		count += 1
		consider_index = list_index.pop()
		start_phrase = consider_index[0]
		end_phrase = consider_index[1]
		if end_phrase - start_phrase < 2:
			continue
		max_score = float('-inf')
		phrase_break = 0
		for i in range(start_phrase, end_phrase):
			if i == start_phrase:
				phrase_score = tree_special_score[start_phrase, start_phrase] + tree_score_matrix[i + 1, end_phrase]
			elif i == end_phrase - 1:
				phrase_score = tree_special_score[end_phrase - 1, end_phrase - 1]+ tree_score_matrix[i, start_phrase]
			else:
				phrase_score = tree_score_matrix[i, start_phrase] + tree_score_matrix[i + 1, end_phrase]
			if phrase_score > max_score:
				max_score = phrase_score
				phrase_break = i
		if phrase_break - start_phrase >= 1:
			list_index.append([start_phrase, phrase_break])
			list_phrase.append([phrase_break, start_phrase])
		else:
			list_special_splitting[phrase_break]=phrase_break
		if end_phrase - (phrase_break + 1) >= 1:
			list_index.append([phrase_break + 1, end_phrase])
			list_phrase.append([phrase_break + 1, end_phrase])
		else:
			list_special_splitting[phrase_break] = phrase_break
	#     print(list_phrase)
	order_dict = {phrase_item[0]: phrase_item[1] for phrase_item in list_phrase}
	order_dict[tree_matrix_size - 1] = 0
	order_list = [order_dict[i] for i in range(tree_matrix_size)]
	return order_list, list_special_splitting
def get_tree_from_score(tree_score_matrix):
	assert len(tree_score_matrix.shape) == 2 and tree_score_matrix.shape[0] == tree_score_matrix.shape[1]
	tree_matrix_size = tree_score_matrix.shape[0]
	list_index = [[0, tree_matrix_size - 1]]
	list_phrase = [[0, tree_matrix_size - 1]]

	count = 0
	while len(list_index) > 0:
		count += 1
		consider_index = list_index.pop()
		start_phrase = consider_index[0]
		end_phrase = consider_index[1]
		if end_phrase - start_phrase < 2:
			continue
		max_score = -100000
		phrase_break = 0
		for i in range(start_phrase, end_phrase):
			if i == start_phrase:
				phrase_score = tree_score_matrix[start_phrase, end_phrase] * tree_score_matrix[i + 1, end_phrase]
			elif i == end_phrase - 1:
				phrase_score = tree_score_matrix[start_phrase, end_phrase] * tree_score_matrix[i, start_phrase]
			else:
				phrase_score = tree_score_matrix[i, start_phrase] * tree_score_matrix[i + 1, end_phrase]
			if phrase_score > max_score:
				max_score = phrase_score
				phrase_break = i
		if phrase_break - start_phrase >= 1:
			list_index.append([start_phrase, phrase_break])
			list_phrase.append([phrase_break, start_phrase])
		if end_phrase - (phrase_break + 1) >= 1:
			list_index.append([phrase_break + 1, end_phrase])
			list_phrase.append([phrase_break + 1, end_phrase])
	#     print(list_phrase)
	order_dict = {phrase_item[0]: phrase_item[1] for phrase_item in list_phrase}
	order_dict[tree_matrix_size - 1] = 0
	order_list = [order_dict[i] for i in range(tree_matrix_size)]
	return order_list

def un_Unary(tree, expandUnary=True, unaryChar="-"):
	# Traverse the tree-depth first keeping a pointer to the parent for modification purposes.
	nodeList = [(tree, [])]
	while nodeList != []:
		node, parent = nodeList.pop()
		if isinstance(node, Tree):
			# expand collapsed unary productions
			if expandUnary == True:
				unaryIndex = node.label().find(unaryChar)
				if unaryIndex != -1:
					newNode = Tree(
						node.label()[unaryIndex + 1:], [i for i in node]
					)
					node.set_label(node.label()[:unaryIndex])
					node[0:] = [newNode]
			for child in node:
				nodeList.append((child, node))
def convert_code_to_tree(pointing_code):
	left_bracket = [0] * len(pointing_code)
	right_bracket = [0] * len(pointing_code)
	list_phrase = [(0, len(pointing_code) - 1)] * len(pointing_code)
	for i, x in enumerate(pointing_code[:-1]):
		if i < x:
			left_bracket[i] += 1
			right_bracket[x] += 1
		else:
			#                 print(i)
			assert not i == x
			left_bracket[x] += 1
			right_bracket[i] += 1
	return left_bracket, right_bracket
def remove_dummy_node_label(tree,childChar="DUMMY_NODE_LABEL"):
	# Traverse the tree-depth first keeping a pointer to the parent for modification purposes.
	nodeList = [(tree, [])]
	while nodeList != []:
		node, parent = nodeList.pop()
		if isinstance(node, Tree):
			# if the node contains the 'childChar' character it means that
			# it is an artificial node and can be removed, although we still need
			# to move its children to its parent
			childIndex = node.label() == childChar
			if childIndex and (node in parent):
				nodeIndex = parent.index(node)
				parent.remove(parent[nodeIndex])
				# Generated node was on the left if the nodeIndex is 0 which
				# means the grammar was left factored.  We must insert the children
				# at the beginning of the parent's children
				if nodeIndex == 0:
					for i in range(len(node)):
						parent.insert(i, node[i])
				elif nodeIndex < len(parent):
					for i in range(len(node)):
						parent.insert(nodeIndex + i, node[i])
				else:
					parent.extend([node[i] for i in range(len(node))])
				node = parent
			for child in node:
				nodeList.append((child, node))
def nary_tree(pointing_code, sentence, tags, word_label,label):
	# print(1~=2)
	#     gold_pointing=transform_test['pointing'][0]
	#     check_sentence=word_vocabulary.convert2word(transform_test['sents'][0])
	#     check_label=label_vocabulary.convert2word(transform_test['labels'][0])
	#     check_tags=tag_vocabulary.convert2word(transform_test['tags'][0])
	left_bracket, right_bracket = convert_code_to_tree(pointing_code)
	left_bracket = ["(ROOT " * x for x in left_bracket]
	right_bracket = [") " * x for x in right_bracket]
	if len(sentence) > 1:
		tree_string = ' '.join([left_bracket[i] + sentence[i] + ' ' + right_bracket[i] for i in range(len(sentence))])
	else:
		tree_string = "(" + label[0] + " " + sentence[0] + ")"
	#     print(tree_string)
	tree = Tree.fromstring(tree_string)
	for i in range(len(tree.leaves()) - 1):
		leaf_pos = tree.leaf_treeposition(i)
		non_zero_list = [j for j in range(len(leaf_pos)) if leaf_pos[j] != 0]
		if len(non_zero_list) == 0:
			tree._label = label[0]
		elif non_zero_list[-1] == len(leaf_pos) - 1:
			zero_list = [k for k in range(len(leaf_pos)) if leaf_pos[k] == 0]
			tree[leaf_pos[:zero_list[-1] + 1]]._label = label[i]
		else:
			tree[leaf_pos[:non_zero_list[-1] + 1]]._label = label[i]
		tree[leaf_pos] = Tree.fromstring("(" + word_label[i] + " " + "(" + tags[i] + " " + tree.leaves()[i] + "))")
	last_point = len(tree.leaves()) - 1
	tree[tree.leaf_treeposition(last_point)] = Tree.fromstring(
		"(" + word_label[last_point] + " " + "(" + tags[last_point] + " " + tree.leaves()[last_point] + "))")
	#     print(' '.join(str(tree).split()))
	remove_dummy_node_label(tree, "DUMMY_WORD_LABEL")
	remove_dummy_node_label(tree)
	un_Unary(tree)
	#     tree.pretty_print()
	return ' '.join(str(tree).split())
def evaluate_anary_tree(model,eval_data,model_tag,vocab_dict,device,with_special_splitting=True, save_dir='save_result/'):
	# device = model.device
	tag_vocabulary = vocab_dict['tag_vocab']
	word_vocabulary = vocab_dict['word_vocab']
	label_vocabulary = vocab_dict['phrase_label_vocab']
	word_label_vocabulary = vocab_dict['word_label_vocab']
	gold_tree = []
	for i in range(len(eval_data['pointing'])):
		gold_pointing = eval_data['pointing'][i]
		#         print(gold_pointing)
		# sentence = word_vocabulary.convert2word(eval_data['sents'][i])
		sentence = eval_data['sents'][i]
		# tags = tag_vocabulary.convert2word(eval_data['tags'][i])
		tags = eval_data['tags'][i]
		gold_label = label_vocabulary.convert2word(eval_data['labels'][i])
		gold_wordlabel = word_label_vocabulary.convert2word(eval_data['word_labels'][i])
		gold_tree.append(nary_tree(gold_pointing, sentence, tags, gold_wordlabel, gold_label))
	pred_tree = []
	for i in range(len(eval_data['pointing'])):
		sentence = [eval_data['sents'][i]]
		tag = [eval_data['tags'][i]]
		# sentence = torch.as_tensor(sentence, dtype=torch.long)
		# tag = torch.as_tensor(tag, dtype=torch.long)
		# sentence = sentence.to(device)
		# tag = tag.to(device)
		seq_len=[len(eval_data['sents'][i])]
		# seq_idx=np.zeros(seq_len)
		# seq_idx = BatchIndices(seq_idx)
		model.eval()
		eval_input = {'sents': sentence, 'tags': tag}
		label_scores, pointing_scores, wordlabel_scores,splitting_scores = model(eval_input)
		pointing_score_data = pointing_scores.data.cpu().numpy()[0]
		splitting_score_data= splitting_scores.data.cpu().numpy()[0]
		if with_special_splitting:
			pred_pointing = get_tree_from_score_with_special_splitting(pointing_score_data,splitting_score_data)
		else:
			pred_pointing = get_tree_from_score(pointing_score_data)
		#         print(pred_pointing)
		pred_label = torch.max(label_scores, 2).indices.tolist()[0]
		pred_label = label_vocabulary.convert2word(pred_label)

		pred_wordlabel = torch.max(wordlabel_scores, 2).indices.tolist()[0]
		pred_wordlabel = word_label_vocabulary.convert2word(pred_wordlabel)
		# sentence = word_vocabulary.convert2word(eval_data['sents'][i])
		sentence = eval_data['sents'][i]
		# tags = tag_vocabulary.convert2word(eval_data['tags'][i])
		tags = eval_data['tags'][i]
		pred_tree.append(nary_tree(pred_pointing, sentence, tags, pred_wordlabel, pred_label))
	with open(save_dir+'pred_tree_' + model_tag + '.txt', 'w') as f:
		for item in pred_tree:
			f.write("%s\n" % item)
		f.close()
	with open(save_dir+'gold_tree_' + model_tag + '.txt', 'w') as f:
		for item in gold_tree:
			f.write("%s\n" % item)
		f.close()
	result = evalb("./EVALB", save_dir+"gold_tree_" + model_tag + ".txt", save_dir+ "pred_tree_" + model_tag + ".txt",
	               save_dir+"output_" + model_tag + ".txt")
	return result