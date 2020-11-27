import time
import warnings
import random
from collections import namedtuple
import os
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import math
from model_transformer_pretrained import Transformer_ProdAtt_ExternalEmbedding_Label
from preprocess_data_spmrl_transformer_predpos import prepare_data, convert_data
from utils import build_vocab, padding_batch
from evaluate_largest_phrase_spmrl_transformer_external_embedding import evaluate_anary_tree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import get_gpu_memory_map,create_logger
import itertools
import ast
import argparse

def keep_last_n_checkpoint(checkpoint_dir,n=5):
	import os
	import glob
	checkpoints = glob.glob(checkpoint_dir+'*.pt')
	checkpoints.sort(key=os.path.getmtime)
	num_checkpoints=len(checkpoints)
	if num_checkpoints >n:
		for old_chk in checkpoints[:num_checkpoints-n]:
			if os.path.lexists(old_chk):
				os.remove(old_chk)
def external_embedding_tag(params):
	if params.use_chars_lstm:
		return 'char_lstm'
	elif params.use_elmo:
		return 'elmo'
	elif params.use_bert:
		return 'bert'
	else:
		return 'no_external_embedding'

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
class PARAMS(object):
	def __init__(self, learning_rate, learning_rate_warmup_steps, clip_grad_norm, step_decay,
	             step_decay_factor, step_decay_patience, max_consecutive_decays, partitioned,
	             num_layers_position_only, num_layers, d_model, num_heads, d_kv, d_ff, d_label_hidden, d_tag_hidden,
	             tag_loss_scale, attention_dropout, embedding_dropout, relu_dropout, residual_dropout, use_tags,
	             use_words, tag_emb_dropout, word_emb_dropout, morpho_emb_dropout,
	             vocab_size, tagset_size, labelset_size, word_labelset_size,
	             model_architecture, batch_size, eval_interval ,save_file_name, model_tag , save_model_dir,
	             use_chars_lstm,use_elmo,use_bert,use_bert_only, tag_vocab,	word_vocab,	char_vocab,
	             d_char_emb,char_lstm_input_dropout,elmo_dropout,bert_model,bert_do_lower_case,
	             previous_training_checkpoint):
		self.learning_rate=learning_rate
		self.learning_rate_warmup_steps=learning_rate_warmup_steps
		self.clip_grad_norm=clip_grad_norm
		self.step_decay=step_decay
		self.step_decay_factor=step_decay_factor
		self.step_decay_patience=step_decay_patience
		self.max_consecutive_decays=max_consecutive_decays
		self.partitioned=partitioned
		self.num_layers_position_only=num_layers_position_only
		self.num_layers=num_layers
		self.d_model=d_model
		self.num_heads=num_heads
		self.d_kv=d_kv
		self.d_ff=d_ff
		self.d_label_hidden=d_label_hidden
		self.d_tag_hidden=d_tag_hidden
		self.tag_loss_scale=tag_loss_scale
		self.attention_dropout=attention_dropout
		self.embedding_dropout=embedding_dropout
		self.relu_dropout=relu_dropout
		self.residual_dropout=residual_dropout
		self.use_tags=use_tags
		self.use_words=use_words
		self.tag_emb_dropout=tag_emb_dropout
		self.word_emb_dropout=word_emb_dropout
		self.morpho_emb_dropout=morpho_emb_dropout
		self.vocab_size=vocab_size
		self.tagset_size=tagset_size
		self.labelset_size=labelset_size
		self.word_labelset_size=word_labelset_size
		self.model_architecture=model_architecture
		self.batch_size=batch_size
		self.eval_interval=eval_interval
		self.save_file_name=save_file_name
		self.model_tag=model_tag
		self.save_model_dir=save_model_dir
		self.timing_dropout=0.0
		self.sentence_max_len=300
		self.device=torch.device("cuda")
		self.use_chars_lstm=use_chars_lstm
		self.use_elmo=use_elmo
		self.use_bert=use_bert
		self.use_bert_only=use_bert_only
		self.tag_vocab=tag_vocab
		self.word_vocab=word_vocab
		self.char_vocab=char_vocab
		self.d_char_emb=d_char_emb
		self.char_lstm_input_dropout=char_lstm_input_dropout
		self.elmo_dropout=elmo_dropout
		self.bert_model=bert_model
		self.bert_do_lower_case=bert_do_lower_case
		self.previous_training_checkpoint = previous_training_checkpoint
		self.bert_transliterate=False

def train_parameters_parsing(params_setting, train_set,dev_set,test_set,vocab_dict,tunning_tag, current_tuning_params, language):
	device = params_setting.device
	# device = torch.device("cuda")
	BATCH_SIZE = params_setting.batch_size
	EVAL_INTERVAL = params_setting.eval_interval
	ARCHITECTURE_DIR=params_setting.save_model_dir + params_setting.model_architecture
	if not os.path.exists(ARCHITECTURE_DIR):
		os.makedirs(ARCHITECTURE_DIR)

	SAVE_FILE_NAME = ARCHITECTURE_DIR + '/'+external_embedding_tag(params_setting)+ "_pred_"
	MODEL_TAG = params_setting.model_architecture + '_pred_'+language+'_'+external_embedding_tag(params_setting)+'_'+ str(params_setting.tunning_tag)+'_'+ tunning_tag
	# logger=create_logger(params_setting.save_model_dir + params_setting.model_architecture + str(params_setting.tunning_tag)+'_'+ tunning_tag +'.log','logger_'+ tunning_tag, mode='w+')
	logger=create_logger(MODEL_TAG+'.log','logger_'+ tunning_tag, mode='w+')
	# for batch_size=20,num_layers=4,embed_dim=384,hidden_dim=384, get 90.97% f1 score in test

	if params_setting.model_architecture == 'mix_attention_external_embedding':
		model = Transformer_ProdAtt_ExternalEmbedding_Label(params_setting.tag_vocab, params_setting.word_vocab, params_setting.char_vocab,
		                                                    params_setting.labelset_size, params_setting.word_labelset_size, params_setting)

	model = model.to(params_setting.device)
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	grad_clip_threshold = np.inf if params_setting.clip_grad_norm == 0 else params_setting.clip_grad_norm
	params = sum([np.prod(p.size()) for p in model_parameters])
	logger.info('Number of parameters: %d ' %(params))
	# loss_function = nn.NLLLoss()
	# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	optimizer = torch.optim.Adam(model.parameters(), lr=1., betas=(0.9, 0.98), eps=1e-9)
	if params_setting.model_architecture == 'bilstm_product_attention' or params_setting.model_architecture == 'bilstm_product_attention_word_label' or params_setting.model_architecture == 'bilstm_product_attention_adjust':
		scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, min_lr=0.000001, verbose=True)
	elif params_setting.model_architecture in ['self_attention','mix_attention','mix_attention_external_embedding']:
		warmup_coeff = params_setting.learning_rate / params_setting.learning_rate_warmup_steps
		scheduler = ReduceLROnPlateau(optimizer, 'max', factor=params_setting.step_decay_factor,
		                              patience=params_setting.step_decay_patience, verbose=True)

		def set_lr(new_lr):
			for param_group in optimizer.param_groups:
				param_group['lr'] = new_lr

		def schedule_lr(iteration):
			iteration = iteration + 1
			if iteration <= params_setting.learning_rate_warmup_steps:
				set_lr(iteration * warmup_coeff)

	start = time.time()
	if params_setting.previous_training_checkpoint is not None:
		model.load_state_dict(torch.load(params_setting.previous_training_checkpoint))
	# with torch.no_grad():
	#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
	#     tag_scores = model(inputs)
	#     #print(tag_scores)
	best_dev_metrics = {'f1': 0, 'precision': 0, 'recall': 0}

	total_processed =0
	for epoch in range(1, params_setting.epochs + 1):  #
		total_loss = 0
		running_loss = 0
		running_loss_pointing = 0
		running_loss_label = 0
		running_loss_wordlabel = 0
		num_batches = 0
		transform_train_sents = train_set['sents']
		#     transform_train_sent_char=transform_train['sent_char']
		transform_train_tags = train_set['tags']
		#     transform_train_tag_char=transform_train['tag_char']
		transform_train_pointing = train_set['pointing']
		transform_train_labels = train_set['labels']
		transform_train_wordlabels = train_set['word_labels']
		transform_train_special_splitting = train_set['special_splitting']
		zip_list = list(zip(transform_train_sents, transform_train_tags,
		                    transform_train_pointing, transform_train_special_splitting, transform_train_labels,
		                    transform_train_wordlabels))
		random.shuffle(zip_list)
		transform_train_sents, transform_train_tags, transform_train_pointing, transform_train_special_splitting, transform_train_labels, transform_train_wordlabels = zip(
			*zip_list)
		for i in range(0, len(train_set['sents']), BATCH_SIZE):
			if i + BATCH_SIZE > len(train_set['sents']):
				continue
			model.train()
			# model.hidden = model.init_hidden()
			batch_sentence = transform_train_sents[i:i + BATCH_SIZE]
			batch_tag = transform_train_tags[i:i + BATCH_SIZE]
			batch_index = transform_train_pointing[i:i + BATCH_SIZE]
			batch_special_splitting = transform_train_special_splitting[i:i + BATCH_SIZE]
			batch_target = transform_train_labels[i:i + BATCH_SIZE]
			batch_wordtarget = transform_train_wordlabels[i:i + BATCH_SIZE]

			batch_sentence_length = [len(seq) for seq in batch_sentence]
			# batch_num_tokens = sum(batch_sentence_length)
			# batch_idxs = np.zeros(batch_num_tokens, dtype=int)
			# j = 0
			# for snum, sentence in enumerate(batch_sentence):
			# 	for _ in sentence:
			# 		batch_idxs[j] = snum
			# 		j+=1
			# batch_idxs = BatchIndices(batch_idxs)
			# batch_sentence= [word for word_seq in batch_sentence for word in word_seq]
			# batch_tag = [tag for tag_seq in batch_tag for tag in tag_seq]
			# batch_sentence = padding_batch(batch_sentence)
			# batch_tag = padding_batch(batch_tag)
			# batch_sentence = torch.as_tensor(batch_sentence, dtype=torch.long)
			# batch_tag = torch.as_tensor(batch_tag, dtype=torch.long)
			# batch_sentence = batch_sentence.to(device)
			# batch_tag = batch_tag.to(device)
			batch_target = padding_batch(batch_target)
			batch_target = torch.as_tensor(batch_target, dtype=torch.long)
			batch_target = batch_target.to(device)

			batch_wordtarget = padding_batch(batch_wordtarget)
			batch_wordtarget = torch.as_tensor(batch_wordtarget, dtype=torch.long)
			batch_wordtarget = batch_wordtarget.to(device)

			batch_index = padding_batch(batch_index, pad_idx=-1)
			batch_index = torch.as_tensor(batch_index, dtype=torch.long)
			batch_index = batch_index.to(device)

			batch_special_splitting = padding_batch(batch_special_splitting, pad_idx=-1)
			batch_special_splitting = torch.as_tensor(batch_special_splitting, dtype=torch.long)
			batch_special_splitting = batch_special_splitting.to(device)

			optimizer.zero_grad()
			# schedule_lr(num_batches)
			schedule_lr(total_processed)
			if params_setting.use_tags:
				batch_input={'sents':batch_sentence,'tags':batch_tag}
			else:
				batch_input = {'sents': batch_sentence}
			batch_label_scores, batch_pointing_scores, batch_wordlabel_scores, batch_splitting_scores = model(
				batch_input)
			#         loss = model.loss_label(batch_tag_scores,batch_target,batch_sentence_length)

			loss_pointing = model.loss_pointing(batch_pointing_scores, batch_index, batch_sentence_length,
			                                    label_pad_token=-1)
			loss_special_splitting = model.loss_pointing(batch_splitting_scores, batch_special_splitting,
			                                             batch_sentence_length, label_pad_token=-1)
			loss_label = model.loss_label(batch_label_scores, batch_target, batch_sentence_length)
			loss_wordlabel = model.loss_wordlabel(batch_wordlabel_scores, batch_wordtarget, batch_sentence_length)
			loss = loss_pointing + loss_label + loss_wordlabel + loss_special_splitting
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)
			optimizer.step()
			running_loss += loss.item()
			running_loss_pointing += loss_pointing.item()
			running_loss_label += loss_label.item()
			running_loss_wordlabel += loss_wordlabel.item()
			num_batches += 1
			total_processed +=1
			if (i / BATCH_SIZE) % EVAL_INTERVAL == 0:
				curr_loss = running_loss / num_batches
				curr_loss_pointing = running_loss_pointing / num_batches
				curr_loss_label = running_loss_label / num_batches
				curr_loss_wordlabel = running_loss_wordlabel / num_batches
				logger.info('==============================================================================')
				logger.info('epoch= %d, step= %d / %d \t exp(loss)= %.5f\t exp(loss_pointing)= %.5f'
				            '\t exp(loss_label)= %.5f\t exp(loss_wordlabel)= %.5f '
				            % (epoch, i / BATCH_SIZE, len(train_set['sents']) // BATCH_SIZE, math.exp(curr_loss), math.exp(curr_loss_pointing),
				               math.exp(curr_loss_label), math.exp(curr_loss_wordlabel)))
				# logger.info('epoch= %d, step= %d / %d \t exp(loss)= %.5f\t exp(loss_pointing)= %.5f'
				#             '\t exp(loss_label)= %.5f\t exp(loss_wordlabel)= %.5f '
				#             % (epoch, i / BATCH_SIZE, len(train_set['sents']) // BATCH_SIZE, curr_loss, curr_loss_pointing,
				#                curr_loss_label, curr_loss_wordlabel))
				logger.info('********************************************')
				curr_dev_metrics = evaluate_anary_tree(model, dev_set, MODEL_TAG, vocab_dict, params_setting.use_tags,save_dir='save_result/'+language+'_'+external_embedding_tag(params_setting)+'_')

				logger.info('dev_f1 = %.2f' % (curr_dev_metrics.fscore))
				curr_test_metrics = evaluate_anary_tree(model, test_set, MODEL_TAG, vocab_dict, params_setting.use_tags,save_dir='save_result/'+language+'_'+external_embedding_tag(params_setting)+'_')

				logger.info('test_f1 =%.2f' % (curr_test_metrics.fscore))

				if curr_dev_metrics.fscore > best_dev_metrics['f1']:
					save_best_eval = SAVE_FILE_NAME + '{0:.2f}'.format(curr_dev_metrics.fscore) + ".pt"
					torch.save(model.state_dict(), save_best_eval)
					keep_last_n_checkpoint(SAVE_FILE_NAME,n=5)
					best_dev_metrics['precision'] = curr_dev_metrics.precision
					best_dev_metrics['recall'] = curr_dev_metrics.recall
					best_dev_metrics['f1'] = curr_dev_metrics.fscore
				logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
				logger.info('best_dev_f1 = %.2f' %(best_dev_metrics['f1']))

		if total_processed > params_setting.learning_rate_warmup_steps:
			scheduler.step(best_dev_metrics['f1'])


		total_loss = running_loss / num_batches
		total_loss_label = running_loss_label / num_batches
		total_loss_wordlabel = running_loss_wordlabel / num_batches
		total_loss_pointing = running_loss_pointing / num_batches
		elapsed = time.time() - start
		logger.info('============================================')
		logger.info('epoch= %d\t exp(loss)= %.5f\t exp(loss_pointing)= %.5f'
		            '\t exp(loss_label)= %.5f\t exp(loss_wordlabel)= %.5f '
		            % (epoch, math.exp(total_loss),
		               math.exp(total_loss_pointing), math.exp(total_loss_label), math.exp(total_loss_wordlabel)))
		logger.info('********************************************')
		curr_dev_metrics = evaluate_anary_tree(model, dev_set, MODEL_TAG, vocab_dict, params_setting.use_tags,save_dir='save_result/'+language+'_'+external_embedding_tag(params_setting)+'_')
		logger.info('dev_precision=%.2f'% (curr_dev_metrics.precision))
		logger.info('dev_recall =%.2f' % (curr_dev_metrics.recall))
		logger.info('dev_f1 =%.2f' % (curr_dev_metrics.fscore))
		curr_test_metrics = evaluate_anary_tree(model, test_set, MODEL_TAG, vocab_dict, params_setting.use_tags,save_dir='save_result/'+language+'_'+external_embedding_tag(params_setting)+'_')
		logger.info('test_precision=%.2f' % (curr_test_metrics.precision))
		logger.info('test_recall =%.2f' % (curr_test_metrics.recall))
		logger.info('test_f1 =%.2f' % (curr_test_metrics.fscore))
		if curr_dev_metrics.fscore > best_dev_metrics['f1']:
			save_best_eval=SAVE_FILE_NAME+'{0:.2f}'.format(curr_dev_metrics.fscore)+".pt"
			torch.save(model.state_dict(), save_best_eval)
			keep_last_n_checkpoint(SAVE_FILE_NAME, n=5)
			best_dev_metrics['precision'] = curr_dev_metrics.precision
			best_dev_metrics['recall'] = curr_dev_metrics.recall
			best_dev_metrics['f1'] = curr_dev_metrics.fscore
		logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		logger.info('best_dev_f1 = %.2f' % ( best_dev_metrics['f1']))


	return {"best_dev_f1": best_dev_metrics['f1'],"save_model_name":MODEL_TAG, 'current_tuning_params':current_tuning_params}

def main():
	tuning_parser = argparse.ArgumentParser(description='Supervised Parsing')
	tuning_parser.add_argument("--tunning_dict", type=str, default='./tunning_dict_mix_attention_external_embedding.txt', help="tunning dictionary directory")
	tuning_parser.add_argument("--start_tunning", type=int, default=0,help='starting point')
	tuning_parser.add_argument("--end_tunning", type=int, default=1, help='end point')
	tuning_parser.add_argument("--external_embedding", type=str, default='char_lstm', help='external_embedding')
	tuning_parser.add_argument("--language", type=str, default="", help="language dataset")
	tuning_parser.add_argument("--use_tags", action='store_true', help="language dataset")
	tuning_parser.add_argument("--bert_model", type=str, help="bert_model")
	tuning_parser.add_argument("--learning_rate", type=float,default=0.0008, help="learning_rate")
	tuning_parser.add_argument("--previous_training_checkpoint", type=str, default='', help='end point')
	tuning_parser.add_argument("--bert_transliterate", action='store_true', help="language dataset")
	tuning_params = tuning_parser.parse_args()
	language = tuning_params.language
	Languages = ['Basque', 'French', 'German', 'Hebrew', 'Hungarian', 'Korean', 'Polish', 'swedish']
	assert language in Languages, "We do not support ['Basque','French','German','Hebrew','Hungarian','Korean','Polish','swedish'] here"
	train_data,dev_data,test_data=prepare_data(language)
	vocab_dict=build_vocab(train_data)
	tag_vocabulary = vocab_dict['tag_vocab']
	word_vocabulary = vocab_dict['word_vocab']
	label_vocabulary = vocab_dict['phrase_label_vocab']
	word_label_vocabulary = vocab_dict['word_label_vocab']
	char_vocabulary=vocab_dict['char_word_vocab']
	transform_train=convert_data(train_data,vocab_dict)
	transform_dev=convert_data(dev_data,vocab_dict)
	transform_test=convert_data(test_data,vocab_dict)
	# Config = namedtuple('Config',field_names='EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAGSET_SIZE, LABELSET_SIZE, WORD_LABELSET_SIZE,'
	#                     'NUM_LAYERS_PHRASE, NUM_LAYERS_WORD, BATCH_SIZE, EVAL_INTERVAL, SAVE_FILE_NAME, MODEL_TAG, SAVE_MODEL_DIR,'
	#                     'lr,lr_warmup_steps, epochs, weight_decay,step_decay_patience,step_decay_factor,'
	#                     'DATA_DIR, device, with_special_splitting, model_architecture, num_head')
	# args = Config( 384, 384, word_vocabulary.size(),tag_vocabulary.size(),label_vocabulary.size(),word_label_vocabulary.size(),
	#                4,1,20,1000,'dummy_label_bilstm','largest_phrase_pointing','../Save_models/Low_cost_parsing',
	#                0.001,500,100,1e-6,2,0.5,
	#                '../Data/WSJ_parsing_clean',torch.device("cuda"),True,'bilstm_product_attention_word_label',4)
	# Config = namedtuple('Config',
	#                     field_names='learning_rate, learning_rate_warmup_steps, clip_grad_norm, step_decay, step_decay_factor,'
	#                                 'step_decay_patience, max_consecutive_decays, partitioned, num_layers_position_only, num_layers, d_model,'
	#                                 'num_heads, d_kv, d_ff, d_label_hidden, d_tag_hidden, tag_loss_scale, attention_dropout, embedding_dropout,'
	#                                 'relu_dropout, residual_dropout, use_tags, use_words, tag_emb_dropout, word_emb_dropout, morpho_emb_dropout,'
	#                                 'VOCAB_SIZE, TAGSET_SIZE, LABELSET_SIZE, WORD_LABELSET_SIZE,'
	#                                 'model_architecture, BATCH_SIZE, EVAL_INTERVAL ,SAVE_FILE_NAME, MODEL_TAG , SAVE_MODEL_DIR',
	#                                 'use_chars_lstm,use_elmo,use_bert,use_bert_only, tag_vocab,word_vocab,char_vocab'
	#                                 'd_char_emb,char_lstm_input_dropout,elmo_dropout,bert_model,bert_do_lower_case')
	standard_params=PARAMS( 0.0008,200,0.0,True, 0.5,
	                        5, 3, True, 0, 8, 1024,
	                        8, 64, 2048, 250, 250, 5.0, 0.2, 0.0,
	                        0.1, 0.2, True, True, 0.2,0.4,0.2,
	                        word_vocabulary.size(),tag_vocabulary.size(), label_vocabulary.size(), word_label_vocabulary.size(),
	                        'mix_attention',100,100,'mix_attention','largest_phrase_pointing','../Save_models/Low_cost_parsing/',
	                        False,False,False,False,tag_vocabulary,word_vocabulary,char_vocabulary,64,0.2,0.5,"bert-base-uncased", True,None)
	if os.path.exists(tuning_params.previous_training_checkpoint):
		standard_params.previous_training_checkpoint=tuning_params.previous_training_checkpoint
	else:
		print('we do not find any previous checkpoint')
	standard_params.use_tags=tuning_params.use_tags
	standard_params.learning_rate = tuning_params.learning_rate
	standard_params.bert_model = tuning_params.bert_model
	standard_params.bert_transliterate = tuning_params.bert_transliterate
	if standard_params.bert_model=="bert-base-multilingual-cased":
		standard_params.bert_model="../Data/pretrain_embedding/bert/bert-base-multilingual-cased.tar.gz"
		standard_params.bert_do_lower_case=False
	if tuning_params.external_embedding=="char_lstm":
		standard_params.use_chars_lstm=True
	elif tuning_params.external_embedding=="elmo":
		standard_params.use_elmo = True
	elif tuning_params.external_embedding=="bert":
		standard_params.use_bert = True
	with open(tuning_params.tunning_dict) as f:
		str_tunning_dicts=f.readline()
	tunning_dicts = ast.literal_eval(str_tunning_dicts)
	tunning_dict_list = []
	hps, values = zip(*tunning_dicts.items())
	for v in itertools.product(*values):
		tunning_dict_list.append(dict(zip(hps, v)))
	# tunning_dict_list=[ast.literal_eval(x) for x in str_tunning_dict_list]
	# tunning_dict = {'embedding_dim': [384], 'hidden_dim': [384], 'num_layers_phrase': [4],
	#                 'batch_size': [20], 'eval_interval': [1000], 'epochs': [100]}
	embedding_tag=external_embedding_tag(standard_params)
	SAVE_MODEL_DIR = standard_params.save_model_dir+language
	if not os.path.exists(SAVE_MODEL_DIR):
		os.makedirs(SAVE_MODEL_DIR)

	general_log_name=standard_params.save_model_dir + standard_params.model_architecture +'_pred_'+language+'_'+embedding_tag +'_'+str(tunning_dict_list[0]['tunning_tag'])+'.log'
	summarize_log=create_logger(general_log_name, 'summerize', mode='a')
	for i in range(tuning_params.start_tunning, min(len(tunning_dict_list),tuning_params.end_tunning)):
		cur_hps=tunning_dict_list[i]
		cur_params=deepcopy(standard_params)
		for key_item in cur_hps:
			setattr(cur_params,key_item,cur_hps[key_item])
		summarize_log.info(vars(cur_params))
		cur_params.save_model_dir = cur_params.save_model_dir + language + "/"
		result_iter = train_parameters_parsing(cur_params, transform_train,transform_dev,transform_test, vocab_dict,str(i+1),cur_hps,language)
		summarize_log.info(result_iter)

	# hps, values = zip(*tunning_dict.items())
	# summarize_log = create_logger(standard_params.save_model_dir + standard_params.model_architecture + 'tunning.log',
	#                               mode='w+')
	# for v in itertools.product(*values):
	# 	cur_hps = dict(zip(hps, v))
	# 	cur_params = deepcopy(standard_params)
	# 	for key_item in cur_hps:
	# 		setattr(cur_params, key_item, cur_hps[key_item])
	# 	count += 1
	# 	summarize_log.info(vars(cur_params))
	# 	result_iter = train_parameters_parsing(cur_params, transform_train, transform_dev, transform_test,
	# 	                                       vocab_dict, str(count))
	# 	summarize_log.info(result_iter)

if __name__ == "__main__":
	main()