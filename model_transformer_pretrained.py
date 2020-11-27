import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
torch_t = torch.cuda

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--", # en dash
    "\u2014": "--", # em dash
    }

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

class CharacterLSTM(nn.Module):
	def __init__(self, num_embeddings, d_embedding, d_out,
	             char_dropout=0.0,
	             normalize=False,
	             **kwargs):
		super().__init__()

		self.d_embedding = d_embedding
		self.d_out = d_out

		self.lstm = nn.LSTM(self.d_embedding, self.d_out // 2, num_layers=1, bidirectional=True)

		self.emb = nn.Embedding(num_embeddings, self.d_embedding, **kwargs)
		# TODO(nikita): feature-level dropout?
		self.char_dropout = nn.Dropout(char_dropout)

		if normalize:
			print("This experiment: layer-normalizing after character LSTM")
			self.layer_norm = LayerNormalization(self.d_out, affine=False)
		else:
			self.layer_norm = lambda x: x

	def forward(self, chars_padded_np, word_lens_np, batch_idxs):
		# copy to ensure nonnegative stride for successful transfer to pytorch
		decreasing_idxs_np = np.argsort(word_lens_np)[::-1].copy()
		decreasing_idxs_torch = from_numpy(decreasing_idxs_np)

		chars_padded = from_numpy(chars_padded_np[decreasing_idxs_np])
		word_lens = from_numpy(word_lens_np[decreasing_idxs_np])
		# print(chars_padded_np)
		# print(len(chars_padded_np))
		# input()

		inp_sorted = nn.utils.rnn.pack_padded_sequence(chars_padded, word_lens_np[decreasing_idxs_np], batch_first=True)
		inp_sorted_emb = nn.utils.rnn.PackedSequence(
			self.char_dropout(self.emb(inp_sorted.data)),
			inp_sorted.batch_sizes)
		_, (lstm_out, _) = self.lstm(inp_sorted_emb)

		lstm_out = torch.cat([lstm_out[0], lstm_out[1]], -1)

		# Undo sorting by decreasing word length
		res = torch.zeros_like(lstm_out)
		res.index_copy_(0, decreasing_idxs_torch, lstm_out)

		res = self.layer_norm(res)
		return res

# %%
def get_elmo_class():
	# Avoid a hard dependency by only importing Elmo if it's being used
	from allennlp.modules.elmo import Elmo
	return Elmo

# %%
def get_bert(bert_model, bert_do_lower_case):
	# Avoid a hard dependency on BERT by only importing it if it's being used
	from pytorch_pretrained_bert import BertTokenizer, BertModel
	if bert_model.endswith('.tar.gz'):
		tokenizer = BertTokenizer.from_pretrained(bert_model.replace('.tar.gz', '-vocab.txt'),
		                                          do_lower_case=bert_do_lower_case)
	else:
		tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bert_do_lower_case)
	bert = BertModel.from_pretrained(bert_model)
	return tokenizer, bert
class TransformerEncoderProdAtt(nn.Module):
	def __init__(self, embedding,
	             num_layers=1, num_heads=2, d_kv=32, d_ff=1024,
	             d_positional=None,
	             num_layers_position_only=0,
	             relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1):
		super().__init__()
		# Don't assume ownership of the embedding as a submodule.
		# TODO(nikita): what's the right thing to do here?
		self.embedding_container = [embedding]
		d_model = embedding.d_embedding

		d_k = d_v = d_kv

		self.stacks = []
		for i in range(num_layers):
			attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, residual_dropout=residual_dropout,
			                          attention_dropout=attention_dropout, d_positional=d_positional)
			if d_positional is None:
				ff = PositionwiseFeedForward(d_model, d_ff, relu_dropout=relu_dropout,
				                             residual_dropout=residual_dropout)
			else:
				ff = PartitionedPositionwiseFeedForward(d_model, d_ff, d_positional, relu_dropout=relu_dropout,
				                                        residual_dropout=residual_dropout)

			self.add_module(f"attn_{i}", attn)
			self.add_module(f"ff_{i}", ff)
			self.stacks.append((attn, ff))

		self.num_layers_position_only = num_layers_position_only
		if self.num_layers_position_only > 0:
			assert d_positional is None, "num_layers_position_only and partitioned are incompatible"
	def forward(self, xs, batch_idxs, extra_content_annotations=None):
		emb = self.embedding_container[0]
		res, timing_signal, batch_idxs = emb(xs, batch_idxs, extra_content_annotations=extra_content_annotations)

		for i, (attn, ff) in enumerate(self.stacks):
			if i >= self.num_layers_position_only:
				res, current_attns = attn(res, batch_idxs)
			else:
				res, current_attns = attn(res, batch_idxs, qk_inp=timing_signal)
			res = ff(res, batch_idxs)

		return res, batch_idxs


class LayerNormalization(nn.Module):
	def __init__(self, d_hid, eps=1e-3, affine=True):
		super(LayerNormalization, self).__init__()

		self.eps = eps
		self.affine = affine
		if self.affine:
			self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
			self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

	def forward(self, z):
		if z.size(-1) == 1:
			return z

		mu = torch.mean(z, keepdim=True, dim=-1)
		sigma = torch.std(z, keepdim=True, dim=-1)
		ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
		if self.affine:
			ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

		# NOTE(nikita): the t2t code does the following instead, with eps=1e-6
		# However, I currently have no reason to believe that this difference in
		# implementation matters.
		# mu = torch.mean(z, keepdim=True, dim=-1)
		# variance = torch.mean((z - mu.expand_as(z))**2, keepdim=True, dim=-1)
		# ln_out = (z - mu.expand_as(z)) * torch.rsqrt(variance + self.eps).expand_as(z)
		# ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

		return ln_out


class ScaledDotProductAttention(nn.Module):
	def __init__(self, d_model, attention_dropout=0.1):
		super(ScaledDotProductAttention, self).__init__()
		self.temper = d_model ** 0.5
		self.dropout = nn.Dropout(attention_dropout)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, q, k, v, attn_mask=None):
		# q: [batch, slot, feat]
		# k: [batch, slot, feat]
		# v: [batch, slot, feat]

		attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

		if attn_mask is not None:
			assert attn_mask.size() == attn.size(), \
				'Attention mask shape {} mismatch ' \
				'with Attention logit tensor shape ' \
				'{}.'.format(attn_mask.size(), attn.size())

			attn.data.masked_fill_(attn_mask, -float('inf'))

		attn = self.softmax(attn)
		# Note that this makes the distribution not sum to 1. At some point it
		# may be worth researching whether this is the right way to apply
		# dropout to the attention.
		# Note that the t2t code also applies dropout in this manner
		attn = self.dropout(attn)
		output = torch.bmm(attn, v)

		return output, attn


class MultiHeadAttention(nn.Module):
	"""
	Multi-head attention module
	"""

	def __init__(self, n_head, d_model, d_k, d_v, residual_dropout=0.1, attention_dropout=0.1, d_positional=None):
		super(MultiHeadAttention, self).__init__()

		self.n_head = n_head
		self.d_k = d_k
		self.d_v = d_v

		if d_positional is None:
			self.partitioned = False
		else:
			self.partitioned = True

		if self.partitioned:
			self.d_content = d_model - d_positional
			self.d_positional = d_positional

			self.w_qs1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_k // 2))
			self.w_ks1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_k // 2))
			self.w_vs1 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_content, d_v // 2))

			self.w_qs2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_k // 2))
			self.w_ks2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_k // 2))
			self.w_vs2 = nn.Parameter(torch_t.FloatTensor(n_head, self.d_positional, d_v // 2))

			init.xavier_normal_(self.w_qs1)
			init.xavier_normal_(self.w_ks1)
			init.xavier_normal_(self.w_vs1)

			init.xavier_normal_(self.w_qs2)
			init.xavier_normal_(self.w_ks2)
			init.xavier_normal_(self.w_vs2)
		else:
			self.w_qs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
			self.w_ks = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
			self.w_vs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_v))

			init.xavier_normal_(self.w_qs)
			init.xavier_normal_(self.w_ks)
			init.xavier_normal_(self.w_vs)

		self.attention = ScaledDotProductAttention(d_model, attention_dropout=attention_dropout)
		self.layer_norm = LayerNormalization(d_model)

		if not self.partitioned:
			# The lack of a bias term here is consistent with the t2t code, though
			# in my experiments I have never observed this making a difference.
			self.proj = nn.Linear(n_head * d_v, d_model, bias=False)
		else:
			self.proj1 = nn.Linear(n_head * (d_v // 2), self.d_content, bias=False)
			self.proj2 = nn.Linear(n_head * (d_v // 2), self.d_positional, bias=False)

		self.residual_dropout = FeatureDropout(residual_dropout)

	def split_qkv_packed(self, inp, qk_inp=None):
		v_inp_repeated = inp.repeat(self.n_head, 1).view(self.n_head, -1, inp.size(-1))  # n_head x len_inp x d_model
		if qk_inp is None:
			qk_inp_repeated = v_inp_repeated
		else:
			qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(self.n_head, -1, qk_inp.size(-1))

		if not self.partitioned:
			q_s = torch.bmm(qk_inp_repeated, self.w_qs)  # n_head x len_inp x d_k
			k_s = torch.bmm(qk_inp_repeated, self.w_ks)  # n_head x len_inp x d_k
			v_s = torch.bmm(v_inp_repeated, self.w_vs)  # n_head x len_inp x d_v
		else:
			q_s = torch.cat([
				torch.bmm(qk_inp_repeated[:, :, :self.d_content], self.w_qs1),
				torch.bmm(qk_inp_repeated[:, :, self.d_content:], self.w_qs2),
			], -1)
			k_s = torch.cat([
				torch.bmm(qk_inp_repeated[:, :, :self.d_content], self.w_ks1),
				torch.bmm(qk_inp_repeated[:, :, self.d_content:], self.w_ks2),
			], -1)
			v_s = torch.cat([
				torch.bmm(v_inp_repeated[:, :, :self.d_content], self.w_vs1),
				torch.bmm(v_inp_repeated[:, :, self.d_content:], self.w_vs2),
			], -1)
		return q_s, k_s, v_s

	def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
		# Input is padded representation: n_head x len_inp x d
		# Output is packed representation: (n_head * mb_size) x len_padded x d
		# (along with masks for the attention and output)
		n_head = self.n_head
		d_k, d_v = self.d_k, self.d_v

		len_padded = batch_idxs.max_len
		mb_size = batch_idxs.batch_size
		q_padded = q_s.new_zeros((n_head, mb_size, len_padded, d_k))
		k_padded = k_s.new_zeros((n_head, mb_size, len_padded, d_k))
		v_padded = v_s.new_zeros((n_head, mb_size, len_padded, d_v))
		invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=torch.uint8)

		for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
			q_padded[:, i, :end - start, :] = q_s[:, start:end, :]
			k_padded[:, i, :end - start, :] = k_s[:, start:end, :]
			v_padded[:, i, :end - start, :] = v_s[:, start:end, :]
			invalid_mask[i, :end - start].fill_(False)

		return (
			q_padded.view(-1, len_padded, d_k),
			k_padded.view(-1, len_padded, d_k),
			v_padded.view(-1, len_padded, d_v),
			invalid_mask.unsqueeze(1).expand(mb_size, len_padded, len_padded).repeat(n_head, 1, 1),
			(~invalid_mask).repeat(n_head, 1),
		)

	def combine_v(self, outputs):
		# Combine attention information from the different heads
		n_head = self.n_head
		outputs = outputs.view(n_head, -1, self.d_v)

		if not self.partitioned:
			# Switch from n_head x len_inp x d_v to len_inp x (n_head * d_v)
			outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, n_head * self.d_v)

			# Project back to residual size
			outputs = self.proj(outputs)
		else:
			d_v1 = self.d_v // 2
			outputs1 = outputs[:, :, :d_v1]
			outputs2 = outputs[:, :, d_v1:]
			outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(-1, n_head * d_v1)
			outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(-1, n_head * d_v1)
			outputs = torch.cat([
				self.proj1(outputs1),
				self.proj2(outputs2),
			], -1)

		return outputs

	def forward(self, inp, batch_idxs, qk_inp=None):
		residual = inp

		# While still using a packed representation, project to obtain the
		# query/key/value for each head
		q_s, k_s, v_s = self.split_qkv_packed(inp, qk_inp=qk_inp)

		# Switch to padded representation, perform attention, then switch back
		q_padded, k_padded, v_padded, attn_mask, output_mask = self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)

		outputs_padded, attns_padded = self.attention(
			q_padded, k_padded, v_padded,
			attn_mask=attn_mask,
		)
		outputs = outputs_padded[output_mask]
		outputs = self.combine_v(outputs)

		outputs = self.residual_dropout(outputs, batch_idxs)

		return self.layer_norm(outputs + residual), attns_padded
class PositionwiseFeedForward(nn.Module):
	"""
	    A position-wise feed forward module.

	    Projects to a higher-dimensional space before applying ReLU, then projects
	    back.
	    """

	def __init__(self, d_hid, d_ff, relu_dropout=0.1, residual_dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_hid, d_ff)
		self.w_2 = nn.Linear(d_ff, d_hid)

		self.layer_norm = LayerNormalization(d_hid)
		# The t2t code on github uses relu dropout, even though the transformer
		# paper describes residual dropout only. We implement relu dropout
		# because we always have the option to set it to zero.
		self.relu_dropout = FeatureDropout(relu_dropout)
		self.residual_dropout = FeatureDropout(residual_dropout)
		self.relu = nn.ReLU()

	def forward(self, x, batch_idxs):
		residual = x

		output = self.w_1(x)
		output = self.relu_dropout(self.relu(output), batch_idxs)
		output = self.w_2(output)

		output = self.residual_dropout(output, batch_idxs)
		return self.layer_norm(output + residual)

class PartitionedPositionwiseFeedForward(nn.Module):
	def __init__(self, d_hid, d_ff, d_positional, relu_dropout=0.1, residual_dropout=0.1):
		super().__init__()
		self.d_content = d_hid - d_positional
		self.w_1c = nn.Linear(self.d_content, d_ff // 2)
		self.w_1p = nn.Linear(d_positional, d_ff // 2)
		self.w_2c = nn.Linear(d_ff // 2, self.d_content)
		self.w_2p = nn.Linear(d_ff // 2, d_positional)
		self.layer_norm = LayerNormalization(d_hid)
		# The t2t code on github uses relu dropout, even though the transformer
		# paper describes residual dropout only. We implement relu dropout
		# because we always have the option to set it to zero.
		self.relu_dropout = FeatureDropout(relu_dropout)
		self.residual_dropout = FeatureDropout(residual_dropout)
		self.relu = nn.ReLU()

	def forward(self, x, batch_idxs):
		residual = x
		xc = x[:, :self.d_content]
		xp = x[:, self.d_content:]

		outputc = self.w_1c(xc)
		outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
		outputc = self.w_2c(outputc)

		outputp = self.w_1p(xp)
		outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
		outputp = self.w_2p(outputp)

		output = torch.cat([outputc, outputp], -1)

		output = self.residual_dropout(output, batch_idxs)
		return self.layer_norm(output + residual)
class FeatureDropout(nn.Module):
	"""
	    Feature-level dropout: takes an input of size len x num_features and drops
	    each feature with probabibility p. A feature is dropped across the full
	    portion of the input that corresponds to a single batch element.
	    """

	def __init__(self, p=0.5, inplace=False):
		super().__init__()
		if p < 0 or p > 1:
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {}".format(p))
		self.p = p
		self.inplace = inplace

	def forward(self, input, batch_idxs):
		return FeatureDropoutFunction.apply(input, batch_idxs, self.p, self.training, self.inplace)

class FeatureDropoutFunction(torch.autograd.function.InplaceFunction):
	@classmethod
	def forward(cls, ctx, input, batch_idxs, p=0.5, train=False, inplace=False):
		if p < 0 or p > 1:
			raise ValueError("dropout probability has to be between 0 and 1, "
			                 "but got {}".format(p))

		ctx.p = p
		ctx.train = train
		ctx.inplace = inplace

		if ctx.inplace:
			ctx.mark_dirty(input)
			output = input
		else:
			output = input.clone()

		if ctx.p > 0 and ctx.train:
			ctx.noise = input.new().resize_(batch_idxs.batch_size, input.size(1))
			if ctx.p == 1:
				ctx.noise.fill_(0)
			else:
				ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
			ctx.noise = ctx.noise[batch_idxs.batch_idxs_torch, :]
			output.mul_(ctx.noise)

		return output

	@staticmethod
	def backward(ctx, grad_output):
		if ctx.p > 0 and ctx.train:
			return grad_output.mul(ctx.noise), None, None, None, None
		else:
			return grad_output, None, None, None, None
class MultiLevelEmbedding(nn.Module):
	def __init__(self,
	             num_embeddings_list,
	             d_embedding,
	             d_positional=None,
	             max_len=300,
	             normalize=True,
	             dropout=0.1,
	             timing_dropout=0.0,
	             emb_dropouts_list=None,
	             extra_content_dropout=None,
	             **kwargs):
		super().__init__()

		self.d_embedding = d_embedding
		self.partitioned = d_positional is not None

		if self.partitioned:
			self.d_positional = d_positional
			self.d_content = self.d_embedding - self.d_positional
		else:
			self.d_positional = self.d_embedding
			self.d_content = self.d_embedding

		if emb_dropouts_list is None:
			emb_dropouts_list = [0.0] * len(num_embeddings_list)
		assert len(emb_dropouts_list) == len(num_embeddings_list)

		embs = []
		emb_dropouts = []
		for i, (num_embeddings, emb_dropout) in enumerate(zip(num_embeddings_list, emb_dropouts_list)):
			emb = nn.Embedding(num_embeddings, self.d_content, **kwargs)
			embs.append(emb)
			emb_dropout = FeatureDropout(emb_dropout)
			emb_dropouts.append(emb_dropout)
		self.embs = nn.ModuleList(embs)
		self.emb_dropouts = nn.ModuleList(emb_dropouts)

		if extra_content_dropout is not None:
			self.extra_content_dropout = FeatureDropout(extra_content_dropout)
		else:
			self.extra_content_dropout = None

		if normalize:
			self.layer_norm = LayerNormalization(d_embedding)
		else:
			self.layer_norm = lambda x: x

		self.dropout = FeatureDropout(dropout)
		self.timing_dropout = FeatureDropout(timing_dropout)

		# Learned embeddings
		self.position_table = nn.Parameter(torch_t.FloatTensor(max_len, self.d_positional))
		init.normal_(self.position_table)

	def forward(self, xs, batch_idxs, extra_content_annotations=None):

		content_annotations = [
			emb_dropout(emb(x), batch_idxs)
			for x, emb, emb_dropout in zip(xs, self.embs, self.emb_dropouts)
		]
		# print(xs)
		# print(batch_idxs.boundaries_np)
		# print(batch_idxs.batch_idxs_np)
		# print(content_annotations)
		# print(content_annotations[0])
		# print(content_annotations[1])
		content_annotations = sum(content_annotations)
		# print(content_annotations.size())
		# input()
		if extra_content_annotations is not None:
			if self.extra_content_dropout is not None:
				content_annotations += self.extra_content_dropout(extra_content_annotations, batch_idxs)
			else:
				content_annotations += extra_content_annotations

		timing_signal = torch.cat([self.position_table[:seq_len, :] for seq_len in batch_idxs.seq_lens_np], dim=0)
		timing_signal = self.timing_dropout(timing_signal, batch_idxs)

		# Combine the content and timing signals
		if self.partitioned:
			annotations = torch.cat([content_annotations, timing_signal], 1)
		else:
			annotations = content_annotations + timing_signal

		# TODO(nikita): reconsider the use of layernorm here
		annotations = self.layer_norm(self.dropout(annotations, batch_idxs))

		return annotations, timing_signal, batch_idxs
class Transformer_ProdAtt_ExternalEmbedding_Label(nn.Module):
	def __init__(
			self,
			tag_vocab,
			word_vocab,
			char_vocab,
			labelset_size,
			word_labelset_size,
			hparams,
			squeeze_dim_pointing=True,
	):
		super().__init__()
		# self.spec['hparams'] = hparams.to_dict()

		self.tag_vocab = tag_vocab
		self.word_vocab = word_vocab
		# self.label_set = label_set
		# self.word_labelset=word_labelset
		self.char_vocab = char_vocab

		# self.tagset_size=tagset_size
		# self.vocab_size=vocab_size
		self.labelset_size=labelset_size
		self.word_labelset_size=word_labelset_size

		self.d_model = hparams.d_model
		self.partitioned = hparams.partitioned
		self.d_content = (self.d_model // 2) if self.partitioned else self.d_model
		self.d_positional = (hparams.d_model // 2) if self.partitioned else None

		num_embeddings_map = {
			'tags': tag_vocab.size(),
			'words': word_vocab.size(),
			'chars': char_vocab.size(),
		}
		emb_dropouts_map = {
			'tags': hparams.tag_emb_dropout,
			'words': hparams.word_emb_dropout,
		}

		self.emb_types = []
		if hparams.use_words:
			self.emb_types.append('words')
		if hparams.use_tags:
			self.emb_types.append('tags')


		self.use_tags = hparams.use_tags

		self.morpho_emb_dropout = None
		if hparams.use_chars_lstm or hparams.use_elmo or hparams.use_bert or hparams.use_bert_only:
			self.morpho_emb_dropout = hparams.morpho_emb_dropout
		else:
			assert self.emb_types, "Need at least one of: use_tags, use_words, use_chars_lstm, use_elmo, use_bert, use_bert_only"

		self.char_encoder = None
		self.elmo = None
		self.bert = None
		if hparams.use_chars_lstm:
			assert not hparams.use_elmo, "use_chars_lstm and use_elmo are mutually exclusive"
			assert not hparams.use_bert, "use_chars_lstm and use_bert are mutually exclusive"
			assert not hparams.use_bert_only, "use_chars_lstm and use_bert_only are mutually exclusive"
			self.char_encoder = CharacterLSTM(
				num_embeddings_map['chars'],
				hparams.d_char_emb,
				self.d_content,
				char_dropout=hparams.char_lstm_input_dropout,
			)
		elif hparams.use_elmo:
			assert not hparams.use_bert, "use_elmo and use_bert are mutually exclusive"
			assert not hparams.use_bert_only, "use_elmo and use_bert_only are mutually exclusive"
			self.elmo = get_elmo_class()(
				options_file="../Data/pretrain_embedding/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
				weight_file="../Data/pretrain_embedding/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
				num_output_representations=1,
				requires_grad=False,
				do_layer_norm=False,
				keep_sentence_boundaries=False,
				dropout=hparams.elmo_dropout,
			)
			d_elmo_annotations = 1024

			# Don't train gamma parameter for ELMo - the projection can do any
			# necessary scaling
			self.elmo.scalar_mix_0.gamma.requires_grad = False

			# Reshapes the embeddings to match the model dimension, and making
			# the projection trainable appears to improve parsing accuracy
			self.project_elmo = nn.Linear(d_elmo_annotations, self.d_content, bias=False)
		elif hparams.use_bert or hparams.use_bert_only:
			self.bert_tokenizer, self.bert = get_bert(hparams.bert_model, hparams.bert_do_lower_case)
			if hparams.bert_transliterate:
				from transliterate import TRANSLITERATIONS
				self.bert_transliterate = TRANSLITERATIONS['hebrew']
			else:
				self.bert_transliterate = None
			self.bert_transliterate = None
			d_bert_annotations = self.bert.pooler.dense.in_features
			self.bert_max_len = self.bert.embeddings.position_embeddings.num_embeddings

			if hparams.use_bert_only:
				self.project_bert = nn.Linear(d_bert_annotations, hparams.d_model, bias=False)
			else:
				self.project_bert = nn.Linear(d_bert_annotations, self.d_content, bias=False)

		# if not hparams.use_bert_only:
		self.embedding = MultiLevelEmbedding(
			[num_embeddings_map[emb_type] for emb_type in self.emb_types],
			hparams.d_model,
			d_positional=self.d_positional,
			dropout=hparams.embedding_dropout,
			timing_dropout=hparams.timing_dropout,
			emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
			extra_content_dropout=self.morpho_emb_dropout,
			max_len=hparams.sentence_max_len,
		)

		self.encoder = TransformerEncoderProdAtt(
			self.embedding,
			num_layers=hparams.num_layers,
			num_heads=hparams.num_heads,
			d_kv=hparams.d_kv,
			d_ff=hparams.d_ff,
			d_positional=self.d_positional,
			num_layers_position_only=hparams.num_layers_position_only,
			relu_dropout=hparams.relu_dropout,
			residual_dropout=hparams.residual_dropout,
			attention_dropout=hparams.attention_dropout,
		)
		# else:
		# 	self.embedding = None
		# 	self.encoder = None

		self.f_label = nn.Sequential(
			nn.Linear(hparams.d_model, hparams.d_label_hidden),
			# LayerNormalization(hparams.d_label_hidden),
			nn.ReLU(),
			nn.Linear(hparams.d_label_hidden, labelset_size),
		)
		self.f_wordlabel = nn.Sequential(
			nn.Linear(hparams.d_model, hparams.d_label_hidden),
			# LayerNormalization(hparams.d_label_hidden),
			nn.ReLU(),
			nn.Linear(hparams.d_label_hidden, word_labelset_size),
		)
		if squeeze_dim_pointing:
			# d_pointing_hidden=hparams.d_label_hidden
			d_pointing_hidden=hparams.d_model
		else:
			d_pointing_hidden=hparams.d_model
		self.f_pointing_query = nn.Sequential(
			nn.Linear(hparams.d_model, d_pointing_hidden),
			# LayerNormalization(hparams.d_model),
			nn.ReLU(),
			nn.Linear(d_pointing_hidden, d_pointing_hidden),
		)
		self.f_pointing_key = nn.Sequential(
			nn.Linear(hparams.d_model, d_pointing_hidden),
			# LayerNormalization(hparams.d_model),
			nn.ReLU(),
			nn.Linear(d_pointing_hidden, d_pointing_hidden),
		)
		self.f_splitting_query = nn.Sequential(
			nn.Linear(hparams.d_model, d_pointing_hidden),
			# LayerNormalization(hparams.d_model),
			nn.ReLU(),
			nn.Linear(d_pointing_hidden, d_pointing_hidden),
		)
		self.f_splitting_key = nn.Sequential(
			nn.Linear(hparams.d_model, d_pointing_hidden),
			# LayerNormalization(hparams.d_model),
			nn.ReLU(),
			nn.Linear(d_pointing_hidden, d_pointing_hidden),
		)
		# self.f_pointing_query = nn.Sequential(
		# 	nn.Linear(hparams.d_model, d_pointing_hidden,bias=False),
		# 	# LayerNormalization(hparams.d_model),
		# 	nn.ReLU(),
		# 	nn.Linear(d_pointing_hidden, d_pointing_hidden,bias=False),
		# )
		# self.f_pointing_key = nn.Sequential(
		# 	nn.Linear(hparams.d_model, d_pointing_hidden,bias=False),
		# 	# LayerNormalization(hparams.d_model),
		# 	nn.ReLU(),
		# 	nn.Linear(d_pointing_hidden, d_pointing_hidden,bias=False),
		# )
		# self.f_splitting_query = nn.Sequential(
		# 	nn.Linear(hparams.d_model, d_pointing_hidden,bias=False),
		# 	# LayerNormalization(hparams.d_model),
		# 	nn.ReLU(),
		# 	nn.Linear(d_pointing_hidden, d_pointing_hidden,bias=False),
		# )
		# self.f_splitting_key = nn.Sequential(
		# 	nn.Linear(hparams.d_model, d_pointing_hidden,bias=False),
		# 	# LayerNormalization(hparams.d_model),
		# 	nn.ReLU(),
		# 	nn.Linear(d_pointing_hidden, d_pointing_hidden,bias=False),
		# )
	def forward(self, inputs, padding=True):
		#input is a dictionary of sentences and tags
		has_tag='tags' in inputs
		has_sent='sents' in inputs
		assert has_sent
		sentences=inputs['sents']
		packed_len = sum([len(sentence) for sentence in sentences])

		i = 0
		tag_idxs = np.zeros(packed_len, dtype=int)
		word_idxs = np.zeros(packed_len, dtype=int)
		batch_idxs = np.zeros(packed_len, dtype=int)
		for snum, sentence in enumerate(sentences):
			for j in range(len(sentence)):
				# tag=inputs['tags'][snum][j]
				word=sentence[j]
				tag_idxs[i] = 0 if not has_tag else self.tag_vocab.getIdx(inputs['tags'][snum][j])
				word_idxs[i] = self.word_vocab.getIdx(word)
				batch_idxs[i] = snum
				i += 1
		assert i==packed_len
		batch_idxs = BatchIndices(batch_idxs)
		emb_idxs_map = {
			'tags': tag_idxs,
			'words': word_idxs,
		}
		emb_idxs = [
			from_numpy(emb_idxs_map[emb_type])
			for emb_type in self.emb_types
		]
		extra_content_annotations = None
		if self.char_encoder is not None:
			assert isinstance(self.char_encoder, CharacterLSTM)
			max_word_len = max([max([len(word) for word in sentence]) for sentence in sentences])
			# max_word_len = max(max_word_len, 3) + 2
			char_idxs_encoder = np.zeros((packed_len, max_word_len+2), dtype=int)
			word_lens_encoder = np.zeros(packed_len, dtype=int)

			i = 0
			for snum, sentence in enumerate(sentences):
				for wordnum, word in enumerate(sentence):
					j = 0
					char_idxs_encoder[i, j] = self.char_vocab.getIdx("<CHAR_START>")
					j += 1
					# if word in (START, STOP):
					# 	char_idxs_encoder[i, j:j + 3] = self.char_vocab.index(
					# 		CHAR_START_SENTENCE if (word == START) else CHAR_STOP_SENTENCE
					# 	)
					# 	j += 3
					# else:
					for char in word:
						char_idxs_encoder[i, j] = self.char_vocab.getIdx(char.lower())
						j += 1
					char_idxs_encoder[i, j] = self.char_vocab.getIdx("<CHAR_END>")
					word_lens_encoder[i] = j + 1
					i += 1
			assert i == packed_len
			extra_content_annotations = self.char_encoder(char_idxs_encoder, word_lens_encoder, batch_idxs)
		elif self.elmo is not None:
			# See https://github.com/allenai/allennlp/blob/c3c3549887a6b1fb0bc8abf77bc820a3ab97f788/allennlp/data/token_indexers/elmo_indexer.py#L61
			# ELMO_START_SENTENCE = 256
			# ELMO_STOP_SENTENCE = 257
			ELMO_START_WORD = 258
			ELMO_STOP_WORD = 259
			ELMO_CHAR_PAD = 260

			# Sentence start/stop tokens are added inside the ELMo module
			max_sentence_len = max([(len(sentence)) for sentence in sentences])
			max_word_len = 50
			char_idxs_encoder = np.zeros((len(sentences), max_sentence_len, max_word_len), dtype=int)

			for snum, sentence in enumerate(sentences):
				for wordnum, word in enumerate(sentence):
					char_idxs_encoder[snum, wordnum, :] = ELMO_CHAR_PAD

					j = 0
					char_idxs_encoder[snum, wordnum, j] = ELMO_START_WORD
					j += 1
					# assert word not in (START, STOP)
					for char_id in word.encode('utf-8', 'ignore')[:(max_word_len - 2)]:
						char_idxs_encoder[snum, wordnum, j] = char_id
						j += 1
					char_idxs_encoder[snum, wordnum, j] = ELMO_STOP_WORD

					# +1 for masking (everything that stays 0 is past the end of the sentence)
					char_idxs_encoder[snum, wordnum, :] += 1

			char_idxs_encoder = from_numpy(char_idxs_encoder)

			elmo_out = self.elmo.forward(char_idxs_encoder)
			elmo_rep0 = elmo_out['elmo_representations'][0]
			elmo_mask = elmo_out['mask']

			elmo_annotations_packed = elmo_rep0[elmo_mask.byte()].view(packed_len, -1)

			# Apply projection to match dimensionality
			extra_content_annotations = self.project_elmo(elmo_annotations_packed)

		elif self.bert is not None:
			all_input_ids = np.zeros((len(sentences), self.bert_max_len), dtype=int)
			all_input_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)
			all_word_start_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)
			all_word_end_mask = np.zeros((len(sentences), self.bert_max_len), dtype=int)
			all_except_cls_sep= np.zeros((len(sentences), self.bert_max_len), dtype=int)

			subword_max_len = 0
			# print(sentences)
			# print([len(s) for s in sentences])
			# input()
			for snum, sentence in enumerate(sentences):
				tokens = []
				word_start_mask = []
				word_end_mask = []
				except_cls_sep =[]

				tokens.append("[CLS]")
				word_start_mask.append(1)
				word_end_mask.append(1)
				except_cls_sep.append(0)

				if self.bert_transliterate is None:
					cleaned_words = []
					for word in sentence:
						word = BERT_TOKEN_MAPPING.get(word, word)
						# This un-escaping for / and * was not yet added for the
						# parser version in https://arxiv.org/abs/1812.11760v1
						# and related model releases (e.g. benepar_en2)
						word = word.replace('\\/', '/').replace('\\*', '*')
						# Mid-token punctuation occurs in biomedical text
						word = word.replace('-LSB-', '[').replace('-RSB-', ']')
						word = word.replace('-LRB-', '(').replace('-RRB-', ')')
						if word == "n't" and cleaned_words:
							cleaned_words[-1] = cleaned_words[-1] + "n"
							word = "'t"
						cleaned_words.append(word)
				else:
					# When transliterating, assume that the token mapping is
					# taken care of elsewhere
					cleaned_words = [self.bert_transliterate(word) for _, word in sentence]
				# print(cleaned_words)
				# print(len(cleaned_words))

				for word in cleaned_words:
					word_tokens = self.bert_tokenizer.tokenize(word)
					for _ in range(len(word_tokens)):
						word_start_mask.append(0)
						word_end_mask.append(0)
						except_cls_sep.append(0)
					word_start_mask[len(tokens)] = 1
					word_end_mask[-1] = 1
					except_cls_sep[-1]= 1
					tokens.extend(word_tokens)
				tokens.append("[SEP]")
				word_start_mask.append(1)
				word_end_mask.append(1)
				except_cls_sep.append(0)
				# print(tokens)
				# print(len(tokens))
				# print(word_start_mask)
				# print(word_end_mask)
				# input()

				input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

				# The mask has 1 for real tokens and 0 for padding tokens. Only real
				# tokens are attended to.
				input_mask = [1] * len(input_ids)

				subword_max_len = max(subword_max_len, len(input_ids))

				all_input_ids[snum, :len(input_ids)] = input_ids
				all_input_mask[snum, :len(input_mask)] = input_mask
				all_word_start_mask[snum, :len(word_start_mask)] = word_start_mask
				all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask
				all_except_cls_sep[snum, :len(except_cls_sep)] = except_cls_sep

			all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids[:, :subword_max_len]))
			all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask[:, :subword_max_len]))
			all_word_start_mask = from_numpy(np.ascontiguousarray(all_word_start_mask[:, :subword_max_len]))
			all_word_end_mask = from_numpy(np.ascontiguousarray(all_word_end_mask[:, :subword_max_len]))
			all_except_cls_sep_mask=from_numpy(np.ascontiguousarray(all_except_cls_sep[:, :subword_max_len]))
			all_encoder_layers, _ = self.bert(all_input_ids, attention_mask=all_input_mask)
			del _
			features = all_encoder_layers[-1]
			# print(features.size())
			# print(all_word_start_mask)
			# print(all_word_end_mask)
			# print(all_except_cls_sep_mask)
			# print(features.shape[-1])
			# print(all_word_end_mask.to(torch.uint8))
			# input()
			if self.encoder is not None:
				# features_packed = features.masked_select(all_word_end_mask.to(torch.uint8).unsqueeze(-1)).reshape(-1, features.shape[-1])
				features_packed = features.masked_select(all_except_cls_sep_mask.to(torch.uint8).unsqueeze(-1)).reshape(-1, features.shape[-1])

				# For now, just project the features from the last word piece in each word
				extra_content_annotations = self.project_bert(features_packed)

		annotations, _ = self.encoder(emb_idxs, batch_idxs, extra_content_annotations=extra_content_annotations)
		if self.partitioned:
			# Rearrange the annotations to ensure that the transition to
			# fenceposts captures an even split between position and content.
			# TODO(nikita): try alternatives, such as omitting position entirely
			annotations = torch.cat([
				annotations[:, 0::2],
				annotations[:, 1::2],
			], 1)
		len_padded = batch_idxs.max_len
		mb_size = batch_idxs.batch_size
		annotations_padded = annotations.new_zeros((mb_size, len_padded, annotations.size(-1)))
		padding_mask = annotations.new_ones((mb_size, len_padded), dtype=torch.uint8)
		for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
			annotations_padded[i, :end - start, :] = annotations[start:end, :]
			padding_mask[i, :end - start].fill_(False)

		label_space = self.f_label(annotations_padded.view(-1, annotations_padded.shape[2]))
		wordlabel_space = self.f_wordlabel(annotations_padded.view(-1, annotations_padded.shape[2]))
		pointing_query = self.f_pointing_query(annotations_padded)
		pointing_key = self.f_pointing_key(annotations_padded)
		pointing_prod = torch.matmul(pointing_query, pointing_key.transpose(-2, -1))

		if padding:
			pointing_prod = pointing_prod.masked_fill(padding_mask.unsqueeze((2)), float('-inf'))
			pointing_prod = pointing_prod.masked_fill(padding_mask.unsqueeze((1)), float('-inf'))
		# print(pointing_prod)

		pointing_scores = F.log_softmax(pointing_prod, dim=-1)
		special_splitting_query = self.f_splitting_query(annotations_padded)
		special_splitting_key = self.f_splitting_key(annotations_padded)
		# special_splitting_query = self.f_pointing_query(annotations_padded)
		# special_splitting_key = self.f_pointing_key(annotations_padded)
		special_splitting_prod = torch.matmul(special_splitting_query, special_splitting_key.transpose(-2, -1))
		if padding:
			special_splitting_prod = special_splitting_prod.masked_fill(padding_mask.unsqueeze((2)), float('-inf'))
			special_splitting_prod = special_splitting_prod.masked_fill(padding_mask.unsqueeze((1)), float('-inf'))
		# print(special_splitting_prod)
		# input()
		special_splitting_scores = F.log_softmax(special_splitting_prod, dim=-1)
		label_scores = F.log_softmax(label_space, dim=1)
		label_scores = label_scores.view(mb_size, len_padded, self.labelset_size)
		wordlabel_scores = F.log_softmax(wordlabel_space, dim=1)
		wordlabel_scores = wordlabel_scores.view(mb_size, len_padded, self.word_labelset_size)
		return label_scores, pointing_scores, wordlabel_scores, special_splitting_scores


	# def _forward(self, xs, batch_idxs, padding=True):
	# 	#xs is the the inputs: [word_idx, tag_idx]
	# 	#batch_idxs is in the format [0,0,0,0,1,1,1,...]
	# 	annotations, _ = self.encoder(xs, batch_idxs, extra_content_annotations=None)
	# 	if self.partitioned:
	# 		# Rearrange the annotations to ensure that the transition to
	# 		# fenceposts captures an even split between position and content.
	# 		# TODO(nikita): try alternatives, such as omitting position entirely
	# 		annotations = torch.cat([
	# 			annotations[:, 0::2],
	# 			annotations[:, 1::2],
	# 		], 1)
	# 	len_padded = batch_idxs.max_len
	# 	mb_size = batch_idxs.batch_size
	# 	# print(mb_size)
	# 	# print(annotations)
	# 	# input()
	# 	annotations_padded = annotations.new_zeros((mb_size, len_padded, annotations.size(-1)))
	# 	# print(batch_idxs.batch_idxs_np)
	# 	# print(batch_idxs.boundaries_np)
	# 	# print(annotations_padded.size())
	#
	# 	padding_mask = annotations.new_ones((mb_size, len_padded), dtype=torch.uint8)
	# 	for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], batch_idxs.boundaries_np[1:])):
	# 		annotations_padded[i, :end - start, :] = annotations[start:end, :]
	# 		padding_mask[i, :end - start].fill_(False)
	# 	# print(padding_mask)
	# 	# print(annotations_padded)
	# 	# input()
	# 	label_space = self.f_label(annotations_padded.view(-1, annotations_padded.shape[2]))
	# 	wordlabel_space = self.f_wordlabel(annotations_padded.view(-1, annotations_padded.shape[2]))
	# 	pointing_query = self.f_pointing_query(annotations_padded)
	# 	pointing_key = self.f_pointing_key(annotations_padded)
	# 	pointing_prod = torch.matmul(pointing_query, pointing_key.transpose(-2, -1))
	#
	# 	if padding:
	# 		pointing_prod = pointing_prod.masked_fill(padding_mask.unsqueeze((2)), float('-inf'))
	# 		pointing_prod = pointing_prod.masked_fill(padding_mask.unsqueeze((1)), float('-inf'))
	# 	# print(pointing_prod)
	#
	# 	pointing_scores = F.log_softmax(pointing_prod, dim=-1)
	# 	special_splitting_query = self.f_pointing_query(annotations_padded)
	# 	special_splitting_key = self.f_pointing_key(annotations_padded)
	# 	special_splitting_prod = torch.matmul(special_splitting_query, special_splitting_key.transpose(-2, -1))
	# 	if padding:
	# 		special_splitting_prod = special_splitting_prod.masked_fill(padding_mask.unsqueeze((2)), float('-inf'))
	# 		special_splitting_prod = special_splitting_prod.masked_fill(padding_mask.unsqueeze((1)), float('-inf'))
	# 	# print(special_splitting_prod)
	# 	# input()
	# 	special_splitting_scores = F.log_softmax(special_splitting_prod, dim=-1)
	# 	label_scores = F.log_softmax(label_space, dim=1)
	# 	label_scores = label_scores.view(mb_size, len_padded, self.labelset_size)
	# 	wordlabel_scores = F.log_softmax(wordlabel_space, dim=1)
	# 	wordlabel_scores = wordlabel_scores.view(mb_size, len_padded, self.word_labelset_size)
	# 	return label_scores, pointing_scores, wordlabel_scores, special_splitting_scores

	#     now we define loss function
	def loss_label(self, label_scores, true_label, sent_lengths):
		Y = true_label.view(-1)
		Y_hat = label_scores.view(-1, self.labelset_size)
		label_pad_token = 0
		mask = (Y > 0).float()
		num_tokens = int(torch.sum(mask).item())
		Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
		cross_entropy_loss = -torch.sum(Y_hat) / num_tokens
		return cross_entropy_loss

	def loss_wordlabel(self, wordlabel_scores, true_wordlabel, sent_lengths):
		Y = true_wordlabel.view(-1)
		Y_hat = wordlabel_scores.view(-1, self.word_labelset_size)
		label_pad_token = 0
		mask = (Y > 0).float()
		num_tokens = int(torch.sum(mask).item())
		Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
		cross_entropy_loss = -torch.sum(Y_hat) / num_tokens
		return cross_entropy_loss

	def loss_pointing(self, pointing_scores, true_pointing, sent_lengths, label_pad_token):
		Y = true_pointing.view(-1)
		max_lengths = max(sent_lengths)
		Y_hat = pointing_scores.view(-1, max_lengths)
		#         label_pad_token=10000
		mask = (Y != label_pad_token).float()
		num_tokens = int(torch.sum(mask).item())
		Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
		# this operations is to replace nan value with 0
		Y_hat[Y_hat != Y_hat] = 0
		cross_entropy_loss = -torch.sum(Y_hat) / num_tokens
		# print(cross_entropy_loss)
		# input()
		return cross_entropy_loss
















