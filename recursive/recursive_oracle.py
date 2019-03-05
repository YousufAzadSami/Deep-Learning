import torch
import random
import math
import numpy as np
import copy

# First, we define our tree datastructure. A tree is a tuple of a label and a list of trees.
class Tree:
	def __init__(self, label, children = []):
		self.label = label
		self.children = children

	def __str__(self):
		if(not self.children):
			return self.label
		strs = []
		for c in self.children:
			strs.append(c.__str__())
		return self.label + '(' + ', '.join(strs) + ')'

	def __repr__(self):
		return self.__str__()

# first, we have a utility function which can generate trees according to a
# given probabilistic grammar
def generate_tree(start, rules, P):
	# initialize the tree with a $nil node and the start nonterminal
	# as single child
	T = Tree('$nil', [start])
	# initialize a stack of parent trees and child indices to
	# maintain a reference to the nonterminal symbol that we need
	# to decode next
	stk = [(T, 0)]
	# we keep generating until the stack is empty
	while(stk):
		# get the reference to the next nonterminal
		parent, child_idx = stk.pop()
		# retrieve the nonterminal itself
		symbol = parent.children[child_idx]
		# retrieve the possible replacements for this symbol
		options = rules[symbol]
		# ... and their probabilities
		ps = P[symbol]
		# choose an option at random with the given
		# distribution
		if(len(options) != len(ps)):
			raise ValueError('For symbol %s, there are %d rule options but %d probabilities' % (symbol, len(options), len(ps)))
		r = np.random.choice(list(range(len(options))), 1, p = ps)[0]
		# apply this rule
		parent.children[child_idx] = copy.deepcopy(options[r])
		# get the newly generated subtree
		new_parent = parent.children[child_idx]
		# push new nonterminals on the stack
		for c in range(len(new_parent.children)-1, -1, -1):
			stk.append((new_parent, c))
	return T.children[0]

def tree_size(tree):
	size = 0
	for child in tree.children:
		size += tree_size(child)
	return size + 1

logical_rules = {'S' : [
	Tree('not', ['S']),
	Tree('and', ['S', 'S']),
	Tree('or', ['S', 'S']),
	Tree('x', []),
	Tree('y', [])
]}

logical_P = {'S' : [ 0.2, 0.15, 0.15, 0.25, 0.25] }

def evaluate_logical(tree, x_val, y_val):
	if(tree.label == 'x'):
		return x_val
	elif(tree.label == 'y'):
		return y_val
	elif(tree.label == 'not'):
		return 1 - evaluate_logical(tree.children[0], x_val, y_val)
	elif(tree.label == 'or'):
		return min(1, evaluate_logical(tree.children[0], x_val, y_val) + evaluate_logical(tree.children[1], x_val, y_val))
	elif(tree.label == 'and'):
		return evaluate_logical(tree.children[0], x_val, y_val) * evaluate_logical(tree.children[1], x_val, y_val)
	else:
		raise ValueError('unexpected tree label: %s' % tree.label)

def generate_logical_tree():
	# generate a tree with the logical rules and probabilities
	x = generate_tree('S', logical_rules, logical_P)
	# get the logical value
	y = evaluate_logical(x, 0, 1)
	return (x, y)


rna_rules = {
	# rules for structures
	'S' : [
		# dangling base
		Tree('dangle', ['B', 'D']),
		# split into multiple structures
		Tree('split', ['S', 'S']),
		# valid base pairs
		Tree('pair', ['C', 'I', 'G']),
		Tree('pair', ['G', 'I', 'C']),
		Tree('pair', ['A', 'I', 'U']),
		Tree('pair', ['U', 'I', 'A']),
		Tree('pair', ['U', 'I', 'G']),
		Tree('pair', ['G', 'I', 'U'])
	],
	# dangling bases
	'D' : [Tree('dangle', ['B', 'D']), Tree('dangle_end', ['B'])],
	# rules for continuations within a stack
	'I' : [
		# branch off new substructure
		Tree('branch', ['I', 'I']),
		# hairpins
		Tree('pair', ['C', 'H0', 'G']),
		Tree('pair', ['G', 'H0', 'C']),
		Tree('pair', ['A', 'H0', 'U']),
		Tree('pair', ['U', 'H0', 'A']),
		Tree('pair', ['U', 'H0', 'G']),
		Tree('pair', ['G', 'H0', 'U']),
		# continue stacking base pairs
		Tree('pair', ['C', 'I', 'G']),
		Tree('pair', ['G', 'I', 'C']),
		Tree('pair', ['A', 'I', 'U']),
		Tree('pair', ['U', 'I', 'A']),
		Tree('pair', ['U', 'I', 'G']),
		Tree('pair', ['G', 'I', 'U'])
	],
	# hairpins
	'H0' : [Tree('hairpin', ['B', 'H1'])],
	'H1' : [Tree('hairpin', ['B', 'H2'])],
	'H2' : [Tree('hairpin', ['B', 'H'])],
	'H' : [Tree('hairpin', ['B', 'H']), Tree('hairpin_end', ['B'])],
	# bases
	'B' : [Tree('c', []), Tree('g', []), Tree('a', []), Tree('u', [])],
	'C' : [Tree('c', [])],
	'G' : [Tree('g', [])],
	'A' : [Tree('a', [])],
	'U' : [Tree('u', [])]
}

rna_P = {
	'S' : [0.1, 0.05, 0.2, 0.15, 0.15, 0.15, 0.1, 0.1],
	'D' : [0.3, 0.7],
	'I' : [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.1, 0.1, 0.1],
	'H0' : [1.],
	'H1' : [1.],
	'H2' : [1.],
	'H' : [0.2, 0.8],
	'B' : [0.25, 0.25, 0.25, 0.25],
	'C' : [1.],
	'G' : [1.],
	'A' : [1.],
	'U' : [1.]
}

rna_arity_alphabet = {
	'dangle' : 2,
	'dangle_end' : 1,
	'split' : 2,
	'pair' : 3,
	'branch' : 2,
	'hairpin' : 2,
	'hairpin_end' : 1,
	'c' : 0,
	'g' : 0,
	'a' : 0,
	'u' : 0
}

def count_dangling(dangle_subtree):
	if(dangle_subtree.label == 'dangle'):
		return 1 + count_dangling(dangle_subtree.children[1])
	elif(dangle_subtree.label == 'dangle_end'):
		return 1

def count_hairpin(hairpin_subtree):
	if(hairpin_subtree.label == 'hairpin'):
		return 1 + count_hairpin(hairpin_subtree.children[1])
	elif(hairpin_subtree.label == 'hairpin_end'):
		return 1

rna_stack_energy_table = np.array([
	# A/U   C/G  G/U
	[-0.9, -1.9, -0.9], # A/U
	[-2.0, -2.8, -1.6], # C/G
	[-0.8, -1.6, -0.5]  # G/U
])

def index_pair(left, right):
	if((left == 'a' and right == 'u') or (left == 'u' and right == 'a')):
		return 0
	if((left == 'c' and right == 'g') or (left == 'g' and right == 'c')):
		return 1
	if((left == 'g' and right == 'u') or (left == 'u' and right == 'g')):
		return 2
	raise ValueError('Unexpected base pair: (%s, %s)' % (left, right))

def estimate_energy(rna_tree):
	# estimate the free energy based on page 17 from
	# http://www.phys.ens.fr/~monasson/Houches/Westhof/L7.pdf
	# check the current label
	if(rna_tree.label == 'dangle'):
		# count the number of bases in the dangling strand
		num_dangling = count_dangling(rna_tree)
		# we assume a roughly logarithmic energy relation
		return 3.9 + 0.75 * np.log(num_dangling)
	if(rna_tree.label == 'split' or rna_tree.label == 'branch'):
		return estimate_energy(rna_tree.children[0]) + estimate_energy(rna_tree.children[1])
	if(rna_tree.label == 'pair'):
		parent_pair = (rna_tree.children[0].label, rna_tree.children[2].label)
		# we reward pairs not by themselves but only if they are
		# stacked with other pairs
		child = rna_tree.children[1]
		if(child.label == 'pair'):
			child_pair = (child.children[0].label, child.children[2].label)
			# so if we find two stacked pairs, get the index for both
			parent_idx = index_pair(parent_pair[0], parent_pair[1])
			child_idx = index_pair(child_pair[0], child_pair[1])
			# and add the free energy to the free energy of the child
			return rna_stack_energy_table[parent_idx, child_idx] + estimate_energy(child)
		else:
			# otherwise just copy the energy of the child
			return estimate_energy(child)
	if(rna_tree.label == 'hairpin'):
		# count the number of bases in the hairpin
		num_hairpin = count_hairpin(rna_tree)
		# we assume a roughly logarithmic energy relation
		return 2.5 + 1.2 * np.log(num_hairpin)
	raise ValueError('Unexpected label: %s' % rna_tree.label)

def generate_rna_tree():
	# generate a tree with the rna rules and probabilities
	x = generate_tree('S', rna_rules, rna_P)
	# get the minimum free energy value
	y = estimate_energy(x)
	return (x, y)
