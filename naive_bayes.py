# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import math

import numpy as np
from tqdm import tqdm
from collections import Counter
import reader
import nltk
from nltk.corpus import stopwords

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


"""
load_data calls the provided utility to load in the dataset.
You can modify the default values for stemming and lowercase, to improve performance when
    we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


def ln(x):
    n = 1000.0
    return n * ((x ** (1/n)) - 1)
"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=5.0, pos_prior=0.8,silently=False):
    print_paramter_vals(laplace, pos_prior)
    pos_words, neg_words = getWordCount(train_set, train_labels, False)
    pos_prob, pos_unk = getProbability(pos_words, laplace)
    neg_prob, neg_unk = getProbability(neg_words, laplace)
    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        res_pos_prob = ln(pos_prior)
        res_neg_prob = ln(1-pos_prior)
        for term in doc:
            if pos_prob.get(term):
                res_pos_prob += ln(pos_prob[term])
            else:
                res_pos_prob += pos_unk
            if neg_words.get(term):
                res_neg_prob += ln(neg_prob[term])
            else:
                res_neg_prob += neg_unk
        if res_pos_prob <= res_neg_prob:
            yhats.append(0)
        else:
            yhats.append(1)
    return yhats

def getWordCount(train_set, train_labels, bigram):
    pos_words = Counter()
    neg_words = Counter()
    for index, label in enumerate(train_labels):
        if bigram:
            input = list(nltk.bigrams(train_set[index]))
        else:
            input = train_set[index]
        if label == 1:
            pos_words.update(input)
        else:
            neg_words.update(input)
    return pos_words, neg_words

def getProbability(words, laplace):
    probability = {}
    total_words = sum(words.values())
    total_word_type = len(words) + 1
    denominator = total_words + (laplace * total_word_type)
    unknown = laplace/denominator
    for word, value in words.items():
        probability[word] = (value + laplace)/denominator
    return probability, ln(unknown)


def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")

def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    return [w for w in tokens if not w.lower() in stop_words]

"""
You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

# main function for the bigrammixture model
def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.01, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.5, silently=False):
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    pos_words, neg_words = getWordCount(train_set, train_labels, False)
    pos_prob, pos_unk = getProbability(pos_words, unigram_laplace)
    neg_prob, neg_unk = getProbability(neg_words, unigram_laplace)
    bi_pos_words, bi_neg_words = getWordCount(train_set, train_labels, True)
    bi_pos_prob, bi_pos_unk = getProbability(bi_pos_words, bigram_laplace)
    bi_neg_prob, bi_neg_unk = getProbability(bi_neg_words, bigram_laplace)
    yhats = []
    for doc in tqdm(dev_set,disable=silently):
        res_pos_prob = ln(pos_prior)
        res_neg_prob = ln(1 - pos_prior)
        bi_res_pos_prob = ln(pos_prior)
        bi_res_neg_prob = ln(1 - pos_prior)
        for term in doc:
            if pos_prob.get(term):
                res_pos_prob += ln(pos_prob[term])
            else:
                res_pos_prob += pos_unk
            if neg_words.get(term):
                res_neg_prob += ln(neg_prob[term])
            else:
                res_neg_prob += neg_unk
        bg = list(nltk.bigrams(doc))
        for term in bg:
            if bi_pos_prob.get(term):
                bi_res_pos_prob += ln(bi_pos_prob[term])
            else:
                bi_res_pos_prob += bi_pos_unk
            if bi_neg_words.get(term):
                bi_res_neg_prob += ln(bi_neg_prob[term])
            else:
                bi_res_neg_prob += bi_neg_unk
        res_pos = ((1-bigram_lambda)*res_pos_prob) + (bigram_lambda*bi_res_pos_prob)
        res_neg = ((1-bigram_lambda)*res_neg_prob) + (bigram_lambda*bi_res_neg_prob)
        if res_pos < res_neg:
            yhats.append(0)
        else:
            yhats.append(1)
    return yhats

