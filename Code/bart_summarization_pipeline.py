# -*- coding: utf-8 -*-

from transformers import pipeline
import csv
import re
import nltk
from nltk.tokenize import WhitespaceTokenizer
import spacy
import torch
import os
from deprecated import deprecated


@deprecated(reason="This class is no longer used")
class Triplets:
    # How to use it:
    # triplets = Triplets('./../Data/embedRelations/', input_file, grouped_aliases)
    # triplets.summarize()
    # triplets.triplet_generate()
    # triplets.extract_triplets()

    def __init__(self, report_folder, book_filename, aliases):
        self.nlp = pipeline('summarization')
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sp = spacy.load("en_core_web_sm")
        self.aliases = aliases
        book = book_filename.split('.')
        if len(book) == 1:
            book = book_filename
        else:
            book = book[0]
        self.report = report_folder + book + '/' + book + '_report.txt'

        models_folder = './../Data/triplets'
        if not os.path.isdir(models_folder):
            os.makedirs(models_folder)

        book_folder = models_folder + '/' + book
        if not os.path.isdir(book_folder):
            os.makedirs(book_folder)
        self.results_folder = book_folder + '/'

    def summarize(self):
        self.cluster_sentences_actual = {}
        self.cluster_sentences = {}
        with open(self.report) as f:
            f = f.readlines()
            doc = [x.split('\t') for x in f]

            for item in doc:
                if len(item) > 1:
                    # sent contains the original sentence while sent1 contains dealiased sentence
                    sent1 = item[-1]
                    sent = item[-1]
                    m1 = re.findall('\w+\d+', sent)
                    for x in m1:
                        p1 = sent.split(m1[0])[-1]
                        # list index out of range
                        if len(m1) < 2:
                            print('Error')
                        p2 = p1.split(m1[1])[0]
                        if len(p2) > 3:
                            if x in self.aliases.keys():
                                sent = sent.replace(x, self.aliases[x][0])
                        else:
                            sent = ''
                        if len(p2) > 3:
                            sent1 = sent1
                        else:
                            sent1 = ''
                    if len(sent) > 0:
                        if item[1] not in self.cluster_sentences_actual.keys():
                            # sent=item[-1].replace('\w\d*>))
                            self.cluster_sentences_actual[item[1]] = [sent]
                        else:
                            self.cluster_sentences_actual[item[1]].append(sent)
                    if len(sent1) > 0:
                        if item[1] not in self.cluster_sentences.keys():
                            # sent=item[-1].replace('\w\d*>))
                            self.cluster_sentences[item[1]] = [sent1]
                        else:
                            self.cluster_sentences[item[1]].append(sent1)
        # summarize_pipeline_act contains representative (original) sentences for each cluster
        self.summarize_pipeline_act = {}
        for items in self.cluster_sentences_actual.keys():
            text = self.cluster_sentences_actual[items]
            text = '.'.join(text).lower()
            result = self.nlp(text, min_length=5, max_length=100)
            self.summarize_pipeline_act[items] = result

        # summarize_pipeline_act contains representative (dealiased) sentences for each cluster
        self.summarize_pipeline = {}
        for items in self.cluster_sentences.keys():
            text = self.cluster_sentences[items]
            text = '.'.join(text).lower()
            # print(''.join(text).lower())
            result = self.nlp(text, min_length=5, max_length=100)
            self.summarize_pipeline[items] = result

        with open(self.results_folder + 'summary_sentences.tsv', 'w', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            for _, r11 in enumerate(self.summarize_pipeline.keys()):
                r1 = self.summarize_pipeline[r11][0]['summary_text']
                writer.writerow([r11, r1])

        with open(self.results_folder + 'summary_sentences_act.tsv', 'w', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            for _, r11 in enumerate(self.summarize_pipeline_act.keys()):
                r1 = self.summarize_pipeline_act[r11][0]['summary_text']
                writer.writerow([r11, r1])

    def triplet_generate(self):
        self.phrases = {}
        self.tags = {}
        for i, r11 in enumerate(self.summarize_pipeline.keys()):
            s = self.summarize_pipeline[r11][0]['summary_text']
            s1 = s.split('.')
            s1 = [x for x in s1 if len(x) > 1]
            for sent in s1:
                sent = sent.lower()
                m1 = re.findall('\w+\d+', sent)
                if len(m1) > 1:
                    p1 = sent.split(m1[0])[-1]
                    p2 = p1.split(m1[1])[0]
                    if i not in self.phrases.keys():
                        self.phrases[i] = [p2]
                    else:
                        self.phrases[i].append(p2)

        for phr in self.phrases.keys():
            for item in self.phrases[phr]:
                # split sentence in words
                toks = WhitespaceTokenizer().tokenize(item)
                t = nltk.pos_tag(toks)
                # takes only verbs,...
                w = [x[0] for x in t if x[1] in ['VBN', 'VBD', 'VBG', 'VB', 'VBS']]
                if phr not in self.tags.keys():
                    self.tags[phr] = w
                else:
                    self.tags[phr].extend(w)

        with open(self.results_folder + 'summary_triplets.tsv', 'w', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            for _, r11 in enumerate(self.tags.keys()):
                r1 = self.tags[r11]
                writer.writerow([r11, r1])
