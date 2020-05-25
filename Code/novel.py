import os
import os.path
import string
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wtk
import re
import difflib
import logging
from sklearn.cluster import DBSCAN
from sklearn import metrics
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
from itertools import groupby
from fuzzywuzzy import fuzz
import collections

CONJUNCTIONS = ['THEREFORE', 'DOES', 'WHEN', 'AND', 'THIS']
FORENAMES = ['MISS', 'UNCLE', 'MADAM']


def plot_characters_minsample_DBSCAN(similarities):
    samples = 1
    for i in range(0, 10):
        eps = 0.001
        maxs = []
        for j in range(0, 1000):
            db = DBSCAN(metric='precomputed', min_samples=samples, algorithm='brute', eps=eps).fit(1 - similarities)
            labels = db.labels_
            max = labels.max()
            # logging.info('samples: %s, eps: %s, max: %s', samples, eps, max)
            maxs.append(max)
            eps += 0.001
        plt.plot(np.linspace(0.001, 1, 1000), maxs, label=str(samples) + ' minimum samples per cluster')
        # logging.info("labels: %s", labels)
        samples += 1
    plt.legend(loc='upper right')
    plt.title("Variation of epsilon over minimum number of element per cluster")
    plt.xlabel("Epsilon")
    plt.ylabel("Number of cluster")
    plt.show()


def plot_epsilon_best_DBSCAN(similarities, samples=1):
    eps = 0.001
    maxs = []
    avg_length = []
    for i in range(0, 1000):
        db = DBSCAN(metric='precomputed', min_samples=samples, algorithm='brute', eps=eps).fit(1 - similarities)
        labels = db.labels_
        avg_length.append(np.average([len(list(group)) for key, group in groupby(labels)]))
        maxs.append(labels.max())
        eps += 0.001
    plt.xlabel("Epsilon")
    plt.ylabel("Number of cluster")
    plt.title("Epsilon variation over best cluster cardinality (min = " + str(samples) + ")")
    plt.plot(np.linspace(0.001, 1, 1000), maxs, label=str(samples) + ' minimum samples per cluster')
    plt.plot(np.linspace(0.001, 1, 1000), avg_length, label='Average number of names per cluster')
    plt.legend(loc='upper right')
    plt.show()


def plot_silhouette(similarities):
    eps = 0.001
    silhouette = []
    for i in range(0, 1000):
        db = DBSCAN(metric='precomputed', min_samples=1, algorithm='brute', eps=eps).fit(1 - similarities)
        if max(db.labels_) + 1 < 2 or max(db.labels_) + 1 >= len(similarities):
            silhouette.append(0)  # there is a cluster for each name or there is only a cluster
        else:
            silhouette.append(metrics.silhouette_score(1 - similarities, db.labels_))
        eps += 0.001
    plt.xlabel("Epsilon")
    plt.ylabel("Silhouette")
    plt.title("Silhouette's evolution")
    plt.plot(np.linspace(0.001, 1, 1000), silhouette)
    plt.show()


def find_best_eps(similarities):
    eps = 0.0001
    best_silhouette = -1
    best_eps = eps
    for i in range(0, 10000):
        db = DBSCAN(metric='precomputed', min_samples=1, algorithm='brute', eps=eps).fit(1 - similarities)
        if max(db.labels_) + 1 < 2 or max(db.labels_) + 1 >= len(similarities):
            eps = eps + 0.0001
            continue  # there is a cluster for each name or there is only a cluster

        silhouette = metrics.silhouette_score(1 - similarities, db.labels_)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_eps = eps
        eps = eps + 0.0001

    return best_eps


def color_wrap(text):
    return "\x1b[31m%s\x1b[0m" % text


def replace_words(text, replacements, debug=False):
    """
    Replaces words in text with 'replacements' word mapping
    :param text: str (string to replace words in)
    :param replacements: dict (each key is the original word, and value is the word to be replaced. for example {'small': 'big'})
    :param debug: boolean
    :return: str
    """
    # Escape each word in the 'replacements' mapping
    replacements = {re.escape(orig): re.escape(alt) for orig, alt in replacements.items()}

    if debug:
        # Add the original word in brackets and add colors to the modified text
        replacements = {orig: color_wrap("%s (%s)" % (alt, orig)) for orig, alt in replacements.items()}

    for orig, alt in replacements.items():
        orig_reg = r"(^|\s|[^a-zA-Z0-9])(%s)(\s|[^a-zA-Z0-9\-]|$)" % orig
        alt_reg = r"\1%s\3" % alt
        text = re.sub(orig_reg, alt_reg, text)

    return text


class Novel:

    def __init__(self, txt_file):
        CLASSIFIER = 'english.muc.7class.distsim.crf.ser.gz'
        root = os.path.join(os.getcwd(), '..', 'libraries', 'stanford-ner-2018-10-16')
        ner_jar_file = os.path.join(root, 'stanford-ner.jar')
        ner_classifier = os.path.join(root, 'classifiers/' + CLASSIFIER)
        self.tagger = StanfordNERTagger(ner_classifier, ner_jar_file, encoding='utf-8')
        np.set_printoptions(threshold=sys.maxsize)
        logging.getLogger().setLevel(logging.INFO)
        STOP = stopwords.words('english') + list(string.punctuation)
        self.file = txt_file
        self.text = ''
        self.persons = []
        self.sentences = []
        self.aliases = []

    def read(self, path=''):
        if os.path.isfile(path + self.file):
            try:
                #TODO 'cp1252' with Crime-and-Punishment.txt
                file = open(path + self.file, 'r', encoding='utf8')
                self.text = file.read()
                # \s includes [\t\n\r\f\v]
                # self.text = re.sub(pattern='\s+', repl=' ', string=self.text).strip()
                self.tokens = wtk(self.text)  # Tokenization
            except IOError:
                logging.ERROR('\t Cannot open ' + self.file)

    def parse_persons(self):
        people = {}
        words_tag = self.tagger.tag(self.tokens)
        name = ""

        for word, tag in words_tag:
            # a name is made of 1 or more names, read all
            if tag == 'PERSON':
                word = word.upper()
                word = word.translate(word.maketrans('', '', '!"”“#$%"&\'’()*+,./:;<=>?@[]^_`{|}~ʹ'))
                if word == 'DON':
                    continue
                if name == "":
                    name += word
                else:
                    name += " " + word
            else:
                # name is not empty
                if name:
                    name = name.strip()
                    current_name = name.split(" ")
                    # Usually and/ed/or are identified as name, e.g. Tom and Jerry
                    if len(current_name) == 3 and (current_name[1] == 'AND' or current_name[1] == 'TO' or \
                                                   current_name[1][-2:] == 'ED' or current_name[1] == 'OR' or \
                                                   current_name[1] == 'NOR'):
                        people[current_name[0]] = people.get(current_name[0], 0) + 1
                        people[current_name[2]] = people.get(current_name[2], 0) + 1
                    # Usually 2 words name contains adverbs or adjectives (...ly) verb (...ed), remove them
                    elif len(current_name) == 2 and ((current_name[1] in string.punctuation) or \
                                                     (current_name[1][-2:] == 'ED') or \
                                                     (current_name[1][-2:]) == 'LY' or \
                                                     (current_name[1] in CONJUNCTIONS)):
                        people[current_name[0]] = people.get(current_name[0], 0) + 1
                    elif len(current_name) == 2 and current_name[0] in FORENAMES:
                        people[current_name[1]] = people.get(current_name[1], 0) + 1
                    else:
                        people[name] = people.get(name, 0) + 1
                    name = ""

        self.persons = collections.OrderedDict(sorted(people.items()))
        return

    def cluster_aliases(self):
        alphabet = collections.defaultdict(list)
        for name in self.persons:
            alphabet[name[0].upper()].append(name)

        clusters_number = 0

        db_names = defaultdict(list)
        for letter, names in alphabet.items():
            n_persons = len(names)
            similarities = np.empty((n_persons, n_persons))
            if len(names) == 1:
                db_names[clusters_number].append(names[0])
                clusters_number += 1
                continue

            for i, person1 in enumerate(names):
                for j, person2 in enumerate(names):
                    # differ = difflib.SequenceMatcher(None, person1, person2) similarities[i][j] = differ.ratio()
                    # similarities[i][j] = fuzz.ratio(person1, person2)/100.
                    # similarities[i][j] = fuzz.token_sort_ratio(person1, person2) / 100.
                    # similarities[i][j] = fuzz.token_set_ratio(person1, person2) / 100.

                    # take the shortest word and find the
                    # similarity between this name and each subslice of the longer name (with the same length). It
                    # returns the higher value.
                    similarities[i][j] = fuzz.partial_ratio(person1, person2) / 100.

            # eps = find_best_eps(similarities)
            # print(letter, ': ', eps)
            eps = 0.3
            db = DBSCAN(metric='precomputed', min_samples=1, algorithm='brute', eps=eps).fit(1 - similarities)

            labels = db.labels_
            if -1 in labels:
                print('Some names are not clustered')
            for i, name in enumerate(names):
                db_names[labels[i] + clusters_number].append(name)
            unique = np.unique(labels, return_counts=False)
            clusters_number += len(unique)

        cluster_rep = {}
        for id, names in db_names.items():
            repetitions = []
            for name in names:
                repetitions.append(self.persons[name])
            cluster_rep[id] = (names, repetitions)

        self.cluster_repetitions = cluster_rep

    def associate_single_names(self):
        single_names = []
        single_ids = []
        multiple_names = []
        multiple_ids = []
        for id, names_repetitions in self.cluster_repetitions.items():
            names = names_repetitions[0]
            if len(names) == 1:
                single_names.append(names_repetitions)
                single_ids.append(id)
            else:
                multiple_names.append(names_repetitions)
                multiple_ids.append(id)

        similarity = {}
        for id_single, single_name_repetitions in zip(single_ids, single_names):
            single_name = single_name_repetitions[0][0]
            single_repetition = single_name_repetitions[1][0]

            for id, names_repetitions in self.cluster_repetitions.items():
                if id_single != id:
                    names = names_repetitions[0]
                    for name in names:
                        if single_name in name:
                            # print(single_name, ' - ', names)
                            if id_single not in similarity:
                                similarity[id_single] = []

                            similarity[id_single].append(id)
                            break
        # print(similarity)
        # add similar names to a cluster
        new_cluster = self.cluster_repetitions
        for id_single, ids_simile in similarity.items():
            add_user = new_cluster[id_single][0][0]
            add_repetition = new_cluster[id_single][1][0]
            id_to_update = ids_simile[0]
            # Winsley is contained in many cluster, insert it into the cluster with more repetitions
            if len(ids_simile) > 1:
                id_best = -1
                best = -1
                for id in ids_simile:
                    repetitions = new_cluster[id][1]
                    sum_repetitions = sum(repetitions)
                    if sum_repetitions > best:
                        best = sum_repetitions
                        id_best = id
                id_to_update = id_best

            cluster_repetitions = new_cluster[id_to_update]
            cluster_repetitions[0].append(add_user)
            cluster_repetitions[1].append(add_repetition)

        # delete bottom up, to eliminate problem with indexes
        similarity = sorted(list(similarity.items()), key=lambda x: x, reverse=True)
        for id_single, ids_simile in similarity:
            del new_cluster[id_single]

        fix_indexes_cluster = {}
        for i, values in enumerate(new_cluster.values()):
            fix_indexes_cluster[i] = values

        self.cluster_repetitions = fix_indexes_cluster

    def dealiases(self):
        text = self.text.upper()
        replacements = {}
        for id, names_rep in self.cluster_repetitions.items():
            character = 'CCHARACTER' + str(id)
            names = names_rep[0]
            for name in names:
                replacements[name] = character

        ordered_replacements = {}
        for k in sorted(replacements, key=len, reverse=True):
            ordered_replacements[k] = replacements[k]

        self.dealiased_text = replace_words(text, ordered_replacements)
        return

    def store(self, filename, data, type='csv'):
        if type == 'csv':
            try:
                with open(filename, 'w', newline='', encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    for key, value in data.items():
                        writer.writerow([key, value])
            except IOError:
                print("I/O error")
        else:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(data)

    def remove_less_than(self, occurrences):
        new_persons = {}
        for name, occurrence in self.persons.items():
            if occurrence <= occurrences:
                continue
            else:
                new_persons[name] = occurrence

        self.persons = new_persons
        return
