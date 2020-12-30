import os
import os.path
import string
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as wtk
import re
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
from nltk import sent_tokenize
import neuralcoref
import spacy
import pandas as pd

java_path = "C:/Program Files/Java/jre1.8.0_271/bin/java.exe"
os.environ['JAVAHOME'] = java_path

CONJUNCTIONS = ['therefore', 'does', 'when', 'and', 'this', 'from', 'that', 'at', 'to', 'i', 'in', 'said', 'of',
                'he', 'by']
FALSE_POSITIVES = ['Don', 'don']
PRE_NAMES = set()
ENCODING = ['utf8', 'cp1252']


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
            for encode in ENCODING:
                try:
                    file = open(path + self.file, 'r', encoding=encode)
                    text = file.read()
                    self.original_text = text
                    self.text = re.sub(pattern='\s+', repl=' ', string=self.text).strip()
                    # text = text.replace('\n', ' ')
                    text = re.sub(' +', ' ', text)
                    text = text.strip()
                    self.text = text
                    self.sentences = sent_tokenize(text)
                    break
                except IOError:
                    logging.error('\t Cannot open ' + self.file)
                    exit(-1)
                except UnicodeDecodeError:
                    logging.warning('\t Cannot open file using encoding ' + encode + ' trying a new encoding!')

    def custom_coref_resolved(self, doc):
        ''' Use this method instead of doc._.coref_resolved, here we clean the character's name before to replace it.
        That because sometimes the coref method identifies commas, quotes,... as part of the name'''

        clusters = doc._.coref_clusters
        resolved = list(tok.text_with_ws for tok in doc)
        for cluster in clusters:
            for coref in cluster:
                if coref != cluster.main:
                    new_name = cluster.main.text.translate(str.maketrans('', '', string.punctuation)).strip()
                    resolved[coref.start] = new_name + doc[coref.end - 1].whitespace_
                    for i in range(coref.start + 1, coref.end):
                        resolved[i] = ""
        return ''.join(resolved)

    def coreference(self):
        nlp = spacy.load("en_core_web_sm")
        coref = neuralcoref.NeuralCoref(nlp.vocab)
        nlp.add_pipe(coref, name='neuralcoref')
        words = self.dealiased_text.split(' ')
        words_number = len(words)
        badge_size = 100000
        if words_number > badge_size:
            if words_number % badge_size == 0:
                iterations = int(words_number / badge_size)
            else:
                iterations = int(words_number / badge_size)
                iterations += 1

            new_text = ""
            for i in range(0, iterations):
                logging.info('Coreferencing part ' + str(i + 1) + ' of ' + str(iterations))
                from_index = i * badge_size
                to_index = (i+1) * badge_size
                sub_text = ' '.join(words[from_index:to_index])

                text_coreference = nlp(sub_text)
                # text = text_coreference._.coref_resolved
                new_text += self.custom_coref_resolved(text_coreference)
        else:
            new_text = self.dealiased_text

        self.dealiased_text = new_text

    def create_cluster_repetitions_df(self):
        self.cluster_repetitions_df = pd.DataFrame(
            data=[['CCHARACTER' + str(key), val[0], val[1]] for key, val in self.cluster_repetitions.items()],
            columns=['Alias', 'Names', 'Occurrences'])

    def parse_persons(self):
        people = {}
        name = ""
        # contains_punctuations = False
        tokenized_sentences = [wtk(sentence) for sentence in self.sentences]
        tagged_sentences = self.tagger.tag_sents(tokenized_sentences)
        for sentence in tagged_sentences:
            for word, tag in sentence:
                # a name is made of 1 or more names, read all
                if tag == 'PERSON':
                    if len(word) == 1:
                        # print(word)
                        continue
                    # all strange symbols: '!"”“#$%"&\'’()*+,./:;<=>?@[]^_`{|}~ʹ'
                    # if word start or end with special characters, drop them
                    # if word[0] in '!"”"“#$%"&\'’()*+,/:;<=>?@[]^_`{|}~ʹ':
                    #     word = word[1:]
                    #     contains_punctuations = True
                    # if word[-1] in '!"”"“#$%"&\'’()*+,/:;<=>?@[]^_`{|}~ʹ':
                    #     word = word[:-1]
                    #     contains_punctuations = True
                    if name == "":
                        name += word
                    else:
                        name += " " + word
                else:
                    # name is not empty
                    if name:
                        name = name.strip()
                        current_name = name.split(" ")
                        # if len(current_name) >= 2 and contains_punctuations:
                        #     print(name)
                        # Usually and/ed/or are identified as name, e.g. Tom and Jerry
                        if len(current_name) == 3 and (current_name[1] == 'and' or current_name[1] == 'to' or \
                                                       current_name[1][-2:] == 'ed' or current_name[1] == 'or' or \
                                                       current_name[1] == 'nor'):
                            people[current_name[0]] = people.get(current_name[0], 0) + 1
                            people[current_name[2]] = people.get(current_name[2], 0) + 1
                        # Usually 2 words name contains adverbs or adjectives (...ly) verb (...ed), remove them
                        elif len(current_name) == 2 and ((current_name[1] in string.punctuation) or \
                                                         (current_name[1][-2:] == 'ed') or \
                                                         (current_name[1][-2:]) == 'ly' or \
                                                         (current_name[1].lower() in CONJUNCTIONS)):
                            people[current_name[0]] = people.get(current_name[0], 0) + 1
                        elif len(current_name) == 1 and current_name[0] in FALSE_POSITIVES:
                            name = ""
                        else:
                            people[name] = people.get(name, 0) + 1
                        name = ""
                        # contains_punctuations = False

        self.persons = collections.OrderedDict(sorted(people.items()))
        return

    def cluster_aliases(self):
        complete_alphabet_names = collections.defaultdict(list)
        simplified_alphabet_names = collections.defaultdict(list)
        for name in self.persons:
            split_name = name.lower().split()
            new_name = ""
            if len(split_name) == 1:  # single names do not have pre-names
                new_name = split_name[0]
            else:
                for name_part in split_name:
                    is_prename = False
                    for pre_name in PRE_NAMES:
                        if name_part == pre_name:
                            is_prename = True
                    if not is_prename:
                        new_name += " " + name_part
                new_name = new_name.strip()
            if len(new_name) == 0:
                new_name = name
            complete_alphabet_names[new_name[0].upper()].append(name)
            simplified_alphabet_names[new_name[0].upper()].append(new_name)

        clusters_number = 0
        db_names = defaultdict(list)
        db_simplified_names = defaultdict(list)
        for letter, names in simplified_alphabet_names.items():
            n_persons = len(names)
            similarities = np.empty((n_persons, n_persons))
            if len(names) == 1:
                db_names[clusters_number].append(complete_alphabet_names[letter][0])
                db_simplified_names[clusters_number].append(simplified_alphabet_names[letter][0])
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
                logging.info('Some names are not clustered')
            for i, name in enumerate(complete_alphabet_names[letter]):
                db_names[labels[i] + clusters_number].append(name)
                simplified_name = simplified_alphabet_names[letter][i]
                db_simplified_names[labels[i] + clusters_number].append(simplified_name)

            unique = np.unique(labels, return_counts=False)
            clusters_number += len(unique)

        cluster_rep = {}
        simple_cluster_rep = {}
        for id, some_names in db_names.items():
            repetitions = []
            for name in some_names:
                repetitions.append(self.persons[name])
            cluster_rep[id] = (some_names, repetitions)
            simple_cluster_rep[id] = (db_simplified_names[id], repetitions)

        # Debug here to discover which names are correctly clustered
        self.cluster_repetitions = cluster_rep
        self.simple_cluster_repetitions = simple_cluster_rep

    def find_persons_title(self):
        text = self.text.replace('\n', ' ')
        new_names = {}
        for name, occurrence in self.persons.items():
            pre_names = re.findall(r'([^ \r\n]+)( ' + name + ')([\r\n]| |$|.)', text, re.IGNORECASE)
            if len(pre_names) == 0:
                continue
            pre_names_occurrences = collections.defaultdict(int)
            for pre_name in pre_names:
                # skip prename which end with punctuations, it is not in the same phrase as the subject
                if pre_name[0][-1] in '!"”“#$%"&\'’()*+,./:;<=>?@[]^_`{|}~ʹ':
                    continue
                pre_names_occurrences[pre_name[0]] += 1
            if len(pre_names_occurrences) == 0:
                continue
            max_index = np.argmax(pre_names_occurrences.values())
            max_occurrence = list(pre_names_occurrences.values())[max_index]
            new_prename = list(pre_names_occurrences.keys())[max_index]
            if float(max_occurrence) / float(occurrence) > 0.5 and max_occurrence > 1:
                # skip special starting character in the pre-name
                if new_prename[0] in '!"”"“#$%"&\'’()*+,./:;<=>?@[]^_`{|}~ʹ':
                    new_prename = new_prename[1:] + ' ' + name

                if new_prename.lower() not in CONJUNCTIONS:
                    new_name = new_prename + ' ' + name
                    logging.info('Adding new name: %s', new_name)
                    new_names[new_name] = max_occurrence
                    PRE_NAMES.add(new_prename.lower())
                    logging.info('Adding new pre-name: %s', new_prename.lower())

        persons = self.persons
        for new_name, occurrence in new_names.items():
            if new_name not in persons:
                persons[new_name] = occurrence
            if new_name in persons:
                persons[new_name] += occurrence

        self.persons = collections.OrderedDict(sorted(persons.items()))

    def filter_similar_names(self, similarity):
        # Winsley is contained in many cluster, insert it into the cluster with more repetitions
        old_similarity = similarity
        for key_a, value_a in old_similarity.items():
            if len(value_a) > 1:
                id_best = -1
                best = -1
                for id in value_a:
                    repetitions = self.cluster_repetitions[id][1]
                    sum_repetitions = sum(repetitions)
                    if sum_repetitions > best:
                        best = sum_repetitions
                        id_best = id
                similarity[key_a] = [id_best]

        # add similar names to a cluster
        new_cluster = self.cluster_repetitions
        new_simple_cluster = self.simple_cluster_repetitions
        to_remove = set()
        to_delete_at_end = set()
        for key_a, value_a in similarity.items():
            if key_a in to_remove:
                continue

            # find other names with the same preference
            same_preferences = set()
            for key_b, value_b in similarity.items():
                if value_b[0] == value_a[0]:
                    same_preferences.add(key_b)

            # more key with the same preference
            selected_cluster = -1
            if len(same_preferences) > 1:
                # take the max
                max = -1
                best_key = -1
                for preference in same_preferences:
                    occurrences = sum(self.cluster_repetitions[preference][1]) + sum(
                        self.cluster_repetitions[similarity[preference][0]][1])
                    if occurrences > max:
                        max = occurrences
                        best_key = preference

                for preference in same_preferences:
                    if preference != key_a:
                        to_remove.add(preference)
                selected_cluster = best_key
            else:
                selected_cluster = list(same_preferences)[0]

            # check if the value of the best is also a key
            value = similarity[selected_cluster][0]
            if value in similarity:
                # the similarity is symmetric? A wants B and B wants A?
                if similarity[value] != selected_cluster:
                    # take the max and remove the other, a=AB and b=BC
                    occurrences_a = sum(self.cluster_repetitions[selected_cluster][1]) + sum(
                        self.cluster_repetitions[value][1])
                    occurrences_b = sum(self.cluster_repetitions[value][1]) + sum(
                        self.cluster_repetitions[similarity[value][0]][1])

                    non_selected_cluster = selected_cluster if np.argmin([occurrences_a, occurrences_b]) == 0 else value
                    selected_cluster = selected_cluster if np.argmax([occurrences_a, occurrences_b]) == 0 else value
                    to_remove.add(selected_cluster)
                    to_remove.add(non_selected_cluster)
                else:
                    to_remove.add(selected_cluster)
                    to_remove.add(value)
            else:
                to_remove.add(selected_cluster)

            # Update both the list with original names and the one with simplified names
            add_user = new_cluster[selected_cluster][0]
            add_repetition = new_cluster[selected_cluster][1]
            add_simple_user = new_simple_cluster[selected_cluster][0]

            cluster_repetitions = new_cluster[similarity[selected_cluster][0]]
            cluster_repetitions[0].extend(add_user)
            # next operation will update both the original and the simple names list
            cluster_repetitions[1].extend(add_repetition)
            cluster_repetitions = new_simple_cluster[similarity[selected_cluster][0]]
            cluster_repetitions[0].extend(add_simple_user)
            to_delete_at_end.add(selected_cluster)

        return to_delete_at_end, new_cluster, new_simple_cluster

    def associate_simple_single_names(self):
        single_names = []
        single_ids = []
        multiple_names = []
        multiple_ids = []
        # find clusters composed by only 1 name and clusters with more names
        for id, names_repetitions in self.simple_cluster_repetitions.items():
            names = names_repetitions[0]
            if len(names) == 1 or all(name == names[0] for name in names):
                single_names.append(names_repetitions)
                single_ids.append(id)
            else:
                multiple_names.append(names_repetitions)
                multiple_ids.append(id)

        # compute the similarity between the single names and all other clusters (also other single names)
        similarity = {}
        for key_a, single_name_repetitions in zip(single_ids, single_names):
            single_name = single_name_repetitions[0][0]
            # single_repetition = single_name_repetitions[1][0]

            for id, names_repetitions in self.simple_cluster_repetitions.items():
                if key_a != id:
                    names = names_repetitions[0]
                    for name in names:
                        if single_name in name or name in single_name:
                            # print(single_name, ' - ', names)
                            if key_a not in similarity:
                                similarity[key_a] = []

                            similarity[key_a].append(id)
                            break

        to_delete_at_end, new_cluster, new_simple_cluster = self.filter_similar_names(similarity)

        fix_indexes_cluster, fix_simple_indexes_cluster = self.delete_names_bottom_up(to_delete_at_end, new_cluster,
                                                                                      new_simple_cluster)
        self.cluster_repetitions = fix_indexes_cluster
        self.simple_cluster_repetitions = fix_simple_indexes_cluster

    def delete_names_bottom_up(self, to_delete_at_end, new_cluster, new_simple_cluster):
        # delete bottom up, to eliminate problem with indexes
        to_delete_at_end = sorted(list(to_delete_at_end), key=lambda x: x, reverse=True)
        for key_a in to_delete_at_end:
            del new_cluster[key_a]
            del new_simple_cluster[key_a]

        fix_indexes_cluster = {}
        fix_simple_indexes_cluster = {}
        i = 0
        for cluster_idx, values in new_cluster.items():
            fix_indexes_cluster[i] = values
            fix_simple_indexes_cluster[i] = new_simple_cluster[cluster_idx]
            i += 1

        return fix_indexes_cluster, fix_simple_indexes_cluster

    def associate_single_names(self):
        similarity = {}
        for id1, value1 in self.cluster_repetitions.items():
            if len(value1[0]) == 1:
                for id2, value2 in self.cluster_repetitions.items():
                    if id1 != id2:
                        single_name = value1[0][0]
                        if any(single_name in name for name in value2[0]):
                            if id1 not in similarity:
                                similarity[id1] = []

                            similarity[id1].append(id2)

        to_delete_at_end, new_cluster, new_simple_cluster = self.filter_similar_names(similarity)
        fix_indexes_cluster, fix_simple_indexes_cluster = self.delete_names_bottom_up(to_delete_at_end, new_cluster,
                                                                                      new_simple_cluster)
        self.cluster_repetitions = fix_indexes_cluster
        self.simple_cluster_repetitions = fix_simple_indexes_cluster

    def dealiases(self):
        replacements = {}
        for id, names_rep in self.cluster_repetitions.items():
            character = 'CCHARACTER' + str(id)
            names = names_rep[0]
            for name in names:
                replacements[name] = character

        ordered_replacements = {}
        for k in sorted(replacements, key=len, reverse=True):
            ordered_replacements[k] = replacements[k]

        self.dealiased_text = replace_words(self.text, ordered_replacements)
        return

    def store(self, filename, data, type='csv'):
        if type == 'csv':
            try:
                with open(filename, 'w', newline='', encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    for key, value in data.items():
                        writer.writerow([key, value])
            except IOError:
                logging.info("I/O error")
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
