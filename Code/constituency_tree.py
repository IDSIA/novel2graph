import pandas as pd
from stanza.server import CoreNLPClient
from nltk.tree import *
import logging
from nltk.translate.ribes_score import position_of_ngram


class ConstituencyTree:
    def __init__(self, original_sentences, lemmatize_sentences):
        self.to_delete = []
        self.sentences = pd.Series([' '.join(sent) for sent in original_sentences],
                                   index=original_sentences.index)
        self.lemmatize_sentences = pd.Series([' '.join(sent) for sent in lemmatize_sentences],
                                             index=lemmatize_sentences.index)

        self.sentences_zero = []
        self.sentences_few = []
        self.sentences_right = []
        self.sentences_more = []

    # get noun phrases with tregex
    def noun_phrases(self, client, sentence, annotators=None):
        pattern = 'NP VP'
        matches = client.tregex(sentence, pattern, annotators=annotators)
        # print("\n".join(
        #     ["\t" + sentence[match_id]['spanString'] for sentence in matches['sentences'] for match_id in sentence]))

    def get_sentences(self):
        cols = ['Characters', 'Original_token', 'Lemmatize_token', 'Tag']
        sentences_zero = pd.DataFrame(data=self.sentences_zero,
                                      columns=cols)
        sentences_few = pd.DataFrame(data=self.sentences_few,
                                     columns=cols)
        sentences_right = pd.DataFrame(data=self.sentences_right,
                                       columns=cols)
        sentences_more = pd.DataFrame(data=self.sentences_more,
                                      columns=cols)

        return sentences_zero, sentences_few, sentences_right, sentences_more

    def find_vpnp_sentences(self, tree):
        for subtree in tree:
            # we are in a leaf
            if type(subtree) != ParentedTree:
                continue
            if subtree.parent().label() != 'ROOT' and subtree.label() == 'S':
                # verify if it has NP and VP
                children_types = subtree.productions()[0]._rhs
                children_types = [child._symbol for child in children_types]
                if 'NP' in children_types and 'VP' in children_types:
                    # print('Found subphrase: ' + ' '.join(subtree.leaves()))
                    self.to_delete.append(subtree.treeposition())
            self.find_vpnp_sentences(subtree)

    # def find_s_sentences(self, tree):
    #     for subtree in tree:
    #         # we are in a leaf
    #         if type(subtree) != ParentedTree:
    #             continue
    #         if subtree.parent().label() != 'ROOT' and subtree.label() == 'S':
    #             print(' '.join(tree.leaves()))
    #             print(' '.join(subtree.leaves()))
    #             self.to_delete.append(subtree.treeposition())
    #         self.find_s_sentences(subtree)
    #
    #
    # def traverse_tree(self, tree):
    #     # print("tree:", tree)
    #     # t .hight(), .label(), .parent()
    #     subject_str = ""
    #     verb_str = ""
    #     complement_str = ""
    #     subject = []
    #     verb = []
    #     complement = []
    #     for subtree in tree:
    #         if type(subtree) == ParentedTree:
    #             # verify how many cchar there are in this sentence, if to many:
    #             # if subtree == S
    #             # search last/next NP and use it as subject
    #             # if there is at least one NP and a VP
    #             # all before NP (PP,...) is read as subject toghether with NP
    #             # if VP has a verb and more complement
    #             # the verb of VP (VBD,..) goes with NP and they are connected to each complement(NP, FRAG,...) that are brothers to make a sentence
    #             # if VP has more VP inside
    #             # each intern VP goes with NP and make a sentence
    #             # if NP has more NP
    #             # each NP is connected to the VP and make a sentence
    #
    #             if subtree.label() == 'S':
    #                 self.traverse_tree(subtree)
    #
    #             if subtree.label() == 'VP':
    #                 self.traverse_tree(subtree)
    #                 subject_type = self.traverse_vp_tree(subtree)
    #                 if new_subject is not None:
    #                     subject = new_subject
    #             else:
    #                 if subtree.parent().label() == 'S' and subtree.label() != '.':
    #                     leaves_str = ' '.join(subtree.leaves())
    #                     subject_str += leaves_str
    #                     subject.append(subtree.label())
    #                     print('New subj:', subject)
    #
    #                 if subtree.parent().label() == 'VP':
    #                     if subtree.label() != 'NP':
    #                         leaves_str = ' '.join(subtree.leaves())
    #                         verb_str += leaves_str
    #                         verb.append(subtree.label())
    #                         print('New verb: ', verb_str)
    #                         if subtree.label() == 'CC':
    #                             print(subject_str, ' ', verb_str, ' ', complement_str)
    #                             verb_str = ""
    #                             verb = []
    #                     if subtree.label() == 'NP':
    #                         leaves_str = ' '.join(subtree.leaves())
    #                         complement_str += leaves_str
    #                         complement.append(subtree.label())
    #                         print('New complement: ', verb_str)
    #                         if subtree.label() == 'CC':
    #                             print(subject_str, ' ', verb_str, ' ', complement_str)
    #                             verb_str = ""
    #                             verb = []

    def add_sentence(self, sentence_tree, sentence_tree_lemmatize):
        number_of_char = 2
        chars = [word for word in sentence_tree.leaves() if 'CCHARACTER' in word]
        count = len(chars)
        try:
            tag = [prod._lhs._symbol for prod in sentence_tree.productions() if
                   len(prod._rhs) > 0 and type(prod._rhs[0]) == str]

            leaves = sentence_tree.leaves()

            if len(tag) != len(leaves):
                logging.info('Someting wrong with this subsentence.')
                return
            if count == 0:
                self.sentences_zero.append([chars, leaves, sentence_tree_lemmatize, tag])
            elif count < number_of_char:
                self.sentences_few.append([chars, leaves, sentence_tree_lemmatize, tag])
            elif count > 2:
                self.sentences_more.append([chars, leaves, sentence_tree_lemmatize, tag])
            else:
                self.sentences_right.append([chars, leaves, sentence_tree_lemmatize, tag])
        except:
            logging.info("An exception occurred while processing this subtree.")

    def delete_subtree(self, tree, lemmatize_sentence):
        if len(self.to_delete) > 0:
            for j in range(len(self.to_delete), 0, -1):
                sub_tree_index = self.to_delete[j - 1]
                try:
                    sub_tree = tree[sub_tree_index]
                    #start index is 1 based
                    start_index = position_of_ngram(tuple(sub_tree.leaves()), tree.leaves())
                    end_index = start_index + len(sub_tree.leaves())
                    sub_tree_lemmatize = lemmatize_sentence[start_index:end_index]
                    self.add_sentence(sub_tree, sub_tree_lemmatize)
                    del tree[sub_tree_index]
                    del lemmatize_sentence[start_index:end_index]
                except:
                    logging.info("An exception occurred, while deleting a subphrase in this tree.")

            remaining_sentence_str = ' '.join(tree.leaves())
            if remaining_sentence_str != "":
                self.add_sentence(tree, lemmatize_sentence)

            # print('Remaining sentence: ', remaining_sentence_str)

    def parse_sentences(self, sentences):
        sentences_str = []
        tags = []
        lemmas = []
        CUSTOM_PROPS = {"parse.model": "edu/stanford/nlp/models/srparser/englishSR.ser.gz"}
        with CoreNLPClient(properties=CUSTOM_PROPS,
                           annotators=['pos', 'parse', 'lemma'],
                           timeout=30000,
                           memory='2G', output_format="json") as client:
            # noun_phrases(client,sentence_str,annotators="parse")
            for sentence in sentences:
                annotations = client.annotate(sentence)['sentences'][0]
                sentences_str.append(annotations['parse'])
                tag = [sentence['pos'] for sentence in annotations['tokens']]
                lemma = [sentence['lemma'] for sentence in annotations['tokens']]
                if len(tag) != len(lemma):
                    logging.info('Check this sentence: ', tag, lemma)
                tags.append(tag)
                lemmas.append(lemma)
        return sentences_str, tags, lemmas

    def extract_sentences(self):
        parsed_sentences, tagged_sentences, lemmas = self.parse_sentences(self.sentences)
        for i, complete_tree in enumerate(parsed_sentences):
            # tag = tagged_sentences[i]
            self.to_delete = []

            # tree.leaves(), tree.treeposition(), tree.subtrees(), tree.productions(), tree.parents()
            # ParentedTree.draw(tree)
            tree = ParentedTree.fromstring(complete_tree)
            # tree = Tree.fromstring(str_tree)
            # print('----')
            # print('Initial sentence:' + ' '.join(tree.leaves()))

            self.find_vpnp_sentences(tree)
            self.delete_subtree(tree, lemmas[i])

        # TODO improve phrases extraction with find_s_sentences traverse_tree?
        # for i, complete_tree in enumerate(sentences_more):
        #     self.to_delete = []
        #     find_s_sentences(complete_tree)
        #     delete_subtree(complete_tree)
        # traverse_tree(tree)
