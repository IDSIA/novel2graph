from embedding import group_aliases, read_alias_occurrences
from constituency_tree import ConstituencyTree
import networkx as nx
from graphviz import Source
from BERT_cluster_sentences import Bert_cluster
# parsetree only works with python <= 3.6!!
from pattern.en import parsetree

from nltk import sent_tokenize
import nltk
import numpy as np
import os
import random
import re
import logging
import spacy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from spacy.matcher import Matcher
import pandas as pd

if os.name == 'nt':
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["CORENLP_HOME"] = "..\\libraries\\stanford-corenlp-4.2.0"
result_folder = './../Data/embedRelations/'
book = None


def plot_grouped_relations(clusters_labels, occurrences, relations_pair_occurrences, result_folder, k,
                           min_entity=15, min_relation=1):
    """Plot a graph (graphviz) in wich the dimension of each character is proportional to his occurrences in the
        novel, and the dimension of each relation is proportional to the number in which two characters are
        in the same sentence."""
    graph = nx.DiGraph()

    max_occurrences_index = np.argmax(occurrences)
    max_occurrence = occurrences[max_occurrences_index]
    i = 0
    show_char = []
    for key, value in clusters_labels.items():
        if occurrences[i] >= min_entity:
            label = value[0].title()
            graph.add_node(key, width=str(2 + (occurrences[i] / max_occurrence) * 5), label=label, key=key,
                           shape="circle",
                           fontname="Arial", fontsize=35)
            show_char.append(True)
        else:
            show_char.append(False)
        i += 1

    relations_color = []
    for i in range(0, k):
        relations_color.append("#{:06x}".format(random.randint(0, 0xFFFFFF)))

    best_occurrence = -1
    for relation_id, pair_occurrences in relations_pair_occurrences.items():
        for pair, occurrence in pair_occurrences.items():
            if occurrence > best_occurrence:
                best_occurrence = occurrence

    for relation_id, pair_occurrences in relations_pair_occurrences.items():
        for pair, occurrence in pair_occurrences.items():
            char1id = int(pair[0].split('CCHARACTER')[1])
            char2id = int(pair[1].split('CCHARACTER')[1])
            if show_char[char1id] and show_char[char2id]:
                if occurrence > min_relation:
                    character1 = pair[0]
                    character2 = pair[1]
                    graph.add_edge(character1, character2, color=relations_color[relation_id],
                                   penwidth=str(occurrence / best_occurrence * 20), key=relation_id)
                    logging.info('%s\t%s\t%s', character1, str(relation_id), character2)

    nx.write_gpickle(graph, result_folder + book + '/' + book + "_graph_gpickle")
    # G = nx.read_gpickle("test.gpickle")

    dot_file = nx.nx_pydot.to_pydot(graph)
    s = Source(dot_file, format="pdf")
    s.render(result_folder + book + '/' + book + "_graph_dot", view=False)


def parse_cooccurrences(phrases):
    """Given some sentences, it computes how many times two characters are in the same sentence. It returns a
    map in which (CHARACTER0, CHARACTER1) is the key and the value are the occurrences of this relation (found in
    text) and another map with the same keys but containing the list of phrases in which character are contained."""
    relations = {}
    relation_phrases = {}
    asymmetric_relations = []
    regexp = re.compile(r"CCHARACTER([0-9]+)\sAND\sCCHARACTER([0-9]+)")
    for phrase in phrases:
        phrase = phrase.string
        count = phrase.count('CCHARACTER')
        if count < 2 or count > 2:
            continue
        if regexp.search(phrase):
            asymmetric_relations.append(phrase)
        characters = [word for word in phrase.split() if 'CCHARACTER' in word]
        if len(characters) != 2:
            logging.info('---Something went wrong with this sentence: %s', phrase)
            continue
        char0 = re.search('(CCHARACTER)([0-9]+)', characters[0]).group(0)
        char1 = re.search('(CCHARACTER)([0-9]+)', characters[1]).group(0)

        key = (char0, char1)
        if key in relations:
            relations[key] += 1
            relation_phrases[key].append(phrase)
        if key not in relations:
            relations[key] = 1
            relation_phrases[key] = []
            relation_phrases[key].append(phrase)

    return relations, relation_phrases, asymmetric_relations


def remove_tail_head(sentences):
    # deprecated
    relation_phrases = pd.DataFrame(columns=['Characters', 'Sentence'])
    for id, tree in sentences.iterrows():
        sentence_str = tree['Sentence'].text
        characters = [word.text for word in tree['Sentence'] if 'CCHARACTER' in word.text]
        if len(characters) != 2:
            logging.info('---Something went wrong with this sentence: %s', sentence_str)
            continue

        if characters[0] != characters[1]:
            relation_phrases.loc[id] = tree

    return relation_phrases


def remove_included_sentences(sentences):
    # deprecated
    ids_to_remove = []
    for id1, row in sentences.iterrows():
        tree1 = row['Sentence']
        sentence1_str = tree1.text
        for id2, row2 in sentences.iterrows():
            tree2 = row2['Sentence']
            sentence2_str = tree2.text
            if (sentence1_str != sentence2_str) and (sentence1_str in sentence2_str) and (id1 != id2):
                logging.info('Removing sentence: %s', sentence1_str)
                ids_to_remove.append(id1)

    new_sentences = pd.DataFrame(columns=['Characters', 'Sentence'])
    for id, tree in sentences.iterrows():
        if id in ids_to_remove:
            continue
        new_sentences.loc[id] = tree

    return new_sentences


def group_relations(relations):
    """In case the relations you receive contain Harry->Ron and Ron->Harry, this method merge these two relations
    into a single one named Harry-Ron, which is equal to the sum of the two relation values. """
    new_relations = {}
    processed = set()
    for key, value in relations.items():
        char1 = key[0]
        char2 = key[1]
        if key in processed:
            continue

        if (char2, char1) in relations:
            # print(char2, char1, value, relations[(char2, char1)], value + relations[(char2, char1)])
            value = value + relations[(char2, char1)]
            processed.add((char2, char1))
        new_relations[key] = value
    return new_relations


def prepare_data_openke(alias, tags, aliases):
    # Here I save files for OpenKE
    entity = open("entity2id.txt", "w")
    entity.write(str(len(alias)) + "\n")
    entity2id = {}
    for i in range(0, len(alias)):
        entity2id[alias[i]] = i
        entity.write(alias[i] + "\t" + str(i) + "\n")
    entity.close()

    list_of_tags = {}
    i = 0
    for key, relations_tag in tags.items():
        for relation_tag in relations_tag:
            tg = relation_tag[0]
            if tg not in list_of_tags:
                list_of_tags[tg] = i
                i += 1

    entity = open("relation2id.txt", "w")
    entity.write(str(len(list_of_tags)) + "\n")
    for key, value in list_of_tags.items():
        entity.write(key.replace(' ', '_') + "\t" + str(value) + "\n")
    entity.close()

    all2id = []
    probability = []
    for key, values in tags.items():
        for value in values:
            text = str(entity2id[key[0]]) + "\t" + str(entity2id[key[1]]) + "\t" + str(list_of_tags[value[0]]) + "\n"
            # Remember only most probable relations
            if value[1] > 0.5:
                if entity2id[key[0]] != entity2id[key[1]]:
                    probability.append(value[1])
                    all2id.append(text)

    entity = open("summary.txt", "w")
    entity.write(str(len(all2id)) + "\n")
    for i, data in enumerate(all2id):
        datas = data.split('\t')
        relation = int(datas[-1].split('\n')[0])
        entity.write(
            aliases[int(datas[0])][0] + '\t' + list(list_of_tags.keys())[relation].replace(' ', '_') + "(" + str(
                probability[i]) + ")" + '\t' + aliases[int(datas[1])][0] + '\n')
    entity.close()

    entity = open("train2id.txt", "w")
    train_len = int(0.7 * len(all2id))
    entity.write(str(train_len) + "\n")
    for i in range(0, train_len):
        entity.write(all2id[i])
    entity.close()

    entity = open("test2id.txt", "w")
    test_len = int(0.15 * len(all2id))
    entity.write(str(test_len) + "\n")
    for i in range(train_len, train_len + test_len):
        entity.write(all2id[i])
    entity.close()

    entity = open("valid2id.txt", "w")
    val_len = int(0.15 * len(all2id))
    entity.write(str(val_len) + "\n")
    for i in range(train_len + test_len, train_len + test_len + val_len):
        entity.write(all2id[i])
    entity.close()


def split_direct_speech(nlp, sentence):
    pattern = [{'IS_PUNCT': True, 'ORTH': '"'}]
    matcher = Matcher(nlp.vocab)
    matcher.add("DirectSpeech", None, pattern)

    matches = matcher(sentence)

    if len(matches) == 0:
        return [sentence]
    else:
        if len(matches) % 2 != 0:
            logging.info("Found too many \", dropping sentence: %s", sentence)
            return [sentence]

        start = 0
        sentences = []
        for i in range(0, len(matches) + 1):
            if i < len(matches):
                match_id, start_punc, end_punc = matches[i]
                span = sentence[start:start_punc]  # The matched span
                start = end_punc
            else:
                # read the last phrase until the end
                span = sentence[start:]  # The matched span

            if span.text != "":
                sentences.append(span)

        return sentences


def set_custom_boundaries(doc):
    # Adds support to use `"`, '--' and `;` as the delimiter for sentence detection
    for token in doc[:-1]:
        if token.text == '"':
            doc[token.i + 1].is_sent_start = True
        if token.text == ';':
            doc[token.i + 1].is_sent_start = True
        if token.text == '--':
            doc[token.i + 1].is_sent_start = True

    return doc


def extract_sentences(dealiased_book_path, nlp):
    """Given a text it extractes phrases and sentences."""
    dealiased_book = open(dealiased_book_path, "r", encoding='utf8')
    book_str = dealiased_book.read()
    dealiased_book.close()

    if "set_custom_boundaries" not in nlp.pipe_names:
        nlp.add_pipe(set_custom_boundaries, before='parser')

    words = book_str.split(' ')
    words_number = len(words)
    badge_size = 100000
    if words_number > badge_size:
        if words_number % badge_size == 0:
            iterations = int(words_number / badge_size)
        else:
            iterations = int(words_number / badge_size)
            iterations += 1

        sent_text = []
        for i in range(0, iterations):
            logging.info('Extracting sentences in part ' + str(i + 1) + ' of ' + str(iterations))
            from_index = i * badge_size
            to_index = (i + 1) * badge_size
            sub_text = ' '.join(words[from_index:to_index])

            custom_doc = nlp(sub_text)
            partial_sentences = list(custom_doc.sents)
            sent_text.extend(partial_sentences)
    else:
        custom_doc = nlp(book_str)
        sentences = list(custom_doc.sents)
        sent_text = sentences

    # now loop over each sentence and tokenize it separately
    index_sentences = {}
    i = 0
    for sentence in sent_text:
        to_save = sentence
        if sentence.text != "\"" and len(sentence) > 2:
            if "\"" in sentence.text:
                to_save = sentence[:-1]
            index_sentences[i] = to_save
            i += 1
        else:
            i += 1
            continue
    return index_sentences


def extract_sentences_with_verbs(sentences):
    verified_sentences = {}
    no_verb_sentences = {}
    for id, tree in sentences.items():
        verb = False
        for word in tree:
            type = word.pos_
            if type == 'VERB':
                verb = True
                break
        if verb:
            verified_sentences[id] = tree
        else:
            no_verb_sentences[id] = tree

    return verified_sentences, no_verb_sentences


def extract_nchar_sentences(sentences, number_of_char=2):
    cols = ['Characters', 'Original_token', 'Lemmatize_token', 'Tag']
    sentences_zero = pd.DataFrame(columns=cols)
    sentences_few = pd.DataFrame(columns=cols)
    sentences_right = pd.DataFrame(columns=cols)
    sentences_more = pd.DataFrame(columns=cols)

    for id, sentence in sentences.items():
        # Sometimes CCHARACTERX is identified as VB
        original_words = [token.text for token in sentence]
        lemmatize_words = [token.lemma_ for token in sentence]
        tags = [token.tag_ for token in sentence]
        chars = [word for word in original_words if 'CCHARACTER' in word]
        sentence_str = sentence.text
        count = sentence_str.count('CCHARACTER')
        if count == 0:
            sentences_zero.loc[id] = [[], original_words, lemmatize_words, tags]
        elif count < number_of_char:
            sentences_few.loc[id] = [chars, original_words, lemmatize_words, tags]
        elif count > 2:
            sentences_more.loc[id] = [chars, original_words, lemmatize_words, tags]
        else:
            sentences_right.loc[id] = [chars, original_words, lemmatize_words, tags]

    return sentences_zero, sentences_few, sentences_right, sentences_more


def map_chars_sentences(sentences):
    relation_phrases = {}
    for id, sentence in sentences.items():
        sentence_str = sentence.string
        characters = [word for word in sentence_str.split() if 'CCHARACTER' in word]
        if len(characters) != 2:
            logging.info('---Something went wrong with this sentence: %s', sentence_str)
            continue
        char0 = re.search('(CCHARACTER)([0-9]+)', characters[0]).group(0)
        char1 = re.search('(CCHARACTER)([0-9]+)', characters[1]).group(0)

        key = (char0, char1)
        if key in relation_phrases:
            relation_phrases[key].append(sentence)
        if key not in relation_phrases:
            relation_phrases[key] = []
            relation_phrases[key].append(sentence)

    return relation_phrases


def remove_said_cchar(sentences):
    # deprecated
    relation_phrases = pd.DataFrame(columns=['Characters', 'Sentence'])
    for id, tree in sentences.iterrows():
        sentence_str = tree.text
        if 'said CCHARACTER' in sentence_str:
            logging.info('Dropping because contains said: %s', sentence_str)
            continue
        relation_phrases.loc[id] = tree
    return relation_phrases


def extract_asymmetric_sentences(sentences):
    # TODO "A and B walk" can be split to "A walks" and "B walks"?
    asymmetric_relations = []
    regexp = re.compile(r"CCHARACTER([0-9]+)\sand\sCCHARACTER([0-9]+)")
    # start_time = datetime.datetime.now()
    sentences_str = pd.Series([' '.join(sent) for sent in sentences['Original_token']])
    for id, sentence in sentences_str.items():
        if not regexp.search(sentence):
            values = sentences.iloc[id]
            asymmetric_relations.append(
                [values['Characters'], values['Original_token'], values['Lemmatize_token'], values['Tag']])
        # Uncomment to use the tree of each sentence instead of the regexp above
        # for chunk in tree.sentences[0].chunk:
        #     if not (chunk.string.count('CCHARACTER') == 2 and chunk.type == 'NP'):
        #         asymmetric_relations[id] = tree
    # print(str(datetime.datetime.now() - start_time))
    return pd.DataFrame(data=asymmetric_relations, columns=['Characters', 'Original_token', 'Lemmatize_token', 'Tag'])


def remove_verb_not_between_chars(sentences):
    sentences_str = pd.Series([' '.join(sent) for sent in sentences['Original_token']])
    relation_phrases = []
    for id, sentence in sentences_str.items():
        read = False
        verb_found = False
        tags = sentences.iloc[id]['Tag']
        tokens = sentences.iloc[id]['Original_token']
        for i, token in enumerate(tokens):
            type = tags[i]
            if 'CCHARACTER' in token:
                read = not read
                continue
            if read:
                if 'VB' in type:
                    values = sentences.iloc[id]
                    relation_phrases.append(
                        [values['Characters'], values['Original_token'], values['Lemmatize_token'], values['Tag']])
                    verb_found = True
                    break
        if not verb_found:
            logging.info('Dropping because no verb between subjects: %s', sentence)

    return pd.DataFrame(data=relation_phrases, columns=['Characters', 'Original_token', 'Lemmatize_token', 'Tag'])


def find_family_relation(sentences, info):
    family_rel = {"brotherhood": ["brother", "sister"],
                  "parent_son": ["mom" "mother", "father", "dad", "daddy"],
                  "others": ["uncle", "aunt", "grandmother", "grandfather",
                             "cousin", "mother-in-law", "father-in-law", "daughter-in-law", "niece", "nephew",
                             "sister-in-law", "daughter", "wife"]}

    for id, tree in sentences.items():
        sentence_str = tree.text
        for relation_type, relations in family_rel.items():
            for relation in relations:
                if " " + relation + " " in sentence_str:
                    last_character = find_last_character(tree)
                    first_character = find_first_character(tree)
                    if last_character is None or first_character is None or last_character == first_character:
                        continue
                    if first_character not in info:
                        info[first_character] = {}
                    if last_character not in info:
                        info[last_character] = {}

                    if first_character in info and "relations" not in info[first_character]:
                        info[first_character]["relations"] = []
                    if last_character in info and "relations" not in info[last_character]:
                        info[last_character]["relations"] = []

                    info[last_character]["relations"].append(relation_type + '_' + first_character)
                    info[first_character]["relations"].append(relation_type + '_' + last_character)

                    continue
    return info


def find_last_character(tree):
    last_name = None
    for word in tree:
        word_str = word.lemma_
        if word_str.startswith('CCHARACTER'):
            last_name = word_str
    return last_name


def find_first_character(tree):
    for word in tree:
        word_str = word.lemma_
        if word_str.startswith('CCHARACTER'):
            return word_str
    return None


def find_sex(sentences):
    info = {}
    sentences_position = {}
    man = ["him", "his", "he"]
    woman = ["her", "hers", "she"]
    for id, tree in sentences.items():
        first_word = tree[0].lemma_
        sex = None
        if first_word.lower() in man:
            sex = 'M'
        if first_word.lower() in woman:
            sex = 'F'
        if sex is None:
            continue

        character = find_last_character(sentences[id - 1])
        if character is None:
            continue

        if character not in info:
            sentences_position[character] = {}
            info[character] = {}
            info[character]["sex"] = {'F': 0, 'M': 0}
            sentences_position[character]["sex"] = {'F': [], 'M': []}

        info[character]["sex"][sex] += 1
        sentences_position[character]["sex"][sex].append(id)

    return info, sentences_position


def replace_starting_pron(sentences_position, sentences, nlp):
    # Rename his, her, him,... with the right subject (only at the beginning of the sentence)
    for char, infos in sentences_position.items():
        sex = find_max_occurrences_dict(infos["sex"])
        for sentence_index in sentences_position[char]["sex"][sex]:
            sentence_parts = sentences[sentence_index].text.split()
            if sentence_parts[0].lower() in ["him", "his", "her", "hers"]:
                sentence_parts[0] = char + '\'s'
            else:
                # She, he
                sentence_parts[0] = char
            new_sentence = " ".join(sentence_parts)
            # recompute the new tree of the dealiased sentence
            # new_sentence_analysis = parsetree(new_sentence, lemmata=True, relations=True)
            new_sentence_analysis = nlp(new_sentence)
            sentences[sentence_index] = new_sentence_analysis


def find_max_occurrences_dict(infos):
    max = None
    max_value = -1
    for sex, sentences_indeces in infos.items():
        value = len(sentences_indeces)
        if value > max_value:
            max = sex
    return max


def find_information(sentences, nlp):
    info, genders_pointer = find_sex(sentences)
    replace_starting_pron(genders_pointer, sentences, nlp)
    info = find_family_relation(sentences, info)
    return info


def filter_sentences(sentences):
    # partial_tokenized_sent = remove_included_sentences(sentences)
    # partial_tokenized_sent = remove_said_cchar(partial_tokenized_sent)
    partial_tokenized_sent = extract_asymmetric_sentences(sentences)
    # partial_tokenized_sent = remove_tail_head(partial_tokenized_sent)
    partial_tokenized_sent = remove_verb_not_between_chars(partial_tokenized_sent)

    return partial_tokenized_sent


def find_relations(relations_phrases, model_type='wiki80_cnn_softmax'):
    relations_tagger = opennre.get_model(model_type)
    tags = {}
    for relation, phrases in relations_phrases.items():
        if '.' in relation[0] or '.' in relation[1]:
            logging.info('%s %s', relation, phrase)
            continue
        if relation[0] == 'CCHARACTER' or relation[1] == 'CCHARACTER':
            # print(relation, phrase)
            continue
        tags[relation] = []
        for phrase in phrases:
            start_header = phrase.find(relation[0])
            end_header = start_header + len(relation[0])
            start_tail = phrase.find(relation[1])
            end_tail = start_tail + len(relation[1])

            tag = relations_tagger.infer({'text': phrase,
                                          'h': {'pos': (start_header, end_header)},
                                          't': {'pos': (start_tail, end_tail)}})
            tags[relation].append(tag)
    return tags


def tsne(embedding, keys):
    x_std = StandardScaler().fit_transform(embedding)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    principalComponents1 = tsne.fit_transform(x_std)
    results = {}
    for i, coordinates in enumerate(principalComponents1):
        results[keys[i]] = coordinates

    return results


def pca(embedding, keys):
    x_std = StandardScaler().fit_transform(embedding)
    pca = PCA(n_components=2, svd_solver='full')
    principalComponents2 = pca.fit_transform(x_std)
    results = {}
    for i, coordinates in enumerate(principalComponents2):
        results[keys[i]] = coordinates

    return results


def plot_verbs_embedding(embedding, k, verbs, labels):
    x_std = StandardScaler().fit_transform(embedding)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    principalComponents1 = tsne.fit_transform(x_std)

    pca = PCA(n_components=2)
    principalComponents2 = pca.fit_transform(x_std)

    fig, axs = plt.subplots(2)
    # fig.suptitle('Verbs embed clustering')
    colors = [np.random.rand(3, ) for i in range(0, k)]
    selected_colors = []
    for i in range(0, k):
        cluster = np.where(labels == i)
        x = [principalComponents1[index, 0] for index in cluster]
        y = [principalComponents1[index, 1] for index in cluster]
        selected_colors.append(colors[labels[i]])
        axs[0].scatter(x, y, c=colors[labels[i]], label=verbs[i].split(' ')[0])

        x = [principalComponents2[index, 0] for index in cluster]
        y = [principalComponents2[index, 1] for index in cluster]
        axs[1].scatter(x, y, c=colors[labels[i]])

    axs[0].set_title('TSNE projection')
    axs[1].set_title('PCA projection')
    # fig.legend(loc='lower center', handles=scatter.legend_elements()[0], labels=list(bert_cls.verbs.values()), ncol=9)
    # fig.xlabel('Dimension 1')
    # fig.ylabel('Dimension 2')
    fig.legend(loc='lower center', ncol=12)
    plt.show()


def chars_to_dataframe(grouped_aliases, occurrences, result_file_path):
    df = pd.DataFrame(columns=['Alias', 'Names', 'Occurrences'])
    occurrence_i = 0
    dataframe_i = 0
    for alias, names in grouped_aliases.items():
        alias_occurrences = occurrences[occurrence_i]
        for i, name in enumerate(names):
            df.loc[dataframe_i] = [alias, name, alias_occurrences[i]]
            dataframe_i += 1
        occurrence_i += 1
    df.to_pickle(result_file_path)


def embed_to_dataframe(verbs, sentences, chars, embedding_dict, result_file_path, chars_number=2):
    df = pd.DataFrame(
        columns=['Sentence ID', 'Character1', 'Verb', 'Character2', 'Embedding', 'Phrase'])
    dataframe_i = 0
    for id, sentence in sentences.items():
        char1 = chars[id][0]
        if chars_number == 2:
            char2 = chars[id][1]
        elif chars_number == 1:
            char2 = None
        else:
            logging.info('Are you sure about the number of characters? If so, implement this part.')
            exit(-1)
        df.loc[dataframe_i] = [id, char1, verbs[id], char2,
                               embedding_dict[id], sentence]
        dataframe_i += 1
    df.to_pickle(result_file_path)


def embed_cluster_to_dataframe(verbs, sentences, all_cluster, chars, embedding_dict, pca_tsne, label_verbs,
                           result_file_path,
                           chars_number=2):
    df = pd.DataFrame(
        columns=['Cluster ID', 'Sentence ID', 'Character1', 'Verb', 'Label-verb', 'Character2', 'Embedding', 'PCA-x',
                 'PCA-y',
                 'tSNE-x', 'tSNE-y', 'Phrase'])
    dataframe_i = 0
    pca_s, tsne_s = pca_tsne
    for cluster_id, sentences_id in all_cluster.items():
        for sentence_id in sentences_id:
            char1 = chars[sentence_id][0]
            if chars_number == 2:
                char2 = chars[sentence_id][1]
            elif chars_number == 1:
                char2 = None
            else:
                logging.info('Are you sure about the number of characters? If so, implement this part.')
                exit(-1)
            df.loc[dataframe_i] = [cluster_id, sentence_id, char1, verbs[sentence_id], label_verbs[cluster_id], char2,
                                   embedding_dict[sentence_id],
                                   pca_s[sentence_id][0], pca_s[sentence_id][1], tsne_s[sentence_id][0],
                                   tsne_s[sentence_id][1], sentences[sentence_id]]
            dataframe_i += 1
    df.to_pickle(result_file_path)


def embedding_verbs(book_filename, grouped_aliases, sentences_file, k):
    book = book_filename.split('.')
    if len(book) == 1:
        book = book_filename
    else:
        book = book[0]

    bert_cls = Bert_cluster(result_folder, book, grouped_aliases, sentences_file=sentences_file)
    bert_cls.extract_verbs()
    # bert_cls.remove_char_from_sentences()
    bert_cls.embedding()
    embed_to_dataframe(bert_cls.verbs, bert_cls.sentences_dictionary, bert_cls.removed_chars, bert_cls.embedding,
                       result_folder + book + '/' + book + '_embeddings' + str(bert_cls.chars_number) + '.pkl',
                       bert_cls.chars_number)
    return bert_cls


def embed_and_cluster(book_filename, grouped_aliases, sentences_file, k):
    book = book_filename.split('.')
    if len(book) == 1:
        book = book_filename
    else:
        book = book[0]
    bert_cls = embedding_verbs(book_filename, grouped_aliases, sentences_file, k)

    chars_number = bert_cls.chars_number
    clusters = bert_cls.kmeans(k)
    # bert_cls.silhouette_kmean()
    bert_cls.generate_triplets(clusters)
    label_verbs = bert_cls.label_verbs(clusters)
    bert_cls.generate_reports(clusters)

    pca_s = pca(list(bert_cls.embedding.values()), list(bert_cls.embedding.keys()))

    tsne_s = tsne(list(bert_cls.embedding.values()), list(bert_cls.embedding.keys()))
    embed_cluster_to_dataframe(bert_cls.verbs, bert_cls.sentences_dictionary, bert_cls.all_cluster,
                           bert_cls.removed_chars, bert_cls.embedding, (pca_s, tsne_s), label_verbs,
                           result_folder + book + '/' + book + '_relations' + str(chars_number) + '.pkl', chars_number)

    # plot_verbs_embedding(list(bert_cls.embedding.values()), k, list(bert_cls.verbs.values()), clusters)
    # (unique, counts) = np.unique(clusters, return_counts=True)
    # relations_counter = {}
    # for i in range(0, len(unique)):
    #     relations_counter[i] = {}
    #
    # for pair, relations in bert_cls.chars_relations.items():
    #     for relation in relations:
    #         if pair not in relations_counter[int(relation)]:
    #             relations_counter[int(relation)][pair] = 1
    #         else:
    #             relations_counter[int(relation)][pair] += 1
    #
    #
    # plot_grouped_relations(grouped_aliases, occurrences, relations_counter, result_folder, book, min_entity=100,
    #                       min_relation=1, k=len(unique))
    # return relations_counter
    return

def extract_relations(book_filename, aliases_occurrences, k=50):
    book = book_filename.split('.')
    if len(book) == 1:
        book = book_filename
    else:
        book = book[0]

    dealiased_folder = './../Data/clust&Dealias/' + book + '/'
    dealiased_book_path = dealiased_folder + book + '_out.txt'
    chars_dataframe_path = dealiased_folder + book + '_characters.pkl'
    aliases = list(aliases_occurrences['Names'])
    occurrences = list(aliases_occurrences['Occurrences'])
    grouped_aliases = group_aliases(aliases)
    chars_to_dataframe(grouped_aliases, occurrences, chars_dataframe_path)
    occurrences = [sum(values) for values in occurrences]
    alias_occurrences = {key: occurrences[i] for i, key in enumerate(grouped_aliases)}
    logging.info('Aliases list: %s', grouped_aliases)
    nlp = spacy.load("en_core_web_sm")
    all_sentences = extract_sentences(dealiased_book_path, nlp)
    all_sentences, no_verb_sentences = extract_sentences_with_verbs(all_sentences)
    # find information about characters
    info = find_information(all_sentences, nlp)
    # Find sentences and phrases and save them
    sentences_zero, sentences_few, sentences_right, sentences_more = extract_nchar_sentences(number_of_char=2,
                                                                                             sentences=all_sentences)
    constituency = ConstituencyTree(sentences_more['Original_token'], sentences_more['Lemmatize_token'])
    constituency.extract_sentences()
    zero, few, right, more = constituency.get_sentences()
    sentences_zero = pd.concat([sentences_zero, zero], ignore_index=True)
    sentences_few = pd.concat([sentences_few, few], ignore_index=True)
    sentences_right = pd.concat([sentences_right, right], ignore_index=True)
    sentences_more = pd.concat([more], ignore_index=True)

    sentences_right = filter_sentences(sentences_right)
    if not os.path.isdir(result_folder + book):
        os.makedirs(result_folder + book)
    sentences_few.to_pickle(result_folder + book + '/' + book + '_few_char_sentences.pkl')
    sentences_right.to_pickle(result_folder + book + '/' + book + '_right_char_sentences.pkl')
    sentences_more.to_pickle(result_folder + book + '/' + book + '_more_char_sentences.pkl')
    sentences_zero.to_pickle(result_folder + book + '/' + book + '_zero_char_sentences.pkl')

    # prepare_data_openke()
    return grouped_aliases
