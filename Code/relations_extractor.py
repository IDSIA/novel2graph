from embedding import group_aliases, read_alias_occurrences
from graphviz import Digraph
from BERT_cluster_sentences import Bert_cluster
# parsetree only works with python <= 3.6!!
from pattern.en import parsetree
import numpy as np
import os
from nltk.corpus import stopwords
import string
# import opennre
import random
import re

if os.name == 'nt':
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


def plot_grouped_relations(clusters_labels, occurrences, relations_pair_occurrences, result_folder, book, k,
                           min_entity=15, min_relation=1):
    """Plot a graph (graphviz) in wich the dimension of each character is proportional to his occurrences in the
        novel, and the dimension of each relation is proportional to the number in which two characters are
        in the same sentence."""
    graph = Digraph('finite_state_machine', encoding='utf-8')
    graph.attr('node', shape='circle')
    graph.attr('node', fontname='Arial')
    graph.attr('node', fontsize=str(35))

    max_occurrences_index = np.argmax(occurrences)
    max_occurrence = occurrences[max_occurrences_index]
    i = 0
    show_char = []
    for key, value in clusters_labels.items():
        if occurrences[i] >= min_entity:
            label = value[0].title()
            graph.node(label, width=str(2 + (occurrences[i] / max_occurrence) * 5), label=label)
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
                    character1 = clusters_labels[pair[0]][0].title()
                    character2 = clusters_labels[pair[1]][0].title()
                    graph.edge(character1, character2, color=relations_color[relation_id],
                               penwidth=str(occurrence / best_occurrence * 20))
                    print(character1 + '\t' + str(relation_id) + '\t' + character2)
    graph.render(filename=result_folder + book + '/' + book)


def find_sentences(dealiased_book_path):
    """Given a text it extractes phrases and sentences."""
    dealiased_book = open(dealiased_book_path, "r", encoding='utf8')
    book = dealiased_book.read()
    dealiased_book.close()
    sss1 = parsetree(book, relations=True, lemmata=True)
    chunk_phrases = []
    sentenced_chunks = []
    sent_chunks = []
    chunks = []
    STOP = stopwords.words('english') + list(string.punctuation)
    for sentence in sss1:
        sentenced_chunks.append(sentence.chunks)
        stchk = []
        for chunk in sentence.chunks:
            chnks = (chunk.type, [(w.string, w.type) for w in chunk.words])
            ch_str = [w.string for w in chunk.words if len(w.string) > 2 and w.string not in STOP]
            chunk_phrases.append(' '.join(ch_str))
            chunks.append(chnks)
            stchk.append(chnks)
        sent_chunks.append(stchk)

    return chunk_phrases, sent_chunks


def parse_cooccurrences(phrases):
    """Given some phrases or sentences, it computes how many times two characters are in the same sentence. It returns a
    map in which (CHARACTER0, CHARACTER1) is the key and the value are the occurrences of this relation (found in
    text) and another map with the same keys but containing the list of phrases in which character are contained."""
    relations = {}
    relation_phrases = {}
    asymmetric_relations = []
    regexp = re.compile(r"CCHARACTER([0-9]+)\sAND\sCCHARACTER([0-9]+)")
    for phrase in phrases:
        count = phrase.count('CCHARACTER')
        if count < 2 or count > 2:
            continue
        if regexp.search(phrase):
            asymmetric_relations.append(phrase)
        characters = [word for word in phrase.split() if 'CCHARACTER' in word]
        if len(characters) != 2:
            print('---Something went wrong with this sentence:' + phrase)
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


def remove_tail_head(relations_phrases):
    all_relations = []
    for pair, relations in relations_phrases.items():
        # skip sentences in which head=tail
        if pair[0] != pair[1]:
            for relation in relations:
                all_relations.append(relation)
    all_relations.sort(key=len)
    return all_relations


def remove_included_sentences(all_relations):
    remove_double_sentence = []
    for sentence1 in all_relations:
        for sentence2 in all_relations:
            if sentence1 in sentence2 and sentence1 != sentence2:
                remove_double_sentence.append(sentence1)

    print('Removing useless sentences:')
    for to_remove in remove_double_sentence:
        if to_remove in all_relations:
            print(to_remove)
            all_relations.remove(to_remove)
    return all_relations


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


def find_phrases_or_sentences(dealiased_book_path, sentences_flag=True):
    phrases, sentences = find_sentences(dealiased_book_path)
    if sentences_flag:
        new_sentences = []
        for sentence_list in sentences:
            new_sentence = ""
            for sentence in sentence_list:
                for word_type in sentence[1]:
                    new_sentence += " " + word_type[0]
            new_sentences.append(new_sentence)
        return parse_cooccurrences(new_sentences)
    else:
        return parse_cooccurrences(phrases)


def find_relations(relations_phrases, model_type='wiki80_cnn_softmax'):
    relations_tagger = opennre.get_model(model_type)
    tags = {}
    for relation, phrases in relations_phrases.items():
        if '.' in relation[0] or '.' in relation[1]:
            print(relation, phrase)
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


def extract_relations(book_filename, clusters):
    book = book_filename.split('.')
    if len(book) == 1:
        book = book_filename
    else:
        book = book[0]

    aliases, occurrences = read_alias_occurrences(clusters)
    occurrences = [sum(values) for values in occurrences]
    grouped_aliases = group_aliases(aliases)
    dealiased_book_path = './../Data/clust&Dealias/' + book + '/' + book + '_out.txt'
    alias = list(grouped_aliases.keys())
    clusters_labels = [x[-1] for x in list(grouped_aliases.values())]
    print(grouped_aliases)
    relations, relations_phrases, asymmetric_relations = find_phrases_or_sentences(dealiased_book_path, True)

    all_relations = remove_tail_head(relations_phrases)
    all_relations = remove_included_sentences(all_relations)

    # prepare_data_openke()
    result_folder = './../Data/embedRelations/'
    bert_cls = Bert_cluster(all_relations, asymmetric_relations, result_folder, book,
                            grouped_aliases)
    bert_cls.remove_char_from_sentences()
    sentence_embeddings = bert_cls.embedding()
    k = 200
    clusters = bert_cls.kmeans(sentence_embeddings, k)
    bert_cls.generate_triplets(clusters)
    bert_cls.generate_reports(clusters)

    relations_counter = {}
    for i in range(0, k):
        relations_counter[i] = {}

    for pair, relations in bert_cls.chars_relations.items():
        for relation in relations:
            if pair not in relations_counter[int(relation)]:
                relations_counter[int(relation)][pair] = 1
            else:
                relations_counter[int(relation)][pair] += 1

    plot_grouped_relations(grouped_aliases, occurrences, relations_counter, result_folder, book, min_entity=100,
                           min_relation=1, k=k)

    return grouped_aliases
