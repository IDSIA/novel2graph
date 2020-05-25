# This script train the TWEC model with default parameters
import re
from chapter_splitter import Book
from gensim.models.word2vec import Word2Vec, LineSentence, PathLineSentences
from gensim import utils
import os
import numpy as np
import glob
import logging
import copy
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class TwecModel(object):
    def __init__(self, book_title, slice_type='chapters', size=100, siter=5, window=5, diter=5,
                 cache_folder='./../Data/embedding/', sg=0, ns=10, alpha=0.025, min_count=5, workers=2, delta=1000,
                 init_mode='hidden'):
        self.size = size
        self.sg = sg
        self.static_iter = siter
        self.dynamic_iter = diter
        self.negative = ns
        self.window = window
        self.static_alpha = alpha
        self.dynamic_alpha = alpha
        self.min_count = min_count
        self.workers = workers
        self.init_mode = init_mode
        self.compass = None
        self.delta = delta
        self.books_title = book_title
        self.slice_type = slice_type

        if not os.path.isdir(cache_folder):
            os.makedirs(cache_folder)

        models_folder = cache_folder + '/models'
        if not os.path.isdir(models_folder):
            os.makedirs(models_folder)
        book_models_folder = models_folder + '/' + self.books_title
        if not os.path.isdir(book_models_folder):
            os.makedirs(book_models_folder)

        self.models_folder = book_models_folder + '/' + self.slice_type
        if not os.path.isdir(self.models_folder):
            os.makedirs(self.models_folder)

        self.dynamic_folder = self.models_folder + '/dynamic'
        if not os.path.isdir(self.dynamic_folder):
            os.makedirs(self.dynamic_folder)

        self.static_folder = self.models_folder + '/static'
        if not os.path.isdir(self.static_folder):
            os.makedirs(self.static_folder)

        self.data = cache_folder + 'embeddings'
        if not os.path.isdir(self.data):
            os.makedirs(self.data)

        slices_folder = cache_folder + 'slices'
        if not os.path.isdir(slices_folder):
            os.makedirs(slices_folder)

        self.slices_folder = slices_folder + '/' + self.books_title
        if not os.path.isdir(self.slices_folder):
            os.makedirs(self.slices_folder)

        with open(os.path.join(self.models_folder, "log.txt"), "w") as f_log:
            f_log.write(slice_type + ' ' + str(size) + ' ' + str(siter) + ' ' + str(window) + ' ' + str(
                diter) + ' ' + cache_folder + str(sg) + ' ' + str(ns) + ' ' + str(alpha) + ' ' + str(
                min_count) + ' ' + str(
                workers) + ' ' + str(delta) + ' ' + init_mode)
            f_log.write('\n')
            logging.basicConfig(filename=os.path.realpath(f_log.name),
                                format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def initialize_from_compass(self, model):
        print("Initializing temporal embeddings from the atemporal compass.")
        if self.init_mode == "copy":
            model = copy.deepcopy(self.compass)
        else:
            vocab_m = model.wv.index2word
            indices = [self.compass.wv.vocab[w].index for w in vocab_m]
            new_syn1neg = np.array([self.compass.syn1neg[index] for index in indices])
            model.syn1neg = new_syn1neg
            if self.init_mode == "both":
                new_syn0 = np.array([self.compass.wv.syn0[index] for index in indices])
                model.wv.syn0 = new_syn0
        model.learn_hidden = False
        model.alpha = self.dynamic_alpha
        model.iter = self.dynamic_iter
        return model

    def train_model(self, slices):
        model = None
        if self.compass is None or self.init_mode != "copy":
            model = Word2Vec(sg=self.sg, size=self.size, alpha=self.static_alpha, iter=self.static_iter,
                             negative=self.negative,
                             window=self.window, min_count=self.min_count, workers=self.workers)
            model.build_vocab(slices, trim_rule=my_rule if self.compass is not None else None)
        if self.compass is not None:
            model = self.initialize_from_compass(model)
        model.train(slices, total_words=sum([len(s) for s in slices]), epochs=model.iter, compute_loss=True)
        return model

    # open or load model from all slices.txt toghether
    def train_static(self):
        if os.path.isfile(os.path.join(self.static_folder, "static.model")):
            self.compass = Word2Vec.load(os.path.join(self.static_folder, "static.model"))
            print("Static model loaded.")
        else:
            files = PathLineSentences(self.slices_folder + '/' + self.slice_type)
            files.input_files = [file for file in files.input_files if not os.path.basename(file).startswith('.')]
            print("Training static embeddings.")
            self.compass = self.train_model(files)
            self.compass.save(os.path.join(self.static_folder, "static.model"))
        global gvocab
        gvocab = self.compass.wv.vocab

    # Train sub-slices and save each in twec_model_sub, each model is trained on a specific slice of the book
    def train_temporal_embeddings_slices(self):
        if self.compass is None:
            self.train_static()

        files = glob.glob(self.slices_folder + '/' + self.slice_type + '/*.txt')
        tot_n_files = len(files)
        files.sort(key=lambda f: int(re.sub('\D', '', f)))
        for n_file, fn in enumerate(files):
            print("Training sliced temporal embeddings: slice {} of {}.".format(n_file + 1, tot_n_files))
            sentences = LineSentence(fn)
            model = self.train_model(sentences)
            model.save(self.dynamic_folder + '/' + str(n_file) + ".model")

        print('creation of subslices completed')

    #    get the vector of the input character from the given slices required
    def get_embeddings(self, aliases, models_folder):
        all_char_vec = [None] * len(aliases)
        for i, alias in enumerate(all_char_vec):
            all_char_vec[i] = []

        models_files = glob.glob(models_folder + '/*.model')
        for n_file, model in enumerate(sorted(models_files)):
            print('Processing model: ', n_file)
            word2vec = Word2Vec.load(model)
            for i, alias in enumerate(aliases):
                char_vec = all_char_vec[i]
                if alias in word2vec.wv.vocab:
                    embed = word2vec[alias]
                    char_vec.append(embed)
                else:
                    char_vec.append(None)

        self.all_char_vec = all_char_vec

    def evaluate(self):
        mfiles = glob.glob(self.opath + '/*.model')
        mods = []
        vocab_len = -1
        for fn in sorted(mfiles):
            if "static" in os.path.basename(fn): continue
            m = Word2Vec.load(fn)
            m.cbow_mean = True
            m.negative = self.negative
            m.window = self.window
            m.vector_size = self.size
            if vocab_len > 0 and vocab_len != len(m.wv.vocab):
                print(
                    "ERROR in evaluation: models with different vocab size {} != {}".format(vocab_len, len(m.wv.vocab)))
                return
            vocab_len = len(m.wv.vocab)
            mods.append(m)
        tfiles = glob.glob(self.test + '/*.txt')
        if len(mods) != len(tfiles):
            print(
                "ERROR in evaluation: number mismatch between the models ({}) in the folder {} and the test files ({}) in the folder {}".format(
                    len(mods), self.opath, len(tfiles), self.test))
            return
        mplps = []
        nlls = []
        for n_tfn, tfn in enumerate(sorted(tfiles)):
            sentences = LineSentence(tfn)
            # Taddy's code (see https://github.com/piskvorky/gensim/blob/develop/docs/notebooks/deepir.ipynb)
            llhd = np.array([m.score(sentences) for m in mods])  # (mods,sents)
            lhd = np.exp(llhd - llhd.max(axis=0))  # subtract row max to avoid numeric overload
            probs = (lhd / lhd.sum(axis=0)).mean(axis=1)  # (sents, mods)
            mplp = np.log(probs[n_tfn])
            mplps.append(mplp)

            nwords = len([w for s in sentences for w in s if w in mods[n_tfn].wv.vocab])
            nll = sum(llhd[n_tfn]) / (nwords)
            nlls.append(nll)
            print("Slice {} {}\n\t- Posterior log probability {:.4f}\n\tNormalized log likelihood {:.4f}".format(n_tfn,
                                                                                                                 tfn,
                                                                                                                 mplp,
                                                                                                                 nll))
        print
        print("Mean posterior log probability: {:.4f}".format(sum(mplps) / (len(mplps))))
        print("Mean normalized log likelihood: {:.4f}".format(sum(nlls) / (len(nlls))))


def plot_traj(data, number_of_character=5):
    char_names = data['a']
    n_chars = number_of_character
    vectors = data['b'][:]
    remove_None = []

    for character in vectors:
        new_character = []
        for points in character:
            if points is not None:
                new_character.append(points)
        remove_None.append(new_character)

    vectors = remove_None
    arr = []
    for character in vectors:
        for points in character:
            arr.append(points)

    np_arr = np.array(arr)
    # pca = PCA(n_components=2)
    # pca.fit(np_arr)
    # principalComponents = pca.fit_transform(np_arr)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    principalComponents = tsne.fit_transform(np_arr)

    xx = {}
    yy = {}
    for character in range(n_chars):
        xx[character] = []
        yy[character] = []

    k = 0
    for i, character in enumerate(vectors):
        n_points = len(character)
        for n_points in range(n_points):
            a = principalComponents[k]
            xx[i].append(a[0])
            yy[i].append(a[1])
            k += 1

    for k in range(n_chars):
        plt.plot(xx[k], yy[k], '-o', label=char_names[k], markevery=len(xx[k]))
        # plt.arrow(xx[k][-1], yy[k][-1], 5, 5, head_width=13, length_includes_head=True, shape='full')
        # print(char_names[k], ': ', xx[k], ' ', yy[k])

    plt.legend(loc=2, ncol=2)
    plt.show()


def plot_dist(models_path, aliases, names_to_plot):
    model_file = glob.glob(models_path + '/*.model')

    character_distances = [None] * len(aliases)
    for i, distance in enumerate(character_distances):
        character_distances[i] = []

    for n_file, fn in enumerate(model_file):
        word2vec = Word2Vec.load(fn)
        for i, alias in enumerate(aliases):
            character_distance = character_distances[i]
            if alias in word2vec.wv.vocab:
                # similarity compute the cosine distance!
                sim = 1 - word2vec.similarity(aliases[0], alias)
            else:
                sim = None
            character_distance.append(sim)

    plt.figure(figsize=(10, 10))
    years = list(range(1, len(character_distances[0]) + 1))
    for i, k in enumerate(character_distances):
        plt.plot(years, k, markersize=7, label=names_to_plot[i])
    leg = plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.25)

    plt.show()


def _get_vector_hp(model, character):
    char_vec = {}
    vec_list = []
    j = 0
    for root, dirs, files in os.walk(model):
        if len(files) > 0:
            j += 1
            char_vec1 = []
            model_file = glob.glob(root + '/*.model')
            for n_file, fn in enumerate(sorted(model_file)):
                t = Word2Vec.load(fn)
                if character in t.wv.vocab:
                    embed = t[character]
                    char_vec1.append(embed)
            if len(char_vec1) > 0:
                vec_list.append(char_vec1)

    return char_vec, vec_list


def my_rule(word, count, min_count):
    if word in gvocab:
        return utils.RULE_KEEP
    else:
        return utils.RULE_DISCARD


def split_chapter(text, slices_folder, chapter_per_slice=6):
    splitter = Book(text, nochapters=False, stats=False)
    chapters = splitter.writeChapters()
    # more_chapters is a list containing #chapters elements, each containing 20 chapters (1-20,2-21,...)
    more_chapters = []
    i = 0
    while True:
        more_chapters.append('\n'.join(chapters[i:i + chapter_per_slice]))
        if i + chapter_per_slice == len(chapters):
            break
        i += 1

    out_slices_path = slices_folder + '/chapters'
    if not os.path.isdir(out_slices_path):
        os.makedirs(out_slices_path)

    for i, tex in enumerate(more_chapters):
        with open(out_slices_path + '/%s.txt' % (i), 'w', encoding='utf8') as f:
            f.write(tex)


def group_aliases(aliases):
    character_ids = {}
    for i, alias in enumerate(aliases):
        lower_character_alias = [x.lower() for x in alias]
        char_id = 'CCHARACTER' + str(i)
        character_ids[char_id] = lower_character_alias
    return character_ids


def ns_type(x):
    x = int(x)
    if x < 1:
        raise argparse.ArgumentTypeError("Min negative sample number is 1.")
    return x


def static_dynamic_embed(book, clusters, dynamic, chapters):
    book = book.split('.')
    if len(book) == 1:
        book = book
    else:
        book = book[0]

    aliases, occurrences = read_alias_occurrences(clusters)
    dealiased_book = './../Data/clust&Dealias/' + book + '/' + book + '_out.txt'
    model = TwecModel(book)
    aliases_group = group_aliases(aliases)
    split_chapter(dealiased_book, model.slices_folder, chapter_per_slice=chapters)
    alias = list(aliases_group.keys())
    chars_label = [x[-1] for x in list(aliases_group.values())]

    model.train_static()
    model.train_temporal_embeddings_slices()

    if dynamic:
        model_path = model.dynamic_folder
        output_name = model.data + '/' + model.books_title + '_dynamic.npz'
    else:
        model_path = model.static_folder
        output_name = model.data + '/' + model.books_title + '_static.npz'

    model.get_embeddings(alias, model_path)
    np.savez(output_name, a=chars_label, b=model.all_char_vec)
    return output_name, model_path, alias


def read_alias_occurrences(data):
    aliases = []
    occurrences = []
    for aliases_occurrences in data.values():
        aliases.append(aliases_occurrences[0])
        occurrences.append(aliases_occurrences[1])
    return aliases, occurrences
