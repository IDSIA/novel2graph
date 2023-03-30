#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:32:16 2022

@author: vanikanjirangat
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import torch
import scipy
import os
import matplotlib.pyplot as plt
import logging
from collections import Counter
import pandas as pd
from scipy.spatial import distance
import itertools
from nltk.corpus import wordnet as wn
import pickle


vst= pd.read_csv("../Verb_senses_streusle_test/.csv)

#vst=pd.read_csv("/content/gdrive/My Drive/Colab Notebooks/Narratives_Supersenses/streusle/Verb_senses_streusle_test.csv")

vst=vst[vst.supersense_tag!="B-DISC"]
verbs=vst.verb_lemma.values
verbs_dict={}
for i,verb in enumerate(verbs):
    verbs_dict[i]=verb
true_supersense=vst.supersense_tag.values

assert len(verbs)==len(true_supersense)

class Bert_cluster:
    def __init__(self, verbs,true_supersense):
        
        self.path_supersense='../my_semcor1.csv'
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbs=verbs
    def extract_supersenses(self):
        subset=pd.read_csv(self.path_supersense)
        cols = subset.columns
        subset1=subset.drop(columns=['supersense','count_in_semcor'])
        cols1=subset1.columns
        bt = subset1.apply(lambda x: x > 0)
        self.values=bt.apply(lambda x: list(cols1[x.values]), axis=1)
        self.keys=subset['supersense'].values
        r=list(zip(self.keys, self.values))
        self.dict_supersense={}
        for i,j in enumerate(r):
            self.dict_supersense[j[0]]=j[1]
            
        return self.dict_supersense
    
    def embedding(self):
        #verb embeddings
        
        embedding = self.model.encode(list(self.verbs.values()))
        
        
        id_embedding = {}
        i = 0
        for id, verb in self.verbs.items():
            
            id_embedding[id] = embedding[i]
            i += 1
##        print('id and verb',id,verb)
        self.embedding = id_embedding

    def verb_context_embedding(self):
        # considering the complete sentence as the context
        
        embedding = self.model.encode(list(self.verbs.values()))
        
        
        id_embedding = {}
        i = 0
        for id, verb in self.verbs.items():
            
            id_embedding[id] = embedding[i]
            i += 1
##        print('id and verb',id,verb)
        self.embedding = id_embedding

    def embedding_supersense(self):
        #individual embeddings of the supersenses for each sense
        #dictionary of dictionary, where main key is supersense and second level 
        #key is senses and values are their embeddings
        self.supersense_embeddings={}
        for key in self.dict_supersense:
            sense_values=self.dict_supersense[key]
            embedding = self.model.encode(list(sense_values))
            sense_embedding = {}
            i = 0
            for sense in sense_values:
                sense_embedding[sense] = embedding[i]
                i += 1
                
            self.supersense_embeddings[key]=sense_embedding
            
    def average_embedding_supersense(self,verb_list):
        #Average embedding of  a supersense category if word is ambiguous 
        
        self.supersense_embeddings_average={}
        embedding = self.model.encode(list(verb_list))
        average_embedding=np.array(embedding).mean(axis=0)
        return average_embedding
    
    def average_static_embedding_supersense(self):
        #Static Average embedding of a supersense category, if the word is completely absent 
        
        supersense_static_embeddings_average={}
        for key in self.dict_supersense:
            sense_values=self.dict_supersense[key]
            embedding = self.model.encode(list(sense_values))
            average_embedding=np.array(embedding).mean(axis=0)
            supersense_static_embeddings_average[key]=average_embedding
        return supersense_static_embeddings_average
        
            
        
    def relate_verb_supersense(self):
        #returns the dictionary with verbs related to supersense
        self.store_embeddings={}
        embed_values=list(self.embedding.values())
        self.verb_supersense_dict={}
        self.id_verb_supersense_dict={}
        self.verbid_superid={}
        self.super_list=[]
        self.supersense_ids={}
        print('verbs',self.verbs)
        print('no. of verbs detected',len(self.verbs))
        for ids,keys in enumerate(self.dict_supersense.keys()):
            self.supersense_ids[ids]=keys
        print('supersense',self.supersense_ids)
            
##        verb_set=self.verbs.values()
##        print('verb_count',len(verb_set))
##        verb_set=list(set(verb_set))
##        print('After removing dupicate verbs')
##        print(len(verb_set))
##        print('lets remove auxiliary verbs')
##        verb_set=list(filter(lambda a: a != 'be', verb_set))
##        print('Final verb_count',len(verb_set))
##        print('\n\n')
        '''
        Assign  an id to each supersense and do the clustering : Return dict (supersense_id:verb)
        In short for each verb, a cluster id should be assigned.
        For supersenses, we have 15 pre-defined clusters
        For the hypernyms, it can vary.
        Anyways, the final cluster id will be hypernym-related and the cluster label will be the hypernym (lch) itself.
        #supersense: (verb,verbid,supersenseid)
        '''
        supersense_path="./opt/anaconda3/envs/NLP/novel2graph_sample/supersenses/supersense_cluster_dict_semeval_test.pkl"
        if os.path.exists(supersense_path):
            print('supsersense path exists')
            with open("./opt/anaconda3/envs/NLP/novel2graph_sample/supersenses/supersense_cluster_dict_semeval_test.pkl","rb") as f:
                self.verb_supersense_dict=pickle.load(f) 
            
            # k=pd.read_pickle(supersense_path)
            # k=k.fillna(0)
            # dictionaryObject = k.to_dict('list')
            # for item in dictionaryObject:
            #     s=dictionaryObject[item]
            #     s1=[]
            #     for m in s:
            #         if m not in[0]:
            #             s1.append(m)
            #     self.verb_supersense_dict[item]=s1
    
        else:    
            unique_verb=0
            ambi_verb=0
            absent_verb=0
            for id,verb in self.verbs.items():
                
                f1=0
                ct=0
                
                for sid,key in enumerate(self.dict_supersense):
                    
                    if verb in self.dict_supersense[key]:
                        f1=1
                        ct+=1
                        keys=key
                        verbs=verb
                        sids=sid
                if ct==1:# checking thenumber of times verb appears in supersense dict
                    #print('Unique presence',verbs,keys)
                    unique_verb+=1
                    if keys not in self.verb_supersense_dict:
                        self.verb_supersense_dict[keys]=[(verbs,id,sids)]
                    # self.store_embeddings[self.verbs[id]]=self.embed_values[id]
                        
                        
                    else:
                        if verbs not in self.verb_supersense_dict[keys]:
                            self.verb_supersense_dict[keys].append((verbs,id,sids))
                            
                            
                    
    ##                self.id_verb_supersense_dict[sids]=verbs
                    
                elif ct>1:# This is ambiguous
                    #print('Presence Ambiguous_compute average dynamic embedding',verb)
                    ambi_verb+=1
                    #'when the verb is present in multiple supersense list
                    #we remove the verb from the list of supersense and compute the average embeddings 
                    #for other words and ccompute cosine distance between verb embedding and
                    # average sence embedding to select the closest supersense
                    min_dist=[]
                    v_embedding = np.array([list(embed_values[id])])
                    
                    
                    avg_embeds=[]
                    for sid,key in enumerate(self.dict_supersense):
                        verbs=self.dict_supersense[key]
                        if verb in self.dict_supersense[key]:
                            verbs1=verbs
                            verbs1.remove(verb)
                            avg_embedding=self.average_embedding_supersense(verbs1)
                        else:
                            avg_embedding=self.average_embedding_supersense(verbs)
                        avg_embeds.append(avg_embedding)
                    
                    avg_sense_embed=np.array(avg_embeds)
                    print(len(avg_embeds))
                    
                    dist=distance.cdist(avg_sense_embed, v_embedding, 'cosine')
                    
                    index_min = min(range(len(dist)), key=dist.__getitem__)
                    key=list(self.dict_supersense.keys())[index_min]
                    #print('Presence Ambiguous',key)
                    if key not in self.verb_supersense_dict:
                        self.verb_supersense_dict[key]=[(verb,id,index_min)]
                        
                    else:
                        if verb not in self.verb_supersense_dict[key]:
                            self.verb_supersense_dict[key].append((verb,id,index_min))
                            
    ##                self.id_verb_supersense_dict[index_min]=verbs
                    
                elif f1==0:#no verbs directly in the supersense list, then go for embeddings
                    absent_verb+=1
                    #print('Not present: compute average static embedding',verb)
                    min_dist=[]
                    
                    v_embedding = np.array([list(embed_values[id])])
                    embed=self.average_static_embedding_supersense()
                    avg_sense_embed=np.array(list(embed.values()))
    
                    
    ##                for key in self.supersense_embeddings:
    ##                    embed=self.supersense_embeddings[key]
    ##                    
    ##                    sense_embed=np.array(list(embed.values()))
    ##                    dist=distance.cdist(sense_embed, v_embedding, 'cosine')
    ##                    min_dist.append(min(dist))
    
                    dist=distance.cdist(avg_sense_embed, v_embedding, 'cosine')
                    index_min = min(range(len(dist)), key=dist.__getitem__)
                    key=list(self.dict_supersense.keys())[index_min]
                    #print('Not present',key)
                    if key not in self.verb_supersense_dict:
                        self.verb_supersense_dict[key]=[(verb,id,index_min)]
                        
                    else:
                        if verb not in self.verb_supersense_dict[key]:
                            self.verb_supersense_dict[key].append((verb,id,index_min))
                            
                                                              
##                self.id_verb_supersense_dict[index_min]=verbs
            #print('verb_supersense_dict',self.verb_supersense_dict)
            df1 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in self.verb_supersense_dict.items() ]))
            df1.to_pickle("./opt/anaconda3/envs/NLP/novel2graph_sample/supersenses/supersense_cluster_average_semevaltest.pkl")
            with open("./opt/anaconda3/envs/NLP/novel2graph_sample/supersenses/supersense_cluster_dict_semevaltest.pkl","wb") as f:
                pickle.dump(self.verb_supersense_dict,f)
        print("STATISTICS--unique--ambi--absent",unique_verb,ambi_verb,absent_verb)
        for items in self.verb_supersense_dict.keys():
                tup=self.verb_supersense_dict[items]
                for t in tup:
                    self.verbid_superid[t[1]]=t[2]
                    
        #print('verbid_superid',self.verbid_superid)
        #verbid mapped with supersense id
        self.verbid_superid=sorted(self.verbid_superid.items(),key=lambda x: x[0])
        #print('verbid')
        self.super_list=[x[1] for x in self.verbid_superid]
        #print('super_list',self.super_list)
        self.super_list=np.array(self.super_list)
        self.super_labels={}
        self.super_labels_id={}
        verbs = list(self.verbs.values())
        for i in range(0, 16):
            t= np.where(self.super_list== i)
            #print("The supersense id %s is associated with verb ids %s"%(i,t[0]))
            cluster_verbs = [verbs[index] for index in t[0]]
            self.super_labels[i] = cluster_verbs
            self.super_labels_id[i]=t[0]
        
        print('We have the verb clusters related to each supersense id', self.super_labels)    
            
# =============================================================================
#             for i in range(0, len(Counter(pp).keys())):
#                 involved_triplet = np.where(np.array(pp) == 13)
# =============================================================================
            #SOrt the dict based on verb ids and get the list of superids
            #We will have all the relation between supersense and corresponding verbs
            #Once we have this , we can group the verbs of same supersense id
            #Then we have to go for labelling based on lch
        l={}
        for m in self.super_labels.keys():
            for v in k[m]:
                if m not in l.keys():
                    l[m]=[v[0]]
                else:
                    l[m].append(v[0])
                    
                    
        z=[]
        for m in k.keys():
            for v in k[m]:
                z.append((v[0],m))
        print('Supersense tags', z)
            
        return self.verb_supersense_dict,self.super_labels,z

bert_cls = Bert_cluster(verbs_dict,true_supersense)
supersense=bert_cls.extract_supersenses()
emb=bert_cls.embedding()
print('finding clusters based on supersense')

embedding_supersense=bert_cls.embedding_supersense()
supersense_clusters_id,supersense_clusters_name,z=bert_cls.relate_verb_supersense()



#k={'verb.communication': [('cont', 0, 3), ('suggests', 10, 3), ('called', 55, 3), ('tell', 62, 3), ('asks', 76, 3), ('miss', 93, 3), ('told', 147, 3), ('call', 148, 3), ('complains', 158, 3), ('speak', 215, 3), ('Asked', 235, 3), ('said', 262, 3), ('tryna', 272, 3), ('say', 306, 3), ('miss', 334, 3), ('said', 360, 3), ('call', 371, 3), ('fakin', 372, 3), ('blessed', 379, 3), ('said', 386, 3), ('cum', 390, 3), ('saying', 397, 3), ('bring', 400, 3), ('talking', 435, 3), ('forgave', 458, 3), ('told', 464, 3), ('teaching', 498, 3), ('cont', 504, 3), ('say', 516, 3), ('text', 532, 3), ('wrote', 556, 3), ('cussin', 557, 3), ('BEG', 571, 3), ('bring', 608, 3), ('asked', 613, 3), ('called', 664, 3), ('asked', 668, 3), ('call', 670, 3), ('remind', 693, 3), ('saying', 700, 3), ('criticize', 719, 3), ('talking', 730, 3), ('Told', 734, 3), ('r', 752, 3), ('whispering', 766, 3), ('tryna', 774, 3), ('dnt', 777, 3), ('talk', 778, 3), ('tell', 792, 3), ('responded', 794, 3), ('call', 804, 3), ('persist', 805, 3), ('talking', 833, 3), ('said', 836, 3), ('requesting', 849, 3), ('miss', 862, 3), ('call', 868, 3), ('said', 873, 3), ('wyd', 881, 3), ('brings', 885, 3), ('miss', 927, 3), ('thank', 948, 3), ('sign', 965, 3), ('investigating', 986, 3), ('tell', 993, 3), ('say', 994, 3), ('say', 996, 3), ('pressed', 1011, 3), ('say', 1015, 3), ('say', 1025, 3), ('post', 1033, 3), ('say', 1049, 3), ('ask', 1073, 3), ('cum', 1076, 3), ('chasin', 1079, 3), ('makin', 1080, 3), ('wrote', 1087, 3), ('miss', 1098, 3), ('miss', 1145, 3), ('say', 1160, 3), ('shoot', 1174, 3), ('informed', 1178, 3), ('qualifying', 1193, 3), ('bring', 1225, 3), ('miss', 1254, 3), ('Saying', 1260, 3), ('thank', 1286, 3), ('said', 1293, 3), ('call', 1339, 3), ('miss', 1342, 3), ('Say', 1353, 3)], 'verb.consumption': [('compliment', 1, 5), ('relaxin', 5, 5), ('TALKING', 16, 5), ('Lettin', 58, 5), ('reads', 72, 5), ('ThAnK', 112, 5), ('eaten', 157, 5), ('reading', 179, 5), ('helping', 189, 5), ('Eating', 208, 5), ('Eat', 234, 5), ('eaten', 333, 5), ('entertained', 344, 5), ('enjoy', 358, 5), ('Thank', 417, 5), ('read', 445, 5), ('fill', 461, 5), ('sitting', 480, 5), ('sitting', 534, 5), ('READING', 568, 5), ('stay', 605, 5), ('eating', 610, 5), ('stay', 614, 5), ('read', 623, 5), ('relax', 626, 5), ('paid', 636, 5), ('Eats', 694, 5), ('drinking', 735, 5), ('support', 813, 5), ('suck', 815, 5), ('cooked', 835, 5), ('stay', 837, 5), ('eatin', 842, 5), ('cookin', 843, 5), ('care', 878, 5), ('cooking', 888, 5), ('stayin', 952, 5), ('sharing', 957, 5), ('party', 962, 5), ('party', 963, 5), ('party', 964, 5), ('stay', 970, 5), ('stayin', 971, 5), ('use', 976, 5), ('sitting', 1007, 5), ('sharing', 1017, 5), ('appreciate', 1115, 5), ('Drink', 1130, 5), ('clean', 1165, 5), ('pet', 1177, 5), ('drinking', 1192, 5), ('Enjoy', 1220, 5), ('used', 1227, 5), ('Honored', 1257, 5), ('party', 1268, 5)], 'verb.cognition': [('believe', 2, 2), ('doin', 3, 2), ('do', 8, 2), ('GUESS', 15, 2), ('skip', 22, 2), ('think', 23, 2), ('is', 27, 2), ('Check', 28, 2), ('knows', 33, 2), ('confirmed', 36, 2), ('Is', 38, 2), ('do', 39, 2), ('hope', 40, 2), ('realized', 45, 2), ('KNOW', 49, 2), ('did', 51, 2), ('was', 53, 2), ('is', 54, 2), ('did', 56, 2), ('do', 57, 2), ('was', 61, 2), ('know', 64, 2), ('do', 67, 2), ('let', 70, 2), ('get', 74, 2), ('is', 75, 2), ('needs', 78, 2), ('fixed', 79, 2), ('given', 80, 2), ('wants', 81, 2), ('do', 83, 2), ('forget', 84, 2), ('see', 86, 2), ('can', 87, 2), ('was', 94, 2), ('thinking', 95, 2), ('were', 96, 2), ('got', 97, 2), ('get', 99, 2), ('get', 100, 2), ('do', 101, 2), ('get', 102, 2), ('guess', 104, 2), ('believe', 107, 2), ('are', 109, 2), ('think', 110, 2), ('checks', 114, 2), ('think', 115, 2), ('let', 118, 2), ('can', 127, 2), ('get', 128, 2), ('Hope', 129, 2), ('Got', 133, 2), ('agree', 135, 2), ('is', 136, 2), ('do', 140, 2), ('got', 141, 2), ('was', 144, 2), ('got', 146, 2), ('gets', 149, 2), ('is', 150, 2), ('check', 152, 2), ('was', 153, 2), ('meeting', 154, 2), ('do', 160, 2), ('responds', 161, 2), ('do', 166, 2), ('know', 167, 2), ('wants', 169, 2), ('is', 173, 2), ('is', 174, 2), ('do', 181, 2), ('feeling', 185, 2), ('getting', 194, 2), ('thought', 196, 2), ('Get', 199, 2), ('know', 201, 2), ('are', 202, 2), ('is', 203, 2), ('waiting', 207, 2), ('are', 211, 2), ('is', 212, 2), ('hope', 229, 2), ('can', 230, 2), ('get', 231, 2), ('get', 233, 2), ('know', 239, 2), ('is', 242, 2), ('knows', 243, 2), ('get', 247, 2), ('get', 249, 2), ('reminded', 252, 2), ('gets', 257, 2), ('Contact', 259, 2), ('got', 260, 2), ('were', 263, 2), ('get', 270, 2), ('know', 271, 2), ('thinks', 276, 2), ('feels', 278, 2), ('get', 284, 2), ('Do', 287, 2), ('is', 289, 2), ('done', 291, 2), ('is', 292, 2), ('done', 293, 2), ('do', 295, 2), ('got', 299, 2), ('get', 302, 2), ('do', 307, 2), ('judge', 308, 2), ('get', 310, 2), ('waiting', 315, 2), ('is', 316, 2), ('Check', 318, 2), ('is', 321, 2), ('had', 322, 2), ('got', 325, 2), ('are', 331, 2), ('is', 340, 2), ('was', 341, 2), ('see', 345, 2), ('is', 352, 2), ('is', 355, 2), ('do', 362, 2), ('think', 363, 2), ('meet', 364, 2), ('do', 382, 2), ('is', 384, 2), ('get', 387, 2), ('know', 395, 2), ('feel', 401, 2), ('let', 404, 2), ('is', 407, 2), ('was', 409, 2), ('tried', 410, 2), ('do', 412, 2), ('getting', 416, 2), ('Check', 418, 2), ('are', 422, 2), ('Knowing', 425, 2), ('is', 431, 2), ('was', 434, 2), ('know', 439, 2), ('can', 441, 2), ('classified', 446, 2), ('hope', 447, 2), ('is', 450, 2), ('Getting', 452, 2), ('was', 453, 2), ('is', 459, 2), ('Take', 460, 2), ('Take', 463, 2), ('get', 466, 2), ('got', 467, 2), ('try', 468, 2), ('get', 472, 2), ('get', 476, 2), ('do', 479, 2), ('is', 482, 2), ('learned', 484, 2), ('doing', 485, 2), ('is', 491, 2), ('is', 492, 2), ('feeling', 494, 2), ('got', 496, 2), ('Try', 497, 2), ('get', 501, 2), ('get', 502, 2), ('Think', 503, 2), ('is', 505, 2), ('is', 508, 2), ('got', 511, 2), ('is', 512, 2), ('doing', 514, 2), ('can', 515, 2), ('can', 518, 2), ('is', 520, 2), ('check', 523, 2), ('see', 524, 2), ('LISTEN', 526, 2), ('are', 536, 2), ('are', 542, 2), ('is', 546, 2), ('is', 549, 2), ('get', 553, 2), ('get', 564, 2), ('IS', 566, 2), ('forget', 569, 2), ('is', 570, 2), ('waiting', 576, 2), ('think', 577, 2), ('know', 579, 2), ('get', 582, 2), ('was', 584, 2), ('see', 590, 2), ('getting', 591, 2), ('is', 592, 2), ('hope', 593, 2), ('is', 594, 2), ('are', 596, 2), ('is', 597, 2), ('Do', 598, 2), ('doing', 607, 2), ('done', 609, 2), ('cleared', 612, 2), ('can', 621, 2), ('was', 624, 2), ('get', 625, 2), ('realised', 627, 2), ('believe', 628, 2), ('picked', 629, 2), ('think', 639, 2), ('are', 640, 2), ('are', 644, 2), ('feelin', 647, 2), ('think', 650, 2), ('knew', 652, 2), ('can', 657, 2), ('was', 660, 2), ('found', 665, 2), ('does', 666, 2), ('Take', 672, 2), ('are', 679, 2), ('is', 682, 2), ('is', 685, 2), ('is', 688, 2), ('Got', 690, 2), ('got', 697, 2), ('meet', 702, 2), ('got', 704, 2), ('feel', 713, 2), ('had', 716, 2), ('had', 717, 2), ('had', 718, 2), ('Are', 721, 2), ('is', 722, 2), ('was', 725, 2), ('gets', 726, 2), ('is', 728, 2), ('is', 729, 2), ('got', 733, 2), ('can', 738, 2), ('get', 739, 2), ('forget', 740, 2), ('know', 741, 2), ('get', 742, 2), ('is', 745, 2), ('is', 746, 2), ('got', 750, 2), ('Try', 751, 2), ('do', 753, 2), ('See', 755, 2), ('get', 763, 2), ('is', 765, 2), ('had', 768, 2), ('is', 786, 2), ('doing', 788, 2), ('see', 791, 2), ('known', 795, 2), ('did', 797, 2), ('get', 798, 2), ('trust', 800, 2), ('was', 809, 2), ('hope', 810, 2), ('let', 816, 2), ('know', 817, 2), ('know', 819, 2), ('get', 823, 2), ('named', 827, 2), ('Find', 828, 2), ('done', 839, 2), ('reminds', 846, 2), ('Are', 850, 2), ('is', 854, 2), ('Get', 857, 2), ('meant', 863, 2), ('was', 864, 2), ('are', 865, 2), ('can', 867, 2), ('Is', 869, 2), ('qualified', 872, 2), ('got', 876, 2), ('had', 877, 2), ('do', 879, 2), ('is', 882, 2), ('is', 883, 2), ('is', 884, 2), ('are', 887, 2), ('see', 890, 2), ('Lets', 891, 2), ('get', 892, 2), ('gives', 898, 2), ('were', 899, 2), ('considering', 901, 2), ('was', 903, 2), ('tried', 904, 2), ('can', 911, 2), ('get', 914, 2), ('are', 915, 2), ('feeling', 924, 2), ('see', 925, 2), ('Do', 926, 2), ('tex', 930, 2), ('get', 931, 2), ('Think', 934, 2), ('is', 941, 2), ('are', 944, 2), ('meet', 950, 2), ('see', 955, 2), ('see', 956, 2), ('see', 958, 2), ('forgotten', 960, 2), ('did', 966, 2), ('got', 967, 2), ('can', 969, 2), ('let', 975, 2), ('thinks', 978, 2), ('was', 979, 2), ('was', 980, 2), ('was', 981, 2), ('get', 982, 2), ('is', 987, 2), ('remembered', 991, 2), ('see', 1000, 2), ('is', 1001, 2), ('get', 1002, 2), ('is', 1003, 2), ('named', 1012, 2), ('was', 1016, 2), ('are', 1018, 2), ('are', 1019, 2), ('is', 1020, 2), ('see', 1022, 2), ('plotting', 1023, 2), ('iis', 1024, 2), ('got', 1027, 2), ('had', 1035, 2), ('Is', 1040, 2), ('does', 1041, 2), ('remember', 1044, 2), ('was', 1045, 2), ('was', 1046, 2), ('was', 1047, 2), ('was', 1048, 2), ('is', 1050, 2), ('is', 1053, 2), ('supposed', 1061, 2), ('is', 1064, 2), ('take', 1065, 2), ('DO', 1067, 2), ('are', 1070, 2), ('was', 1072, 2), ('is', 1077, 2), ('Did', 1083, 2), ('IS', 1085, 2), ('remember', 1092, 2), ('were', 1093, 2), ('know', 1094, 2), ('see', 1097, 2), ('are', 1117, 2), ('were', 1118, 2), ('hope', 1122, 2), ('check', 1124, 2), ('do', 1125, 2), ('are', 1127, 2), ('had', 1132, 2), ('was', 1133, 2), ('think', 1141, 2), ('Hope', 1147, 2), ('supposed', 1151, 2), ('see', 1154, 2), ('is', 1156, 2), ('was', 1158, 2), ('do', 1163, 2), ('was', 1171, 2), ('is', 1180, 2), ('are', 1182, 2), ('are', 1184, 2), ('is', 1188, 2), ('Find', 1189, 2), ('is', 1198, 2), ('got', 1201, 2), ('was', 1206, 2), ('is', 1207, 2), ('tried', 1208, 2), ('were', 1211, 2), ('Explains', 1213, 2), ('thinks', 1214, 2), ('hope', 1217, 2), ('Thinking', 1222, 2), ('is', 1223, 2), ('is', 1226, 2), ('Think', 1232, 2), ('did', 1237, 2), ('think', 1239, 2), ('get', 1244, 2), ('Review', 1247, 2), ('hope', 1255, 2), ('was', 1259, 2), ('is', 1261, 2), ('know', 1262, 2), ('know', 1264, 2), ('are', 1269, 2), ('got', 1270, 2), ('order', 1276, 2), ('feel', 1277, 2), ('got', 1280, 2), ('Try', 1281, 2), ('get', 1285, 2), ('get', 1291, 2), ('think', 1294, 2), ('let', 1296, 2), ('find', 1305, 2), ('know', 1319, 2), ('let', 1321, 2), ('know', 1322, 2), ('IS', 1325, 2), ('doing', 1329, 2), ('had', 1335, 2), ('was', 1336, 2), ('intended)', 1337, 2), ('is', 1347, 2), ('Get', 1355, 2), ('Is', 1356, 2), ('feels', 1360, 2)], 'verb.contact': [('maxin', 4, 6), ('Retweet', 21, 6), ('tweet', 69, 6), ('retweeted', 253, 6), ('slapped', 261, 6), ('peels', 309, 6), ('pulling', 330, 6), ('lick', 356, 6), ('kiss', 361, 6), ('crash', 394, 6), ('tamped', 399, 6), ('Cut', 403, 6), ('f*ck', 599, 6), ('tweet', 658, 6), ('tapping', 684, 6), ('luv', 724, 6), ('spilling', 840, 6), ('tweet', 871, 6), ('slapped', 902, 6), ('Retweet', 1056, 6), ('Luv', 1106, 6), ('killing', 1187, 6), ('pull', 1236, 6), ('bites', 1251, 6), ('lick', 1297, 6), ('lock', 1351, 6), ('CRANK', 1357, 6)], 'verb.stative': [('have', 6, 13), ('HAVE', 14, 13), ("'s", 24, 13), ("'m", 25, 13), ('be', 26, 13), ("'re", 29, 13), ('went', 44, 13), ('have', 46, 13), ("'m", 60, 13), ('be', 71, 13), ("'s", 85, 13), ('Came', 92, 13), ("'m", 98, 13), ("'s", 103, 13), ("'m", 105, 13), ("'s", 106, 13), ('going', 108, 13), ("'s", 111, 13), ('will', 113, 13), ('means', 122, 13), ("'m", 123, 13), ('like', 125, 13), ('like', 126, 13), ("'re", 130, 13), ('having', 131, 13), ('Have', 132, 13), ("'m", 137, 13), ('Hanging', 151, 13), ("'s", 165, 13), ('have', 172, 13), ('has', 176, 13), ('heart', 178, 13), ('ends', 182, 13), ("'re", 183, 13), ("'s", 184, 13), ("'s", 186, 13), ('going', 187, 13), ('be', 197, 13), ('be', 206, 13), ("'s", 210, 13), ('have', 213, 13), ('stick', 217, 13), ('am', 219, 13), ('will', 223, 13), ('be', 224, 13), ("'ve", 225, 13), ('been', 226, 13), ("'s", 238, 13), ('means', 240, 13), ('survive', 241, 13), ('have', 246, 13), ('have', 254, 13), ('have', 256, 13), ('went', 265, 13), ("'s", 266, 13), ("'s", 267, 13), ("'s", 268, 13), ("'s", 274, 13), ('been', 275, 13), ("'m", 282, 13), ('going', 285, 13), ("'s", 290, 13), ("'s", 298, 13), ('has', 319, 13), ('been', 320, 13), ('will', 327, 13), ('being', 335, 13), ('having', 338, 13), ('Has', 342, 13), ('been', 343, 13), ('Headed', 347, 13), ('be', 353, 13), ('be', 366, 13), ('ai', 368, 13), ("'s", 370, 13), ('will', 375, 13), ("'s", 380, 13), ('stay', 388, 13), ("'s", 389, 13), ("'s", 391, 13), ('be', 392, 13), ("'s", 396, 13), ('Keep', 398, 13), ('going', 413, 13), ('been', 414, 13), ('keeps', 415, 13), ('features', 420, 13), ('means', 421, 13), ('CHILLIN', 424, 13), ('have', 426, 13), ('been', 432, 13), ('going', 437, 13), ('be', 438, 13), ("'s", 440, 13), ('be', 451, 13), ('be', 455, 13), ("'m", 470, 13), ('be', 473, 13), ('be', 474, 13), ('covered', 483, 13), ("'s", 488, 13), ('mark', 489, 13), ('marked', 490, 13), ('be', 493, 13), ('been', 510, 13), ("'s", 521, 13), ('Be', 522, 13), ('be', 531, 13), ('ai', 535, 13), ("'m", 537, 13), ('Have', 538, 13), ('Be', 540, 13), ("'s", 545, 13), ('have', 554, 13), ('be', 555, 13), ('been', 583, 13), ('will', 586, 13), ('be', 587, 13), ('be', 600, 13), ('been', 603, 13), ("'m", 606, 13), ('been', 616, 13), ("'m", 617, 13), ("'s", 638, 13), ("'m", 645, 13), ("'m", 646, 13), ("'m", 656, 13), ("'m", 675, 13), ('be', 676, 13), ('extends', 689, 13), ("'m", 695, 13), ("'m", 699, 13), ("'m", 701, 13), ("'s", 703, 13), ('will', 706, 13), ('be', 707, 13), ("'s", 708, 13), ("'s", 709, 13), ('have', 711, 13), ('have', 712, 13), ('be', 715, 13), ('Going', 720, 13), ("'m", 723, 13), ("'s", 727, 13), ('was/is', 731, 13), ('going', 732, 13), ('air', 748, 13), ('be', 754, 13), ('Won', 756, 13), ('have', 760, 13), ('hold', 764, 13), ('have', 767, 13), ('am', 769, 13), ('hold', 770, 13), ("'m", 773, 13), ('be', 775, 13), ("'m", 776, 13), ('Being', 780, 13), ('being', 784, 13), ('have', 789, 13), ('be', 799, 13), ("'s", 803, 13), ('be', 806, 13), ('have', 814, 13), ('havin', 818, 13), ('Going', 824, 13), ('ends', 830, 13), ('Will', 844, 13), ("'m", 848, 13), ('be', 851, 13), ('Be', 858, 13), ('have', 875, 13), ('been', 896, 13), ('have', 897, 13), ("'m", 905, 13), ('having', 918, 13), ("'ve", 920, 13), ("'re", 928, 13), ('having', 929, 13), ('Have', 933, 13), ('have', 936, 13), ("'m", 943, 13), ('be', 945, 13), ('ft', 947, 13), ('have', 951, 13), ('going', 954, 13), ('be', 959, 13), ('cn', 972, 13), ('BE', 990, 13), ('c', 992, 13), ('has', 997, 13), ("'m", 999, 13), ('proving', 1004, 13), ('be', 1005, 13), ('getting', 1010, 13), ('have', 1013, 13), ('going', 1014, 13), ("'m", 1028, 13), ('going', 1029, 13), ('following', 1031, 13), ("'m", 1038, 13), ('be', 1039, 13), ('brought', 1052, 13), ("'re", 1057, 13), ('being', 1058, 13), ("'re", 1060, 13), ('be', 1062, 13), ('GON', 1066, 13), ('getting', 1071, 13), ('am', 1084, 13), ('will', 1090, 13), ("'re", 1102, 13), ("'re", 1103, 13), ("'m", 1110, 13), ('Going', 1113, 13), ('Be', 1114, 13), ('works', 1123, 13), ('going', 1139, 13), ('Like', 1140, 13), ('been', 1142, 13), ('Have', 1146, 13), ("'s", 1150, 13), ("'s", 1155, 13), ('have', 1159, 13), ('have', 1164, 13), ('Headed', 1173, 13), ('hemmed', 1176, 13), ("'s", 1179, 13), ("'re", 1183, 13), ("'s", 1185, 13), ("'re", 1186, 13), ("'s", 1196, 13), ('been', 1197, 13), ('deserve', 1215, 13), ('will', 1216, 13), ('have', 1218, 13), ('b', 1221, 13), ('be', 1228, 13), ("'s", 1229, 13), ('be', 1230, 13), ("'s", 1231, 13), ("'m", 1233, 13), ('going', 1234, 13), ("'s", 1240, 13), ('will', 1241, 13), ('Going', 1243, 13), ("'s", 1246, 13), ('turns', 1250, 13), ('turns', 1253, 13), ('have', 1256, 13), ("'re", 1265, 13), ('getting', 1266, 13), ('has', 1271, 13), ("'ve", 1272, 13), ('LANDED', 1282, 13), ("'ve", 1288, 13), ('been', 1289, 13), ('have', 1292, 13), ('stick', 1298, 13), ('have', 1299, 13), ('will', 1300, 13), ('be', 1301, 13), ('have', 1307, 13), ("'s", 1309, 13), ('will', 1311, 13), ('be', 1312, 13), ('will', 1314, 13), ('be', 1315, 13), ('went', 1316, 13), ('went', 1317, 13), ('will', 1320, 13), ('like', 1326, 13), ('have', 1328, 13), ('being', 1333, 13), ('be', 1344, 13), ('WILL', 1345, 13), ("'s", 1350, 13), ('Have', 1359, 13), ('won', 1361, 13)], 'verb.creation': [('startin', 7, 7), ('FEATURED', 30, 7), ('Make', 31, 7), ('posted', 35, 7), ('comes', 37, 7), ('invited', 47, 7), ('INVITE', 50, 7), ('played', 52, 7), ('made', 65, 7), ('discovered', 73, 7), ('declare', 77, 7), ('follow', 90, 7), ('wasz', 91, 7), ('put', 119, 7), ('work', 120, 7), ('come', 121, 7), ('made', 143, 7), ('thinkk', 145, 7), ('set', 156, 7), ('respond', 159, 7), ('Put', 198, 7), ('iguess', 200, 7), ('appears', 205, 7), ('work', 216, 7), ('building', 227, 7), ('Enlighten', 236, 7), ('make', 248, 7), ('meet', 258, 7), ('joining', 264, 7), ('sync', 279, 7), ('draw', 281, 7), ('put', 286, 7), ('Played', 300, 7), ('design', 304, 7), ('start', 317, 7), ('write', 323, 7), ('work', 328, 7), ('approaching', 332, 7), ('come', 348, 7), ('come', 349, 7), ('3dreaming', 365, 7), ('gettin', 367, 7), ('showin', 369, 7), ('makes', 429, 7), ('write', 442, 7), ('Working', 477, 7), ('put', 481, 7), ('dictate', 486, 7), ('receive', 487, 7), ('Come', 499, 7), ('comes', 507, 7), ('ride', 541, 7), ('working', 543, 7), ('make', 547, 7), ('play', 560, 7), ('join', 561, 7), ('Come', 573, 7), ('responding', 601, 7), ('Make', 602, 7), ('help', 611, 7), ('started', 630, 7), ('writing', 631, 7), ('sketching', 632, 7), ('beginning', 633, 7), ('starting', 641, 7), ('gettin', 648, 7), ('gettin', 651, 7), ('plant', 654, 7), ('Come', 655, 7), ('come', 667, 7), ('Entered', 677, 7), ('published', 678, 7), ('beginning', 680, 7), ('join', 686, 7), ('follow', 687, 7), ('picking', 698, 7), ('comes', 705, 7), ('make', 743, 7), ('getn', 772, 7), ('Write', 779, 7), ('write', 783, 7), ('Start', 787, 7), ('follow', 811, 7), ('show', 829, 7), ('Rise', 845, 7), ('playing', 852, 7), ('host', 861, 7), ('coming', 866, 7), ('leading', 870, 7), ('written', 893, 7), ('identified', 912, 7), ('write', 917, 7), ('writing', 919, 7), ('scheduled', 921, 7), ('make', 935, 7), ('causes', 946, 7), ('fitted', 989, 7), ('setup', 1009, 7), ('coming', 1021, 7), ('started', 1030, 7), ('start', 1042, 7), ('play', 1043, 7), ('Opens', 1055, 7), ('studying/doing', 1063, 7), ('tel', 1075, 7), ('start', 1086, 7), ('make', 1091, 7), ('act', 1095, 7), ('live', 1101, 7), ('come', 1105, 7), ('Come', 1111, 7), ('Make', 1116, 7), ('entering', 1120, 7), ('shows', 1137, 7), ('makes', 1144, 7), ('start', 1149, 7), ('bringing', 1161, 7), ('makes', 1162, 7), ('COMES', 1166, 7), ('Pick', 1168, 7), ('plans', 1195, 7), ('coming', 1199, 7), ('remake', 1203, 7), ('show', 1204, 7), ('Started', 1212, 7), ('live', 1219, 7), ('join', 1242, 7), ('started', 1245, 7), ('happens', 1249, 7), ('stand', 1252, 7), ('dress', 1267, 7), ('help', 1275, 7), ('working', 1278, 7), ('count', 1284, 7), ('configuring', 1290, 7), ('opened', 1302, 7), ('coming', 1310, 7), ("#ff'ed", 1349, 7), ('workz', 1352, 7), ('Play', 1362, 7)], 'verb.perception': [('seems', 9, 10), ('watched', 13, 10), ('ring', 68, 10), ('sounds', 168, 10), ('tweeting', 177, 10), ('saw', 195, 10), ('tweeting', 220, 10), ('Looking', 250, 10), ('saw', 251, 10), ('scan', 283, 10), ('watch', 406, 10), ('Listening', 423, 10), ('seen', 433, 10), ('listens', 449, 10), ('seen', 456, 10), ('listen', 509, 10), ('wait', 525, 10), ('LISTENING', 527, 10), ('saw', 528, 10), ('watched', 580, 10), ('look', 634, 10), ('Saw', 659, 10), ('seen', 663, 10), ('Close', 674, 10), ('look', 681, 10), ('watching', 736, 10), ('noting', 808, 10), ('wait', 821, 10), ('looks', 886, 10), ('smells', 889, 10), ('looking', 973, 10), ('seems', 1034, 10), ('noticed', 1037, 10), ('seen', 1109, 10), ('hear', 1131, 10), ('saw', 1157, 10), ('watching', 1170, 10), ('saw', 1190, 10), ('watch', 1209, 10), ('tweeted', 1238, 10), ('watch', 1248, 10), ('looked', 1263, 10), ('wait', 1287, 10), ('seems', 1330, 10), ('watch', 1338, 10)], 'verb.emotion': [('Thrilled', 11, 8), ('love', 34, 8), ('HATE', 48, 8), ('failed', 117, 8), ('need', 124, 8), ('Need', 139, 8), ('refuses', 163, 8), ('hates', 171, 8), ('need', 232, 8), ('need', 280, 8), ('waste', 288, 8), ('regret', 296, 8), ('impressed', 301, 8), ('hate', 305, 8), ('want', 311, 8), ('need', 313, 8), ('Chilling', 324, 8), ('liked', 329, 8), ('LOVE', 357, 8), ('need', 373, 8), ('want', 374, 8), ('fucked', 378, 8), ('hate', 381, 8), ('love', 427, 8), ('love', 428, 8), ('lose', 443, 8), ('mean', 448, 8), ('worry', 457, 8), ('love', 462, 8), ('missing', 471, 8), ('hate', 513, 8), ('exceeded', 529, 8), ('enjoyed', 530, 8), ('love', 548, 8), ('messed', 550, 8), ('Need', 552, 8), ('need', 562, 8), ('SUCKS', 595, 8), ('need', 615, 8), ('excited', 618, 8), ('goof', 622, 8), ('wish', 635, 8), ('freak', 637, 8), ('winning', 643, 8), ('love', 661, 8), ('hate', 696, 8), ('Wish', 710, 8), ('Love', 758, 8), ('hate', 771, 8), ('LOVE', 793, 8), ('Cant', 820, 8), ('need', 826, 8), ('want', 831, 8), ('wimped', 838, 8), ('impressed', 874, 8), ('loves', 908, 8), ('want', 909, 8), ('hate', 916, 8), ('lost', 939, 8), ('loved', 940, 8), ('love', 949, 8), ('blame', 984, 8), ('want', 995, 8), ('want', 1026, 8), ('better', 1032, 8), ('MISSED', 1036, 8), ('distracted', 1059, 8), ('Boycotting', 1082, 8), ('hurt', 1089, 8), ('love', 1096, 8), ('scared', 1099, 8), ('cant', 1100, 8), ('Need', 1104, 8), ('lose', 1134, 8), ('lose', 1135, 8), ('bored', 1172, 8), ('hate', 1175, 8), ('wish', 1181, 8), ('loved', 1210, 8), ('Wish', 1224, 8), ('surprised', 1258, 8), ('experienced', 1273, 8), ('need', 1295, 8), ('stuck', 1304, 8), ('want', 1306, 8), ('mean', 1323, 8), ('ignore', 1331, 8), ('blame', 1332, 8), ('broke', 1334, 8), ('love', 1341, 8), ('wish', 1343, 8), ('HATE', 1346, 8), ('need', 1348, 8), ('Want', 1354, 8), ('NEEDED', 1358, 8)], 'verb.possession': [('register4', 12, 11), ('WIN', 18, 11), ('subscribe', 32, 11), ('win', 82, 11), ('taken', 88, 11), ('saves', 218, 11), ('give', 221, 11), ('earned', 244, 11), ('bet', 245, 11), ('bet', 303, 11), ('catch', 314, 11), ('give', 326, 11), ('catch', 339, 11), ('give', 385, 11), ('buy', 405, 11), ('purchase', 408, 11), ('bought', 465, 11), ('catch', 478, 11), ('took', 495, 11), ('buy', 506, 11), ('releasing', 544, 11), ('give', 551, 11), ('pay', 642, 11), ('hadd', 669, 11), ('gave', 692, 11), ('took', 749, 11), ('Dealing', 782, 11), ('pass', 807, 11), ('sold', 853, 11), ('commit', 894, 11), ('saved', 922, 11), ('caught', 968, 11), ('give', 977, 11), ('donated', 985, 11), ('buy', 1006, 11), ('#invest', 1129, 11), ('Offering', 1200, 11), ('give', 1235, 11), ('took', 1279, 11), ('buy', 1327, 11)], 'verb.motion': [('DRIVING', 17, 9), ('walks', 42, 9), ('walks', 43, 9), ('Go', 59, 9), ('go', 89, 9), ('go', 116, 9), ('Fallin', 134, 9), ('pull', 164, 9), ('GOO', 175, 9), ('Driving', 191, 9), ('driving', 192, 9), ('driving', 193, 9), ('walking', 204, 9), ('go', 214, 9), ('go', 255, 9), ('takin', 269, 9), ('kick', 273, 9), ('leave', 294, 9), ('turn', 297, 9), ('run', 336, 9), ('dance', 337, 9), ('go', 383, 9), ('drive', 393, 9), ('Launches', 419, 9), ('goin', 430, 9), ('snap', 469, 9), ('trippin', 475, 9), ('twirlllllllllll', 500, 9), ('move', 517, 9), ('cross', 533, 9), ('stoop', 558, 9), ('go', 563, 9), ('kickit', 574, 9), ('punts', 575, 9), ('*throwing', 620, 9), ('goin', 662, 9), ('fly', 673, 9), ('stuffed', 714, 9), ('droppin', 747, 9), ('go', 790, 9), ('run', 802, 9), ('go', 822, 9), ('go', 832, 9), ('turn', 834, 9), ('go', 847, 9), ('droppin', 856, 9), ('Rockin', 859, 9), ('marched', 880, 9), ('running', 895, 9), ('leaving', 906, 9), ('traveling', 907, 9), ('go', 910, 9), ('goin', 953, 9), ('move', 961, 9), ('Go', 974, 9), ('position', 1107, 9), ('fly', 1108, 9), ('moving', 1112, 9), ('driving', 1191, 9), ('move', 1194, 9), ('spinning', 1283, 9), ('go', 1313, 9), ('travelling', 1340, 9)], 'verb.change': [('increased', 19, 1), ('removing', 20, 1), ('dont', 63, 1), ('left', 155, 1), ('finish', 180, 1), ('gone', 222, 1), ('finish', 228, 1), ('slow', 277, 1), ('dont', 312, 1), ('Ending', 354, 1), ('switch', 376, 1), ('end', 402, 1), ('grinds', 454, 1), ('mixed', 565, 1), ('STOP', 567, 1), ('mixing', 572, 1), ('dying', 578, 1), ('left', 585, 1), ('baking', 588, 1), ('gone', 604, 1), ('stop', 649, 1), ('die', 653, 1), ('stopped', 683, 1), ('Rewind', 691, 1), ('change', 762, 1), ('Break', 860, 1), ('removed', 913, 1), ('gone', 942, 1), ('changed', 1303, 1), ('switch', 1324, 1)], 'verb.competition': [('beat', 41, 4), ('shoots', 162, 4), ('aimed', 188, 4), ('fuck', 351, 4), ('kill', 411, 4), ('shooting', 436, 4), ('kill', 519, 4), ('bomb', 559, 4), ('hit', 589, 4), ('Hit', 619, 4), ('Fck', 671, 4), ('require', 737, 4), ('beat', 744, 4), ('charged', 757, 4), ('kill', 801, 4), ('kill', 825, 4), ('Shout', 855, 4), ('beat', 937, 4), ('beat', 938, 4), ('fuck', 983, 4), ('defendi', 988, 4), ('leav', 1081, 4), ('kill', 1088, 4), ('expose', 1126, 4), ('FUCK', 1202, 4), ('FUCK', 1205, 4)], 'verb.body': [('smile', 66, 0), ('Tidyin', 138, 0), ('cry', 170, 0), ('donning', 237, 0), ('shut', 350, 0), ('laughed', 444, 0), ('reeks', 539, 0), ('wear', 759, 0), ('wear', 761, 0), ('spit', 796, 0), ('cry', 812, 0), ('belched', 841, 0), ('sleep', 900, 0), ('sleep', 923, 0), ('wearing', 932, 0), ('wear', 998, 0), ('Wasnt', 1051, 0), ('spent', 1054, 0), ('Sleeping', 1068, 0), ('waking', 1069, 0), ('sleep', 1074, 0), ('sleep', 1078, 0), ('spent', 1119, 0), ('Decline', 1128, 0), ('wear', 1136, 0), ('wearing', 1138, 0), ('didnt', 1153, 0), ('drops', 1167, 0), ('shut', 1308, 0)], 'verb.social': [('further', 142, 12), ('succeed', 190, 12), ('launches', 209, 12), ('celebrating', 346, 12), ('married', 359, 12), ('murdered', 377, 12), ('trying', 581, 12), ('trying', 1008, 12), ('lets', 1121, 12), ('canceled', 1143, 12), ('rape', 1169, 12), ('trying', 1274, 12), ('lets', 1318, 12)], 'verb.weather': [('Fired', 781, 14), ('fired', 785, 14), ('shine', 1148, 14), ('rain', 1152, 14)]}



# tp=[]
# for t in true_supersense:
#   t=t.replace('B-','')
#   tp.append(t)
        
# count=0.0
# errors=[]
# for i,pred in enumerate(z):
#   if pred[1][0]==tp[i]:
#     count+=1
#     print(pred)
#   else:
#     errors.append(pred)    
