from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering,MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.metrics import silhouette_score
import re
import torch
import scipy
from sklearn.cluster import DBSCAN
import os
#import matplotlib.pyplot as plt
import logging
from collections import Counter
import pandas as pd
from scipy.spatial import distance
import itertools
from nltk.corpus import wordnet as wn
import pickle
##__version__ = "0.4.1"
##__DOWNLOAD_SERVER__ = 'https://sbert.net/models/'
##model_path = "bert-base-nli-mean-tokens"
##model_path = __DOWNLOAD_SERVER__ + model_path + '.zip'
##tokenizer = AutoTokenizer.from_pretrained("bert-base-nli-mean-tokens")
##model = AutoModel.from_pretrained("bert-base-nli-mean-tokens")
#Mean Pooling - Take attention mask into account for correct averaging
##def mean_pooling(model_output, attention_mask):
##    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
##    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
##    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
##    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
##    return sum_embeddings / sum_mask
##encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
##
###Compute token embeddings
##with torch.no_grad():
##    model_output = model(**encoded_input)



class Bert_cluster:
    def __init__(self, result_folder, book, aliases, sentences_file=None):
        if not os.path.isdir(result_folder):
            os.makedirs(result_folder)
        self.result_folder = result_folder
        if not os.path.isdir(result_folder + book):
            os.makedirs(result_folder + book)
        self.book = book
        self.path_supersense='.//my_semcor1.csv'
        self.model = SentenceTransformer('bert-base-nli-mean-tokens',device='cuda')
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device',device)
        self.aliases = aliases
        if sentences_file is not None:
            self.sentences = pd.read_pickle(sentences_file)
            sentences_str = [' '.join(sent) for sent in self.sentences['Original_token']]
            sentences_str = [sentence.replace('\n', '') for sentence in sentences_str]
            self.sentences_str = pd.Series(sentences_str)
            sentences_str = [' '.join(sent) for sent in self.sentences['Lemmatize_token']]
            sentences_str = [sentence.replace('\n', '') for sentence in sentences_str]
            self.sentences_lemmatize_str = pd.Series(sentences_str)
            print('*************',self.sentences)
            self.chars_number = len(self.sentences.iloc[0]['Characters'])
            print('chars_number',self.chars_number)
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
        


    def extract_verbs(self):
        verbs_between_chars = {}
        removed_chars = {}
        new_sentences = {}
        for id, row in self.sentences.iterrows():
            try:
                sentence_str = self.sentences_str.iloc[id]
                sentence_lemmatize_str = self.sentences_lemmatize_str.iloc[id]

            except:
                print("An exception occurred while processing this sentence.")
                exit(-1)
            tokens = row['Original_token']
            tags = row['Tag']
            #list of lemmatize verbs
            verbs = []
            #the original tag for each item in verbs
            verbs_tag = []
            chars = []

            if self.chars_number == 2:
                between_names = False
                continues_verb = False
                first_verb = True

                for i, word in enumerate(tokens):
                    if 'CCHARACTER' in word:
                        chars.append(re.search('(CCHARACTER)([0-9]+)', word).group(0))
                        between_names = not between_names
                        continue
                    if between_names:
                        if 'VB' in tags[i]:
                            if first_verb:
                                verbs.append(row['Lemmatize_token'][i])
                                verbs_tag.append(tags[i])
                                first_verb = False
                                continues_verb = True
                                continue
                            if not first_verb and continues_verb:
                                if row['Lemmatize_token'][i]!='be':
                                    verbs.append(row['Lemmatize_token'][i])
                                    verbs_tag.append(tags[i])
                            else:
                                continue
                        else:
                            continues_verb = False
            #single character in the sentence
            elif self.chars_number == 1:
                continues_verb = False
                first_verb = True
                for i, word in enumerate(tokens):
                    if 'CCHARACTER' in word:
                        chars.append(re.search('(CCHARACTER)([0-9]+)', word).group(0))
                        continue
                    if 'VB' in tags[i]:
                        if first_verb:
                            if row['Lemmatize_token'][i]!='be':
                                verbs.append(row['Lemmatize_token'][i])
                                verbs_tag.append(tags[i])
                                first_verb = False
                                continues_verb = True
                                continue
                        if not first_verb and continues_verb:
                            if row['Lemmatize_token'][i]!='be':
                                verbs.append(row['Lemmatize_token'][i])
                                verbs_tag.append(tags[i])
                        else:
                            continue
                    else:
                        continues_verb = False
            else:
                logging.info('Are you sure about the number of characters? If so, implement this part.')
                exit(-1)

            verbs_str = ' '.join(verbs)
            if verbs_str != '':
                verb = None
                for i in range(len(verbs)-1, -1, -1):
                    if 'VB' in verbs_tag[i]:
                        verb = verbs[i]
                        break
                if verb is not None:
                    if len(chars) == self.chars_number:
                        verbs_between_chars[id] = verb
                        removed_chars[id] = chars
                        new_sentences[id] = sentence_str
                    else:
                        logging.info('Someting went wrong with the amount of characters you provide: %s, names: %s', sentence_str, chars)
                else:
                    logging.info('VB not detected in sentence: %s', sentence_str)
            else:
                logging.info('Verb not detected: %s', sentence_str)

        self.verbs = verbs_between_chars
        print('verbs:',len(self.verbs))
        print(self.verbs)
        self.removed_chars = removed_chars
        self.sentences_dictionary = new_sentences
##        for id ,verb in self.verbs.items():
##            if verb=='be':
##                del self.verbs[id]
##                del self.removed_chars[id]
##                del self.sentences_dictionary[id]
##        
        

    def remove_char_from_sentences(self):
        # Deprecated, not working anymore
        # remove character identifier from sentences
        no_char_sentences = {}
        removed_chars = []
        for id, sentence in self.sentences_dictionary.items():
            new_sentence = ""
            chars = []
            for word in sentence.split(' '):
                if 'CCHARACTER' in word:
                    chars.append(re.search('(CCHARACTER)([0-9]+)', word).group(0))
                    continue
                new_sentence += word + ' '
            no_char_sentences[id] = new_sentence
            removed_chars.append(chars)
        self.removed_chars = removed_chars
        self.no_char_sentences = no_char_sentences
        

                           
            

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
        supersense_path=".//supersenses/supersense_cluster_dict_%s.pkl"%(self.book)
        if os.path.exists(supersense_path):
            print('supsersense path exists')
            with open(".//supersenses/supersense_cluster_dict_%s.pkl"%(self.book),"rb") as f:
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
                    print('Unique presence',verbs,keys)
                    if keys not in self.verb_supersense_dict:
                        self.verb_supersense_dict[keys]=[(verbs,id,sids)]
                    # self.store_embeddings[self.verbs[id]]=self.embed_values[id]
                        
                        
                    else:
                        if verbs not in self.verb_supersense_dict[keys]:
                            self.verb_supersense_dict[keys].append((verbs,id,sids))
                            
                            
                    
    ##                self.id_verb_supersense_dict[sids]=verbs
                    
                elif ct>1:# This is ambiguous
                    print('Presence Ambiguous_compute average dynamic embedding',verb)
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
                    print('Presence Ambiguous',key)
                    if key not in self.verb_supersense_dict:
                        self.verb_supersense_dict[key]=[(verb,id,index_min)]
                        
                    else:
                        if verb not in self.verb_supersense_dict[key]:
                            self.verb_supersense_dict[key].append((verb,id,index_min))
                            
    ##                self.id_verb_supersense_dict[index_min]=verbs
                    
                elif f1==0:#no verbs directly in the supersense list, then go for embeddings
                    print('Not present: compute average static embedding',verb)
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
                    print('Not present',key)
                    if key not in self.verb_supersense_dict:
                        self.verb_supersense_dict[key]=[(verb,id,index_min)]
                        
                    else:
                        if verb not in self.verb_supersense_dict[key]:
                            self.verb_supersense_dict[key].append((verb,id,index_min))
                            
                                                              
##                self.id_verb_supersense_dict[index_min]=verbs
            print('verb_supersense_dict',self.verb_supersense_dict)
            df1 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in self.verb_supersense_dict.items() ]))
            df1.to_pickle(".//supersenses/supersense_cluster_average_%s.pkl"%(self.book))
            with open(".//supersenses/supersense_cluster_dict_%s.pkl"%(self.book),"wb") as f:
                pickle.dump(self.verb_supersense_dict,f)
            
        for items in self.verb_supersense_dict.keys():
                tup=self.verb_supersense_dict[items]
                for t in tup:
                    self.verbid_superid[t[1]]=t[2]
                    
        print('verbid_superid',self.verbid_superid)
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
            
        return self.verb_supersense_dict
    
##    def relate_verb_supersense(self):
##        #returns the dictionary with verbs related to supersense
##        embed_values=list(self.embedding.values())
##        self.verb_supersense_dict={}
##        for i,verb in enumerate(self.verbs.values()):
##            f1=0
##            ct=0
##            for key in self.dict_supersense:
##                if verb in self.dict_supersense[key]:
##                    f1=1
##                    ct+=1
##                    if key not in self.verb_supersense_dict:
##                        self.verb_supersense_dict[key]=[verb]
##                    else:
##                        if verb not in self.verb_supersense_dict[key]:
##                            self.verb_supersense_dict[key].append(verb)
##                            
##            if ct>1:# This is ambiguous
##                
##                
##            if f1==0:#no verbs directly in the supersense list, then go for embeddings
##                min_dist=[]
##                
##                v_embedding = np.array([list(embed_values[i])])
##                for key in self.supersense_embeddings:
##                    embed=self.supersense_embeddings[key]
##                    
##                    sense_embed=np.array(list(embed.values()))
##                    dist=distance.cdist(sense_embed, v_embedding, 'cosine')
##                    min_dist.append(min(dist))
##                index_min = min(range(len(min_dist)), key=min_dist.__getitem__)
##                key=list(self.dict_supersense.keys())[index_min]
##                if key not in self.verb_supersense_dict:
##                    self.verb_supersense_dict[key]=[verb]
##                else:
##                    if verb not in self.verb_supersense_dict[key]:
##                        self.verb_supersense_dict[key].append(verb)
##        return self.verb_supersense_dict

    def find_common_hypernyms(self):
      common_hyper=[]
      lchs={}
      lchs_id={}
      self.lch_idmappings={}
      supersense_ids={}
      self.represent_hyper={}
      self.id_represent_hyper={}
      self.outliers_hyper={}
      id_syn=0
      self.cluster_ids_hypernym={}
      self.cluster_ids_super={}
      self.cluster_verbs=[]
      
      
      
      # for ids, supersense in enumerate(self.super_labels):
      #     verb_cluster=self.super_labels[supersense]
      #     print('verb groups',verb_cluster)
          
      #     verb_ids=self.super_labels_id[supersense]
      #     print('verb_ids',verb_ids)
      #     if len(verb_cluster)>0:
      #         verb_pairs=list(itertools.combinations(verb_cluster,2))
              
          
      for ids,supersense in enumerate(self.verb_supersense_dict):
          
          verb_groups=self.verb_supersense_dict[supersense]
          #print('verb groups',verb_groups)
          verb_cluster=[x[0] for x in verb_groups]
          verb_id=[x[1] for x in verb_groups]
          supersense_id=[x[2] for x in verb_groups]
          #if len(verb_cluster)>1:
          #get the pairs 
         
          verb_pairs=list(itertools.combinations(verb_cluster,2))
          verb_pairs_id=list(itertools.combinations(verb_id,2))
          super_pairs_id=list(itertools.combinations(supersense_id,2))
          
          for j,(v1,v2) in enumerate(verb_pairs):
              #print('verb_pair',(v1,v2))
              (id1,id2)=verb_pairs_id[j]
              (sid1,sid2)=super_pairs_id[j]
            
              flag=0
              # finding the lch for each verb pair, if it exists
              
              for ss1 in wn.synsets(v1,pos='v'):
                  for ss2 in wn.synsets(v2,pos='v'):
                      ch=ss1.lowest_common_hypernyms(ss2)
                      if len(ch)>0:
                          #print(v1,v2)
                          flag=1
                          common_hyper.extend(ch)
                          # associate the lch with the verb_pair, with corresponding
                          #ids and supersense_ids
                          for k in ch:
                              if k not in lchs:
                                  lchs[k]=[(v1,v2)]
                                  lchs_id[k]=[(id1,id2)]
                                  supersense_ids[k]=[(sid1,sid2)]
                              else:
                                  lchs[k].append((v1,v2))
                                  lchs_id[k].append((id1,id2))
                                  supersense_ids[k].append((sid1,sid2))
                                   #if (v1,v2) not in lchs[k]:
                                


          # count the extracted hypernyms to rank them, finding most_common                       
          x=Counter(common_hyper)

          ct=x.most_common()
          #print('ct',ct)
          lch_verbs={}
          lch_verbs_id={}
          super_verbs_id={}
          #print('lch',lchs)
          # for each lch
          for k in lchs.keys():
              m=list(itertools.chain(*lchs[k]))
              lch_verbs[k]=m
              lch_verbs_id[k]=list(itertools.chain(*lchs_id[k]))
##              super_verbs_id[k]=list(itertools.chain(*super_verbs_id[k]))
              super_verbs_id[k]=list(itertools.chain(*supersense_ids[k]))
          m1=[]
          verbs_syn={}
          verbs_syn_id={}
          verbs_super_id={}
          
##          we take the each synsets based on the most common occurrences and iteratively remove the verbs that already appear in a previous more frequent synset
          # verbs_syn will have the synset (lch): [associated verb_list]
          for i,syn in enumerate(ct):
              m2=lch_verbs[syn[0]]
              m2_id=lch_verbs_id[syn[0]]
              m2_sid=super_verbs_id[syn[0]]
              for kk,j in  enumerate(m2):
                  #if j not in m1:
                    if syn[0] not in verbs_syn.keys():
                        verbs_syn[syn[0]]=[j]
                        verbs_syn_id[syn[0]]=[m2_id[kk]]
                        verbs_super_id[syn[0]]=[m2_sid[kk]]
                    else:
                        verbs_syn[syn[0]].append(j)
                        verbs_syn_id[syn[0]].append(m2_id[kk])
                        verbs_super_id[syn[0]].append(m2_sid[kk])
                    self.cluster_ids_hypernym[m2_id[kk]]=id_syn
                    self.cluster_ids_super[m2_sid[kk]]=id
                    m1.append(j)
              #print(id_syn)
              #print('id_syn',(id_syn,syn[0]))
              self.lch_idmappings[id_syn]=syn[0].name().split('.')[0]
              id_syn+=1
            
          
        
          # print('ids_hyper',self.cluster_ids_hypernym)
          # print('verbs_syn:lch and associated verbs',verbs_syn)
         # print(len(lchs))
          
            

          #get the dictionary of outlliers, which are the verbs not associated with any of the lch/synsets
              
          c=list(itertools.chain(*verbs_syn.values()))
          
         
          outliers=list(set(verb_cluster).difference(set(c)))
          print('outliers',outliers)
          
          cid=list(itertools.chain(*verbs_syn_id.values()))
          
          outliers_id=list(set(verb_id).difference(set(cid)))
          
          cids=list(itertools.chain(*verbs_super_id.values()))
          
          # outliersuper_ids=list(set(supersense_id).difference(set(cids)))
          outliersuper_ids=[supersense_id[0]]*len(outliers)
          
          #print(outliers,outliers_id,outliersuper_ids)
          # id_syn+=1
          #print(self.cluster_ids_hypernym)
          
          for ids,out in enumerate(outliers):
              self.cluster_ids_hypernym[outliers_id[ids]]=id_syn
              self.cluster_ids_super[outliersuper_ids[ids]]=id
              
              self.lch_idmappings[id_syn]=out
              id_syn+=1
          #print(self.cluster_ids_hypernym)  
          #print('self.cluster_ids_super',self.cluster_ids_super) 
          if supersense not in self.represent_hyper:
              self.represent_hyper[supersense]=[verbs_syn]
          else:
              self.represent_hyper[supersense].append(verbs_syn)
              
          if supersense not in self.outliers_hyper:
              self.outliers_hyper[supersense]=[verbs_syn]
          else:
              self.outliers_hyper[supersense].append(verbs_syn)
          #print(self.represent_hyper)
          #print(self.outliers_hyper)
      print('lch_id_maps',self.lch_idmappings)
      
      print(self.cluster_ids_hypernym)
      
      self.cluster_ids_hypernym=sorted(self.cluster_ids_hypernym.items(),key=lambda x: x[0])
      self.hypernym_ids=np.array([x[1] for x in self.cluster_ids_hypernym])
      print('hypernym_ids',self.hypernym_ids)
      verbs=list(self.verbs.values())
      verbs1= list(set(list(self.verbs.values())))
      print(verbs)
      verb_ids=[x[0] for x in self.cluster_ids_hypernym]
      # labels = {}
      
      self.hypernym_ids=np.array(self.hypernym_ids)
      # for i in self.hypernym_ids:
      #     involved_triplet = np.where(self.hypernym_ids== i)
      #     if len(involved_triplet[0]) == 0:
      #         logging.info('----No Relation---\n')
      #         continue
            
      #     cluster_verbs = [verbs[index] for index in involved_triplet[0]]
      #     labels[i] = cluster_verbs
     
      # print(labels)
      print('****clusters*****',self.hypernym_ids)
      return self.hypernym_ids                  
                
  # def kmeans(self, k):
  #       km = KMeans(n_clusters=k)
  #       # normalize vectors before the kmean
  #       sentence_embeddings = [vector / np.linalg.norm(vector) for vector in list(self.embedding.values())]

  #       km.fit(sentence_embeddings)
  #       self.cluster_model = km
  #       return np.array(km.labels_.tolist())

    def generate_triplets(self, clusters):
        triplets = []
        i = 0
        print("printing information:",len(clusters), len(self.removed_chars), len(self.verbs), len(self.sentences_dictionary))
        print(clusters)
        print('\n')
        print(self.removed_chars)
        print('\n')
        print(self.sentences_dictionary)
        for key, value in self.removed_chars.items():
            if self.chars_number == 2:
                triplet = np.array([value[0], clusters[i], value[1]])
            elif self.chars_number == 1:
                triplet = np.array([value[0], clusters[i]])
            else:
                logging.info('Are you sure about the number of characters? If so, implement this part.')
                exit(-1)
            triplets.append(triplet)
            i += 1
        self.triplets = triplets
        print('triplets generated',self.triplets)
        

    def label_verbs(self, clusters):
        verbs = list(self.verbs.values())
        print('verbs',verbs)
        labels = {}
        labels1={}
        for i in clusters:
            involved_triplet = np.where(clusters == i)
            if len(involved_triplet[0]) == 0:
                logging.info('----No Relation---\n')
                continue
            cluster_verbs = [verbs[index] for index in involved_triplet[0]]
            labels[i] = self.lch_idmappings[i]
            labels1[i]=cluster_verbs
        print('labels',labels)
        return labels

    def generate_reports(self, clusters):
        file = open(self.result_folder + self.book + '/' + self.book + '_report_' + str(self.chars_number) + '.txt', "w+")
        file1=open(self.result_folder + self.book + '/' + self.book + '_report_verb' + str(self.chars_number) + '.txt', "w+")#added
        data = {}
        all_cluster = {}
        sentences = list(self.sentences_dictionary.values())
        verbs = list(self.verbs.values())
        print('aliases',self.aliases)
        print('clusters',clusters,type(clusters))
        for i in clusters:
            print('cluster i',i)
            involved_triplet = np.where(clusters == i)
            if len(involved_triplet[0]) == 0:
                file.write('----No Relation---\n')
                continue
            cluster_verbs = [verbs[index] for index in involved_triplet[0]]
            verb = self.lch_idmappings[i]
            file.write('----Relation ' + verb + '---\n')
##            file1.write('----Relation ' + str(verbs1) + '---\n')#added
            all_cluster[i] = []
            print('**involved_triplet',involved_triplet)
            #DEBUG HERE!!!
            for index in involved_triplet[0]:
                current_triplet = self.triplets[int(index)]
                print('current_triplet',current_triplet)
                sentence = sentences[index]
                
                # print((self.aliases[current_triplet[0]][0],self.lch_idmappings[int(current_triplet[1])],self.aliases[current_triplet[-1][0]]))
                #sentence = sentence.replace('\n', '')
                if self.chars_number == 2:
                    print('Triplets')
                    if self.aliases[current_triplet[0]][0]!=self.aliases[current_triplet[2]][
                                   0]:# no self relations
                        print('('+self.aliases[current_triplet[0]][0] + ',' + verbs[index] + ',' +
                               self.aliases[current_triplet[2]][
                                   0]+')')
                        file.write(self.aliases[current_triplet[0]][0] + '\t' + verbs[index] + '\t' +
                               self.aliases[current_triplet[2]][
                                   0] + '\t' + sentence + '\n')
                        file1.write(str(self.aliases[current_triplet[0]][0]) + ',' + str(verbs[index])+ ',' +
                               str(self.aliases[current_triplet[2]][
                                   0])+ '\n')

                        key = (current_triplet[0], current_triplet[2])
                        if key not in data:
                            data[key] = []
                        data[key].append(str(current_triplet[1]))
                elif self.chars_number == 1:
                    file.write(self.aliases[current_triplet[0]][0] + '\t' + verbs[index] + '\t' + sentence + '\n')
                    key = (current_triplet[0])
                    if key not in data:
                        data[key] = []
                    data[key].append(str(current_triplet[1]))
                else:
                    logging.info('Are you sure about the number of characters? If so, implement this part.')
                    exit(-1)
                all_cluster[i].append(list(self.sentences_dictionary.keys())[index])
        file.close()
        file1.close()
        self.chars_relations = data
        self.all_cluster = all_cluster              
            
             
                       
       
            
            
    

    def dbscan(self, metric='precomputed', min_samples=1, algorithm='brute', eps=0.27):
        embedding = list(self.embedding.values())
        similarities = np.empty((len(embedding), len(embedding)))
        embedding = [vector / np.linalg.norm(vector) for vector in embedding]

        for i, sentence1 in enumerate(embedding):
            for j, sentence2 in enumerate(embedding):
                similarities[i][j] = 1. + scipy.spatial.distance.cosine(sentence1, sentence2)

        db = DBSCAN(metric=metric, min_samples=min_samples, algorithm=algorithm, eps=eps).fit(similarities)
        self.cluster_model = db
        return db.labels_

    def affinity_propagation(self):
        sentence_embeddings = [vector / np.linalg.norm(vector) for vector in list(self.embedding.values())]
        af = AffinityPropagation().fit(sentence_embeddings)

        return af.labels_

    def mean_shift(self):
        sentence_embeddings = [vector / np.linalg.norm(vector) for vector in list(self.embedding.values())]
        bandwidth = estimate_bandwidth(sentence_embeddings)

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(sentence_embeddings)

        return ms.labels_



    def silhouette_kmean(self, max_iter=50):
        silh = []
        j = 2
        best_silhouette = -1
        best_k = -1
        embeddings = [vector / np.linalg.norm(vector) for vector in list(self.embedding.values())]
        sse = {}
        while j <= max_iter:
            km = KMeans(n_clusters=j)
            km = km.fit(embeddings)
            y = km.predict(embeddings)

            sse[j] = km.inertia_

            silhouette_avg = silhouette_score(embeddings, y)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_k = j
            silh.append(silhouette_avg)
            j = j + 1

        # the K-value corresponding to maximum silhoutte score is selected
        m = max(silh)
        max_indices = [i for i, j in enumerate(silh) if j == m]
        maximums = max_indices[0] + 2

        fig, axs = plt.subplots(2)
        fig.suptitle('Best Kmean evalution')
        axs[0].plot(list(sse.keys()), list(sse.values()))
        axs[0].set(ylabel='SSE')
        axs[1].plot(list(sse.keys()), silh)
        axs[1].set(ylabel='Silhouette')
        plt.xlabel("Number of cluster")
        plt.show()

        return silh, maximums

    def silhouette_dbscan(self, max_min_sample=100, max_eps=5):
        silh = []
        epsi = []
        clusters = []
        eps = 0.2
        embeddings = [vector / np.linalg.norm(vector) for vector in list(self.embedding.values())]
        best_silh = -1

        while eps <= max_eps:
            y = self.dbscan(min_samples=1, eps=eps)
            if max(y) < 2:
                eps = eps + 0.2
                continue
            silhouette_avg = silhouette_score(embeddings, y)
            silh.append(silhouette_avg)
            epsi.append(eps)
            clusters.append(max(y))
            if silhouette_avg > best_silh:
                best_silh = silhouette_avg

            eps = eps + 0.2

        # the K-value corresponding to maximum silhoutte score is selected
        #m = max(silh)
        #max_indices = [i for i, j in enumerate(silh) if j == m]
        #maximums = max_indices[0] + 2


        #fig.suptitle('Best Kmean evalution')
        plt.plot(epsi, silh)
        plt.ylabel("Avg. silhouette")
        plt.xlabel("Evolution of epsilon")
        plt.show()

        #return silh, maximums
# with open("./opt/anaconda3/envs/NLP/novel2graph_sample/Data/embedRelations/hp2_sample2/hp2_sample2_report_verb2.txt") as f:
#     k=f.readlines()
# a=k[0].strip('\n')
# res = tuple(map(str, a.split(',')))
