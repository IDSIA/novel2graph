# NOVEL2GRAPH
The algorithm receives a book and it discovers main characters, main relations between characters and more powerful information. In addition it is able to plot the evolution of a character as also the distances between different characters (for more infos see: https://arxiv.org/pdf/2003.08811.pdf).

## Quickstart
- Clone the project
- Download the latest version of https://nlp.stanford.edu/software/CRF-NER.shtml#Download
- Unzip the downloaded folder and put the folder `stanford-ner-20XX-XX-XX` in `./libraries`
- Install requirements.txt
- start test_embedding.py or test_relations_clustering.py providing a book in txt format


### Start
Start the program simply by typing:
```shell
$ python Code/test_embedding.py myBook.txt
```

### Output 
Results are generated in *Data* folder, in particular this folder could contain:
- *clust&Dealias*: for each book contains the following files
    - *bookName_occurrences.csv* contains a list of character name and its occurrences
    - *bookName_more_than_1.csv* as before but only with names occurring more than once
    - *bookName_clusters.csv* contains all clusters, each row shows a cluster id, contained names and occurrences list
    - *bookName_out.txt* contains the story in which names are replaced with an identifier ("CHARACTERX")
- *embedding*:
    - *embeddings* contains the embedding of the book
    - *models/bookName/chapters* contains dynamic and static trained models
    - *slices/bookName/chapters* contains slices which are use to train each model
- *embedRelations/book*:
    - *bookName* contains the graphiz version of the pdf
    - *bookName.pdf* contains main characters and main relations between them
    - *bookName_report.txt* contains all relations retrieved using embedding and kmean, a row contains the sentence, involved characters, the relation identifier and a flag showing if the action is performed by both actors
- *triplets*:
    - *summary_sentences.tsv* contains foreach cluster a representative sentence 
    - *summary_sentences_act.tsv* as before but with a cleaned version of the sentence
    - *summary_triplets.tsv* shows the identified relation for each cluster
### Bugs fix
- To install Graphviz: ```$ sudo apt install python-pydot python-pydot-ng graphviz```
- To install mysql-config: ```$ sudo apt-get install libmysqlclient-dev```
- To install resource stopwords (or punkt), please use the NLTK Downloader:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
- Often it is useful to download: 
    - ```python -m spacy download en```
    - ```python -m spacy download en_core_web_sm```