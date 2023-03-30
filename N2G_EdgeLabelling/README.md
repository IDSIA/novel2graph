Corpus of coarse-grained (supersense) annotated tweets
=======================

## Introduction

This repository hosts two datasets with part-of-speech and coarse-grained sense-annotated Twitter data. 
The datasets were introduced in the "More or less supervised super-sense tagging of Twitter" paper, by Johannsen et. al, 2014. 
Details about the annnotation process, including inter-annotator agreement, can also be found in the paper. See more on this below.

There are four data files in the repository and they belong to two different datasets, **Ritter** and **In-house**.

The **Ritter** dataset (files starting with `ritter`) adds supersense annotations on top of a named-entity-recognition dataset originally released by Ritter et. al, 2011. The train-dev-test splits are those suggested by Dercynski et. al, 2013.

The tweets of the **In-house** dataset were gathered in 2013 by searching for tweets that link to external homepages (i.e. that contain the string "http"). Prior to the supersense annotation the dataset was released with part-of-speech tags by Plank et. al, 2014. In that paper the dataset was referred to as  **Lowlands.test**.

## Cite

If you use these data in a scientific publication, please cite the paper:

````
@inproceedings{Johannsen:ea:14,
	Author = {Anders Johannsen and Dirk Hovy and Héctor Martinez and Barbara Plank and Anders Søgaard},
	Booktitle = {The 3rd Joint Conference on Lexical and Computational Semantics (*SEM)},
	Title = {More or less supervised super-sense tagging of Twitter},
	Address = {Dublin, Ireland},
	Year = {2014}}
````

## References

Barbara Plank, Dirk Hovy, Anders Søgaard. 2014. *Learning part-of-speech taggers with inter-annotator agreement loss*. In EACL. 

Leon Derczynski, Alan Ritter, Sam Clark, and Kalina Bontcheva. 2013. Twitter part-of-speech tagging for all: overcoming sparse and noisy data. In RANLP.

Anders Johannsen, Dirk Hovy, Héctor Martinez, Barbara Plank, and Anders Søgaard. 2014. *More or less supervised super-sense tagging of Twitter*. In \*SEM. Dublin, Ireland. 

Alan Ritter, Sam Clark, Mausam, and Oren Etzioni. 2011. *Named entity recognition in tweets: an experimental study*. In EMNLP.
