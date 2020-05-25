
#!/bin/sh
#$ -S /bin/sh
#$ -N starter
#$ -cwd
#$ -j y
#$ -m eas
#$ -M simone@idsia.ch
#$ -l h_rt=1::
declare -a russian_books=("anna-karenina.txt", "Crime-and-Punishment.txt", "The-Brothers-Karamazov.txt", "The-Idiot.txt", "The-Possessed.txt", "war-and-peace.txt")
for book in "${russian_books[@]}"
do
    qsub -b yes -N "b_${book}" -S /bin/sh -cwd -j y -m eas -M simone@idsia.ch -l h_rt=23:: -l mem_free=10G ~/anaconda3/envs/novel2graph/bin/python ~/novel2graph/Code/test_relations_clustering.py "$book"
done