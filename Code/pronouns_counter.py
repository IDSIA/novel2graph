from nltk.tokenize import word_tokenize

first = ['i', 'me', 'my', 'mine', 'myself']
second_fifth = ['you', 'your', 'yours', 'yourself', 'yourselves']
third_m = ['he', 'him', 'his', 'himself']
third_f = ['she', 'her', 'hers', 'herself']
third_o = ['it', 'its', 'itself']
fourth = ['we', 'us', 'our', 'ours', 'ourselves']
sixth = ['they', 'them', 'their', 'theirs', 'themselves']

pronouns = [first, second_fifth, third_m, third_f, third_o, fourth, sixth]

files = ['original.txt', 'after.txt', 'coref_before_my_dealias.txt', 'coref_after_my_dealias.txt']


for file in files:
    print('---- file: ', file)
    file = open(file, 'r', encoding='cp1252')
    text = file.read()
    text = text.lower()
    tokens = word_tokenize(text)

    for pron_group in pronouns:
        counter = 0
        for pron in pron_group:
            counter += tokens.count(pron)
        print(pron_group, ' : ', str(counter))

