from BERT_family_cls import Bert_family_cls


#Be sure to have some GPUs to run this code
def main():
    bert = Bert_family_cls('all_hp')
    bert.upload_files()
    bert.train()


if __name__ == '__main__':
    main()
