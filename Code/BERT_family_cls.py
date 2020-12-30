import os
import tensorflow as tf
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import matthews_corrcoef
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


class Bert_family_cls:
    #from paper: Temporal Embeddings and Transformer Models for Narrative Text Understanding.
    def __init__(self, book, result_folder='./../Data/family_relations/'):
        if not os.path.isdir(result_folder):
            os.makedirs(result_folder)

        self.results_folder = result_folder + book + '/'
        if not os.path.isdir(self.results_folder):
            os.makedirs(result_folder)

        if not os.path.isdir(self.results_folder + '/input'):
            raise Exception('Please insert input files in Data/family_relations/book/input!')

        self.img_folder = result_folder + 'imgs/'
        if not os.path.isdir(self.img_folder):
            os.makedirs(self.img_folder)

        self.models_folder = result_folder + 'models/'
        if not os.path.isdir(self.models_folder):
            os.makedirs(self.models_folder)

        # device_name = tf.test.gpu_device_name()
        # if device_name != '/device:GPU:0':
        #     raise SystemError('GPU device not found')
        # print('Found GPU at: {}'.format(device_name))

        self.book = book
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    def upload_files(self):
        dfs = []
        i = 0
        for file in os.listdir(self.results_folder + 'input'):
            print('User uploaded file "{name}" with length {length} bytes'.format(
                name=file, length=len(file)))
            df = pd.read_csv(self.results_folder + 'input/' + file, delimiter='\t', header=None,
                             names=['sentence', 'entity_pair', 'label'])
            df.replace(np.nan, 'NIL', inplace=True)
            dfs.append(df)
            i += 1
        self.dfs = dfs

    def train(self):
        with open(os.path.join(self.results_folder, "log.txt"), "w") as f_log:
            for train, test in LeaveOneOut().split(self.dfs):
                train_set = [self.dfs[i] for i in train]
                test_set = self.dfs[test[0]]
                # Create sentence and label lists
                sentences_list = []
                labels_list = []
                for i, book in enumerate(train_set):
                    sentences_list.extend(book.sentence.values)
                    labels_list.extend(book.label.values)
                    f_log.write("Length book: " + str(len(sentences_list[i])) + '\n')
                f_log.write("Sentences: " + str(len(sentences_list)) + ", labels:" + str(len(labels_list)) + '\n')

                MAX_LEN = 128
                # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
                sentences_train = [self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=MAX_LEN) for
                                   i, sent in
                                   enumerate(sentences_list)]

                le = LabelEncoder()
                labels_train = labels_list
                f_log.write(str(labels_train[:10]) + '\n')
                f_log.write('Analyze labels' + '\n')
                le.fit(labels_train)
                le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                f_log.write(str(le_name_mapping) + '\n')
                labels_train = le.fit_transform(labels_train)

                # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
                input_ids_train = [inputs["input_ids"] for inputs in sentences_train]

                # Pad our input tokens
                input_ids_train = pad_sequences(input_ids_train, maxlen=MAX_LEN, truncating="post", padding="post")
                # Create attention masks
                attention_masks_train = []

                # Create a mask of 1s for each token followed by 0s for padding
                for seq in input_ids_train:
                    seq_mask_train = [float(i > 0) for i in seq]
                    attention_masks_train.append(seq_mask_train)

                # Use train_test_split to split our data into train and validation sets for training
                train_inputs, train_labels = input_ids_train, labels_train
                train_masks, _ = attention_masks_train, input_ids_train

                # Convert all of our data into torch tensors, the required datatype for our model
                train_inputs = torch.tensor(train_inputs).to(torch.int64)
                train_labels = torch.tensor(train_labels).to(torch.int64)
                train_masks = torch.tensor(train_masks).to(torch.int64)

                batch_size = 32
                # Create an iterator of our data with torch DataLoader. This helps save on memory during training
                # because, unlike a for loop, with an iterator the entire dataset does not need to be loaded into
                # memory
                train_data = TensorDataset(train_inputs, train_masks, train_labels)
                train_sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
                torch.cuda.empty_cache()

                # BINARY CLASSIFIER
                model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
                model.cuda()
                param_optimizer = list(model.named_parameters())
                no_decay = ['bias', 'gamma', 'beta']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                     'weight_decay_rate': 0.01},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                     'weight_decay_rate': 0.0}
                ]

                # This variable contains all of the hyperparemeter information our training loop needs
                optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

                train_loss_set = []

                # Number of training epochs (authors recommend between 2 and 4)
                epochs = 10

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                torch.cuda.get_device_name(0)

                for _ in trange(epochs, desc="Epoch"):
                    # Training
                    # Set our model to training mode (as opposed to evaluation mode)
                    model.train()

                    # Tracking variables
                    tr_loss = 0
                    nb_tr_examples, nb_tr_steps = 0, 0

                    # Train the data for one epoch
                    for step, batch in enumerate(train_dataloader):
                        # Add batch to GPU
                        batch = tuple(t.to(device) for t in batch)
                        # Unpack the inputs from our dataloader
                        b_input_ids, b_input_mask, b_labels = batch
                        # Clear out the gradients (by default they accumulate)
                        optimizer.zero_grad()
                        # Forward pass
                        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                        train_loss_set.append(loss.item())
                        # Backward pass
                        loss.backward()
                        # Update parameters and take a step using the computed gradient
                        optimizer.step()

                        # Update tracking variables
                        tr_loss += loss.item()
                        nb_tr_examples += b_input_ids.size(0)
                        nb_tr_steps += 1

                    f_log.write("Train loss: {}".format(tr_loss / nb_tr_steps) + '\n')

                plt.figure(figsize=(15, 8))
                plt.title("Training loss")
                plt.xlabel("Batch")
                plt.ylabel("Loss")
                plt.plot(train_loss_set)
                plt.savefig(self.img_folder + 'train' + str(test[0]) + '.png')

                model_to_save = model
                WEIGHTS_NAME = "BERT_Novel_test" + str(test[0]) + ".bin"
                OUTPUT_DIR = self.models_folder
                output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
                f_log.write(str(output_model_file) + '\n')
                torch.save(model_to_save.state_dict(), output_model_file)
                state_dict = torch.load(output_model_file)
                model.load_state_dict(state_dict)

                sentences6 = test_set.sentence.values
                f_log.write(str(len(sentences6)) + '\n')
                labels6 = test_set.label.values

                labels_test = labels6
                sentences11 = sentences6
                sentences_test = [self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=MAX_LEN) for
                                  i, sent in
                                  enumerate(sentences11)]

                f_log.write('Analyze labels test' + '\n')
                le.fit(labels_test)
                le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                f_log.write(str(le_name_mapping) + '\n')
                labels_test = le.fit_transform(labels_test)
                MAX_LEN = 128

                # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
                input_ids1 = [inputs["input_ids"] for inputs in sentences_test]
                # Pad our input tokens
                input_ids1 = pad_sequences(input_ids1, maxlen=MAX_LEN, truncating="post", padding="post")
                # Create attention masks
                attention_masks1 = []

                # Create a mask of 1s for each token followed by 0s for padding
                for seq in input_ids1:
                    seq_mask1 = [float(i > 0) for i in seq]
                    attention_masks1.append(seq_mask1)

                f_log.write(str(len(attention_masks1[0])) + '\n')

                prediction_inputs = torch.tensor(input_ids1).to(torch.int64)
                prediction_masks = torch.tensor(attention_masks1).to(torch.int64)

                prediction_labels = torch.tensor(labels_test).to(torch.int64)

                batch_size = 32
                prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
                prediction_sampler = SequentialSampler(prediction_data)
                prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

                # Prediction on test set
                # Put model in evaluation mode
                model.eval()
                # Tracking variables
                predictions, true_labels = [], []
                # Predict
                for batch in prediction_dataloader:
                    # Add batch to GPU
                    batch = tuple(t.to(device) for t in batch)
                    # Unpack the inputs from our dataloader
                    b_input_ids, b_input_mask, b_labels = batch
                    # Telling the model not to compute or store gradients, saving memory and speeding up prediction
                    with torch.no_grad():
                        # Forward pass, calculate logit predictions
                        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()

                    # Store predictions and true labels
                    predictions.append(logits)
                    true_labels.append(label_ids)

                f_log.write(str(len(predictions)) + ' ' + str(len(true_labels)) + '\n')
                f_log.write(str(predictions[0][0]) + '\n')

                # Import and evaluate each test batch using Matthew's correlation coefficient
                matthews_set = []

                for i in range(len(true_labels)):
                    matthews = matthews_corrcoef(true_labels[i],
                                                 np.argmax(predictions[i], axis=1).flatten())
                    matthews_set.append(matthews)

                # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
                flat_predictions = [item for sublist in predictions for item in sublist]
                flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
                flat_true_labels = [item for sublist in true_labels for item in sublist]

                f_log.write(str(len(flat_predictions) + ' ' + len(flat_true_labels)) + '\n')
                f_log.write(str(flat_predictions[989:994] + ' ' + flat_true_labels[989:994]) + '\n')
                f_log.write(str(flat_predictions[0:11] + ' ' + flat_true_labels[0:11]) + '\n')
                f_log.write('Classification Report' + '\n')
                f_log.write(str(classification_report(flat_true_labels, flat_predictions)) + '\n')
                f_log.write(str(confusion_matrix(flat_true_labels, flat_predictions)) + '\n')

    def weighted_avg_performances(self, partial_cls_reports):
        # partial_cls_reports contains a row for each epoch, the row contains the same information of the
        # classification report matrix (but as array) these information are stored in the train report
        partial_cls_reports = np.array(partial_cls_reports)
        precisions = []
        recalls = []
        f_values = []
        positive_rates = [0.3, 0.392, 0.289, 0.386, 0.626, 0.4762]

        for i in range(0, len(partial_cls_reports)):
            precisions.append(
                partial_cls_reports[i, 0] * (1 - positive_rates[i]) + partial_cls_reports[i, 4] * positive_rates[i])
            # print(partial_cls_reports[i,0], partial_cls_reports[i, 4], positive_rates[i])
            recalls.append(
                partial_cls_reports[i, 1] * (1 - positive_rates[i]) + partial_cls_reports[i, 5] * positive_rates[i])
            # print(partial_cls_reports[i,1], partial_cls_reports[i, 5], positive_rates[i])
            f_values.append(
                partial_cls_reports[i, 2] * (1 - positive_rates[i]) + partial_cls_reports[i, 6] * positive_rates[i])
            # print(partial_cls_reports[i,2], partial_cls_reports[i, 6], positive_rates[i])

        print(np.mean(precisions), np.std(precisions))
        print(np.mean(recalls), np.std(recalls))
        print(np.mean(f_values), np.std(f_values))

        # for i in range(0, len(a[0])):
        # print(np.mean(partial_cls_reports[:, i]), np.std(partial_cls_reports[:, i]))
