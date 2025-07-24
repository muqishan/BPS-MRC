import torch
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from torch.optim import AdamW
import flair
import os
# Check if multiple GPUs are available
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     flair.device = torch.device("cuda:1")

# define columns
columns = {0: 'text', 1: 'ner'}
# this is the folder in which train, test and dev files reside
data_folder = 'datasets20240805_4391'


# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train_data.txt',
                              test_file='test_data.txt',
                              dev_file='val_data.txt')

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
print(label_dict)

# 4. initialize fine-tuneable transformer embeddings WITH document context
embeddings = TransformerWordEmbeddings(model='pubmedbert',
                                       layers="-1",
                                       subtoken_pooling="mean",
                                    #    subtoken_pooling="first",
                                       fine_tune=True,
                                       use_context=True,
                                       allow_long_sentences=True,
                                       model_max_length=768,
                                       )
# store_embeddings(evaluation_split_data, embeddings_storage_mode)
# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type='ner',
                        tag_format='BIO',
                        use_crf=True,
                        use_rnn=True,
                        reproject_embeddings=True,
                        )

# If multiple GPUs are available, wrap the model with DataParallel
# if torch.cuda.device_count() > 1:
#     tagger = torch.nn.DataParallel(tagger)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. run fine-tuning
trainer.fine_tune(
    'runs/cells_ner2_20240805',
    learning_rate=5.0e-5,
    mini_batch_size=32,
    max_epochs=20,
    optimizer=AdamW,
    embeddings_storage_mode="none",
    use_final_model_for_eval=True,
    # train_with_dev=True
    # mini_batch_chunk_size=1,  # Large mini-batch sizes may improve speed if using large GPU memory
)
