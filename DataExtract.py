import torch
import torchtext
import random
import numpy as np
import spacy
import en_core_web_sm

def dataExtract():

    Vocab_size = 20000
    # FIX THE RANDOM
    RANDOM_SEED = 1234
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # EXTRACT DATA FROM DATAFRAME
    text = torchtext.legacy.data.Field(
        tokenize="spacy",
        tokenizer_language= 'en_core_web_sm'
    )

    label = torchtext.legacy.data.LabelField(dtype=torch.long)
    fields = [('TEXT_COLUMN_NAME', text), ('LABEL_COLUMN_NAME', label)]
    dataset = torchtext.legacy.data.TabularDataset(
        path="../data/movie_data.csv", format='csv',
        skip_header=True, fields = fields
    )

        # SPLIT DATASET
    train_data, test_data = dataset.split(
        split_ratio=[0.75, 0.25],
        random_state = random.seed(RANDOM_SEED)
    )
    train_data, valid_data = train_data.split(
        split_ratio=[0.8, 0.2],
        random_state = random.seed(RANDOM_SEED)
    )

    # BUILD VOCABULARY
    text.build_vocab(train_data, max_size=Vocab_size)
    label.build_vocab(train_data)

    return text, label