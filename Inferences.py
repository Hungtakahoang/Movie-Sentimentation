import torch
import spacy
import en_core_web_sm
from Model import RNN
import numpy as np
from DataExtract import dataExtract

def inference_model(sentence):

    # DEFINE NECESSARY VARIABLES
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 2

    text, label = dataExtract()

    model = RNN(input_dim=len(text.vocab), embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=num_classes)

    # LOADING MODEL
    nlp = spacy.blank("en")
    device = torch.device('cpu')
    model = torch.load('../Movie Sentimentation/model/model1.pth', map_location=device)
    def predict_sentiment(model, sentence):
        model.eval()
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        indexed = [text.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        prediction = torch.nn.functional.softmax(model(tensor), dim=1)
        return prediction[0][0].item()

    value_1 = np.multiply(np.round(predict_sentiment(model, sentence), 1), 100)
    value_2 = np.multiply(np.round((1 - predict_sentiment(model, sentence)), 1), 100)
    return value_1, value_2

# value_1, value_2 = inference_model("This is a good movie I have ever seen.")
# print(value_1)