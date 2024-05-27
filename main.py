import streamlit as st
from streamlit_chat import message as msgchat
from openai import OpenAI
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import os, json, requests, torch, pickle
import numpy as np 
import torch.nn as nn



# Session States
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

model_url = "https://drive.google.com/file/d/12Jm-PRrLHF0_l9pNV8oQmCxxCX8kRFT3/view?usp=sharing" # link to the trained sentiment model, sentiment_pt.pt 
label_list = ['medical doctor', 'veterinarian', 'others']
def download_model(model_url): # downloads the pytorch model from google drive.
    cmd = "python3 gdrivedl.py %s -O %s" % (model_url, "model/sentiment_pt.pt")
    os.system(cmd)
    return

def load_vocab_size(pickle_file ="model/lookup_dict.pkl"):
    # Open the file in binary mode 
    with open(pickle_file, 'rb') as file: 
        # Call load method to deserialze 
        return pickle.load(file) 
    
# download model Url
download_model(model_url)

#load stoi
stoi = load_vocab_size()
n_embd = 500 # the dimensionality of the character embedding vectors
n_hidden = 1000 # the number of neurons in the hidden layer of the MLP
output_dim = 3
context_length = 17000

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Embedding(context_length, n_embd),
            nn.Linear(n_embd , n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, output_dim),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# load trained torch model from torch
model = torch.load("model/sentiment_pt.pt")
model.eval()


def encode_message(user_input): # encode the text data from users into numbers.
    block_size = 17000
    x_len = len(user_input[:block_size]) # limit user input to block_size
    context = [0] * (block_size - x_len)
    context_trans = [stoi[ch] for ch in user_input.lower()]
    context.extend(context_trans)
    return np.array(context).reshape(1, -1) # as array


def decode_sentiment(encoded_message): # predict if this is text is that of a medical doctor, veterinarian or others.
    
    Xb = torch.Tensor(encoded_message)

    out = model(Xb)
    _, predicted = torch.max(out, 1)
    val = torch.argmax(predicted, dim = 1).item()

    
    return label_list[val]


def handle_user_input(user_question):
    st.session_state.chat_history.append({'question': user_question})
    encoded_message = encode_message(user_question)
    response = decode_sentiment(encoded_message)
    st.session_state.chat_history.append({'answer': response})
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            msgchat(message['question'], key=str(i), is_user=True)
        else:
            msgchat(message['answer'], key=str(i))




def main():
    user_question = st.text_input("Understand the sentiment of a text as either a medical doctor, veterinarian or others.")


    if user_question:
        handle_user_input(user_question)
    


if __name__ == "__main__":
    main()