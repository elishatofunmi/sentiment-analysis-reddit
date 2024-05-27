import pandas as pd
import torch, pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
%matplotlib inline


df_train = pd.read_csv('dataset.csv')
np.unique(df_train['label'],return_counts = True)

df_samp = df_train.sample(frac=1).groupby('label', sort=False).head(500)
df_samp.shape

np.unique(df_samp['label'], return_counts = True)

# read in all the words
print(len(df_train['comment']))
print(max(len(w) for w in df_train['comment']))
print(df_train['comment'][:2])

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(df_train['comment']).lower())))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

# build the dataset
block_size = 17000 # context length: how many characters do we take to predict the next one?
label_list = ['medical doctor', 'veterinarian', 'others']

label_list_index = {'medical doctor': 0, 'veterinarian': 1, 'others':2}

def build_dataset(df):  
  X, Y = [], []
  for index, row in df.iterrows():
    x, y = row['comment'], row['label']
    x_len = len(x)
    context = [0]* (block_size - x_len)
    context_trans = [stoi[ch] for ch in x.lower()]
    context.extend(context_trans)
    X.append(context)
    Y.append(label_list_index[y])

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

#df_samp
n1 = int(0.6*len(df_samp))
n2 = int(0.8*len(df_samp))
Xtr,  Ytr  = build_dataset(df_samp[:n1])     # 80%
Xdev, Ydev = build_dataset(df_samp[n1:n2])   # 10%
Xte,  Yte  = build_dataset(df_samp[n2:])     # 10%

n_embd = 500 # the dimensionality of the character embedding vectors
n_hidden = 1000 # the number of neurons in the hidden layer of the MLP
output_dim = 3
context_length = 17000

import torch.nn as nn

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
print(model)


# same optimization as last time
max_steps = 5000
batch_size = 8
lossi = []


optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  

for i in range(max_steps):
  
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
  
  # forward pass
  optimizer.zero_grad()
  logits = model(Xb)
  pred_probab = nn.Softmax(dim=1)(logits)
  y_pred = pred_probab.argmax(1)
  loss = F.cross_entropy(logits, y_pred) # loss function, CrossEntropyLoss
  # Backpropogating gradient of loss
  loss.backward()

  # Updating parameters(weights and bias)
  optimizer.step()
  
  # track stats
  if i % 50 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())


plt.plot(torch.tensor(lossi).view(-1, 1).mean(1))


with torch.no_grad():
    correct = 0
    total = 0
    for i in range(int(max_steps/100)):
      ix = torch.randint(0, Xte.shape[0], (1,))
      Xb, Yb = Xte[ix], Yte[ix] # batch X,Y
  
      out = model(Xb)
      _, predicted = torch.max(out, 1)
      total += 1
      correct += (torch.argmax(predicted, dim = 1) == Yb).sum().item()
    print('Testing accuracy: {} %'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0
    for i in range(int(max_steps/100)):
      ix = torch.randint(0, Xdev.shape[0], (1,))
      Xb, Yb = Xdev[ix], Ydev[ix] # batch X,Y
  
      out = model(Xb)
      _, predicted = torch.max(out, 1)
      total += 1
      correct += (torch.argmax(predicted, dim = 1) == Yb).sum().item()
    print('Dev accuracy: {} %'.format(100 * correct / total))


with torch.no_grad():
    correct = 0
    total = 0
    for i in range(int(max_steps/25)):
      ix = torch.randint(0, Xtr.shape[0], (1,))
      Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
  
      out = model(Xb)
      _, predicted = torch.max(out, 1)
      total += 1
      correct += (torch.argmax(predicted, dim = 1) == Yb).sum().item()
    print('Train accuracy: {} %'.format(100 * correct / total))

df_samp.to_csv("samp.csv", index= False)

torch.save(model, 'sentiment_pt.pt')

# save vocab_size
# Open a file and use dump() 
with open('model/lookup_dict.pkl', 'wb') as file: 
    # A new file will be created 
    pickle.dump(stoi, file)


