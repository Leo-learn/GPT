# Importing the Libraires
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------------

# Importing the dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

#  read it in to inspect the file
with open('input.txt', 'r', encoding="utf-8") as f:
  text = f.read()
print("The length of chararcters in the text: ", len(text))



# Creating our vocabullary and vocab_size and seperating each unique character
chars = sorted(list(set(text)))
vocab_size = len(chars) # possible characters that the model can see or emit

# Tokenization -- mapping unique characters to integer
stoi = { ch:i for i,ch in enumerate(chars)} # characters:integers :: this is creation a look up table
itos = { i:ch for i,ch in enumerate(chars)} # integer:characters
encode = lambda s: [stoi[c] for c in s] # encoder: take string, output list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # encoder: take string, output list of integers

# Train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # The first 90% will be train data rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
  # generate a small batch of data of input x and target y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) # this means will get random chunks of the dataset
  x = torch.stack([data[i:i+block_size] for i in ix]) # taking all the 1D tensors and stacking them up 
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device) # when we load the data we move it to device if cuda available
  return x, y


@torch.no_grad() # this is a CONTEXT MANAGER that disables gradient calculation for whatever happens in this function 
def estimate_loss(): # averages out the loss over multiple batches
  out = {}
  model.eval() # setting the model to evaluation phase
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train() # setting the model back to training phase
  return out


class BigramLanguageModel(nn.Module): # Bigram subclass of the nn.module

  def __init__(self, vocab_size):
    super().__init__()
    # eahc token directly reads of the logits
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    # nn.Embedding is just a wrapper

  def forward(self, idx, targets=None):

    # idx integers are both (B,T) tensors of integers
    logits = self.token_embedding_table(idx) # (B,T,C)
    if targets is None:
      loss= None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C) # this is reshaping the tensor by reducing dimensionality and maintaining the shape of it
      targets  = targets.view(B*T) # making the targets one dimension :: alternatively we can have targets.view(-1) 
      # we use the negative log likelihood loss
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  
  def generate(self, idx, max_new_tokens): # idx is the currenmt context of characters in some batch 
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      # get the predictions
      logits, loss = self(idx)
      # focus only on the last time step
      logits = logits[:, -1, :] # become (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=1) #(B,C)
      # sample from the distributon
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the runnig sequence
      idx = torch.cat((idx, idx_next) , dim=1) # (B, T+1)
    return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device) # moving the model parameters to the device

# create the PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)# zeroing out all the gradients from previous step
    loss.backward()# getting gradients for a;; paarmeters
    optimizer.step() # then using those gradient to update the parameters

# generate from the model 
context = torch.zeros((1,1), dtype=torch.long, device=device) # the zero is how we are going to kick off the generation
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
