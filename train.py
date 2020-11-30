import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open("/content/drive/MyDrive/PROJECTS/Chatbot Transport Rental Agency/intents.json","r") as f:
  intents = json.load(f)


# Separating required field from dictionary
all_words=[]
tags=[]
x_y=[]

for intent in intents["intents"]:
  t=intent["tag"]
  tags.append(t)
  for i in intent["patterns"]:
    each_word=token_ize(i)
    all_words.extend(each_word)
    x_y.append((each_word,t))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stemming(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))


# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in x_y:
    # X: bag of words for each pattern_sentence
    bag = b_o_w(pattern_sentence, all_words)
    X_train.append(bag)
    # y: Label index
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)



# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.0001
input_size = len(X_train[0])
output_size = len(tags)
print(input_size, output_size)



class Chat_Dataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



dataset = Chat_Dataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ChatNet(input_size, output_size).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  for words,labels in train_loader:

    words=words.to(device)
    labels = labels.to(dtype=torch.long).to(device)

    outputs = model(words)

    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "/content/drive/MyDrive/PROJECTS/Chatbot Transport Rental Agency/data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
