import random
import json

import torch

from model import ChatNet
from nltk_utils import b_o_w, token_ize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('C:/Users/Ankan/Desktop/Chatbot/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "C:/Users/Ankan/Desktop/Chatbot/cpu_data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = ChatNet(input_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sammy"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = token_ize(sentence)
    X = b_o_w(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.6:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")