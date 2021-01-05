import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import tokenization as t
import model as m
import sample as s
import pickle
import argparse

parser = argparse.ArgumentParser(description="train the model")
parser.add_argument("--model_checkpoint", type=str, help="")

levels, chars, indx_to_chars, chars_to_indx, n_chars = t.get_everything("levels/*.txt")

with open("character_data.pickle", "wb") as f:
    pickle.dump((chars, indx_to_chars, chars_to_indx, n_chars), f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = m.Model(n_chars, 50, 0.25, device, 2).to(device)

args = parser.parse_args()
if args.model_checkpoint != None:
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.eval()


def random_data(batch_size):
    levels_to_process = []
    x = []
    y = []
    max_length = 0
    for _ in range(batch_size):
        level = random.choice(levels)
        if len(level) > max_length:
            max_length = len(level)
        levels_to_process.append(level)

    for i in range(len(levels_to_process)):
        levels_to_process[i] = levels_to_process[i].ljust(max_length, "@")

    for level in levels_to_process:
        x.append(model.input_process(level, chars_to_indx))
        y.append(model.output_process(level, chars_to_indx))

    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)


def train(input_train, output_train):
    hidden = model.init_hidden(input_train.shape[0])
    model.zero_grad()
    loss = 0
    for i in range(input_train.shape[1]):
        y = output_train[:, i]
        x = input_train[:, i, :]
        x = x.view(x.shape[0], 1, x.shape[1])
        output, hidden = model(hidden, x)
        output = output.squeeze(1)
        l = criterion(output, y)
        loss += l

    loss.backward()
    optimizer.step()

    return output, loss.item() / len(input_train)


n_iters = 5000
print_every = 20
plot_every = 5
all_loses = []
total_loss = 0

for i in range(0, n_iters):
    output, loss = train(*random_data(25))
    total_loss += loss

    if i % print_every == 0:
        print(f"====================EPOCH: {i}=================")
        print(loss)
        print(
            s.sample(model, n_chars, indx_to_chars, chars_to_indx, 255).replace(
                "\n", "°"
            )
        )
        torch.save(model.state_dict(), "model.mod")

    if i % plot_every == 0:
        all_loses.append(loss)

plt.figure()
plt.plot(all_loses)
plt.show()
