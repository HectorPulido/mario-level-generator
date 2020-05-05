import pickle
import torch
import generate as g

with open("character_data.pickle", 'rb') as f:
     (chars, indx_to_chars, chars_to_indx, n_chars) = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = n_chars
hidden_size = 50
dropout = 0.25
n_layers = 2
path = "model.mod" 
gen = g.Generator(input_size, hidden_size, dropout, device, n_layers, path)

print(gen.generate(indx_to_chars, chars_to_indx, -1))