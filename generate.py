import torch
import model as m
import sample as s


class Generator:
    def __init__(self, input_size, hidden_size, dropout, device, n_layers, path):
        self.input_size = input_size
        self.model = m.Model(input_size, hidden_size, dropout, device, n_layers)
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()

    def generate(self, indx_to_chars, chars_to_indx, max_length=-1):
        return s.sample(
            self.model, self.input_size, indx_to_chars, chars_to_indx, max_length
        )
