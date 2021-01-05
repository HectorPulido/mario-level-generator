import torch
import numpy as np
import torch.nn.functional as F


def sample(
    model, n_chars, indx_to_chars, chars_to_indx, max_length=-1, start_letter="-"
):
    with torch.no_grad():  # no need to track history in sampling
        input = model.input_process(start_letter, chars_to_indx)
        input = input.view(1, input.shape[0], input.shape[1])
        hidden = model.init_hidden(input.shape[0])

        output_name = start_letter

        while True:
            output, hidden = model(hidden, input)
            p = F.softmax(output, dim=2).data.cpu()
            p, top_ch = p.topk(3)
            top_ch = top_ch.numpy().squeeze()
            p = p.numpy().squeeze()
            topi = np.random.choice(top_ch, p=p / p.sum())
            if topi == chars_to_indx["@"]:
                break
            else:
                letter = indx_to_chars[int(topi)]
                output_name += letter
            if max_length != -1 and len(output_name) > max_length:
                break

            input = model.input_process(letter, chars_to_indx)
            input = input.view(1, input.shape[0], input.shape[1])

        return output_name
