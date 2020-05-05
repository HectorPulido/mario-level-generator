import glob


def get_paths(path):
    return glob.glob(path)


def read_lines(path):
    replacement_matrix = {
        "<": "p",
        ">": "p",
        "[": "p",
        "]": "p",
        "B": "b",
        "o": "C",
    }

    level = open(path).read().strip()
    for key, val in replacement_matrix.items():
        level = level.replace(key, val)
    level = level.split("\n")

    level_complete = ""
    for i in range(len(level[0])):
        for j in level:
            level_complete += j[i]
        level_complete += "\n"
    return level_complete + "@" # @ is EOL 


def all_levels(paths):
    paths = get_paths(paths)
    levels = []
    for path in paths:
        levels.append(read_lines(path))
    return levels


def tokenization(data):
    # get all levels
    complete_data = ""
    for d in data:
        complete_data += d

    chars = tuple(set(complete_data))
    indx_to_chars = dict(enumerate(chars))
    chars_to_indx = {v: k for k, v in indx_to_chars.items()}

    return chars, indx_to_chars, chars_to_indx


def get_everything(path):
    levels = all_levels(path)
    chars, indx_to_chars, chars_to_indx = tokenization(levels)
    n_chars = len(chars)  # EOL

    return levels, chars, indx_to_chars, chars_to_indx, n_chars
