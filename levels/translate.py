import glob

replace_matrix = {"#": "X", "B": "S", "p": "[", "P": "]"}


def get_paths(path):
    return glob.glob(path)


def read_lines(path):
    level = open(path).read()
    return level


for i in get_paths("*.txt"):
    lines = read_lines(i)
    for key, val in replace_matrix.items():
        lines = lines.replace(key, val)
    new_path = "translate_mario/" + i
    file1 = open(new_path, "w")
    file1.write(lines)
    file1.close()
