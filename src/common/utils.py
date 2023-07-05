
def get_file_lines(file):
    with open(file) as data_file:
        return list(map(lambda x: x.replace('\n', ' '), data_file.readlines()))
