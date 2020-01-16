class NameEntry:
    def __init__(self, text, gender, count):
        self.text = text
        self.gender = gender
        self.count = count

    def increase_occurrences(self, count):
        self.count += count

    # Possible Genders: M = Male, F = Female, B = Both
    def update_gender(self, new_gender):
        if self.gender == 'B' or new_gender == self.gender:
            return
        else:
            self.gender = 'B'

    def __str__(self):
        return '{},{},{}'.format(self.text, self.gender, self.count)

    def get_name(self):
        return self.text

def load_names():
    with open('./processed/all_names.txt') as f:
        return [l.strip().split(',')[0].lower() for l in f]


def load_classed_names():
    res = []
    with open('./processed/all_names.txt') as f:
        for line in f:
            name_parts = line.strip().split(',')
            res.append(NameEntry(*name_parts))
    return res

def main():
    print("Loading names")
    names = load_names()
    print("Loaded names")

    print("Checking names for spaces...")
    contains_space = [' ' in name for name in names]

    frequency = sorted(names,key=lambda e:e.)
    result = any(contains_space)

    print("Name lists {} contain names with spaces".format("does" if result else "does not"))


if __name__ == "__main__":
    main()
