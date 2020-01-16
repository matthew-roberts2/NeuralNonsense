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


start_date = 1880
end_date = 2017


def main():
    names = {}
    for i in range(0, end_date - start_date):
        date = start_date + i
        with open('./data/yob{}.txt'.format(date), 'r') as f:
            for line in f:
                name, gender, count = line.strip().split(',')
                count = int(count)
                if name in names:
                    names[name].increase_occurrences(count)
                    names[name].update_gender(gender)
                else:
                    names[name] = NameEntry(name, gender, count)

    print(names['Matthew'])
    print(names['Phillip'])
    names = sorted(names.values(), key=lambda x: x.count, reverse=True)

    for i in range(5):
        print(str(names[i]))

    # with open('./data/all_names.txt', 'w') as f:
    #     for name in names:
    #         f.write('{}\n'.format(str(name)))


main()


























































































































