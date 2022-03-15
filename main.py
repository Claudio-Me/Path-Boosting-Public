import timeit
from collections import defaultdict
# https://stackoverflow.com/questions/41165664/convert-list-to-dictionary-with-duplicate-keys-using-dict-comprehension
def testing():
    a = ['rosso', 'rosso', 'rosso', 'rosso', 'giallo', 'giallo', 'blu', 'arancione']

    for i in range(23):
        a = a + a

    my_list = list(zip(a, range(len(a))))
    print("start")
    start = timeit.timeit()

    new_dict = defaultdict(list)
    for (key, value) in my_list:
        new_dict[key].append(value)

    end = timeit.timeit()
    print(end - start)


if __name__ == '__main__':
    testing()
