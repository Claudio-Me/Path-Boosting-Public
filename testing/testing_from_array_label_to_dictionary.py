from timeit import default_timer as timer
from collections import defaultdict
import numpy as np


# test which is the best method to go from the array "my_list" that contains in the ith position the label for node `i`
# to a dictionary that associates to each label the list of nodes with that label
# seems that the 3rd methods is the fastest

# https://stackoverflow.com/questions/41165664/convert-list-to-dictionary-with-duplicate-keys-using-dict-comprehension


def method_1(my_list):
    new_dict = {}
    for (key, value) in my_list:
        if key in new_dict:
            new_dict[key].append(value)
        else:
            new_dict[key] = [value]


def method_2(my_list):
    new_dict = {}
    for (key, value) in my_list:
        new_dict.setdefault(key, []).append(value)


def method_3(my_list):
    new_dict = defaultdict(list)
    for (key, value) in my_list:
        new_dict[key].append(value)


def testing(method_number, number_of_tests=20, approx_length=67108864):
    base = ['rosso', 'rosso', 'rosso', 'rosso', 'giallo', 'giallo', 'blu', 'arancione']
    a = base
    while len(a) < approx_length:
        a = a + a
    print(len(a))

    my_list = list(zip(a, range(len(a))))
    print("start")

    timing = []

    for i in range(number_of_tests):
        start = timer()

        if method_number == 1:
            method_1(my_list)
        elif method_number == 2:
            method_2(my_list)
        elif method_number == 3:
            method_3(my_list)
        else:
            print("Error, method " + str(method_number) + " not found")
        end = timer()
        # print(end - start)
        timing.append(end - start)

    timing = np.array(timing)
    print("Method " + str(method_number) + " timing: " + str(np.average(timing)))
