import csv
import math
import numpy as np

from operator import itemgetter


def save_txt(mat, name):
    np.savetxt(name, mat, delimiter=",", fmt="%.2e")


def card_expiry_to_full_time(str):
    return "20" + str[2:4] + "/" + str[0:2] + "/01"


def check_and_update_list(list, id):
    try:
        return list.index(id)
    except ValueError:
        list.append(id)
        return len(list) - 1


# count, mean, E[x**2], max, min
def collect_statistics(ref, value):
    count = ref[0] + 1
    ref[1] = (ref[1] * (count - 1.0) + value) / count
    ref[2] = (ref[2] * (count - 1.0) + value**2) / count

    if count is 1:
        ref[3] = value
        ref[4] = value
    else:
        if value > ref[3]:
            ref[3] = value
        if value < ref[4]:
            ref[4] = value

    ref[0] = count
    return value


def normalize(value, ref, mode=0):
    return (value - ref[4]) * 1.0 / (ref[3] - ref[4])


def get_start_stat():
    return [0, 0, 0, 0, 0]


def zeropad_2_digits(str):
    if len(str) is 0:
        return "00"
    elif len(str) is 1:
        return "0" + str
    else:
        return str


def to_one_hot(num, bin_size):
    if bin_size <= 1:
        return num
    else:
        out = np.zeros((bin_size), dtype=np.float32)
        out[num] = 1
        return out


def sort_tuples(tuples):
    return sorted(tuples, key=itemgetter(0))


def sequence_summarize(sorted_transactions):
    out = []
    temp = [0, 0, 0]
    for t in sorted_transactions:
        if t[1] == temp[1] and t[2] == temp[2]:
            temp[0] = temp[0] + t[0]
        else:
            out.append(temp)
            temp = [t[0], t[1], t[2]]
    out.append(temp)
    return out[1:]


if __name__ == "__main__":

    print(card_expiry_to_full_time("0101"))
    stat = util.get_start_stat()
    stat2 = util.get_start_stat()
    collect_statistics(stat, 10)
    collect_statistics(stat2, 0.1)
    collect_statistics(stat, 0.1)
    print(stat, stat2)

    print(sort_tuples([(0, 'a'), (2, 'b'), (1, 'c')]))
    print(sequence_summarize([[1, '0', 'a'], [2, '0', 'a'], [3, '0', 'b'], [4, '1', 'b'], [5, '1', 'b'], [6, '1', 'c']]))
