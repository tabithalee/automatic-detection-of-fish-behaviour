import numpy as np


def find_abs_array_max(my_array):
    my_array = np.absolute(my_array)
    return my_array.tolist().index(max(my_array))


def find_peak_duration(my_array, sample_interval):
    my_max_index = find_abs_array_max(my_array)
    my_extrema = my_array[my_max_index]

    # find left endpoint
    index_offset = 1
    left_count = 1
    prev_diff = my_extrema - my_array[my_max_index-index_offset]
    next_diff = my_array[my_max_index - index_offset - 1] - my_array[my_max_index-index_offset]

    while np.sign(prev_diff) == np.sign(next_diff):
        index_offset += 1
        left_count += 1
        prev_diff = next_diff
        next_diff = my_array[my_max_index - index_offset - 1] - my_array[my_max_index - index_offset]

    # find right endpoint
    index_offset = 1
    right_count = 1
    if (my_max_index + index_offset) < len(my_array.tolist()):
        prev_diff = my_extrema - my_array[my_max_index + index_offset]
        next_diff = my_array[my_max_index + index_offset] - my_array[my_max_index + index_offset + 1]

        while np.sign(prev_diff) == np.sign(next_diff):
            index_offset += 1
            right_count += 1
            prev_diff = next_diff
            if (my_max_index + index_offset + 1) < len(my_array.tolist()):
                next_diff = my_array[my_max_index + index_offset + 1] - my_array[my_max_index + index_offset]
            else:
                break

    return left_count + right_count










if __name__ == '__main__':
    print('hello')
