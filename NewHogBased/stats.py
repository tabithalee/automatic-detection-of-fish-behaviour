def find_mean(data):
    return sum(data) / len(data)


def calc_moment(data, degree):
    my_mean = find_mean(data)
    my_list = [((item - my_mean) ** degree) for item in data]
    return sum(my_list) / len(data)


# using fisher-pearson coefficient of skewness
def my_skew(data):
    moment3 = calc_moment(data, 3)
    moment2 = calc_moment(data, 2)
    return moment3 / (moment2 ** (3 / 2))

# using the kurtosis excess formula that should be centered around 0
def my_kurtosis(data):
    moment4 = calc_moment(data, 4)
    moment2 = calc_moment(data, 2)
    return moment4 / (moment2 ** 2)-3
