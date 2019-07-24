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


def find_weighted_mean(data, weights):
    my_product = [xi * wi for xi, wi in zip(data, weights)]
    return sum(my_product) / sum(weights)


def find_weighted_moment(data, weights, deg):
    my_mean = find_weighted_mean(data, weights)
    if deg is 2:
        sp = 1
    else:
        sp = find_weighted_moment(data, weights, 2)
    my_dist = [(xi - my_mean) / sp for xi in data]
    my_product = [wi * (di ** deg) for wi, di in zip(weights, my_dist)]
    return sum(my_product) / sum(weights)


def find_weighted_skew(data, weights):
    return find_weighted_moment(data, weights, 3)


def find_weighted_kurtosis(data, weights):
    return find_weighted_moment(data, weights, 4) - 3

