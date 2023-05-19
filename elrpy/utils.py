def get_dims(group_data):
    group_Xs, group_Ys, group_Ns = group_data
    k = len(group_Ns)
    d = next(iter(group_Xs.values())).shape[1]
    p = next(iter(group_Ys.values())).shape[0]
    return k, d, p

