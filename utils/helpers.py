

def column_names(gridpoints, colnames):
    # colnames = itertools.product(cfg.data.ar.colnames, cfg.data.ar.gridpoints)
    # colnames = list(map(lambda x: x[0] + str(x[1]), colnames))
    columns = []
    for gp in gridpoints:
        for col in colnames:
            columns.append(col + str(gp))

    return columns
