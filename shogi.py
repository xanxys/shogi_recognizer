import warnings


all_types = [
    'FU', 'KY', 'KE', 'GI', 'KI', 'KA', 'HI',
    'TO', 'NY', 'NK', 'NG', 'UM', 'RY', 'OU']


def get_initial_configuration_with_dir():
    """
    Return densely populated {key: cell}
        where
            key: 2-letter (e.g. "11", "91")
            cell: {"state": state, "type": type}
            state: "empty", "up", or "down"
            type: "FU" ... "OU"
    """
    initial_state_top = {
        (1, 1): "KY",
        (2, 1): "KE",
        (3, 1): "GI",
        (4, 1): "KI",
        (5, 1): "OU",
        (6, 1): "KI",
        (7, 1): "GI",
        (8, 1): "KE",
        (9, 1): "KY",
        (2, 2): "KA",
        (8, 2): "HI",
        (1, 3): "FU",
        (2, 3): "FU",
        (3, 3): "FU",
        (4, 3): "FU",
        (5, 3): "FU",
        (6, 3): "FU",
        (7, 3): "FU",
        (8, 3): "FU",
        (9, 3): "FU",
    }
    config = {}
    for ix in range(1, 10):
        for iy in range(1, 10):
            key = "%d%d" % (ix, iy)
            config[key] = {
                "state": "empty",
                "type": "empty"
            }
    for ((x, y), ty) in initial_state_top.items():
        key_gote = "%d%d" % (x, y)
        key_sente = "%d%d" % (10 - x, 10 - y)
        config[key_gote] = {
            "state": "down",
            "type": ty
        }
        config[key_sente] = {
            "state": "up",
            "type": ty
        }
    return config


def get_initial_configuration():
    """
    Return (pos, type)
    pos: (1, 1) - (9, 9)
    type will be 2-letter strings like CSA format.
    (e.g. "FU", "HI", etc.)
    """
    warnings.warn(
        """get_initial_configuration() returns ambiguous cell state.
        Use get_initial_configuration_with_dir() instead.""",
        DeprecationWarning)

    initial_state_top = {
        (1, 1): "KY",
        (2, 1): "KE",
        (3, 1): "GI",
        (4, 1): "KI",
        (5, 1): "OU",
        (6, 1): "KI",
        (7, 1): "GI",
        (8, 1): "KE",
        (9, 1): "KY",
        (2, 2): "KA",
        (8, 2): "HI",
        (1, 3): "FU",
        (2, 3): "FU",
        (3, 3): "FU",
        (4, 3): "FU",
        (5, 3): "FU",
        (6, 3): "FU",
        (7, 3): "FU",
        (8, 3): "FU",
        (9, 3): "FU",
    }
    initial_state = {}
    for (pos, ty) in initial_state_top.items():
        x, y = pos
        initial_state[pos] = ty
        initial_state[(10 - x, 10 - y)] = ty
    return initial_state
