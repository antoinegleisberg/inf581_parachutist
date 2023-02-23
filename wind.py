# create a class that contains functions that return wind vectors
def constant_wind(x, y):
    """Return a constant wind vector"""
    return [4.0, 0.0]


def linear_wind(x, y):
    """Return a wind vector that is linear in y"""
    return [(250 - y) / 25, 0.0]


class Wind:
    """Implement a spatial representation of the wind"""

    def __init__(self, wind_function=constant_wind):
        self.wind_function = wind_function

    def get_wind(self, x, y):
        return self.wind_function(x, y)
