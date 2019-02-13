import numpy as np

# Must have latex installed.
def clean_poly_eq(coefficients, scientific_notation=True):
    n = len(coefficients)
    degs = list(range(n))
    coefficients = ["{0:.2E}".format(i).split('E') for i in coefficients]
    coefficients.reverse()
    pieces = []
    for ((cof1,cof2), deg) in zip(coefficients, degs):
        if deg == 0:
            if float(cof1) > 0:
                piece = "{0}{{E}}^{{{1}}}".format(cof1, cof2)
            else:
                piece = "{0}{{E}}^{{{1}}}".format(cof1, cof2)

        elif deg == 1:
            if float(cof1) > 0:
                piece = "+{0}{{E}}^{{{1}}}{{x}}".format(cof1, cof2)
            else:
                piece = "{0}{{E}}^{{{1}}}{{x}}".format(cof1, cof2)

        else:
            if float(cof1) > 0:
                piece = "+{0}{{E}}^{{{1}}}{{x}}^{{{2}}}".format(cof1, cof2, deg)
            else:
                piece = "{0}{{E}}^{{{1}}}{{x}}^{{{2}}}".format(cof1, cof2, deg)

        pieces.append(piece)

    pieces.reverse()

    equation = r"\[{y} = " + "".join(pieces[::-1]) + "\]"

    return equation


def get_poly_hat(x_values, y_values, poly_degree):
    coeffs = np.polyfit(x_values, y_values, poly_degree)
    poly_eqn = np.poly1d(coeffs)

    y_bar = np.sum(y_values) / len(y_values)
    ssreg = np.sum((poly_eqn(x_values) - y_bar) ** 2)
    sstot = np.sum((y_values - y_bar) ** 2)
    r_square = ssreg / sstot

    return (coeffs, poly_eqn, r_square)