#!/usr/bin/env python

from typing import NamedTuple

import numpy as np
from pyscipopt import Model, quicksum


def example():
    # Current holdings of five different securities
    holdings = np.array([100, 100, 500, 250, 50])

    # Composition (proportion of each asset type, e.g. stock, bonds, cash) of each security
    fund_compositions = np.array(
        [
            [0.6, 0.9, 0.9, 0.0, 0.0],
            [0.4, 0.1, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.1, 0.0, 1.0],
        ]
    )
    # Desired proportion of each asset type in overall portfolio
    target_composition = np.array([0.75, 0.25, 0.05])

    print(fund_compositions @ holdings)

    return holdings, fund_compositions, target_composition


def rebalance_portfolio(holdings, fund_compositions, target_composition):
    m, n = fund_compositions.shape
    assert (m, n) == (*target_composition.shape, *holdings.shape)

    big_m = holdings.sum()

    target_composition_scaled = big_m * target_composition

    model = Model()

    # x[i, j]:  amount (in currency units) of security i that should be exchange
    # for security j. Nonnegative; x[j, i] gives the other direction.
    x = model.addMatrixVar((n, n), name="x", lb=0, ub=None)

    # z[i, j] = 1 if x[i, j] positive, 0 otherwise
    z = model.addMatrixVar((n, n), name="z", vtype="binary")
    model.addMatrixCons(x <= big_m * z)
    model.setObjective(z.sum(), sense="minimize")

    # y[i]: amount of security i that I hold as a result of these exchanges.
    # Must be nonnegative.
    y = model.addMatrixVar(n, name="y", lb=0, ub=None)
    for i in range(n):
        # Must agree with x as defined
        model.addCons(y[i] == holdings[i] + x[:, i].sum() - x[i, :].sum())

    for j in range(1):
        model.addMatrixCons(fund_compositions[j, :] @ y == target_composition_scaled[j])
    model.optimize()
    sol = model.getBestSol()

    print(sol[x])
    print([model.getVal(y[i]) for i in range(n)])
    return sol


if __name__ == "__main__":
    sol = rebalance_portfolio(*example())
    print(sol)
