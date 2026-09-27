#!/usr/bin/env python

from sys import stderr

import numpy as np
from py_markdown_table.markdown_table import markdown_table
from pyscipopt import Model


def main():
    initial_holdings, fund_compositions, target_composition = example_problem()
    transactions, final_holdings = rebalance_portfolio(
        initial_holdings, fund_compositions, target_composition
    )
    fund_names = [f"Fund {i}" for i in range(5)]
    component_names = ["US equities", "Foreign equities", "Bonds"]

    data = [
        {"Fund": fund_name}
        | {
            component_name: as_percentage(proportion)
            for component_name, proportion in zip(component_names, composition)
        }
        for fund_name, composition in zip(fund_names, fund_compositions.T)
    ]
    write_stderr_table(data)

    pretty_print(initial_holdings, transactions, final_holdings)


def example_problem():
    "Provide typical inputs to `rebalance_portfolio()`."
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

    return holdings, fund_compositions, target_composition


def rebalance_portfolio(holdings, fund_compositions, target_composition):
    """Compute the minimal set of transactions to balance the portfolio.

    Return the matrix of exchanges and resulting holdings.
    """
    m, n = fund_compositions.shape
    assert (m, n) == (*target_composition.shape, *holdings.shape)

    # For use in logic constraints
    big_m = holdings.sum()

    # Scale target composition from proportions to holding units
    target_composition_scaled = big_m * target_composition

    model = Model()

    # x[i, j]: amount (in currency units) of security i that should be exchanged
    # for security j. Nonnegative; x[j, i] gives the other direction.
    x = model.addMatrixVar((n, n), name="x", lb=0, ub=None)

    # z[i, j] = 1 if x[i, j] positive, 0 otherwise; thus z.sum() is the number
    # of transactions.
    z = model.addMatrixVar((n, n), name="z", vtype="binary")
    model.addMatrixCons(x <= big_m * z)

    # Objective: minimize number of transactions
    model.setObjective(z.sum(), sense="minimize")

    # y[i]: amount of security i that I hold as a result of these exchanges.
    # Must be nonnegative.
    y = model.addMatrixVar(n, name="y", lb=0, ub=None)
    for i in range(n):
        # y agrees with x as defined
        model.addCons(y[i] == holdings[i] + x[:, i].sum() - x[i, :].sum())
    for j in range(1):
        # Composition achieved by y agrees with target mix
        model.addMatrixCons(fund_compositions[j, :] @ y == target_composition_scaled[j])

    model.optimize()
    if (status := model.getStatus()) == "optimal":
        return model.getVal(x), model.getVal(y)
    raise RuntimeError(f"SCIP returned inoptimal solver status {status!r}")


def pretty_print(initial_holdings, x, y, security_names=None):
    "Print initial holdings, transactions, and final holdings."
    (n,) = initial_holdings.shape
    assert y.shape == (n,)
    assert x.shape == (n, n)
    epsilon = 1e-8 * y.sum()

    if not security_names:
        security_names = [f"security {i}" for i in range(n)]

    print("Initial holdings:")
    for (i,), value in np.ndenumerate(initial_holdings):
        if value < epsilon:
            continue
        name = security_names[i]
        print(f"  {value:>8.2f} of {name}")

    print("Transactions:")
    for (i, j), value in np.ndenumerate(x):
        if value < epsilon:
            continue
        left = security_names[i]
        right = security_names[j]
        print(f"  Exchange {value:.2f} of {left} for {right}")

    print("Final holdings:")
    for (i,), value in np.ndenumerate(y):
        if value < epsilon:
            continue
        name = security_names[i]
        print(f"  {value:>8.2f} of {name}")


def write_stderr_table(data: list[dict]) -> None:
    stderr.write(markdown_table(data).set_params(quote=False).get_markdown())
    stderr.write("\n\n")


def as_percentage(s: float) -> str:
    if s < 1e-4:
        return ""
    return f"{round(100 * s)}%"


if __name__ == "__main__":
    main()
