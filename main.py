#!/usr/bin/env python

from string import Template
from sys import stderr

import numpy as np
from py_markdown_table.markdown_table import markdown_table
from pyscipopt import Model


def main():
    component_names = ["US equities", "Foreign equities", "Bonds"]
    fund_names = [
        "Whole-world Stock",
        "US Tilt Equity",
        "Strategy 90/10",
        "Ex-US Fund",
        "Bond Fund",
    ]

    initial_holdings, fund_compositions, target_composition = example_problem()
    transactions, final_holdings = rebalance_portfolio(
        initial_holdings, fund_compositions, target_composition
    )

    template_vars = {}

    data = [
        {"Fund": fund_name, "Holding": as_currency(holding)}
        for fund_name, holding in zip(fund_names, initial_holdings)
    ]
    template_vars["current_holdings"] = as_markdown_table(data)

    data = [
        {"Fund": fund_name}
        | {
            component_name: as_percentage(proportion)
            for component_name, proportion in zip(component_names, composition)
        }
        for fund_name, composition in zip(fund_names, fund_compositions.T)
    ]
    template_vars["fund_composition"] = as_markdown_table(data)

    data = [
        {
            "Component": component_name,
            "Current proportion": as_percentage(current_proportion),
            "Target proportion": as_percentage(target_proportion),
        }
        for component_name, current_proportion, target_proportion in zip(
            component_names,
            fund_compositions @ initial_holdings / initial_holdings.sum(),
            target_composition,
        )
    ]
    template_vars["current_composition"] = as_markdown_table(data)

    data = [
        {
            "Fund": fund_name,
            "Current holding": as_currency(initial_holding),
            "Rebalanced holding": as_currency(rebalanced_holding),
            "Different?": "No"
            if abs(initial_holding - rebalanced_holding) < 1e-4
            else "Yes",
        }
        for fund_name, initial_holding, rebalanced_holding in zip(
            fund_names, initial_holdings, final_holdings
        )
    ]
    template_vars["rebalanced_holdings"] = as_markdown_table(data)

    data = [
        {
            "Exchange amount": as_currency(amount),
            "From fund": fund_names[i],
            "To fund": fund_names[j],
        }
        for (i, j), amount in np.ndenumerate(transactions)
        if amount > 1e-4
    ]
    template_vars["transactions"] = as_markdown_table(data)

    markdown_template = Template("""
Currently holding:
$current_holdings

Fund composition:
$fund_composition

Current composition:
$current_composition

Rebalanced holdings:
$rebalanced_holdings

Transactions:
$transactions
""")

    stderr.write(markdown_template.substitute(**template_vars))


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


def as_markdown_table(data: list[dict]) -> str:
    return markdown_table(data).set_params(quote=False).get_markdown()


def as_percentage(s: float) -> str:
    if s < 1e-4:
        return ""
    return f"{round(100 * s)}%"


def as_currency(v: int) -> str:
    if v < 1e-4:
        return ""
    return f"${v:.2f}"


if __name__ == "__main__":
    main()
