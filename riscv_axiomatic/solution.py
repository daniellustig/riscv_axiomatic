#!/usr/bin/env python3

# Built-in modules
import argparse
import io
import sys

# Standard modules
import tabulate  # pip install tabulate
import z3  # pip install z3-solver

# Local modules
from . import assembly
from . import config
from .relation import RelationBuilder, Singleton
from .z3_wrapper import *


class Outcome:
    """The final condition(s) that a model will be checked against.

    An `Outcome` contains two parts:
    1) A named expression that evaluates to true or false for each instance.
       This expression comes with a stated expectation that it holds for at
       least one outcome (exists), holds for no outcome (~exists), or holds for
       all outcomes (forall)
    2) a list of individual sub-expressions that evaluate to anything, and
       which collectively make up the definition of a unique outcome.
    """

    def __init__(
        self,
        condition_string,
        condition_expression,
        expectation,
        state_elements,
        variables,
        addresses=None,
    ):
        self._condition_string = condition_string
        self._condition_expression = condition_expression
        if expectation not in ["exists", "forall", "~exists"]:
            raise Exception(f"Unexpected expectation type '{expectation}'")
        self._expectation = expectation
        self._state_elements = state_elements
        self._variables = dict(variables)
        if addresses:
            self._addresses = addresses
        else:
            self._addresses = set()

    def add_state(self, k, v):
        self._state_elements[k] = v

    def condition_string(self):
        return self._condition_string

    def expectation(self):
        return self._expectation

    def evaluate_condition(self, solution):
        return solution[self._condition_expression]

    def mark_address(self, k):
        self._addresses.add(k)

    def is_address(self, k):
        return k in self._addresses

    def state_string(self, solution):
        state = []
        for k, v in self._state_elements.items():
            if k in self._addresses:
                match = "<Other>"
                for variable, expr in self._variables.items():
                    if true(solution[v] == solution[expr]):
                        match = variable
                        break
                state.append(f"{k}={match};")
            else:
                state.append(f"{k}={hex_(solution[v])};")
        return " ".join(sorted(state))

    def different_state(self, solution):
        conjuncts = []
        for k, v in self._state_elements.items():
            if k in self._addresses:
                v_conjuncts = []
                for variable, expr in self._variables.items():
                    if true(solution[v] == solution[expr]):
                        v_conjuncts.append(v == expr)
                    else:
                        v_conjuncts.append(v != expr)
                conjuncts.append(z3.Not(z3.And(v_conjuncts)))
            else:
                conjuncts.append(z3.Not(v == solution[v]))
        return z3.Or(conjuncts)

    def expectation_met(self, exists, forall):
        if self._expectation == "exists":
            return exists
        elif self._expectation == "forall":
            return forall
        elif self._expectation == "~exists":
            return not exists
        else:
            assert False


def topsort(name, s, r):
    # Algorithm from Cormen, Leiserson, Rivest, Stein, via Wikipedia :)

    # Create a dictionary representation of `r`
    edges = {}
    for k, v in r.items():
        if len(k) != 2:
            raise Exception("Expected an arity-2 relation")

        # Occasionally the solver chooses not to assign a variable, I guess
        # because it doesn't affect the solution.  Exclude such edges, because
        # otherwise we might end up with artificial cycles (e.g., for the
        # relation {(a, b): c, (b, a): Not(c)}).
        if not true(v):
            continue

        edges.setdefault(k[0], [])
        edges[k[0]].append(k[1])

    permanent = set()
    temporary = set()
    unmarked = set()
    for i in s:
        assert isinstance(i, tuple)
        unmarked.add(i[0])
    order = []

    def visit(n):
        if n in permanent:
            return

        if n in temporary:
            print(f"# WARNING: Cycle in {name} relation {temporary}")
            return

        temporary.add(n)

        for adj in edges.get(n, []):
            visit(adj)

        temporary.remove(n)
        permanent.add(n)
        order.append(n)

    while unmarked or temporary:
        n = unmarked.pop()
        visit(n)

    return order[::-1]


def print_verbose_solution(
    builder,
    model,
    solution,
    outcome,
    prefix="# ",
    expressions=[],
    verbose=False,
):
    s = ""

    s += (
        "gmo:"
        + str(topsort("gmo", solution["MemoryOp"], solution["gmo"]))
        + "\n"
    )

    if verbose:
        s += "All relations:\n"
        for k, v in model.items():
            s += f"{k} = {solution[v]}\n"

    for e in expressions:
        s += f"Evaluation of {e}: {solution[e]}\n"

    # Group implicit accesses with the associated explicit access,
    # in cases of thread interleaving, i.e., from sfence.vma to walk1/0
    #
    #            A
    #           /|\
    #       /--- | \ po
    # B_walk1    |  \
    #     |      |   sfence.vma (via IPI)
    # B_walk0    |  /
    #       \--- | / po
    #           \|/
    #            B

    print_order = topsort(
        "print_order",
        solution["Op"],
        solution["^(po + implicit_program_order + po.^~translation_order)"],
    )

    for k, v in outcome._variables.items():
        s += f"{k} = {hex_(solution[v])}\n"

    def _address(i):
        if i in builder.base_address:
            base = builder.base_address[i]
        else:
            return ""
        if base is None:
            return ""
        base = solution[base]
        a = hex_(base)
        for k, v in outcome._variables.items():
            if true(resize(solution[v], base.size()) == base):
                if isinstance(k, int):
                    continue
                break
        return f"Address={a}"

    def _value(i):
        if i in builder.return_value:
            v = builder.return_value[i]
        elif i in builder.write_value:
            v = builder.write_value[i]
        else:
            return ""
        return f"Value={hex_(solution[v])}"

    def _inst(i):
        meta = builder.meta[i]
        return [
            i,
            str(meta),
            _address(i),
            _value(i),
        ]

    def _hart(hart):
        t = [[f"Hart {hart[0]}:"]]
        h = Singleton(hart)
        h_events = h.join(solution["hart_ops"])
        if not h_events:
            return []
        for i in print_order:
            r = Singleton(i)
            if not true(r.in_(h_events)):
                continue
            t.append(_inst(i))
        return t

    t = []
    for h in solution["Thread"]:
        t += _hart(h)
    s += tabulate.tabulate(t) + "\n"

    return "\n".join([prefix + i for i in s.split("\n")])
