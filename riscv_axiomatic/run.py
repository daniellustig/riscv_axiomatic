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
from . import execution
from . import solution
from .z3_wrapper import *


def run(args):
    output = ""

    def p(*s):
        nonlocal output
        output += " ".join(map(str, s)) + "\n"

    # Parse the command line arguments
    options = config.parse_args(args)

    # Parse the input file
    def parse(f):
        return assembly.Parser(options).parse(f.read())

    if options.input_file:
        with open(options.input_file, "r") as f:
            title, builder, harts, outcome = parse(f)
    else:
        title, builder, harts, outcome = parse(sys.stdin)

    model = builder.model(verbose=options.verbose > 1)
    verbose_solver = options.verbose or options.verbose_solver

    # Add forced return values
    for f in options.return_values:
        reg, value = [i.strip() for i in f.split("=")]
        thread, reg = [i.strip() for i in reg.split(":")]
        thread = f"P{thread}"
        reg_value = execution.final_register_value(thread, reg, options.xlen)
        model.fact(reg_value == int(value), f"force {thread}:{reg} == {value}")
        p(f"# Forced {thread}:{reg} == {value}")

    # Check assertions
    if options.check_assertions:
        model.check_assertions(verbose=verbose_solver)

    # Print a log of the model options
    if options.verbose or options.print_options or options.full_solution:
        o = "; ".join([f"{k}={v}" for k, v in vars(options).items()])
        p(f"# Options: [{o}]")

    expectation = {
        "exists": "Allowed",
        "forall": "Required",
        "~exists": "Forbidden",
    }[outcome.expectation()]

    # Keep track of the outcomes
    solutions = []
    exists = False
    forall = True
    matches = 0
    nonmatches = 0

    # Set up the loop
    current_solution = True  # just to get the loop started

    # Generate all solutions, up to the provided max count
    while current_solution and (
        not options.max_solutions or len(solutions) < options.max_solutions
    ):
        current_solution = model.solve(verbose=verbose_solver)
        if not current_solution:
            break

        solutions.append(current_solution)

        model.fact(
            outcome.different_state(current_solution),
            f"Different solution {len(solutions)}",
        )

    p(f"Test {title} {expectation}")
    p(f"States {len(solutions)}")

    for current_solution in solutions:
        # for k, v in builder.return_value.items():
        #    p(f"return_value[{k}] = {hex_(current_solution[v])}")
        # for h in harts:
        #    for k in h._canonical_registers:
        #        p(f"{h.name}:{k} = {hex_(current_solution[h._get_register(k)])}")

        if options.full_solution or options.verbose:
            p("# Solution\n#")

        if options.verbose or options.full_solution:
            p(
                solution.print_verbose_solution(
                    builder,
                    model,
                    current_solution,
                    outcome,
                    expressions=options.expressions,
                    verbose=options.verbose,
                )
            )

        s = outcome.state_string(current_solution)
        p(s)

        c = outcome.evaluate_condition(current_solution)
        if options.full_solution or options.verbose:
            p("# Condition satisfied:", c)

        if options.full_solution or options.verbose:
            p("#\n# End solution\n#")

        if true(c):
            exists = True
            matches += 1
        elif false(c):
            forall = False
            nonmatches += 1
        else:
            raise Exception(f"Solver could not determine outcome: {c}")

    result = {
        "exists": "Ok" if exists else "No (allowed not found)",
        "forall": "Ok" if forall else "No (forbidden found)",
        "~exists": "Ok" if not exists else "No (forbidden found)",
    }[outcome.expectation()]
    p(f"{result}")
    p(f"Condition {outcome.expectation()} {outcome.condition_string()}")
    p("Hash=0")

    result = {
        (False, False): "Never",
        (False, True): "Never",
        (True, False): "Sometimes",
        (True, True): "Always",
    }[exists, forall]
    p(f"Observation {title} {result} {matches} {nonmatches}")
    p()

    return output


if __name__ == "__main__":
    print(run(sys.argv[1:]))
