#!/usr/bin/env python3

import argparse


def hex_int(n):
    return int(n, 16)


def Parser():
    parser = argparse.ArgumentParser()

    ############################################################################
    # Configuration
    ############################################################################

    main = parser.add_argument_group("Top-level configuration")

    main.add_argument(
        "-a",
        "--check-assertions",
        action="store_true",
    )

    main.add_argument(
        "-e",
        "--expressions",
        dest="expressions",
        type=str,
        action="extend",
        nargs="+",
        default=[],
        help="Relational expressions to print along with every solution "
        "(assuming -f is also set)",
    )

    main.add_argument(
        "-f",
        dest="full_solution",
        action="store_true",
        help="For every solution instance, print the entire model "
        "(default: print just the state)",
    )

    main.add_argument("-i", dest="input_file", type=str, default=None)

    main.add_argument("-n", dest="max_solutions", type=int, default=None)

    main.add_argument(
        "--return-values",
        dest="return_values",
        type=str,
        action="extend",
        nargs="+",
        default=[],
        help="Forced return values, of the form '<thread>:<reg>=<value>'",
    )

    main.add_argument("-p", "--print-options", action="store_true")

    main.add_argument(
        "-v",
        "--verbose",
        action="store",
        nargs="?",
        type=int,
        default=0,
        const=1,
    )
    main.add_argument(
        "--no-verbose", dest="verbose", action="store_false", default=False
    )

    main.add_argument(
        "-s",
        "--verbose-solver",
        action="store_true",
        help="Print more SAT solver information"
        " (e.g., print an UNSAT core when no solution is found)",
    )

    ############################################################################
    # Model settings
    ############################################################################

    model = parser.add_argument_group("Model settings")

    model.add_argument("--allow-misaligned-accesses", action="store_true")
    model.add_argument(
        "--no-allow-misaligned-accesses",
        dest="allow_misaligned_accesses",
        action="store_false",
    )

    model.add_argument("--allow-misaligned-atomics", action="store_true")
    model.add_argument(
        "--no-allow-misaligned-atomics",
        dest="allow_misaligned_atomics",
        action="store_false",
    )

    model.add_argument("--allow-unknown-regs", action="store_true")
    model.add_argument(
        "--no-allow-unknown-regs",
        dest="allow_unknown_regs",
        action="store_false",
    )

    model.add_argument("--arbitrary-initial-values", action="store_true")
    model.add_argument(
        "--no-arbitrary-initial-values",
        dest="arbitrary_initial_values",
        action="store_false",
    )

    model.add_argument("--big-endian", action="store_true")
    model.add_argument(
        "--no-big-endian", dest="big_endian", action="store_false"
    )

    model.add_argument(
        "--hardware-a-d-bit-update", action="store_true", default=True
    )
    model.add_argument(
        "--no-hardware-a-d-bit-update",
        dest="hardware_a_d_bit_update",
        action="store_false",
    )

    model.add_argument("--max-unroll-count", type=int, default=3)

    model.add_argument("--physical-address-bits", type=int, default=16)

    model.add_argument("--rvwmo", action="store_true", default=True)
    model.add_argument("--no-rvwmo", dest="rvwmo", action="store_false")

    model.add_argument("--satp", dest="satp", type=hex_int, default=0)

    model.add_argument("--sc-address-must-match", action="store_true")
    model.add_argument(
        "--no-sc-address-must-match",
        dest="sc_address_must_match",
        action="store_false",
        default=False,
    )

    model.add_argument("--sc-returns-0-or-1", action="store_true")
    model.add_argument(
        "--no-sc-returns-0-or-1",
        dest="sc_returns_0_or_1",
        action="store_false",
        default=False,
    )

    model.add_argument("--start-pc", type=int, default=0x80)

    model.add_argument("--supervisor", action="store_true")
    model.add_argument(
        "--user", dest="supervisor", action="store_false", default=False
    )

    model.add_argument("--use-initial-value-function", action="store_true")

    model.add_argument(
        "--virtual-memory-ordering", action="store_true", default=True
    )
    model.add_argument(
        "--no-virtual-memory-ordering",
        dest="virtual_memory_ordering",
        action="store_false",
    )

    model.add_argument("--xlen", type=int, choices=[32, 64], default=32)

    return parser


def parse_args(args=None):
    if isinstance(args, str):
        args = args.split(" ")
    return Parser().parse_args(args)
