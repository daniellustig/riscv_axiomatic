#!/usr/bin/env python3

import argparse
import concurrent.futures
import glob
import random
import subprocess
import sys
import time

import riscv_axiomatic.run as run
import scripts.parse_log as parse_log

global outcomes
outcomes = {}


def comment(s):
    return "\n".join([f"# {i.rstrip()}" for i in s.split("\n")])


def launch(filename):
    args = [
        "python3",
        "-m",
        "riscv_axiomatic",
        "-i",
        filename,
        "--physical-address-bits=16",  # for solver performance
        "--xlen=64",
        "--sc-returns-0-or-1",
        "--max-unroll-count=5",
    ]
    cmd = " ".join(args)

    try:
        proc = subprocess.run(args, timeout=60, capture_output=True, check=True)
    except subprocess.TimeoutExpired as e:
        if e.stdout is not None:
            out = comment(e.stdout.decode())
        else:
            out = ""
        if e.stderr is not None:
            err = comment(e.stderr.decode())
        else:
            err = ""
        return f"# Result: Timeout\n# {cmd}\n{out}\n{err}\n"
    except subprocess.CalledProcessError as e:
        err = comment(e.stderr.decode())
        return f"# Result: Error\n# {cmd}\n{err}\n"
    except Exception as e:
        err = comment(str(e))
        return f"# Result: Exception\n# {cmd}\n{err}\n"

    out = proc.stdout.decode()
    err = comment(proc.stderr.decode())
    mismatch = ""
    try:
        outcome = parse_log.Outcome(proc.stdout.decode(), "proposed")
        test_name = outcome.title
        expected = outcomes[test_name]
        if outcome != expected:
            result = "Mismatch"
            mismatch = comment(expected.mismatch(outcome) + "\n")
        else:
            result = "Success"
    except KeyError:
        result = "Success (no baseline)"
    except Exception as e:
        result = "Match exception"
        mismatch = comment(str(e))

    return f"# Result: {result}\n# {cmd}\n{out}\n{err}\n{mismatch}\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", dest="baseline", type=str, default=None)
    parser.add_argument("-c", dest="threadcount", type=int, default=4)
    parser.add_argument("-l", dest="limit", type=int, default=None)
    parser.add_argument("-p", dest="pathname", type=str, required=True)
    parser.add_argument("-s", dest="shuffle", type=int, default=0)
    parser.add_argument("-t", dest="timeout", type=int, default=60)
    args = parser.parse_args()

    if args.baseline is not None:
        with open(args.baseline, "r") as f:
            outcomes = parse_log.parse_one_file(f, "baseline")
    else:
        outcomes = {}

    files = glob.glob(args.pathname, recursive=True)
    if not files:
        raise Exception(f"No files found at '{args.pathname}'")
    if args.limit:
        files = files[:args.limit]

    # Shuffle
    for i in range(args.shuffle):
        n = random.randint(0, len(files) - 1)
        files = [files[n]] + files[:n] + files[n + 1 :]

    i = 1
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(args.threadcount) as pool:
        for result in pool.map(launch, files):
            elapsed = time.time() - start
            total = elapsed / i * len(files)
            remaining = total - elapsed
            print(
                f"# Progress: {i} / {len(files)}  "
                f"({elapsed:.1f} seconds elapsed, "
                f"estimate {remaining:.1f} seconds == "
                f"{(remaining/60):.1f} minutes remaining)"
            )
            print(result)
            i += 1
