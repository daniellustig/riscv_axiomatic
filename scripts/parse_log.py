#!/usr/bin/env python3

import sys


class Outcome:
    def __init__(self, text, key):
        self.text = text
        self.key = key

        lines = text.split("\n")
        lines = list(filter(lambda x: x and x[0] != "#", lines))

        # Test CoRW1+fence.rw.rws Allowed
        test_lit, self.title, self.expectation = lines[0].strip().split(" ")
        assert test_lit == "Test"

        # States 1
        states_lit, state_count = lines[1].strip().split(" ")
        assert states_lit == "States"
        self.state_count = int(state_count)

        # 0:x5=0; x=1;
        states = []
        for i in range(self.state_count):
            state_contents = lines[2 + i].strip().split(" ")
            state_elements = []
            for i in state_contents:
                key, val = i.rstrip(";").split("=")
                try:
                    val = str(hex(int(val)))
                except ValueError:
                    try:
                        val = str(hex(int(val, 16)))
                    except ValueError:
                        pass
                state_elements.append(f"{key}={val}")
            state = tuple(sorted(state_elements))
            states.append(state)
        self.states = set(sorted(states))

        # No
        self.outcome = lines[2 + self.state_count].strip().split(" ")[0]

        # Witnesses
        # Positive: 0 Negative: 1
        # Condition exists (not (0:x5=0 /\ x=1))
        i = 3 + self.state_count
        while i < len(lines):
            condition_line = lines[i].strip().split(" ")
            if condition_line[0] == "Condition":
                self.condition_type = condition_line[1]
                break
            i += 1

        # Observation CoRW1+fence.rw.rws Never 0 1
        # Time CoRW1+fence.rw.rws 0.00
        # Hash=f3fb407dcff0267ce87c6f3c7a7dbdcc

    def __eq__(self, other):
        return (
            self.title == other.title
            and self.expectation == other.expectation
            and self.state_count == other.state_count
            and self.states == other.states
            and self.outcome == other.outcome
            and self.condition_type == other.condition_type
        )

    def __ne__(self, other):
        return not self == other

    def mismatch(self, other):
        r = ""

        def p(*s):
            nonlocal r
            r += " ".join(map(str, s)) + "\n"

        assert self.title == other.title
        p(f"\nMismatch: {self.title}")
        if self.title != other.title:
            p("title", self.title, other.title)
        if self.expectation != other.expectation:
            p("expectation", self.expectation, other.expectation)
        if self.state_count != other.state_count:
            p("state_count", self.state_count, other.state_count)
        if self.states != other.states:
            p("states")
            a_minus_b = self.states - other.states
            if a_minus_b:
                p(f"{len(a_minus_b)} unique states in {self.key}")
                p(a_minus_b)
            b_minus_a = other.states - self.states
            if b_minus_a:
                p(f"{len(b_minus_a)} unique states in {other.key}")
                p(b_minus_a)
        if self.outcome != other.outcome:
            p("outcome", self.outcome, other.outcome)
        if self.condition_type != other.condition_type:
            p("condition_type", self.condition_type, other.condition_type)

        return r


class Test:
    def __init__(self):
        self.outcomes = {}

    def __setitem__(self, key, value):
        self.outcomes[key] = value

    def check(self, n):
        keys = list(self.outcomes.keys())
        if len(keys) != n:
            print(
                f"Warning: Test {self.outcomes[keys[0]].title} "
                f"has only {len(keys)} keys"
            )

        for i in range(len(self.outcomes)):
            for j in range(i + 1, len(self.outcomes)):
                a = self.outcomes[keys[i]]
                b = self.outcomes[keys[j]]
                if a != b:
                    print(a.mismatch(b))


def parse(tests, f, key):
    text = f.read()
    texts = text.split("\n\n")

    for text in texts:
        if not text.strip():
            continue
        try:
            outcome = Outcome(text, key)
        except Exception as e:
            print("Exception:", e)
            print(text)
            raise e
        title = outcome.title
        tests.setdefault(title, Test())
        tests[title][key] = outcome

    print(f"{len(tests)} tests parsed")
    return tests


def parse_one_file(f, key):
    text = f.read()
    texts = text.split("\n\n")
    tests = {}

    for text in texts:
        if not text.strip():
            continue
        try:
            outcome = Outcome(text, key)
        except Exception as e:
            print("Exception:", e)
            print(text)
            raise e
        title = outcome.title
        tests[title] = outcome

    return tests


def main():
    tests = {}
    for arg in sys.argv[1:]:
        with open(arg, "r") as f:
            parse(tests, f, arg)

    for t in tests.values():
        t.check(len(sys.argv[1:]))


if __name__ == "__main__":
    main()
