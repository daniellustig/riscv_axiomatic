#!/usr/bin/env python3

# Standard modules
import z3  # pip install z3-solver

# Local modules
from . import alloy
from .z3_wrapper import *


class Relation(dict):
    """A Relation.

    This is a Relation within a relational model of the type that herd, Alloy,
    or isla-axiomatic would use.  Keys have type `tuple` of `str`, and values
    are boolean expressions (either python `bool` or `z3.BoolRef`).

    Relations are not created directly, but instead are created via
    `RelationBuilder` instances.
    """

    def __init__(self):
        """Creates an empty relation.

        Use a `RelationBuilder` to create a non-empty relation.
        """

        super().__init__
        self.arity = None

    def get(self, k, d=False):
        """Like `dict.get()`, except the default is `False` instead of `None`."""

        return super().get(k, d)

    def _set(self, k, v):
        """An internal API used for actually building relations.

        Not meant to be used from outside the module.
        """

        # Typechecking
        if not isinstance(k, tuple):
            raise TypeError(k)
        if not isinstance(v, z3.BoolRef) and not isinstance(v, bool):
            raise TypeError(v)

        # Make sure the arity of `k` matches the arity of other keys in self
        if self.arity is None:
            self.arity = len(k)
        elif self.arity != len(k):
            raise TypeError(
                f"`Relation` has arity {self.arity}, "
                f"but adding a key of arity {len(k)}"
            )

        # No use adding `v` if it is known to be false.
        if v is True:
            # Convert Python type to z3 type
            super().__setitem__(k, z3.BoolVal(True))
        elif v is not False:
            # v = simplify(v)  # slower, but generates simpler expressions
            if not z3.is_false(v):
                super().__setitem__(k, v)

    def __setitem__(self, k, v):
        """Use `update()` instead."""

        raise Exception("setting elements in a Relation not permitted")

    def __str__(self):
        if not self:
            return "{}"
        s = "{\n"
        for k, v in self.items():
            v = simplify(v)
            if not false(v):
                s += f"  {k}: {' '.join([i.strip() for i in str(v).split()])}\n"
        s += "}"
        return s

    def simplify(self):
        """Return a version of the Relation with all variables simplified"""

        r = Relation()
        for k, v in self.items():
            v = simplify(v)
            r._set(k, v)
        return r

    def contains(self, other):
        """`self` contains all elements of `other`"""

        return other.in_(self)

    def cross(self, other):
        """Cross product of `self` and `other`"""

        r = Relation()
        for k1, v1 in self.items():
            for k2, v2 in other.items():
                r._set(k1 + k2, z3.And(v1, v2))
        return r

    def __eq__(self, other):
        """`self` and `other` are equal"""
        return z3.And(self.contains(other), other.contains(self))

    def iden(self):
        """The identity mapping on elements of `self`"""

        r = Relation()
        for k, v in self.items():
            if len(k) != 1:
                raise TypeError(
                    "`iden()` can be called only on relations of arity 1"
                )
            r._set(k + k, v)
        return r

    def if_(self, c):
        """If `c`, then self; else, an empty relation."""

        r = Relation()
        for k, v in self.items():
            r._set(k, z3.And(c, v))
        return r

    def in_(self, other):
        """`self` is contained within `other`"""

        conjuncts = []
        for k, v in self.items():
            conjuncts.append(z3.Implies(v, other.get(k)))
        return z3.And(conjuncts)

    def intersect(self, other):
        """The set intersection of `self` and `other`"""

        r = Relation()
        for k, v in self.items():
            if k in other:
                r._set(k, z3.And(v, other.get(k)))
        return r

    def irreflexive(self):
        """`self` is irreflexive: it has no elements `(a, a)` for any `a`"""

        conjuncts = []
        for k, v in self.items():
            if len(k) != 2:
                raise Exception(
                    "`irreflexive()` requires a `Relation` of arity 2"
                )
            if k[0] == k[1]:
                conjuncts.append(z3.Not(v))
        return z3.And(conjuncts)

    def join(self, other):
        """Return the relational join of `self` with `other`.

        Example: if
        ```
        a = {('a', 'b'): v1, ('a', 'c'): v2}
        b = {('b', 'd'): v3, ('c', 'd'): v4, ('a', 'd'): v5}
        ```
        Then:
        ```
        a.join(b) == {('a', 'd'): Or(And(v1, v3), And(v2, v4))}
        ```
        """

        # Preprocess `other` by sorting its elements into a dict, where keys in
        # the dict are the first atom in each key of `other`
        other_dict = {}
        for k, v in other.items():
            other_dict.setdefault(k[0], {})[k[1:]] = v

        # Find all pairs where the last atom of a member of `self` is the same
        # as the first atom of a member of `other`.
        r = RelationBuilder()
        for k1, v1 in self.items():
            for k2, v2 in other_dict.get(k1[-1], {}).items():
                r.update(k1[:-1] + k2, z3.And(v1, v2))

        return r.relation()

    def __ne__(self, other):
        return z3.Not(self == other)

    def no(self):
        """`self` contains no members"""
        return z3.Not(z3.Or(list(self.values())))

    def some(self):
        """`self` contains at least one member"""
        return z3.Or(list(self.values()))

    def __sub__(self, other):
        """Set difference"""
        r = Relation()
        for k, v in self.items():
            v2 = other.get(k)
            # Optimization: for large expressions, the z3 simplify() function
            # can't always statically simplify And(a, Not(a)) into False
            if v == v2:
                continue
            r._set(k, z3.And(v, z3.Not(v2)))
        return r

    def transitive(self):
        """`self` is transitive"""
        return self.join(self).in_(self)

    def transpose(self):
        """The transpose of `self`"""
        r = Relation()
        for k, v in self.items():
            r._set(k[::-1], v)
        return r

    def transitive_closure(self):
        """The transitive closure of self.

        Transitive closure cannot be produced in general with first-order
        logics, but since we are working only with finite relations here, we
        can just calculate it with no issue.
        """

        # Figure out how many atoms are touched by `self`.
        nodes = set()
        for k, v in self.items():
            if len(k) != 2:
                raise Exception(
                    "transitive closure requires a relation of arity 2"
                )
            nodes.add(k[0])
            nodes.add(k[1])

        # self + self.self + self.self.self + ... + self.<n-2 times>.self
        # is sufficient to produce the transitive closure for a set of edges
        # connecting n nodes.  (self + self.self)^m for 2^m >= n is a
        # conservative way to produce this with less calculation than manually
        # producing all n join patterns in the first formula.
        n = 1
        r = self
        while n < len(nodes):
            r = r.union(r.join(r))
            n <<= 1

        return r

    def union(self, other):
        """Set union"""

        r = RelationBuilder()
        for k, v in self.items():
            r.update(k, v)
        for k, v in other.items():
            r.update(k, v)
        return r.relation()


class RelationBuilder(dict):
    """A class that produces `Relation`s.

    Each instance starts empty, but elements can be added using `update()`.  An
    instance can be converted into a `Relation` using `relation()`.
    """

    def __init__(self):
        """Create an empty RelationBuilder.

        No arguments are needed; members are added using `update()`.
        """

        # Implementation detail: the dict part of self stores all values as a
        # list, and z3.Or() of each list is used when creating the final
        # relation in `relation()`.  This just makes for less nesting of z3.Or
        # calls for keys that are updated many times.

        super().__init__()
        self.arity = None

    def __setitem__(self, k, v):
        """Use `update()` instead."""

        raise Exception("use `update()` instead of `__setitem__()`")

    def update(self, k, v):
        """If `k` is not already a key in the relation being built within
        `self`, add `k: v` to the relation.  If `k` is already a key, then
        replace it with `k: z3.Or(previous_value, v)`.
        """

        # Typechecking the key
        if not isinstance(k, tuple):
            raise TypeError(f"expected `k` to be a tuple, but got {k}")
        for i in k:
            if type(i) != str:
                raise TypeError(
                    "expected each element of `k` to be a str, "
                    f"but got {type(i)} {i}"
                )

        # Typechecking the value
        if not isinstance(v, bool) and not isinstance(v, z3.BoolRef):
            raise TypeError(f"expected `v` to be a boolean type, but got {v}")

        # Making sure the arity matches other elements of the relation
        if self.arity is None:
            self.arity = len(k)
        elif self.arity != len(k):
            raise TypeError(
                f"`RelationBuilder` has arity {self.arity}, "
                f"but adding a key of arity {len(k)}"
            )

        # See note in __init__ about how values are stored
        super().setdefault(k, []).append(v)

    def relation(self):
        """Return the `Relation` built up through `update()` calls."""

        # See note in __init__ about how values are stored
        r = Relation()
        for k, v in self.items():
            if len(v) == 1:
                r._set(k, v[0])
            else:
                r._set(k, z3.Or(v))
        return r


def Singleton(k, v=True):
    """Create a relation with a single element `k`, with condition `v`.

    If Not(v), then the created relation will be empty.

    This can be done with `RelationBuilder`s, but would be more verbose.
    """

    r = Relation()
    if isinstance(k, str):
        k = (k,)
    r._set(k, v)
    return r


################################################################################


class Solution:
    """One valid instance of a solved `RelationalModel`.

    Not meant to be created directly by users.  Get one by calling `solve()` or
    `solutions()` on a `RelationalModel`.
    """

    def __init__(self, relational_model, z3_model):
        self.relational_model = relational_model
        self.z3_model = z3_model

    def __getitem__(self, e):
        """Evaluate expression `e` within this `Solution`.

        If `e` is a boolean or bitvector expression, evaluate it.

        If `e` is a Relation, create a new `Relation` in which all key values
        are replaced with their values evaluated within `self`.

        If `e` is a string, evaluate it as an Alloy expression.
        """

        if isinstance(e, bool):
            return e
        elif isinstance(e, z3.BoolRef) or isinstance(e, z3.BitVecRef):
            return self.z3_model.eval(e)
        elif isinstance(e, Relation):
            r = RelationBuilder()
            for k, v in e.items():
                r.update(k, self.z3_model.eval(v))
            return r.relation()
        elif isinstance(e, str):
            return self[self.relational_model[e]]
        else:
            raise TypeError(f"Cannot evaluate {type(e)} {e}")

    def __str__(self):
        s = ""
        for k, v in self.relational_model.items():
            s += f"{k} = {self[v]}\n"
        return s


class RelationalModel(dict):
    "A relational model, in the style of herd, Alloy, or isla-axiomatic."

    def __init__(self):
        self._relations = {}
        """A mapping from names to `Relation`s"""

        self._parser = alloy.Parser(self)
        """A parser of Alloy language expressions"""

        self._facts = []
        """The facts added to the model"""

        self._assertions = []
        """The assertions added to the model"""

        self._atoms = Relation()
        """The list of all known atoms.  Used for iden() calculation"""

        self._parse_cache = {}
        """A cache of results of prior calls to __getitem__()"""

    def __getitem__(self, e):
        """Parse `e` as an Alloy expression evaluated within the model"""

        if not isinstance(e, str):
            raise TypeError

        # First check the cache
        try:
            return self._parse_cache[e]
        except KeyError:
            pass

        # Then actually parse it as an expression
        r = self._parser.parse(e)

        # ...and cache it
        self._parse_cache[e] = r
        return r

    def __setitem__(self, k, v):
        """Add `k` to the model with value `v`.

        Raise a `KeyError` if `k` is already present in `self`.
        """

        if k in self:
            raise Exception(f"cannot overwrite relation {k}")

        if k in self:
            raise KeyError(f"Overwriting {k}")

        # Actually add it to the dict
        super().__setitem__(k, v)

        # Update "iden" if we are seeing new keys for the first time
        if isinstance(v, Relation):
            r = RelationBuilder()
            for k in v.keys():
                for i in k:
                    r.update((i,), True)
            self._atoms = self._atoms.union(r.relation())
            super().__setitem__("iden", self._atoms.iden())
        else:
            raise TypeError(f"{type(v)}")

    def __str__(self):
        s = ""
        for name, r in self.items():
            s += f"{name} = {r}\n"
        return s

    def _add(self, fact, f, name, suppress_warnings=False):
        """Add `f` as a model fact (`fact`=True) or assertion (`fact`=False)).

        If `name` is None, then a name will be created.
        """

        # If passed a string, parse it into a formula
        if isinstance(f, str):
            if name is None:
                name = f
            f = self[f]

        # Create a name if one wasn't provided
        if name is None:
            if isinstance(f, str):
                name = f
            elif fact:
                name = f"_{len(self._facts)}"
            else:
                name = f"_{len(self._assertions)}"

        # Typechecking
        if not isinstance(f, bool) and not isinstance(f, z3.BoolRef):
            raise TypeError(f"{name} is not a boolean")

        # Checking for trivially false facts
        if (not suppress_warnings) and false(f):
            if fact:
                print(f"# Trivially false fact '{name}'")
            else:
                raise Exception(f"# Trivially false assertion {name}")

        # Add the fact to the appropriate list, so we can `reset_facts()` later
        if fact:
            self._facts.append((name, f))
        else:
            self._assertions.append((name, f))

    def assert_(self, f, name=None):
        """Add `f` as a model assertion (aka constraint) with name `name`.

        If `name` is None, then a name will be created.
        """

        self._add(False, f, name)

    def fact(self, f, name=None, suppress_warnings=False):
        """Add `f` as a model fact (aka constraint) with name `name`.

        If `name` is None, then a name will be created.
        """

        self._add(True, f, name, suppress_warnings)

    def check_assertions(self, verbose=False):
        for name, a in self._assertions:
            solution = self.solve(verbose=verbose, assertion=a)
            if solution is not None:
                raise Exception(f"Assertion {name} violated!")

    def solve(self, verbose=False, assertion=None):
        solver = z3.Solver()
        if verbose:
            solver.set(unsat_core=True)

        for name, fact in self._facts:
            try:
                solver.assert_and_track(fact, z3.Bool(f"Fact: {name}"))
            except z3.z3types.Z3Exception:
                raise Exception(f"Fact {name} multiply defined")

        if assertion is not None:
            try:
                solver.assert_and_track(z3.Not(assertion), f"Assertion")
            except z3.z3types.Z3Exception:
                raise Exception(f"Assertion multiply defined")

        if verbose:
            print("Solving...")

        result = solver.check()
        if result == z3.sat:
            if verbose:
                print("SAT")
            return Solution(self, solver.model())
        else:
            if verbose:
                print("UNSAT")
                core = solver.unsat_core()
                print("UNSAT Core has len", len(core))
                print(core)
            return None
