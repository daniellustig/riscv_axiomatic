#!/usr/bin/env python3

# Standard modules
import lark  # pip install lark-parser

# Local modules
from .z3_wrapper import *


# Precedence rules taken from
# http://alloytools.org/download/alloy4-grammar.txt

# TODO: the missing operations can be easily added as needed.  I just haven't
# gotten around to adding them all.

_grammar = """
  ?start: expr

  expr: expr_or

  // 1)    let    all a:X|F   no a:X|F   some a:X|F   lone a:X|F   one a:x|F   sum a:x|F
  // 2)    ||
  ?expr_or: expr_and
          | expr_or "or" expr_and -> or_

  // 3)    <=>
  // 4)    =>     => else  // both right-associative
  // 5)    &&
  ?expr_and: expr_not
           | expr_and "and" expr_not -> and_
  // 6)    !
  ?expr_not: expr_in
    | "not" expr_not -> not_

  // 7)    in     =        <        >       <=      >=      !in   !=   !<   !>  !<=  !>=
  ?expr_in: expr_some
    | expr_union "in" expr_in -> in_
    | expr_union "=" expr_in -> equal

  // 8)    no X   some X   lone X   one X   set X   seq X
  ?expr_some: expr_union
            | "some" expr_union -> some
            | "no" expr_union -> no

  // 9)    <<     >>       >>>
  // 10)   +      -
  ?expr_union: expr_intersect
    | expr_intersect "+" expr_union -> union
    | expr_intersect "-" expr_union -> difference

  // 11)   #X
  // 12)   ++
  // 13)   &
  ?expr_intersect: expr_cross
    | expr_cross "&" expr_intersect -> intersect

  // 14)   ->  // right-associative
  ?expr_cross: expr_domain
    | expr_cross "->" expr_domain -> cross

  // 15)   <:
  ?expr_domain: expr_range
    | expr_range "<:" expr_domain -> domain

  // 16)   :>
  ?expr_range: expr_join
    | expr_range ":>" expr_join -> range_

  // 17)   []
  // 18)   .
  ?expr_join: expr_tc
    | expr_tc "." expr_join -> join

  // 19)   ~    *     ^
  ?expr_tc: expr_name
    | "^" expr_tc -> tc
    | "*" expr_tc -> rtc
    | "~" expr_tc -> transpose
    | "?" expr_tc -> optional  // NOT PRESENT IN ALLOY

  ?expr_name: expr_paren
    | CNAME -> name
    | SIGNED_INT -> num

  ?expr_paren: "(" expr ")"

  %import common.CNAME
  %import common.SIGNED_INT
  %import common.WS
  %import common.C_COMMENT
  %import common.CPP_COMMENT

  %ignore WS
  %ignore C_COMMENT
  %ignore CPP_COMMENT
"""


@lark.v_args(inline=True)
class _Transformer(lark.Transformer):
    def __init__(self, definitions):
        super().__init__()
        self.definitions = definitions

    def expr(self, e):
        return e

    def and_(self, a, b):
        return z3.And(a, b)

    def cross(self, a, b):
        return a.cross(b)

    def domain(self, a, b):
        return a.iden().join(b)

    def difference(self, a, b):
        return a - b

    def equal(self, a, b):
        return a == b

    def in_(self, a, b):
        return a.in_(b)

    def intersect(self, a, b):
        return a.intersect(b)

    def join(self, a, b):
        return a.join(b)

    def no(self, a):
        return a.no()

    def not_(self, a):
        return z3.Not(a)

    def optional(self, a):
        return a.union(self.name("iden"))

    def or_(self, a, b):
        return z3.Or(a, b)

    def range_(self, a, b):
        return a.join(b.iden())

    def rtc(self, a):
        return a.transitive_closure().union(self.name("iden"))

    def some(self, a):
        return a.some()

    def tc(self, a):
        return a.transitive_closure()

    def transpose(self, a):
        return a.transpose()

    def union(self, a, b):
        return a.union(b)

    def name(self, a):
        if a in self.definitions:
            return self.definitions.get(a)
        else:
            raise KeyError(f"Name {a} undefined")

    def num(self, n):
        return n


class Parser:
    """Parser to parse Alloy language expressions in a `RelationalModel`."""

    def __init__(self, definitions):
        """Create a Parser.

        `definitions` is a `dict` mapping names to `Relation`s.  Note that
        `RelationalModel` is a subclass of `dict` and therefore works here too.
        """

        self._parser = lark.Lark(_grammar)
        self._transformer = _Transformer(definitions)

    def parse(self, e):
        """Parse Alloy expression `e`."""

        return self._transformer.transform(self._parser.parse(e))
