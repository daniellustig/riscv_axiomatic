#!/usr/bin/env python3

# Standard modules
import z3  # pip install z3-solver


################################################################################
# z3py helper functions


def to_bool(value):
    "Convert python booleans into z3 booleans.  Leave other values as-is"

    if isinstance(value, bool):
        return z3.BoolVal(value)
    else:
        return value


def simplify(value):
    """A wrapper for `z3.simplify()` that allows passing in python `True` or
    `False`, since `z3.simplify()` of a bool otherwise raises an exception.
    """

    return z3.simplify(to_bool(value))


def false(value):
    """Syntactic sugar for `z3.is_false(simplify(value))`.

    i.e., return True iff `value` statically simplifies to False.
    """

    return z3.is_false(simplify(value))


def true(value):
    """Syntactic sugar for `z3.is_true(simplify(value))`.

    i.e., return True iff `value` statically simplifies to True.
    """

    return z3.is_true(simplify(value))


def const(value):
    """Syntactic sugar for `z3.is_const(simplify(value))`.

    i.e., return True iff `value` statically simplifies to a constant value.
    """

    return z3.is_const(simplify(value))


def resize(n, size, unsigned=True):
    """Convert `n` into a `z3.BitVecRef` of size `size`.

    If `n` is larger than `size` bits already, then it will be silently
    truncated.  To check for this, use `truncates(n, size)`.

    `n` can be an `int`, a `str`, a `bool`, a `z3.BoolRef`, or a
    `z3.BitVecRef`.  For the two boolean types, size must equal 1.

    If `n` is a `Z3.BitVecRef` and `n.size() < size`, then `n` will be
    zero-extended if `unsigned == True`, or sign-extended if
    `unsigned == False`.
    """

    # Handle the non-BitVecRef cases with one-liners
    if isinstance(n, int):
        return z3.BitVecVal(n, size)
    elif isinstance(n, str):
        return z3.BitVec(n, size)
    elif isinstance(n, bool) or isinstance(n, z3.BoolRef):
        if size != 1:
            raise Exception(
                "in `resize(n, size)`, if `n` is a bool, `size` must equal 1"
            )
        return z3.If(n, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1))
    elif not isinstance(n, z3.BitVecRef):
        raise TypeError(n)

    # Handle the BitVecRef case
    ext_bits = size - n.size()
    if ext_bits > 0:
        if unsigned:
            return z3.ZeroExt(ext_bits, n)
        else:
            return z3.SignExt(ext_bits, n)
    elif ext_bits < 0:
        # Silent truncation: see docstring
        return z3.Extract(size - 1, 0, n)
    else:
        return n


def truncates(n, size):
    """Return a boolean indicating whether turning `n` into a `z3.BitVecRef` of
    size `size` truncates any non-zero bits.
    """

    if n.size() >= size:
        return False

    overflow_bits = z3.Extract(n.size() - 1, size, n)
    return overflow_bits != 0


def hex_(value):
    """If `z3.BitVec` `value` is numeric, convert it to a hex string.

    If `value` is non-numeric, just convert it to a string.
    """

    try:
        return hex(simplify(value).as_long())
    except AttributeError:
        return str(simplify(value))
