#!/usr/bin/env python3

# Standard modules
import lark  # pip install lark-parser
import z3  # pip install z3-solver

# Local modules
from . import alloy
from . import config
from . import execution
from . import relation
from . import rvwmo
from . import solution
from . import virtual
from .z3_wrapper import *


class Meta:
    """Line information and original text for instructions"""

    def __init__(self, text, line, column):
        self.text = text
        self.line = line
        self.column = column
        if column is not None and line is None:
            raise Exception("If column is not None, line must not be None")

    def __str__(self):
        return self.text

    def full(self):
        s = ""
        if self.line is not None:
            s += f"Line {self.line}"
        if self.column is not None:
            assert self.line
            s += f", Column {self.column}"
        if s:
            s += ": "
        s += self.text
        return s


grammar = """
  ?start: test
  
  test: header front init threads locations filter outcome

  header: string string_ext

  front: string_ext*

  init: "{" (init_expr)* "}"

  init_expr: type location ("=" number)? ";"

  type: /int/
      | /int/ /\*/
      | /int32_t/
      | /int32_t/ /\*/
      | /uint32_t/
      | /uint32_t/ /\*/
      | /int64_t/
      | /int64_t/ /\*/
      | /uint64_t/
      | /uint64_t/ /\*/
      |

  threads: litmus_threads | new_threads

  new_threads: new_thread+

  new_thread: thread_name "{" instruction+ "}"

  litmus_threads: litmus_thread_names litmus_insts+

  litmus_thread_names: thread_name ("|" thread_name)* ";"

  litmus_insts: maybe_instruction ("|" maybe_instruction)* ";"

  thread_name: /P[0-9]+/ | /Pi/

  maybe_instruction: instruction |

  ?instruction: /add/ reg "," reg "," reg -> add
              | /addi/ reg "," reg "," number -> addi
              | /amoadd.d/ aq_rl reg "," reg "," addr -> amo
              | /amoadd.w/ aq_rl reg "," reg "," addr -> amo
              | /amoor.d/ aq_rl reg "," reg "," addr -> amo
              | /amoor.w/ aq_rl reg "," reg "," addr -> amo
              | /amoswap.d/ aq_rl reg "," reg "," addr -> amo
              | /amoswap.w/ aq_rl reg "," reg "," addr -> amo
              | /andi/ reg "," reg "," number -> andi
              | /beq/ reg "," reg "," target -> branch
              | /bne/ reg "," reg "," target -> branch
              | /csrw/ reg "," reg -> csrw
              | /fence/ fence_type -> fence
              | /fence\.i/ -> fence_i
              | /lb/ aq_rl reg "," addr -> ld
              | /ld/ aq_rl reg "," addr -> ld
              | /lh/ aq_rl reg "," addr -> ld
              | /li/ reg "," number -> li
              | /lr.d/ aq_rl reg "," addr -> ld
              | /lr.w/ aq_rl reg "," addr -> ld
              | /lw/ aq_rl reg "," addr -> ld
              | /or/ reg "," reg "," reg -> or_
              | /ori/ reg "," reg "," number -> ori
              | /sb/ aq_rl reg "," addr -> st
              | /sbi_remote_sfence_vma/ "(" hart_list ("," reg "," reg)? ")" -> sbi_remote_sfence_vma
              | /sc.d/ aq_rl reg "," reg "," addr -> sc
              | /sc.w/ aq_rl reg "," reg "," addr -> sc
              | /sd/ aq_rl reg "," addr -> st
              | /sfence\.vma/ (reg "," reg)? -> sfence
              | /sh/ aq_rl reg "," addr -> st
              | /sw/ aq_rl reg "," addr -> st
              | /xor/ reg "," reg "," reg -> xor
              | target ":" -> label

  !aq_rl: ".aq" | ".rl" | ".aqrl" | ".aq.rl" |

  hart_list: "{" thread_name ("," thread_name)* "}"

  filter: "filter" condition
        |

  locations: "locations" "[" (location ";")+ "]"
           |

  location: thread_reg
          | string
          | "*" hex

  thread_reg: int ":" string

  outcome: expectation condition

  ?expectation: /exists/ | /forall/ | /~exists/

  condition: condition_or

  ?condition_or: condition_and
               | condition_or "\\\\/" condition_and -> condition_or

  ?condition_and: condition_not
                | condition_and "/\\\\" condition_not -> condition_and

  ?condition_not: condition_eq
                | "not" condition_not -> condition_not
                | "~" condition_not -> condition_not
                | "(" condition_or ")" -> condition_paren

  ?condition_eq: int ":" reg "=" number -> condition_reg
               | string "=" number -> condition_addr
               | "*" hex "=" number -> condition_addr
               | "true" -> condition_true

  reg: string

  addr: number "(" reg ")"
       | "(" reg ")"

  fence_type: fence_rw "," fence_rw
            | /./ /tso/

  !fence_rw: "r" | "w" | "rw"

  ?number: signed_int | hex | pte32 | pte64 | sv32 | variable | addressof

  signed_int: SIGNED_INT -> int

  int: INT

  hex: /0x[0-9a-fA-F]+/

  variable: string

  addressof: "&" string -> variable

  pte32: "pte32" "(" "ppn" "=" number "," \
                      "d" "=" b "," \
                      "a" "=" b "," \
                      "g" "=" b "," \
                      "u" "=" b "," \
                      "x" "=" b "," \
                      "w" "=" b "," \
                      "r" "=" b "," \
                      "v" "=" b ")"

  pte64: "pte64" "("  "n" "=" b "," \
                    "ppn" "=" number "," \
                      "d" "=" b "," \
                      "a" "=" b "," \
                      "g" "=" b "," \
                      "u" "=" b "," \
                      "x" "=" b "," \
                      "w" "=" b "," \
                      "r" "=" b "," \
                      "v" "=" b ")"

  sv32: "sv32" "(" "asid" "=" number "," "ppn" "=" number ")"
  sv39: "sv39" "(" "asid" "=" number "," "ppn" "=" number ")"
  sv48: "sv48" "(" "asid" "=" number "," "ppn" "=" number ")"

  b: /[01]/

  ?target: string

  string: CNAME

  string_ext: /[+-_=A-Za-z0-9()*"'[\]]+/ -> string

  %import common.CNAME
  %import common.INT
  %import common.SIGNED_INT
  %import common.WS
  %import common.C_COMMENT
  %import common.CPP_COMMENT

  %ignore WS
  %ignore C_COMMENT
  %ignore CPP_COMMENT
"""


################################################################################


@lark.v_args(inline=True, meta=True)
class Transformer(lark.Transformer):
    def __init__(self, options, text):
        super().__init__()
        self.options = options
        self.text = text
        self.variables = {}
        self.final_addrs = {}

        self.builder = rvwmo.LitmusTestBuilder(
            physical_address_bits=self.options.physical_address_bits,
            allow_misaligned_atomics=self.options.allow_misaligned_atomics,
            arbitrary_initial_values=self.options.arbitrary_initial_values,
            big_endian=self.options.big_endian,
            sc_address_must_match=self.options.sc_address_must_match,
        )

    def _text(self, meta):
        text = self.text[meta.start_pos : meta.end_pos]
        text = " ".join([i.strip() for i in text.split(" \n\t")])
        return text

    def _meta(self, meta):
        return Meta(self._text(meta), meta.line, meta.column)

    def ld(self, meta, opcode, aq_rl, dst, src):
        meta = self._meta(meta)
        base, offset = src
        return execution.Read(meta, opcode, aq_rl, dst, base, offset)

    def st(self, meta, opcode, aq_rl, src, dst):
        meta = self._meta(meta)
        base, offset = dst
        return execution.Write(meta, opcode, aq_rl, base, offset, src)

    def amo(self, meta, opcode, aq_rl, dst, src, addr):
        meta = self._meta(meta)
        base, offset = addr
        return execution.AMO(meta, opcode, aq_rl, base, offset, src, dst)

    def sc(self, meta, opcode, aq_rl, dst, src, addr):
        meta = self._meta(meta)
        base, offset = addr
        return execution.Write(meta, opcode, aq_rl, base, offset, src, dst)

    def li(self, meta, opcode, dst, value):
        meta = self._meta(meta)
        return execution.LoadImmediate(meta, opcode, dst, value)

    def ori(self, meta, opcode, dst, src1, src2):
        meta = self._meta(meta)
        return execution.OrImmediate(meta, opcode, dst, src1, src2)

    def or_(self, meta, opcode, dst, src1, src2):
        meta = self._meta(meta)
        return execution.Or(meta, opcode, dst, src1, src2)

    def addi(self, meta, opcode, dst, src1, src2):
        meta = self._meta(meta)
        return execution.AddImmediate(meta, opcode, dst, src1, src2)

    def andi(self, meta, opcode, dst, src1, src2):
        meta = self._meta(meta)
        return execution.AndImmediate(meta, opcode, dst, src1, src2)

    def add(self, meta, opcode, dst, src1, src2):
        meta = self._meta(meta)
        return execution.Add(meta, opcode, dst, src1, src2)

    def xor(self, meta, opcode, dst, src1, src2):
        meta = self._meta(meta)
        return execution.XOr(meta, opcode, dst, src1, src2)

    def fence(self, meta, opcode, fence_type):
        meta = self._meta(meta)
        return execution.Fence(meta, fence_type)

    def fence_type(self, meta, a, b=None):
        if a == ".":
            return str(b)
        else:
            return f"{a},{b}"

    def fence_rw(self, meta, a):
        return a

    def fence_i(self, meta, opcode):
        meta = self._meta(meta)
        return execution.FenceI(meta)

    def sfence(self, meta, opcode, addr="x0", satp_asid="x0"):
        meta = self._meta(meta)
        return execution.SFenceVMA(meta, addr, satp_asid)

    def branch(self, meta, opcode, src1, src2, target):
        meta = self._meta(meta)
        return execution.Branch(meta, opcode, src1, src2, target)

    def csrw(self, meta, opcode, csr, value):
        meta = self._meta(meta)
        return execution.CSRRW(meta, opcode, "x0", csr, value)

    def sbi_remote_sfence_vma(self, meta, opcode, harts, addr=None, size=None):
        meta = self._meta(meta)
        return execution.SBIRemoteSFenceVMA(meta, opcode, harts, addr, size)

    def hart_list(self, meta, *harts):
        return harts

    def maybe_instruction(self, meta, i=None):
        return i

    def reg(self, meta, name):
        return name

    def label(self, meta, name):
        return execution.Label(name)

    def aq_rl(self, meta, name=""):
        return name

    def addr(self, meta, a, b=None):
        if b is None:
            return (a, 0)
        else:
            return (b, a)

    def int(self, meta, n):
        return int(n)

    def hex(self, meta, n):
        return int(n, 16)

    def _get_variable(self, v):
        if isinstance(v, int):
            return resize(v, self.options.xlen)

        try:
            return self.variables[v]
        except KeyError:
            pass

        # 8B aligned
        bv = resize(
            z3.Concat(
                resize(v, self.options.physical_address_bits - 3), resize(0, 3)
            ),
            self.options.xlen,
        )

        self.variables[v] = bv
        return bv

    def variable(self, meta, v):
        return self._get_variable(v)

    def string(self, meta, s):
        return s

    def pte32(self, meta, ppn, d, a, g, u, x, w, r, v):
        return virtual.PTE32(ppn, d, a, g, u, x, w, r, v).value()

    def pte64(self, meta, n, ppn, d, a, g, u, x, w, r, v):
        return virtual.PTE64(ppn, d, a, g, u, x, w, r, v, n).value()

    def sv32(self, meta, asid, ppn):
        return virtual.satp32(asid, ppn)

    def sv39(self, meta, asid, ppn):
        return virtual.satp64(8, asid, ppn)

    def sv48(self, meta, asid, ppn):
        return virtual.satp64(9, asid, ppn)

    def b(self, meta, n):
        return int(n)

    def init(self, meta, *init_exprs):
        d = {}
        addresses = set()
        initial_values = {}
        for init_meta, type_, dst, expr in init_exprs:
            init_meta = self._meta(init_meta)

            # Parse the type of the initial value expression
            if not type_:
                width = self.options.xlen / 8
            elif type_ == "int" or type_ == "uint32_t" or type_ == "int32_t":
                width = 4
            elif type_ == "uint64_t" or type_ == "int64_t":
                width = 8
                if self.options.xlen != 64:
                    raise Exception("uint64_t requires --xlen=64")
            elif type_ and type_[-1] == "*":
                width = self.options.xlen / 8
                if isinstance(dst, tuple):
                    reg = execution.x_register(dst[1])
                    addresses.add(f"{dst[0]}:{reg}")
                else:
                    addresses.add(dst)
            else:
                raise Exception(f"Unknown type '{type_}'")

            # Either set the initial value as specified, or create a store in
            # the initial thread "Pi" to do it
            if isinstance(dst, int) or isinstance(dst, str):
                dst = self._get_variable(dst)
                if expr is not None:
                    w = {4: "w", 8: "d"}[width]
                    if self.options.use_initial_value_function:
                        for i in range(w):
                            if self.options.big_endian:
                                raise Exception(
                                    "Big-endian mode not yet supported"
                                )
                            initial_values[dst + i] = z3.BitVec(
                                expr >> (8 * i), 8
                            )
                    else:
                        d.setdefault("Pi", [])
                        d["Pi"] += [
                            execution.LoadImmediate(init_meta, "li", "a0", dst),
                            execution.LoadImmediate(
                                init_meta, "li", "a1", expr
                            ),
                            execution.Write(
                                init_meta,
                                f"s{w}",
                                aq_rl="",
                                src="a1",
                                base="a0",
                                offset=0,
                            ),
                        ]
            else:
                thread, reg = dst
                if width != self.options.xlen / 8:
                    raise Exception("Register width mismatch {thread}:{reg}")
                if expr is not None:
                    thread = f"P{thread}"
                    d.setdefault(thread, []).append(
                        execution.LoadImmediate(init_meta, "li", reg, expr)
                    )
        return d, addresses, initial_values

    def init_expr(self, meta, type_, dst, expr=None):
        return meta, type_, dst, expr

    def type(self, meta, *type_):
        return "".join(type_)

    def thread_name(self, meta, n):
        return f"P{n}"

    def litmus_thread_names(self, meta, *names):
        return names

    def litmus_insts(self, meta, *insts):
        return insts

    def litmus_threads(self, meta, names, *insts):
        d = {}
        for t in names:
            d[t] = []
        for row in insts:
            if len(row) != len(names):
                raise Exception(
                    f"Unexpected number of columns: {self._meta(meta)}"
                )
            for i in range(len(row)):
                inst = row[i]
                if inst:
                    d[names[i]].append(inst)

        threads = {}
        for thread, insts in d.items():
            threads[thread] = insts
        return threads

    def new_thread(self, meta, name, *instructions):
        return name, instructions

    def thread_name(self, meta, name):
        return str(name)

    def new_threads(self, meta, *threads):
        d = {}
        for name, insts in threads:
            if name in threads:
                raise Exception(f"Duplicate thread name {name}")
            d[name] = insts
        return d

    def threads(self, meta, threads):
        for t in threads:
            assert t[:2] != "PP"
        return threads

    def outcome(self, meta, expectation, condition):
        name, expression, state_elements, addrs = condition
        return (
            solution.Outcome(
                name,
                expression,
                expectation,
                state_elements,
                self.variables,
                addrs,
            ),
            meta,
        )

    def condition(self, meta, e):
        expr, regs, addrs = e
        return self._text(meta), expr, regs, addrs

    def condition_or(self, meta, a, b):
        expr_a, regs_a, addrs_a = a
        expr_b, regs_b, addrs_b = b
        return (
            z3.Or(expr_a, expr_b),
            {**regs_a, **regs_b},
            addrs_a | addrs_b,
        )  # dict merge

    def condition_and(self, meta, a, b):
        expr_a, regs_a, addrs_a = a
        expr_b, regs_b, addrs_b = b
        return (
            z3.And(expr_a, expr_b),
            {**regs_a, **regs_b},
            addrs_a | addrs_b,
        )  # dict merge

    def condition_not(self, meta, a):
        expr, regs, addrs = a
        return z3.Not(expr), regs, addrs

    def condition_true(self, meta):
        return True, {}, set()

    def condition_reg(self, meta, thread, reg, value):
        # Herd and rmem always use the x version of the register
        reg = execution.x_register(reg)
        name = f"{thread}:{reg}"
        expr = execution.final_register_value(
            f"P{thread}", reg, self.options.xlen
        )
        if isinstance(value, z3.BitVecRef) and not z3.is_const(value):
            addrs = {name}
        else:
            addrs = set()
        return expr == value, {name: expr}, addrs

    def _addr_final_value_reg(self, addr):
        if isinstance(addr, int):
            bv = resize(addr, self.options.physical_address_bits)
        else:
            bv = self._get_variable(addr)
        return self.final_addrs.setdefault(bv, f"x{len(self.final_addrs) + 1}")

    def condition_addr(self, meta, addr, value):
        reg = self._addr_final_value_reg(addr)
        if isinstance(addr, int):
            name = f"*{hex(addr)}"
        else:
            name = addr
        expr = execution.final_register_value(f"Pf", reg, self.options.xlen)
        if isinstance(value, z3.BitVecRef) and not z3.is_const(value):
            addrs = {name}
        else:
            addrs = set()
        return expr == value, {name: expr}, addrs

    def condition_paren(self, meta, c):
        return c

    def filter(self, meta, condition=None):
        if condition is None:
            return None
        text, expr, regs, addrs = condition
        return expr

    def location(self, meta, location):
        return location

    def locations(self, meta, *locations):
        return locations

    def thread_reg(self, meta, thread, reg):
        return thread, str(reg)

    def header(self, meta, arch, title):
        return arch, title

    def test(
        self, meta, header, front, init, threads, locations, filter_, outcome
    ):
        outcome, outcome_meta = outcome
        meta = self._meta(outcome_meta)

        arch, title = header

        # Initial conditions
        init_insts, location_addresses, initial_values = init
        for thread, insts in init_insts.items():
            threads[thread] = insts + list(threads.setdefault(thread, []))
        for a in location_addresses:
            outcome.mark_address(a)
        for k, v in initial_values.items():
            self.builder.initial_values[k] = v

        # Mark the declared locations as being part of the outcome
        for l in locations:
            if isinstance(l, tuple):
                thread, reg = l
                reg = execution.x_register(reg)
                fv = execution.final_register_value(
                    f"P{thread}", reg, self.options.xlen
                )
                outcome.add_state(f"{thread}:{reg}", fv)
            else:
                reg = self._addr_final_value_reg(l)
                fv = execution.final_register_value(
                    "Pf", reg, self.options.xlen
                )
                if isinstance(l, int):
                    name = f"*{hex(l)}"
                else:
                    name = l
                outcome.add_state(name, fv)

        # Create a final thread to perform the loads used to check for the
        # final state of certain physical addresses
        final_hart_insts = []

        w = "w" if self.options.xlen == 32 else "d"
        for addr, reg in self.final_addrs.items():
            threads.setdefault("Pf", [])
            threads["Pf"] += [
                execution.LoadImmediate(meta, "li", reg, addr),
                execution.Read(
                    meta, f"l{w}", aq_rl="", dst=reg, base=reg, offset=0
                ),
            ]

        # Create and execute the harts
        harts = {}
        thread_order = sorted(list(threads.keys()))
        if "Pi" in thread_order:
            thread_order.remove("Pi")
            thread_order = ["Pi"] + thread_order
        if "Pf" in thread_order:
            thread_order.remove("Pf")
            thread_order = thread_order + ["Pf"]
        for t in thread_order:
            force_supervisor = t == "Pi" or t == "Pf"
            harts[t] = execution.Thread(
                t, t, self.builder, self.options, threads[t], force_supervisor
            )
            harts[t].execute()

        # Constrain the initial and final harts to execute in the right order
        for t in threads:
            if "Pi" in threads and t != "Pi":
                self.builder.hart_order.update(("Pi", t), True)
            if "Pf" in threads and t != "Pf":
                self.builder.hart_order.update((t, "Pf"), True)

        # Filter the outcomes as specified
        if filter_ is not None:
            self.builder.fact(filter_, "filter")

        # Ensure that no two address variables are the same
        for v, bv in self.variables.items():
            for v2, bv2 in self.variables.items():
                if v == v2:
                    break
                self.builder.fact(bv != bv2, f"{v} != {v2}")

        return title, self.builder, harts, outcome


class Parser:
    def __init__(self, options):
        self.parser = lark.Lark(grammar, propagate_positions=True)
        self.options = options

    def parse(self, s):
        # Preprocess
        while True:
            start = s.find("(*")
            if start == -1:
                break
            end = s.find("*)")
            if end == -1 or end < start:
                raise Exception(f"Unmatched comment {start} {end}")
            s = s[:start] + s[end + 2 :]

        transformer = Transformer(self.options, s)
        try:
            return transformer.transform(self.parser.parse(s))
        except lark.exceptions.UnexpectedCharacters as e:
            for t in e.considered_tokens:
                if "instruction ::=" in str(e.considered_tokens):
                    raise Exception(f"line {e.line}: unknown opcode")
            raise e


if __name__ == "__main__":
    import sys

    options = config.parse_args([])
    with open(sys.argv[1], "r") as f:
        title, builder, harts, outcome = Parser(options).parse(f.read())
        model = builder.model()
        print(model)
        print(outcome)
