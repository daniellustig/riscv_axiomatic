#!/usr/bin/env python3

# Built-in modules
import argparse
import unittest

# Standard modules
import z3  # pip install z3-solver

# Local modules
from . import virtual
from .relation import RelationBuilder
from .riscv import *
from .z3_wrapper import *


class Label:
    def __init__(self, name):
        self.name = name


class Instruction:
    def __str__(self):
        return self.opcode


class Add(Instruction):
    def __init__(self, meta, opcode, dst, src1, src2):
        self.meta = meta
        self.opcode = opcode
        self.dst = dst
        self.src1 = src1
        self.src2 = src2


class AddImmediate(Instruction):
    def __init__(self, meta, opcode, dst, src1, src2):
        self.meta = meta
        self.opcode = opcode
        self.dst = dst
        self.src1 = src1
        self.src2 = src2


class AMO(Instruction):
    def __init__(self, meta, opcode, aq_rl, base, offset, src, dst):
        self.meta = meta
        self.opcode = opcode
        self.base = base
        self.offset = offset
        self.src = src
        self.dst = dst
        opcode, width = opcode.split(".")
        try:
            self.op, self.unsigned = {
                "amoadd": (lambda a, b: a + b, True),  # unsigned is N/A
                "amoswap": (lambda a, b: b, True),  # unsigned is N/A
                "amoand": (lambda a, b: a & b, True),  # unsigned is N/A
                "amoor": (lambda a, b: a | b, True),  # unsigned is N/A
                "amoxor": (lambda a, b: a ^ b, True),  # unsigned is N/A
            }[opcode]
        except:
            raise Exception(f"unknown AMO opcode {opcode}")
        self.width = {"w": 4, "d": 8}[width]
        self.aq, self.rl = {
            "": (False, False),
            ".aq": (True, False),
            ".rl": (False, True),
            ".aqrl": (True, True),
            ".aq.rl": (True, True),
        }[aq_rl]


class AndImmediate(Instruction):
    def __init__(self, meta, opcode, dst, src1, src2):
        self.meta = meta
        self.opcode = opcode
        self.dst = dst
        self.src1 = src1
        self.src2 = src2


class Branch(Instruction):
    def __init__(self, meta, opcode, src1, src2, target):
        self.meta = meta
        self.opcode = opcode
        self.src1 = src1
        self.src2 = src2
        self.target = target
        self.op = {
            "beq": lambda a, b: a == b,
            "bne": lambda a, b: a != b,
            "blt": lambda a, b: a < b,
            "bltu": lambda a, b: z3.ULT(a, b),
            "bge": lambda a, b: a >= b,
            "bgeu": lambda a, b: z3.UGE(a, b),
        }[opcode.lower()]


class CSRRW(Instruction):
    def __init__(self, meta, opcode, dst, csr, src):
        self.meta = meta
        self.opcode = opcode
        self.dst = dst
        self.csr = csr
        self.src = src


class Fence(Instruction):
    def __init__(self, meta, fence_type):
        self.meta = meta
        self.opcode = "fence"

        if fence_type not in ["r,r", "w,w", "r,rw", "rw,w", "rw,rw", "tso"]:
            print(f"# WARNING: non-standard fence type {fence_type}")

        try:
            self.pr, self.pw, self.sr, self.sw, self.tso = {
                "r,r": (True, False, True, False, False),
                "r,w": (True, False, False, True, False),
                "w,w": (False, True, False, True, False),
                "w,r": (False, True, True, False, False),
                "r,rw": (True, False, True, True, False),
                "rw,w": (True, True, False, True, False),
                "rw,rw": (True, True, True, True, False),
                "tso": (False, False, False, False, True),
            }[fence_type]
        except KeyError:
            raise KeyError(f"Unknown fence type {fence_type}")


class FenceI(Instruction):
    def __init__(self, meta):
        self.meta = meta
        self.opcode = "fence.i"


class LoadImmediate(Instruction):
    def __init__(self, meta, opcode, dst, value):
        self.meta = meta
        self.opcode = opcode
        self.dst = dst
        self.value = value


class NoOp(Instruction):
    def __init__(self, meta):
        self.meta = meta
        self.opcode = "nop"


class Or(Instruction):
    def __init__(self, meta, opcode, dst, src1, src2):
        self.meta = meta
        self.opcode = opcode
        self.dst = dst
        self.src1 = src1
        self.src2 = src2


class OrImmediate(Instruction):
    def __init__(self, meta, opcode, dst, src1, src2):
        self.meta = meta
        self.opcode = opcode
        self.dst = dst
        self.src1 = src1
        self.src2 = src2


class Read(Instruction):
    def __init__(self, meta, opcode, aq_rl, dst, base, offset):
        self.meta = meta
        self.opcode = opcode
        self.dst = dst
        self.base = base
        self.offset = offset
        self.width, self.unsigned, self.reserve = {
            "lb": (1, False, False),
            "lh": (2, False, False),
            "lw": (4, False, False),
            "ld": (8, False, False),
            "lbu": (1, True, False),
            "lhu": (2, True, False),
            "lwu": (4, True, False),
            "lr.w": (4, False, True),
            "lr.d": (8, False, True),
        }[opcode]
        self.aq, self.rl = {
            "": (False, False),
            ".aq": (True, False),
            ".aqrl": (True, True),
            ".aq.rl": (True, True),
        }[aq_rl]


class SBIRemoteSFenceVMA(Instruction):
    def __init__(self, meta, opcode, harts, addr, size):
        self.meta = meta
        self.opcode = opcode
        self.harts = harts
        self.addr = addr
        self.size = size


class SFenceVMA(Instruction):
    def __init__(self, meta, addr="x0", satp_asid="x0"):
        self.meta = meta
        self.opcode = "sfence.vma"
        self.addr = addr
        self.satp_asid = satp_asid


class Write(Instruction):
    def __init__(
        self, meta, opcode, aq_rl, base, offset, src, conditional_dst=None
    ):
        self.meta = meta
        self.opcode = opcode
        self.base = base
        self.offset = offset
        self.src = src
        self.conditional_dst = conditional_dst
        self.width, self.conditional = {
            "sb": (1, False),
            "sh": (2, False),
            "sw": (4, False),
            "sd": (8, False),
            "sc.w": (4, True),
            "sc.d": (8, True),
        }[opcode]
        self.aq, self.rl = {
            "": (False, False),
            ".rl": (False, True),
            ".aqrl": (True, True),
            ".aq.rl": (True, True),
        }[aq_rl]

        if self.conditional and self.conditional_dst is None:
            raise Exception("Expected dst for conditional store")
        elif (not self.conditional) and self.conditional_dst is not None:
            raise Exception("Received dst for non-conditional store")


class XOr(Instruction):
    def __init__(self, meta, opcode, dst, src1, src2):
        self.meta = meta
        self.opcode = opcode
        self.dst = dst
        self.src1 = src1
        self.src2 = src2


################################################################################


def final_register_value(thread, reg, xlen):
    reg = canonical_register(reg)
    return resize(f"{thread}:{reg}:final_value", xlen)


class Thread:
    def __init__(
        self,
        name,
        hart,
        builder,
        options,
        instructions,
        force_supervisor=False,
    ):
        self.name = name
        self.hart = hart
        self._builder = builder
        self._options = options
        self._instructions = {}
        self._labels = {}

        # Dynamic state
        self._pc = self._options.start_pc
        self._user_mode = not force_supervisor and not options.supervisor
        self._condition = True
        self._registers = {}
        self._deps = {}
        self._branch = []
        self._unroll_count = 0
        self._state_stack = []

        # Map instructions onto PCs

        pc = self._options.start_pc
        for i in instructions:
            # Labels show up in the list of "instructions", but they aren't
            # actually instructions, so they need a bit of special handling
            if isinstance(i, Label):
                if i.name in self._labels:
                    raise Exception(f"Label {i.name} multiply defined")
                self._debug(f"Thread {self.name} Label {i.name} = PC {pc}")
                self._labels[i.name] = pc
                continue

            # Normal instructions
            self._debug(f"Thread {self.name} PC {pc} = {i}")
            self._instructions[pc] = i
            pc += 4

    def _debug(self, *s):
        if self._options.verbose == 1:
            # Print only the first line...some z3 expressions get too long to
            # print in any meaningful way
            print(*[i.split("\n")[0] for i in s])
        elif self._options.verbose:
            print(*s)

    def _push_state(self, branch):
        """Save the state prior to a branch"""

        self._state_stack.append((self._pc, self._user_mode, self._condition))
        self._branch.append(branch)
        self._unroll_count += 1

    def _pop_state(self):
        """Restore the state prior to a branch"""

        self._pc, self._user_mode, self._condition = self._state_stack[-1]
        self._state_stack.pop()
        self._branch.pop()
        self._unroll_count -= 1

    def _operation_name(self, thread=None, branch=None, pc=None):
        """The RVWMO Op name for thread/branch/PC.

        If thread/branch/PC are None, use the current value."""

        if thread is None:
            thread = f"{self.name}"
        if branch is None:
            branch = f"{'_'.join(self._branch)}"
        if pc is None:
            pc = self._pc
        pc = f"pc{hex(pc)}"
        return f"{thread}{branch}_{pc}"

    def _get_register(self, reg):
        """Get the current value of the specified register"""

        reg = canonical_register(reg)
        if reg == canonical_register("zero"):
            return resize(0, self._options.xlen)
        try:
            r = self._registers[reg]
            return r
        except KeyError:
            if self._options.arbitrary_initial_values:
                init = z3.BitVec(f"_{self.name}_{reg}_init", self._options.xlen)
            else:
                init = z3.BitVecVal(0, self._options.xlen)
            return self._registers.setdefault(reg, init)

    def _set_register(self, reg, value, extra_condition=True):
        """Set the value of `reg` to `value`.

        Only updates the value if And(self._condition, extra_condition).
        """

        reg = canonical_register(reg)
        if reg == canonical_register("zero"):
            return

        # Check for illegal CSR values
        if reg == canonical_register("satp"):
            # satp is WARL
            if self._options.xlen == 32:
                valid = z3.Or(
                    value == 0,  # Bare
                    z3.Extract(31, 31, value) == 1,  # Sv32
                )
            elif self._options.xlen == 64:
                valid = z3.Or(
                    value == 0,  # Bare
                    z3.Extract(63, 60, value) == 8,  # Sv39
                    z3.Extract(63, 60, value) == 9,  # Sv48
                )
            else:
                assert False

            if false(valid):
                print(
                    "# WARNING: invalid satp write may be ignored "
                    "or have unpredictable consequences!"
                )
            elif not true(valid):
                print(
                    "# WARNING: possibly-invalid satp write may be ignored "
                    "or have unpredictable events!"
                )

        elif reg in [
            canonical_register("sepc"),
            canonical_register("scause"),
            canonical_register("stval"),
        ]:
            valid = True

        elif reg in csrs.keys():
            raise Exception(f"Unexpected CSR {reg}")

        else:
            valid = True

        condition = z3.And(self._condition, extra_condition, valid)

        if isinstance(value, int):
            value = resize(value, self._options.xlen)
        else:
            if not isinstance(value, z3.BitVecRef):
                raise TypeError
            if value.size() != self._options.xlen:
                raise TypeError("Invalid width")

        if true(condition):
            self._registers[reg] = value
        else:
            old = self._get_register(reg)
            self._registers[reg] = z3.If(condition, value, old)
        self._debug(f"Reg {reg} = {simplify(self._get_register(reg))}")

    def _update_condition(self, value):
        """Continue executing the current control flow only if `value`"""

        self._condition = simplify(z3.And(self._condition, value))
        self._debug(f"Condition = {self._condition}")

    def _addrdep(self, op_name, reg, extra_condition=True):
        """Declare a syntactic address dependency.

        `op_name` will have an address dependency on the Op(s) that wrote the
        value last written into `reg`, as tracked by _set_dep() and
        _carry_dep().
        """

        reg = canonical_register(reg)
        for k, v in self._deps.get(reg, {}).items():
            self._builder.addrdep.update(
                k + (op_name,), z3.And(v, self._condition, extra_condition)
            )

    def _ctrldep(self, op_name, reg, extra_condition=True):
        """Declare a syntactic control dependency.

        `op_name` will have an control dependency on the Op(s) that wrote the
        value last written into `reg`, as tracked by _set_dep() and
        _carry_dep().
        """

        reg = canonical_register(reg)
        for k, v in self._deps.get(reg, {}).items():
            self._builder.ctrldep.update(
                k + (op_name,), z3.And(v, self._condition, extra_condition)
            )

    def _datadep(self, op_name, reg, extra_condition=True):
        """Declare a syntactic data dependency.

        `op_name` will have an data dependency on the Op(s) that wrote the
        value last written into `reg`, as tracked by _set_dep() and
        _carry_dep().
        """

        reg = canonical_register(reg)
        for k, v in self._deps.get(reg, {}).items():
            self._builder.datadep.update(
                k + (op_name,), z3.And(v, self._condition, extra_condition)
            )

    def _set_dep(self, op_name, reg, extra_condition=True):
        """Set `op_name` as the head of a dependency chain in `reg`.

        Only set it if `extra_condition` is true.
        """

        reg = canonical_register(reg)
        if reg == canonical_register("zero"):
            return

        condition = z3.And(self._condition, extra_condition)

        dep = RelationBuilder()

        # If condition, cancel out the old values.  If not, keep the old values
        for k, v in self._deps.get(reg, {}).items():
            dep.update(k, z3.And(v, z3.Not(condition)))

        # If condition, add `op_name` as the dependency source
        if op_name is not None:
            dep.update((op_name,), condition)

        self._deps[reg] = dep.relation()
        self._debug(f"Dep[{reg}] = {dep.relation()}")

    def _carry_dep(self, dst, src):
        """Carry a dependency from `src` to `dst`"""

        src = canonical_register(src)
        if src == canonical_register("zero"):
            return

        dst = canonical_register(dst)
        if dst == canonical_register("zero"):
            return

        dep = RelationBuilder()

        # If self._condition, cancel out the old values.  If not, keep the old
        # values
        for k, v in self._deps.get(dst, {}).items():
            dep.update(k, z3.And(v, z3.Not(self._condition)))

        # If self._condition, carry deps from src to dst
        for k, v in self._deps.get(src, {}).items():
            dep.update(k, z3.And(v, self._condition))

        self._deps[dst] = dep.relation()
        self._debug(f"Dep[{dst}] = {dep.relation()}")

    def _translate(self, name, meta, va, read, write, execute, width):
        """Translate `va` into a physical address, according to satp.mode"""

        satp = self._get_register("satp")
        if z3.is_const(simplify(satp)):
            vm = virtual.vm(self._options, satp)
            pa, faults = vm.translate(
                self._builder,
                meta,
                self.name,
                name,
                self._condition,
                resize(va, self._options.xlen),
                read=read,
                write=write,
                execute=execute,
                user=self._user_mode,
                width=width,
            )

        # Slow path, where satp isn't constant
        # "Slow" in the sense of having a much more complex solver constraint
        elif self._options.xlen == 32:
            mode = z3.Extract(31, 31, satp)
            asid = z3.Extract(30, 22, satp)
            ppn = z3.Extract(21, 0, satp)

            pa_bare, fault_bare = (
                resize(va, self._options.physical_address_bits),
                truncates(va, self._options.physical_address_bits),
            )

            pa_sv32, fault_sv32 = virtual.sv32(
                self._options, asid, ppn
            ).translate(
                self._builder,
                meta,
                self.name,
                f"{name}_sv32",
                z3.And(self._condition, mode == 1),
                resize(va, self._options.xlen),
                read=read,
                write=write,
                execute=execute,
                user=self._user_mode,
            )

            pa, faults = (
                z3.If(mode == 1, pa_sv32, pa_bare),
                z3.If(mode == 1, fault_sv32, fault_bare),
            )

        elif self._options.xlen == 64:
            mode = z3.Extract(63, 60, satp)
            asid = z3.Extract(59, 44, satp)
            ppn = z3.Extract(43, 0, satp)

            pa_bare, fault_bare = (
                resize(va, self._options.physical_address_bits),
                truncates(va, self._options.physical_address_bits),
            )

            pa_sv39, fault_sv39 = virtual.sv39(
                self._options, asid, ppn
            ).translate(
                self._builder,
                meta,
                self.name,
                f"{name}_sv39",
                z3.And(self._condition, mode == 8),
                resize(va, self._options.xlen),
                read=read,
                write=write,
                execute=execute,
                user=self._user_mode,
            )

            pa_sv48, fault_sv48 = virtual.sv48(
                self._options, asid, ppn
            ).translate(
                self._builder,
                meta,
                self.name,
                f"{name}_sv48",
                z3.And(self._condition, mode == 9),
                resize(va, self._options.xlen),
                read=read,
                write=write,
                execute=execute,
                user=self._user_mode,
            )

            pa, faults = (
                z3.If(mode == 8, pa_sv39, z3.If(mode == 9, pa_sv48, pa_bare)),
                z3.If(
                    mode == 8,
                    fault_sv39,
                    z3.If(mode == 9, fault_sv48, fault_base),
                ),
            )

        else:
            assert False

        return pa, faults

    def _fault_if(self, fault_conditions, pc, stval=None):
        for k, v in fault_conditions.items():
            if not isinstance(k, SCauseCodes):
                raise TypeError
            k = int(k)

            for c in v:
                self._set_register("sepc", pc, c)
                # Bit 31 of scause is scause.interrupt, but that's 0 in this
                # case, so it's fine to just set scause to k
                self._set_register("scause", k, c)
                if stval is not None:
                    self._set_register("stval", stval, c)
                self._update_condition(z3.Not(c))

    def execute_instruction(self, i, pc):
        op_name = self._operation_name()
        label = str(i)
        if isinstance(i, Add):
            self._carry_dep(i.dst, i.src1)
            self._carry_dep(i.dst, i.src2)
            src1 = self._get_register(i.src1)
            src2 = self._get_register(i.src2)
            self._set_register(i.dst, src1 + src2)

        elif isinstance(i, AddImmediate):
            self._carry_dep(i.dst, i.src1)
            src1 = self._get_register(i.src1)
            self._set_register(i.dst, src1 + resize(i.src2, self._options.xlen))

        elif isinstance(i, AMO):
            va = self._get_register(i.base) + i.offset
            pa, fault_conditions = self._translate(
                op_name,
                i.meta,
                resize(va, self._options.xlen),
                read=False,
                write=True,
                execute=False,
                width=i.width,
            )
            self._fault_if(fault_conditions, pc, va)

            value = resize(self._get_register(i.src), i.width * 8)

            self._builder.amo(
                i.meta,
                self.name,
                op_name,
                label,
                self._condition,
                i.width,
                pa,
                i.op,
                value,
                i.aq,
                i.rl,
            )

            self._set_register(
                i.dst,
                resize(
                    self._builder.return_value[op_name],
                    self._options.xlen,
                    i.unsigned,
                ),
            )

            self._addrdep(op_name, i.base)
            self._datadep(op_name, i.src)
            self._set_dep(op_name, i.dst)

        elif isinstance(i, AndImmediate):
            self._carry_dep(i.dst, i.src1)
            src1 = self._get_register(i.src1)
            self._set_register(i.dst, src1 & resize(i.src2, self._options.xlen))

        elif isinstance(i, Branch):
            self._builder.branch(
                i.meta, self.name, op_name, label, self._condition
            )
            src1 = self._get_register(i.src1)
            src2 = self._get_register(i.src2)
            condition = i.op(src1, src2)

            self._ctrldep(op_name, i.src1)
            self._ctrldep(op_name, i.src2)

            return i.target, condition

        elif isinstance(i, CSRRW):
            self._fault_if(
                {SCauseCodes.illegal_instruction: [self._user_mode]}, pc, pc
            )

            src = canonical_register(i.src)
            csr = canonical_register(i.csr)
            dst = canonical_register(i.dst)

            value = self._get_register(i.src)

            if dst != canonical_register("x0"):
                self._set_register(i.dst, self._get_register(i.csr))
                # The order of the dep carrying matters
                self._carry_dep(i.dst, i.csr)

            self._set_register(i.csr, value)
            self._carry_dep(i.csr, i.src)

        elif isinstance(i, Fence):
            self._builder.fence(
                i.meta,
                self.name,
                op_name,
                label,
                self._condition,
                i.pr,
                i.pw,
                i.sr,
                i.sw,
                i.tso,
            )

        elif isinstance(i, FenceI):
            pass

        elif isinstance(i, LoadImmediate):
            self._set_register(i.dst, resize(i.value, self._options.xlen))
            self._set_dep(None, i.dst)

        elif isinstance(i, Or):
            self._carry_dep(i.dst, i.src1)
            self._carry_dep(i.dst, i.src2)
            src1 = self._get_register(i.src1)
            src2 = self._get_register(i.src2)
            self._set_register(i.dst, src1 | src2)

        elif isinstance(i, OrImmediate):
            self._carry_dep(i.dst, i.src1)
            src1 = self._get_register(i.src1)
            self._set_register(i.dst, src1 | resize(i.src2, self._options.xlen))

        elif isinstance(i, Read):
            va = self._get_register(i.base) + i.offset
            pa, fault_conditions = self._translate(
                op_name,
                i.meta,
                resize(va, self._options.xlen),
                read=True,
                write=False,
                execute=False,
                width=i.width,
            )
            self._fault_if(fault_conditions, pc, va)
            self._builder.read(
                i.meta,
                self.name,
                op_name,
                label,
                self._condition,
                i.width,
                pa,
                i.aq,
                i.rl,
                i.reserve,
            )
            self._set_register(
                i.dst,
                resize(
                    self._builder.return_value[op_name],
                    self._options.xlen,
                    i.unsigned,
                ),
            )

            self._addrdep(op_name, i.base)
            self._set_dep(op_name, i.dst)

        elif isinstance(i, SBIRemoteSFenceVMA):
            # Create two anchor points in the current thread

            self._builder.dummy_memory_op(
                i.meta,
                self.name,
                op_name + "_w",
                label,
                self._condition,
                aq=False,
                rl=True,
            )

            self._builder.dummy_memory_op(
                i.meta,
                self.name,
                op_name + "_r",
                label,
                self._condition,
                aq=True,
                rl=False,
            )

            def _sfence(op_name, addr):
                """Helper function shortcut for creating remote SFENCEs below"""
                self._builder.sfence_vma(
                    i.meta,
                    thread_name,
                    op_name,
                    label,
                    self._condition,
                    addr,
                    satp_asid=None,
                )

            # Create a new software thread that interleaves an SFENCE.VMA into
            # each specified remote hart
            for h in i.harts:
                # Create a new software thread to interleave into the targeted
                # hart.  the RVWMO assertion "Hart <: iden in threads" ensures
                # that we don't send IPIs to non-existent harts
                thread_name = f"{self._operation_name()}_rfence_{h}"
                self._builder.thread(thread_name, h)

                # Create an anchor point to start the thread
                self._builder.dummy_memory_op(
                    i.meta,
                    thread_name,
                    op_name + f"_r_{h}",
                    label,
                    self._condition,
                    aq=True,
                    rl=False,
                )

                # Insert the right SFENCE.VMA(s)
                if not i.addr and not i.size:
                    _sfence(f"{op_name}_f_{h}", None)
                elif not i.size:
                    _sfence(f"{op_name}_f_{h}", i.addr)
                else:
                    i = 0
                    for addr in range(i.addr, i.addr + i.size, 4096):
                        _sfence(f"{op_name}_f_{h}_{i}")
                        i += 1

                # Create an anchor point to end the thread
                self._builder.dummy_memory_op(
                    i.meta,
                    thread_name,
                    op_name + f"_w_{h}",
                    label,
                    self._condition,
                    aq=False,
                    rl=True,
                )

                # Enforce ordering between the newly-created thread and this
                # thread, in both directions, using the anchor points
                self._builder.force.update(
                    (f"{op_name}_w", f"{op_name}_r_{h}"), self._condition
                )
                self._builder.force.update(
                    (f"{op_name}_w_{h}", f"{op_name}_r"), self._condition
                )

        elif isinstance(i, SFenceVMA):
            self._fault_if(
                {SCauseCodes.illegal_instruction: [self._user_mode]}, pc, pc
            )

            if canonical_register(i.addr) == canonical_register("x0"):
                addr = None
            else:
                addr = resize(
                    self._get_register(i.addr),
                    self._options.physical_address_bits,
                )

            if canonical_register(i.satp_asid) == canonical_register("x0"):
                satp_asid = None
            else:
                satp_asid = self._get_register(i.satp_asid)

            self._builder.sfence_vma(
                i.meta,
                self.name,
                op_name,
                label,
                self._condition,
                addr,
                satp_asid,
            )

        elif isinstance(i, Write):
            va = self._get_register(i.base) + i.offset
            pa, fault_conditions = self._translate(
                op_name,
                i.meta,
                resize(va, self._options.xlen),
                read=False,
                write=True,
                execute=False,
                width=i.width,
            )
            self._fault_if(fault_conditions, pc, va)
            value = resize(self._get_register(i.src), i.width * 8)
            self._builder.write(
                i.meta,
                self.name,
                op_name,
                label,
                self._condition,
                i.width,
                pa,
                value,
                i.aq,
                i.rl,
                i.conditional_dst,
            )

            if i.conditional_dst:
                success = self._builder.condition(op_name)

                return_value = z3.BitVec(f"{op_name}_rv", self._options.xlen)
                self._builder.fact(
                    (return_value == 0) == success,
                    f"{op_name} return value success",
                )
                if self._options.sc_returns_0_or_1:
                    self._builder.fact(
                        (return_value == 1) == z3.Not(success),
                        f"{op_name} return value failure",
                    )

                self._set_register(i.conditional_dst, return_value)
                self._set_dep(op_name, i.conditional_dst, success)
            else:
                success = True

            self._addrdep(op_name, i.base, success)
            self._datadep(op_name, i.src, success)

        elif isinstance(i, XOr):
            self._carry_dep(i.dst, i.src1)
            self._carry_dep(i.dst, i.src2)
            src1 = self._get_register(i.src1)
            src2 = self._get_register(i.src2)
            self._set_register(i.dst, src1 ^ src2)

        else:
            assert False

    def execute(self):
        if self._options.allow_misaligned_accesses:
            raise Exception("allow_misaligned_accesses not implemented")

        self._pc = self._options.start_pc
        self._debug(f"PC = {self._pc}")

        self._builder.thread(self.name, self.hart)
        if self.name != "Pi" and self.name != "Pf":
            self._set_register(
                "satp", resize(self._options.satp, self._options.xlen)
            )
        else:
            self._set_register("satp", resize(0, self._options.xlen))

        self._execute()

        for k, v in canonical_registers.items():
            self._builder.fact(
                final_register_value(self.name, k, self._options.xlen)
                == self._get_register(k),
                f"Final value {self.name} {k}",
            )

    def _execute(self):
        while True:
            self._debug(
                f"Thread {self.name} PC {self._pc} "
                f"Operation {self._operation_name()}"
            )

            if false(self._condition):
                self._debug(f"Condition is false")
                self._debug(f"Execution has terminated")
                return

            if self._unroll_count >= self._options.max_unroll_count:
                self._debug(
                    "Reached unroll limit:", simplify(z3.Not(self._condition))
                )
                self._builder.fact(z3.Not(self._condition), "unroll limit")
                return

            if self._pc not in self._instructions:
                self._debug(f"PC {self._pc} not in instruction stream")
                self._debug(f"Execution has terminated")
                return

            i = self._instructions[self._pc]

            self._debug(f"Execute {self._pc} {i} {self._condition}")
            branch = self.execute_instruction(i, self._pc)
            if branch is not None:
                label, condition = branch
                try:
                    target = self._labels[label]
                except KeyError:
                    raise Exception(f"Unknown label {target}")

                self._debug("Branch", label, target, condition)

                # First execute the taken branch, under "condition"
                self._push_state(self._operation_name())
                self._pc = target
                self._update_condition(condition)
                self._execute()

                # Then execute the not-taken branch, under "Not(condition)"
                self._pop_state()
                self._update_condition(z3.Not(condition))

            # Go to the next instruction
            self._pc += 4
            self._set_register("pc", resize(self._pc, self._options.xlen))
