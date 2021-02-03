#!/usr/bin/env python3

# Built-in modules
import unittest

# Standard modules
import z3  # pip install z3-solver

# Local modules
from .riscv import *
from .z3_wrapper import *


################################################################################
# RISC-V Virtual Memory Architecture


class PTE32:
    def __init__(self, ppn, d, a, g, u, x, w, r, v):
        self.ppn = resize(ppn, 22)
        self.d = d
        self.a = a
        self.g = g
        self.u = u
        self.x = x
        self.w = w
        self.r = r
        self.v = v

        # 32b PTEs don't have an N bit, but it's easier to assume it's present
        # and hard-coded to False
        self.n = False

    def __str__(self):
        return (
            f"{{ppn={hex_(self.ppn)}, d={self.d}, a={self.a}, "
            f"g={self.g}, u={self.u}, x={self.x}, w={self.w}, "
            f"r={self.r}, v={self.v}}}"
        )

    def value(self):
        return simplify(
            z3.Concat(
                self.ppn,
                z3.BitVecVal(0, 2),
                resize(self.d, 1),
                resize(self.a, 1),
                resize(self.g, 1),
                resize(self.u, 1),
                resize(self.x, 1),
                resize(self.w, 1),
                resize(self.r, 1),
                resize(self.v, 1),
            )
        )


def pte32(value):
    if value.size() != 32:
        raise Exception(f"pte32 received a value of size {value.size()}")
    v = simplify(z3.Extract(0, 0, value) == z3.BitVecVal(1, 1))
    r = simplify(z3.Extract(1, 1, value) == z3.BitVecVal(1, 1))
    w = simplify(z3.Extract(2, 2, value) == z3.BitVecVal(1, 1))
    x = simplify(z3.Extract(3, 3, value) == z3.BitVecVal(1, 1))
    u = simplify(z3.Extract(4, 4, value) == z3.BitVecVal(1, 1))
    g = simplify(z3.Extract(5, 5, value) == z3.BitVecVal(1, 1))
    a = simplify(z3.Extract(6, 6, value) == z3.BitVecVal(1, 1))
    d = simplify(z3.Extract(7, 7, value) == z3.BitVecVal(1, 1))
    ppn = z3.Extract(31, 10, value)

    return PTE32(ppn=ppn, d=d, a=a, g=g, u=u, x=x, w=w, r=r, v=v)


class PTE64:
    def __init__(self, ppn, d, a, g, u, x, w, r, v, n):
        self.ppn = resize(ppn, 44)
        self.d = d
        self.a = a
        self.g = g
        self.u = u
        self.x = x
        self.w = w
        self.r = r
        self.v = v
        self.n = n

    def value(self):
        return simplify(
            z3.Concat(
                z3.BitVecVal(0, 1),
                resize(self.n, 1),
                z3.BitVecVal(0, 8),
                self.ppn,
                z3.BitVecVal(0, 2),
                resize(self.d, 1),
                resize(self.a, 1),
                resize(self.g, 1),
                resize(self.u, 1),
                resize(self.x, 1),
                resize(self.w, 1),
                resize(self.r, 1),
                resize(self.v, 1),
            )
        )


def pte64(value):
    if value.size() != 64:
        raise Exception("pte64 requires a bitvector of size 64")
    v = simplify(z3.Extract(0, 0, value) == z3.BitVecVal(1, 1))
    r = simplify(z3.Extract(1, 1, value) == z3.BitVecVal(1, 1))
    w = simplify(z3.Extract(2, 2, value) == z3.BitVecVal(1, 1))
    x = simplify(z3.Extract(3, 3, value) == z3.BitVecVal(1, 1))
    u = simplify(z3.Extract(4, 4, value) == z3.BitVecVal(1, 1))
    g = simplify(z3.Extract(5, 5, value) == z3.BitVecVal(1, 1))
    a = simplify(z3.Extract(6, 6, value) == z3.BitVecVal(1, 1))
    d = simplify(z3.Extract(7, 7, value) == z3.BitVecVal(1, 1))
    n = simplify(z3.Extract(62, 62, value) == z3.BitVecVal(1, 1))
    ppn = z3.Extract(53, 10, value)

    return PTE64(ppn=ppn, d=d, a=a, g=g, u=u, x=x, w=w, r=r, v=v, n=n)


nonleaf = {"d": False, "a": False, "x": False, "w": False, "r": False}
"Used as arguments, e.g., pte64(ppn=0x40000, **nonleaf, g=0, u=1)"


class VMMode:
    def __init__(
        self,
        options,
        mode,
        asid,
        ppn,
        pa_size,
        page_size,
        pte_size,
        levels,
        vpn_bits,
        ppn_bits,
        f_satp,
        f_pte,
    ):
        self.options = options
        self.mode = mode
        self.asid = asid
        self.ppn = resize(ppn, options.physical_address_bits - 12)
        self.pa_size = pa_size
        self.page_size = page_size
        self.pte_size = pte_size
        self.ppn_size = pa_size - 12
        self.levels = levels
        self.vpn_bits = vpn_bits
        self.ppn_bits = ppn_bits
        self.f_satp = f_satp
        self.f_pte = f_pte

    def physical_address(self, a):
        return resize(a, self.options.physical_address_bits)

    def satp(self):
        return self.f_satp(self.asid, self.ppn)

    def PTE(ppn, d, a, g, u, x, w, r, n=False):
        if not self.f_pte:
            raise Exception("no PTE in bare mode")
        return self.f_pte(ppn, d, a, g, u, x, w, r, n)

    def translate(
        self,
        builder,
        meta,
        tid,
        op_name,
        condition,
        va,
        read,
        write,
        execute,
        user,
        width,
    ):
        faults = {}
        previous_implicit = []

        def _fault_if(t, c1, c2):
            """Record the possibility of a fault.

            Return the condition for not faulting.
            """

            if t == "misaligned":
                if write:
                    cause = SCauseCodes.store_amo_misaligned
                else:
                    cause = SCauseCodes.load_misaligned
            elif t == "access":
                if write:
                    cause = SCauseCodes.store_amo_access
                else:
                    cause = SCauseCodes.load_misaligned
            elif t == "page":
                if write:
                    cause = SCauseCodes.store_amo_page
                else:
                    cause = SCauseCodes.load_page
            else:
                raise Exception(f"Unknown fault type {t}")

            faults.setdefault(cause, []).append(z3.And(c1, c2))
            return z3.And(c1, z3.Not(c2))

        def _walk(a, i, c):
            if a.size() != self.options.physical_address_bits:
                raise Exception(
                    f"walk expected a "
                    f"{self.options.physical_address_bits}-bit PA, "
                    f"but got an address of {a.size()} bits"
                )
            hi, lo = self.vpn_bits[i]
            vpn_i = z3.Extract(hi, lo, va)

            possibly_n = i == 0 and self.pte_size == 8
            if possibly_n:
                # offset is further constrained in the second `possibly_n`
                # block below
                offset = self.physical_address(
                    z3.BitVec(f"{op_name}_walk_{i}_offset", 9)
                )
            else:
                offset = self.physical_address(vpn_i)

            # Calculate the PTE address
            pte_addr = a + offset * self.pte_size

            # Generate an Implicit Read
            read_name = f"{op_name}_walk_{i}"
            builder.read(
                meta,
                tid,
                read_name,
                "PT Walk",
                c,
                self.pte_size,
                pte_addr,
                False,
                False,
                reserve=True,
                implicit=True,
                satp_asid=self.asid,
            )

            # Update translation_order
            nonlocal previous_implicit
            for prev_op, prev_condition in previous_implicit:
                builder.translation_order.update(
                    (prev_op, read_name), z3.And(c, prev_condition)
                )
            previous_implicit.append((read_name, c))

            # Interpret the return value as a PTE of the appropriate size
            value = builder.return_value[read_name]
            if self.pte_size == 4:
                pte = pte32(value)
            elif self.pte_size == 8:
                pte = pte64(value)
            else:
                assert False

            # Fault if any of the reserved bits are set
            if self.pte_size == 8:
                custom = z3.Extract(63, 63, pte.value())
                reserved = z3.Extract(61, 54, pte.value())
                c = _fault_if("access", c, z3.Or(custom != 0, reserved != 0))

            # Batch the page faults for better solver performance
            page_fault_conditions = []

            # Check for NAPOT pages
            if possibly_n:
                # For Svnapot translations, the offset must either match the
                # original offset or be to a valid NAPOT PTE in the same region
                napot64k = z3.And(
                    pte.n, z3.Extract(3, 0, pte.ppn) == z3.BitVecVal(0b1000, 4)
                )

                builder.fact(
                    z3.Or(
                        offset == self.physical_address(vpn_i),
                        z3.And(
                            napot64k,
                            z3.Extract(8, 4, offset) == z3.Extract(8, 4, vpn_i),
                        ),
                    ),
                    "Svnapot offset",
                )

                # Fault if the offset points to an invalid NAPOT PTE
                page_fault_conditions.append(z3.And(pte.n, z3.Not(napot64k)))
            else:
                page_fault_conditions.append(pte.n)
                napot64k = False

            # Fault if not V
            page_fault_conditions.append(z3.Not(pte.v))

            # Fault if W and not R
            page_fault_conditions.append(z3.And(pte.w, z3.Not(pte.r)))

            # Check R, W, X, U
            leaf = z3.Or(pte.r, pte.x)
            page_fault_conditions.append(z3.And(leaf, read, z3.Not(pte.r)))
            page_fault_conditions.append(z3.And(leaf, write, z3.Not(pte.w)))
            page_fault_conditions.append(z3.And(leaf, execute, z3.Not(pte.x)))
            page_fault_conditions.append(z3.And(user, z3.Not(pte.u)))

            # Fault if i == 0 and this is not a leaf PTE
            if i == 0:
                page_fault_conditions.append(z3.Not(leaf))

            # Fault if this is a misaligned superpage
            for j in range(i):
                hi, lo = self.ppn_bits[j]
                ppn_j = z3.Extract(hi, lo, pte.value())
                page_fault_conditions.append(
                    z3.And(leaf, ppn_j != z3.BitVecVal(0, ppn_j.size())),
                )

            # Fault if A and D are not set properly
            a_d_insufficient = z3.And(
                c,
                z3.Or(
                    z3.And(leaf, z3.Not(pte.a)),
                    z3.And(leaf, write, z3.Not(pte.d)),
                ),
            )
            if self.options.hardware_a_d_bit_update:
                # That's it for page fault checks
                c = _fault_if("page", c, z3.Or(page_fault_conditions))

                old_pte_value = pte.value()
                updated_pte_value = z3.Concat(
                    z3.Extract(old_pte_value.size() - 1, 8, old_pte_value),
                    z3.If(write, resize(1, 1), z3.Extract(7, 7, old_pte_value)),
                    resize(1, 1),
                    z3.Extract(5, 0, old_pte_value),
                )
                write_name = f"{op_name}_walk_{i}_a_d"

                # We are assuming this SC always succeeds just as a way of
                # filtering out some of the space that needs to be explored
                builder.write(
                    meta,
                    tid,
                    write_name,
                    "A/D Update",
                    a_d_insufficient,
                    self.pte_size,
                    pte_addr,
                    updated_pte_value,
                    aq=False,
                    rl=False,
                    conditional_dst="x0",
                    implicit=True,
                    satp_asid=self.asid,
                )
                builder.implicit_pair.update(
                    (read_name, write_name), a_d_insufficient
                )
                builder.datadep.update(
                    (read_name, write_name), a_d_insufficient
                )
                for prev_op, prev_condition in previous_implicit:
                    builder.translation_order.update(
                        (prev_op, write_name),
                        z3.And(c, prev_condition, a_d_insufficient),
                    )
                previous_implicit.append(
                    (write_name, z3.And(c, a_d_insufficient))
                )
                pte_value = z3.If(
                    a_d_insufficient, old_pte_value, updated_pte_value
                )
            else:
                page_fault_conditions.append(a_d_insufficient)

                # That's it for page fault checks
                c = _fault_if("page", c, z3.Or(page_fault_conditions))

                pte_value = pte.value()

            # Update the PPN to account for PTE.N
            pte_size = pte_value.size()
            pte_value = z3.If(
                napot64k,
                z3.Concat(
                    z3.Extract(pte_size - 1, 14, pte_value),
                    z3.Extract(15, 12, va),
                    z3.Extract(9, 0, pte_value),
                ),
                pte_value,
            )

            # Generate the PTE from the PPNs and offsets
            pa_components = []
            for j in range(self.levels - 1, i - 1, -1):
                hi, lo = self.ppn_bits[j]
                pa_components.append(z3.Extract(hi, lo, pte_value))
            for j in range(i - 1, -1, -1):
                hi, lo = self.vpn_bits[j]
                pa_components.append(z3.Extract(hi, lo, va))
            pa_components.append(z3.Extract(11, 0, va))
            pa_leaf = self.physical_address(z3.Concat(pa_components))

            # Fault if the calculated PA is out of range
            c = _fault_if(
                "access",
                c,
                z3.And(
                    leaf, truncates(pa_leaf, self.options.physical_address_bits)
                ),
            )

            # Recurse to the next level, unless this is a leaf PTE
            if i == 0 or true(leaf):
                return pa_leaf, c
            else:
                hi = self.ppn_bits[-1][0]
                lo = self.ppn_bits[0][1]
                ppn = z3.Extract(hi, lo, pte_value)
                a = self.physical_address(z3.Concat(ppn, z3.BitVecVal(0, 12)))
                c = z3.And(c, z3.Not(leaf))
                pa, c_walk = _walk(a, i - 1, c)
                return z3.If(leaf, pa_leaf, pa), z3.If(leaf, c, c_walk)

        # Walk the page table, unless we're in bare mode
        if execute:
            raise Exception("Execute access not yet implemented")
        if true(self.mode == 0):
            pa = self.physical_address(va)
            c = condition
        elif false(self.mode == 0):
            base = self.physical_address(
                z3.Concat(self.ppn, z3.BitVecVal(0, 12))
            )
            pa, c = _walk(base, self.levels - 1, condition)

            for prev_op, prev_condition in previous_implicit:
                builder.translation_order.update(
                    (prev_op, op_name), z3.And(c, prev_condition)
                )
        else:
            raise Exception("Indeterminate satp not yet supported")

        # Check alignment of the final PA
        log2_width = {1: 0, 2: 1, 4: 2, 8: 3}[width]
        if log2_width > 0:
            m = z3.Extract(log2_width - 1, 0, pa) != 0
            c = _fault_if(
                "misaligned", c, z3.Extract(log2_width - 1, 0, pa) != 0
            )

        return pa, faults


def satp32(asid, ppn):
    return simplify(
        z3.Concat(z3.BitVecVal(1, 1), resize(asid, 9), resize(ppn, 22))
    )


def satp64(mode):
    return lambda asid, ppn: simplify(
        z3.Concat(z3.BitVecVal(mode, 4), resize(asid, 16), resize(ppn, 22))
    )


def sv32(options, asid, ppn):
    return VMMode(
        options,
        mode=z3.BitVecVal(1, 1),
        asid=asid,
        ppn=ppn,
        pa_size=34,
        page_size=4096,
        pte_size=4,
        levels=2,
        vpn_bits=[(21, 12), (31, 22)],
        ppn_bits=[(19, 10), (31, 20)],
        f_satp=satp32,
        f_pte=pte32,
    )


def sv39(options, asid, ppn):
    return VMMode(
        options,
        mode=z3.BitVecVal(8, 4),
        asid=asid,
        ppn=ppn,
        pa_size=56,
        page_size=4096,
        pte_size=8,
        levels=3,
        vpn_bits=[(20, 12), (29, 21), (38, 30)],
        ppn_bits=[(18, 10), (27, 19), (53, 28)],
        f_satp=satp64(8),
        f_pte=pte64,
    )


def sv48(options, asid, ppn):
    return VMMode(
        options,
        mode=z3.BitVecVal(9, 4),
        asid=asid,
        ppn=ppn,
        pa_size=56,
        page_size=4096,
        pte_size=8,
        levels=3,
        vpn_bits=[(20, 12), (29, 21), (38, 30), (47, 39)],
        ppn_bits=[(18, 10), (27, 19), (36, 28), (53, 37)],
        f_satp=satp64(9),
        f_pte=pte64,
    )


def bare32(options):
    return VMMode(
        options,
        mode=z3.BitVecVal(0, 1),
        asid=0,
        ppn=0,
        pa_size=34,
        page_size=None,
        pte_size=None,
        levels=None,
        vpn_bits=[],
        ppn_bits=[],
        f_satp=satp32,
        f_pte=None,
    )


def bare64(options):
    return VMMode(
        options,
        mode=z3.BitVecVal(0, 4),
        asid=0,
        ppn=0,
        pa_size=34,
        page_size=None,
        pte_size=None,
        levels=None,
        vpn_bits=[],
        ppn_bits=[],
        f_satp=satp64,
        f_pte=None,
    )


def vm(options, satp):
    satp = simplify(resize(satp, options.xlen))
    if satp.size() == 32:
        mode = z3.Extract(31, 31, satp)
        if true(mode == 0):
            return bare32(options)
        elif true(mode == 1):
            asid = z3.Extract(30, 22, satp)
            ppn = z3.Extract(21, 0, satp)
            return sv32(options, asid, ppn)
        else:
            raise Exception(f"Illegal satp mode {satp}")
    elif satp.size() == 64:
        mode = z3.Extract(63, 60, satp)
        if true(mode == 0):
            return bare64(options)
        asid = z3.Extract(59, 44, satp)
        ppn = z3.Extract(43, 0, satp)
        if true(mode == 8):
            return sv39(options, asid, ppn)
        elif true(mode == 9):
            return sv48(options, asid, ppn)
        else:
            raise Exception(f"Illegal satp mode {satp}")
    else:
        assert False
