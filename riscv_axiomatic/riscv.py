#!/usr/bin/env python3

# Standard modules
from enum import IntEnum


################################################################################
# RISC-V Architecture Component Definitions
################################################################################

csrs = {0x141: "sepc", 0x142: "scause", 0x143: "stval", 0x180: "satp"}

_canonical_registers = {
    **{"x0": "zero", "x1": "ra", "x2": "sp", "x3": "gp", "x4": "tp"},
    **{f"x{i+5}": f"t{i}" for i in range(3)},
    **{f"x{i+8}": f"s{i}" for i in range(2)},
    **{f"x{i+10}": f"a{i}" for i in range(8)},
    **{f"x{i+18}": f"s{i+2}" for i in range(10)},
    **{f"x{i+28}": f"s{i+3}" for i in range(4)},
    **{"pc": "pc"},
    **csrs,
}
canonical_registers = {
    **_canonical_registers,
    **{v: v for v in _canonical_registers.values()},
}

_x_registers = {
    v: k for k, v in canonical_registers.items() if str(k)[0] == "x"
}
_csr_x_registers = {
    k: v for k, v in canonical_registers.items() if isinstance(k, int)
}
_x_registers = {
    **_x_registers,
    **{"pc": "pc"},
    **_csr_x_registers,
    **{v: v for v in _x_registers.values()},
    **{v: v for v in _csr_x_registers.values()},
}

_next_x_reg = 32
_unknown_regs = set()


def _unknown_reg(reg):
    global _next_x_reg
    if reg not in _unknown_regs:
        print(f"# WARNING: unknown register {reg}")
        _unknown_regs.add(reg)
    x = f"x{_next_x_reg}"
    _x_registers[reg] = x
    canonical_registers[reg] = x
    canonical_registers[x] = x
    _next_x_reg += 1
    return reg


def canonical_register(reg):
    try:
        return canonical_registers[reg]
    except KeyError:
        return canonical_registers[_unknown_reg(reg)]


def x_register(reg):
    try:
        return _x_registers[reg]
    except KeyError:
        return _x_registers[_unknown_reg(reg)]


class SCauseCodes(IntEnum):
    illegal_instruction = 2
    load_misaligned = 4
    load_access = 5
    store_amo_misaligned = 6
    store_amo_access = 7
    load_page = 13
    store_amo_page = 15
