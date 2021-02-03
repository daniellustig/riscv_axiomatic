RISC-V Axiomatic Concurrency Model

By Dan Lustig, @daniellustig on GitHub

# Overview

The official RISC-V [formal model](https://github.com/rems-project/sail-riscv)
is written in Sail, but that model defines the single-threaded semantics only.
Concurrency semantics are exported to an external model such as
[rmem](https://github.com/rems-project/rmem),
[herd](https://github.com/herd/herdtools7), or
[Alloy](https://github.com/daniellustig/riscv-memory-model), but none of these
tools yet support virtual memory and SFENCE.VMA semantics.  Among these, rmem
is the most likely to add support for virtual memory semantics, and rmem uses
an operational approach.  Ideally, the operational approach would be
complemented by an axiomatic approach, and eventually someone could prove that
the two agree in all cases.  This project aims to provide that middle step:
an axiomatic model of RISC-V concurrency, including virtual memory.

This is a research project and a work in progress.  It has not been officially
adopted by RISC-V, and there is no guarantee of correctness or stability, but
hopefully it will help RISC-V formalize its virtual memory ordering rules
officially in conjunction with the other tools used to model RISC-V
concurrency.

# Demo

```console
$ cat litmus/sc_d_bit.litmus

RISCV sc_d_bit

{
  (* Set up the intial state of the page table *)
  uint32_t *0x2040=pte32(ppn=3,d=0,a=1,g=0,u=1,x=0,w=1,r=1,v=1);
  uint32_t *0x1000=pte32(ppn=2,d=0,a=0,g=0,u=1,x=0,w=0,r=0,v=1);
  (* run with --satp=0x80000001 to use the page table created above *)
}

P0;
(* Store 42 to the VA mapped by the PTE.  The SC should either fault or
update the D bit.  If HW updates the D bit, the SC is allowed to succeed. *)
li a1, 0x10000;
li a2, 42;
lr.w a0, 0(a1);
sc.w a3, a2, 0(a1);

(* Either the SC succeeds and writes 42 to PA 0x3000, or the SC fails,
   but there should be no fault *)
forall 0:scause=0 /\ 0:stval=0 /\ ((0:a3=0 /\ *0x3000=42) \/ not(0:a3=0))

$ python3 -m riscv_axiomatic -i litmus/sc_d_bit.litmus --satp=0x80000001 --hardware-a-d-bit-update --sc-returns-0-or-1
Test sc_d_bit Required
States 2
*0x3000=0x0; 0:scause=0x0; 0:stval=0x0; 0:x13=0x1;
*0x3000=0x2a; 0:scause=0x0; 0:stval=0x0; 0:x13=0x0;
Ok
Condition forall 0:scause=0 /\ 0:stval=0 /\ ((0:a3=0 /\ *0x3000=42) \/ not(0:a3=0))
Hash=0
Observation sc_d_bit Always 2 0

$ python3 -m riscv_axiomatic -i litmus/sc_d_bit.litmus --physical_address_bits=32 --satp=0x80000001 --no-hardware-a-d-bit-update
Test sc_d_bit Required
States 1
*0x3000=0x0; 0:scause=0xf; 0:stval=0x10000; 0:x13=0x0;
No (forbidden found)
Condition forall 0:scause=0 /\ 0:stval=0 /\ ((0:a3=0 /\ *0x3000=42) \/ not(0:a3=0))
Hash=0
Observation sc_d_bit Never 0 1
```

In this test, the page table is set up such that virtual address `0x10000`
points to physical address `0x3000`, with the page being marked accessed but
not dirty.  The main thread performs an LR/SC on that virtual address, storing
`42` if successful.

Three possible outcomes are calculated.  The first two apply if the
implementation supports automatic update of the A/D bits.

In the first, the SC fails (for whatever reason), so the final value of
`*0x3000` is still the initial value of `0`, hart 0 register `x13` aka `a2` is
set to `1`, and the `scause` and `sepc` registers are set to their own initial
values of `0`.  In the second, the SC succeeds, in which case the final value
of `*0x3000` is now `42` aka `0x2a`, register `x13` indicates `0` for success,
and there is still no fault.

Note that in this context, even this single-threaded example has more than
one legal outcome.  This is generally true of tests with SC instructions and/or
tests with non-trivial virtual memory interactions.

In the third case, if the implementation does not support automatic A/D bit
updates, the execution will fault upon trying to execute the SC, so the final
value of `*0x3000` will be `0`, the final value of `x13` will be the initial
value `0` (not `0` indicating success...the SC never executed in this
scenario), and `scause` and `sepc` indicate a store/AMO page fault at PC
`0x8c`, the PC of the SC instruction.

```console
$ cat litmus/sbi_remote_sfence_vma.litmus
RISCV sbi_remote_sfence_vma

(* Test the shootdown process. *)
(* The load in P1 should never return 0xdeadbeef *)

{
  uint32_t *0x2040 = pte32(ppn=3,d=1,a=1,g=0,u=1,x=0,w=1,r=1,v=1);
  uint32_t *0x1000 = pte32(ppn=2,d=0,a=0,g=0,u=1,x=0,w=0,r=0,v=1);
  uint32_t *0x3000 = 0xdeadbeef;
}

P0                                     | P1                               ;
  (* In bare mode: migrate PA *)       | (* Enter Sv32 mode *)            ;
  (*  0x3000 to PA 0x5000 *)           | li a0, 0x80000001                ;
                                       | csrw satp, a0                    ;
  (* zero out the PTE *)               |                                  ;
  li a0, 0x2040                        | (* Store to and then load from *);
  sw x0, 0(a0)                         | (* VA 0x10000 *)                 ;
                                       | li a1, 0x10000                   ;
  (* TLB shootdowns *)                 | li a2, 42                        ;
  sfence.vma                           | sw a2, 0(a1)                     ;
  sbi_remote_sfence_vma({P1})          | lw a3, 0(a1)                     ;
                                       |                                  ;
  (* Copy the data from 0x3000 *)      |                                  ;
  (* to 0x5000 *)                      |                                  ;
  li a1, 0x3000                         |                                 ;
  lw a2, 0(a1)                           |                                ;
  li a1, 0x5000                            |                              ;
  sw a2, 0(a1)                               |                            ;
                                               |                          ;
  (* Ensure the copy is done before *)           |                        ;
  (* the new PTE is set up *)                      |                      ;
  fence w,w                                          |                    ;
                                                      |                   ;
  (* set up the new PTE *)                            |                   ;
  li a4, pte32(ppn=5,d=1,a=1,g=0,u=1,x=0,w=1,r=1,v=1) |                   ;
  sw a4, 0(a0)                                        |                   ;

forall 1:a3=42 \/ not 1:scause=0

$ python3 -m riscv_axiomatic -i litmus/sbi_remote_sfence_vma.litmus --supervisor
Test sbi_remote_sfence_vma Required
States 3
1:scause=0xf; 1:x13=0x0;
1:scause=0xd; 1:x13=0x0;
1:scause=0x0; 1:x13=0x2a;
Ok
Condition forall 1:a3=42 \/ not 1:scause=0
Hash=0
Observation sbi_remote_sfence_vma Always 3 0
```

This test checks the remote TLB invalidation protocol.  There are a few
possible outcomes: either one of the P1 accesses faults as it tries to access
the VA in question during the period when the page is invalid (during
migration), or the access completes properly.  There is no legal situation in
which the P1 load returns the initial value of `0xdeadbeef`.

# Proposed Virtual Memory Ordering Rules

These rules are designed to integrate implicit accesses of the
address-translation data structures into the RVWMO model defined in Chapter 14
of the RISC-V Unprivileged Spec.

1. If memory operation `a` precedes memory operation `b` in program order, `i`
   is an implicit access to the address-translation data structures performed
   on behalf of `b`, and there is an SFENCE.VMA instruction with ASID (if
   applicable) and address (if applicable) matching `i` between `a` and `b` in
   program order, then `a` precedes `b` in the global memory order.
2. If `i` is an implicit access to the address-translation data structures
   performed on behalf of memory operation `b`, then `i` precedes `b` in the
   global memory order
3. On implementations that perform A and D bit updates, the update of the A
   and/or D bit appears after the associated read of the relevant page table
   entry in the same way as would occur for a load-reserved/store-conditional
   pair, and the RVWMO atomicity axiom applies as well.

Note that no dependency ordering is assumed to exist between accesses to
consecutive levels of the page tables for a given access.  This represents
the fact that implicit memory accesses may access stale data stored in the
address-translation caches (e.g., TLBs or page walk caches) in the
implementation.

# Svnapot Ordering Rules

[Svnapot](https://github.com/riscv/virtual-memory) is a proposed new extension
that allows Sv39 and Sv48 page table entries to cover a naturally-aligned
power-of-two (NAPOT) range of memory with a single translation.  This model
formalizes this axiomatically, as follows:

In step 2 of the address-translation algorithm, _pte_ is taken to be
the value of _a + j*PTESIZE_, where _j_ is either:
1. _va.vpn[i]_, as in the original algorithm, or
2. any PPNBITS-sized value such that bits \[PPNBITS-1:napot\_bits\]
   of _j_ and of _va.vpn[i]_ are identical, and such that _pte.n_=1.

This somewhat non-intuitive specification represents the fact that an
implicit access to the address-translation data structures may hit on
a NAPOT PTE for any address in the NAPOT region prefetched into or cached in
the hart's address-translation cache, and therefore might not actually interact
with the PTE usually associated with the virtual address.  The operational
style model would capture this constraint more "naturally".

# Modeling Approach

The tool is built using the following layers (top to bottom)

* `run.py`: the main interface to the tool
* `assembly.py`: a parser of litmus tests like
  [these](https://github.com/litmus-tests/litmus-tests-riscv).  For
  consistency, the tool does its best to use the same input and output format
  as the other standard tools in the area, although support for instructions
  and registers related to virtual memory and faults are added.
* `hart.py`: a basic instruction-level execution model.  Contains enough basic
  instructions to support litmus tests, but not nearly a complete model like
  the Sail model provides.
* `rvwmo.py`: an encoding of the RVWMO rules at the level of memory operations
* `relation.py`: a relational modeling library of the style that herd or Alloy
  use.  This is built on top of [z3py](https://github.com/Z3Prover/z3).

The tool is written in Python3, which makes it pretty easy to mess around with
the model and/or modeling approach as needed.  It also makes it easy to
integrate bitvectors into the model, in a way that would be difficult for herd
or Alloy.  Bitvectors are particularly important for being able to model page
table entries and walks in a meaningful way.

# User Documentation

To run all virtual memory litmus tests, run:

```
# ./run_all.py -p "litmus/*.litmus"  # after figuring out what args we'll need
./scripts/run_virtual.sh             # in the meantime
```

To run all litmus tests from https://github.com/litmus-tests/litmus-tests-riscv, run:

```
./run_all.py -p "litmus-tests-riscv/tests/non-mixed-size/**/*.litmus" -b litmus-tests-riscv/model-results/herd.logs
./run_all.py -p "litmus-tests-riscv/tests/mixed-size/**/*.litmus" -b litmus-tests-riscv/model-results/herd.logs
```

(flat.logs can be used instead of herd.logs, for slightly different baseline results)

## Command Line Options

More documentation to come... For now, `python3 -m riscv_axiomatic -h` will
give a full listing.

One notable option is `-f`:

```
$ python3 -m riscv_axiomatic -i litmus/sc_d_bit.litmus --satp=0x80000001 --hardware-a-d-bit-update --sc-returns-0-or-1 -f
# Options: [check_assertions=False; expressions=[]; full_solution=True; input_file=litmus/sc_d_bit.litmus; max_solutions=None; return_values=[]; print_options=False; verbose=0; verbose_solver=False; allow_misaligned_accesses=False; allow_misaligned_atomics=False; allow_unknown_regs=False; arbitrary_initial_values=False; big_endian=False; hardware_a_d_bit_update=True; max_unroll_count=3; physical_address_bits=16; rvwmo=True; satp=2147483649; sc_address_must_match=False; sc_returns_0_or_1=True; start_pc=128; supervisor=False; use_initial_value_function=False; virtual_memory_ordering=True; xlen=32]
Test sc_d_bit Required
States 2
# Solution
#
# gmo:['Pi_pc0x94', 'Pi_pc0x88', 'P0_pc0x8c_walk_0', 'P0_pc0x8c_walk_0_a_d', 'P0_pc0x8c_walk_1', 'P0_pc0x88_walk_0', 'P0_pc0x88_walk_1', 'P0_pc0x88', 'Pf_pc0x84']
# --------------------  -------------------------------------------------------------------------  --------------  -----------
# Hart Pi:
# Pi_pc0x88             uint32_t *0x2040=pte32(ppn=3,d=0,a=1,g=0,u=1,x=0,w=1,r=1,v=1);             Address=0x2040  Value=0xc57
# Pi_pc0x94             uint32_t *0x1000=pte32(ppn=2,d=0,a=0,g=0,u=1,x=0,w=0,r=0,v=1);             Address=0x1000  Value=0x811
# Hart P0:
# P0_pc0x88_walk_1      lr.w a0, 0(a1)                                                             Address=0x1000  Value=0x811
# P0_pc0x88_walk_0      lr.w a0, 0(a1)                                                             Address=0x2040  Value=0xcd7
# P0_pc0x88             lr.w a0, 0(a1)                                                             Address=0x3000  Value=0x0
# P0_pc0x8c_walk_1      sc.w a3, a2, 0(a1)                                                         Address=0x1000  Value=0x811
# P0_pc0x8c_walk_0      sc.w a3, a2, 0(a1)                                                         Address=0x2040  Value=0xc57
# P0_pc0x8c_walk_0_a_d  sc.w a3, a2, 0(a1)                                                         Address=0x2040  Value=0xcd7
# Hart Pf:
# Pf_pc0x84             forall 0:scause=0 /\ 0:stval=0 /\ ((0:a3=0 /\ *0x3000=42) \/ not(0:a3=0))  Address=0x3000  Value=0x0
# --------------------  -------------------------------------------------------------------------  --------------  -----------
# 
*0x3000=0x0; 0:scause=0x0; 0:stval=0x0; 0:x13=0x1;
# Condition satisfied: True
#
# End solution
#
# Solution
#
# gmo:['Pi_pc0x94', 'Pi_pc0x88', 'P0_pc0x8c_walk_0', 'P0_pc0x8c_walk_0_a_d', 'P0_pc0x88_walk_0', 'P0_pc0x8c_walk_1', 'P0_pc0x88_walk_1', 'P0_pc0x88', 'P0_pc0x8c', 'Pf_pc0x84']
# --------------------  -------------------------------------------------------------------------  --------------  -----------
# Hart Pi:
# Pi_pc0x88             uint32_t *0x2040=pte32(ppn=3,d=0,a=1,g=0,u=1,x=0,w=1,r=1,v=1);             Address=0x2040  Value=0xc57
# Pi_pc0x94             uint32_t *0x1000=pte32(ppn=2,d=0,a=0,g=0,u=1,x=0,w=0,r=0,v=1);             Address=0x1000  Value=0x811
# Hart P0:
# P0_pc0x88_walk_1      lr.w a0, 0(a1)                                                             Address=0x1000  Value=0x811
# P0_pc0x88_walk_0      lr.w a0, 0(a1)                                                             Address=0x2040  Value=0xcd7
# P0_pc0x88             lr.w a0, 0(a1)                                                             Address=0x3000  Value=0x0
# P0_pc0x8c_walk_1      sc.w a3, a2, 0(a1)                                                         Address=0x1000  Value=0x811
# P0_pc0x8c_walk_0      sc.w a3, a2, 0(a1)                                                         Address=0x2040  Value=0xc57
# P0_pc0x8c_walk_0_a_d  sc.w a3, a2, 0(a1)                                                         Address=0x2040  Value=0xcd7
# P0_pc0x8c             sc.w a3, a2, 0(a1)                                                         Address=0x3000  Value=0x2a
# Hart Pf:
# Pf_pc0x84             forall 0:scause=0 /\ 0:stval=0 /\ ((0:a3=0 /\ *0x3000=42) \/ not(0:a3=0))  Address=0x3000  Value=0x2a
# --------------------  -------------------------------------------------------------------------  --------------  -----------
# 
*0x3000=0x2a; 0:scause=0x0; 0:stval=0x0; 0:x13=0x0;
# Condition satisfied: True
#
# End solution
#
Ok
Condition forall 0:scause=0 /\ 0:stval=0 /\ ((0:a3=0 /\ *0x3000=42) \/ not(0:a3=0))
Hash=0
Observation sc_d_bit Always 2 0
```

# Architecture Model

The tool models a restricted subset of Rv32 and Rv64 user ISA and user and
supervisor mode of the privileged ISA.  The proposed Svnapot extension is
also supported.

* Page faults, environment calls, access faults, misaligned address faults, and
  illegal instructions are delegated to S-mode, which in practice means that
  `sepc`, `scause`, and `stval` are the CSRs that are updated with the fault
  information.  S-mode then simply terminates execution at that point, making
  the fault information visible as part of the final output of a test.
* Herd/litmus test compatibility
  * lw.aq and sc.rl don't actually exist, but for now are parsed as RCpc, to
    match herd and flat model results
  * A few tests use illegal registers, but the tool tries to be accomodating

## Assembly

* Supports the subset of actual RISC-V assembly shown in assembly.py
* One SBI call: `sbi_remote_sfence_vma(address, harts)`, where `address` is the
  optional address passed to each `sfence`, and `harts` is a list of harts by
  name
  * This is a variant of the real-world SBI call, which takes an address range
    and hart bitmask
  * Convenience that avoids having to expand all the assembly it would take to
    actually implement the syscall

