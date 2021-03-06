#!/bin/bash

set -e
set -x

python3 -m riscv_axiomatic -i litmus/PTE_activate_satp_one_thread.litmus --supervisor
python3 -m riscv_axiomatic -i litmus/PTE_activate_satp_one_thread+sfence.vma.litmus --supervisor

python3 -m riscv_axiomatic -i litmus/PTE_activate_satp_two_thread.litmus --supervisor --max-unroll-count=1
python3 -m riscv_axiomatic -i litmus/PTE_activate_satp_two_thread+sfence.vma.litmus --supervisor --max-unroll-count=1

python3 -m riscv_axiomatic -i litmus/PTE_IRIW.litmus --supervisor

python3 -m riscv_axiomatic -i litmus/PTE_OoO_fetch.litmus --supervisor

python3 -m riscv_axiomatic -i litmus/PTE_update_ad_lrsc.litmus --satp=0x80000001 --sc-returns-0-or-1

python3 -m riscv_axiomatic -i litmus/sbi_remote_sfence_vma.litmus --supervisor

python3 -m riscv_axiomatic -i litmus/sc_d_bit.litmus  --satp=0x80000001 --sc-returns-0-or-1
python3 -m riscv_axiomatic -i litmus/sc_d_bit.litmus  --satp=0x80000001 --no-hardware-a-d-bit-update --sc-returns-0-or-1

python3 -m riscv_axiomatic -i litmus/sw_napot.litmus --xlen=64

python3 -m riscv_axiomatic -i litmus/sw_supervisor.litmus --satp=0x80000001
