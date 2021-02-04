#!/usr/bin/env python3

# Standard modules
import z3  # pip install z3-solver

# Local modules
from . import alloy
from .relation import Relation, RelationBuilder, RelationalModel, Singleton
from .z3_wrapper import *


class RVWMOBuilder:
    """A class to build up to an RVWMO model instance.

    The various `Relation`s can be built up incrementally in any way through
    the `RelationBuilder` members, and then a `RelationalModel` can be
    extracted using `model()`.
    """

    def __init__(
        self,
        physical_address_bits,
        allow_misaligned_atomics=False,
        arbitrary_initial_values=False,
        big_endian=False,
        sc_address_must_match=False,
    ):
        self._physical_address_bits = physical_address_bits
        self._allow_misaligned_atomics = allow_misaligned_atomics
        self._arbitrary_initial_values = arbitrary_initial_values
        self._big_endian = big_endian
        self._sc_address_must_match = sc_address_must_match

        self._model = RelationalModel()

        # Sets (arity-1 relations)

        self.Hart = RelationBuilder()
        "All hardware threads"

        self.Thread = RelationBuilder()
        "All software threads"

        self.Op = RelationBuilder()
        "All operations (memory operations plus fences)"

        self.MemoryOp = RelationBuilder()
        "All memory operations"

        self.DummyMemoryOp = RelationBuilder()
        """All dummy memory operations

        DummyMemoryOps are used to define inter-thread ordering constraints
        without building up some non-trivial locking protocol within the model.
        """

        self.Read = RelationBuilder()
        """All read operations (including implicit reads, load-reserve
        operations, and AMOs)"""

        self.Write = RelationBuilder()
        """All write operations (including implicit writes for A/D bit updates
        and AMOs)"""

        self.ImplicitOp = RelationBuilder()
        "All implicit reads of the address-translation data structures"

        self.LoadReserve = RelationBuilder()
        """All load-reserve operations (including implicit reads for A/D bit
        updates"""

        self.StoreConditional = RelationBuilder()
        """All store-conditional operations (including implicit writes for A/D
        bit updates"""

        self.AMO = RelationBuilder()
        "All AMO operations"

        self.Aq = RelationBuilder()
        "All operations marked .aq"

        self.Rl = RelationBuilder()
        "All operations marked .rl"

        self.RCsc = RelationBuilder()
        "All RCsc operations"

        self.Fence = RelationBuilder()
        "All fences (not counting sfences)"

        self.FencePR = RelationBuilder()
        "All fences with .pr"

        self.FencePW = RelationBuilder()
        "All fences with .pw"

        self.FenceSR = RelationBuilder()
        "All fences with .sr"

        self.FenceSW = RelationBuilder()
        "All fences with .sw"

        self.FenceTSO = RelationBuilder()
        "All fences with .tso"

        self.SFenceVMA = RelationBuilder()
        "All sfence.vmas"

        # Edges (arity-2 relations)

        self.thread_ops = RelationBuilder()
        "Relates a (software) Thread to all ops in that thread"

        self.threads = RelationBuilder()
        "Relates each Hart to the software Threads scheduled on it"

        self.po = RelationBuilder()
        "Program order"

        self.addrdep = RelationBuilder()
        "Syntactic address dependencies"

        self.ctrldep = RelationBuilder()
        "Syntactic control dependencies"

        self.datadep = RelationBuilder()
        "Syntactic data dependencies"

        self.implicit_pair = RelationBuilder()
        """LR/SC pairs for implicit address-translation accesses

        Normal LR/SC pairs use the _pair() calculation and program order, but
        implicit address-translation accesses aren't inserted into program
        order, and so we need another way to get them to be paired properly
        """

        self.translation_order = RelationBuilder()
        """The sequence of address-translation reads, in order, and then
        pointing to the parent Explicit Op"""

        self.hart_order = RelationBuilder()
        """An ordering between Harts, such that for (a, b) in `hart_order`,
        MemoryOps in `a` are ordered before MemoryOps in `b` in the global
        memory order.  This can be used to set up harts that set up the
        initial condition or check the final condition, for example.
        """

        self.force = RelationBuilder()
        """MemoryOp pairs where the first is forced before the second in gmo.

        This is used for things like forcing operations in the initial thread
        to appear in gmo before operations in the main thread, and forcing
        remote SFENCE.VMA instructions to be ordered in the right place wrt
        operations in the thread issuing an sbi_remote_sfence_vma() (as would
        in reality be ordered via IPIs).
        """

        self.implicit_program_order = RelationBuilder()
        """Like program order, but includes ImplicitOps as well."""

        # Other dictionaries tracking per-Op information

        self.base_address = {}
        "For each MemoryOp or SFenceVMA, the base address"

        self.width_bytes = {}
        "For each MemoryOp or SFenceVMA, the width of the operation"

        self.return_value = {}
        "For each Read, the value returned (of size width_bytes)"

        self.write_value = {}
        "For each Write, the value written (of size matching width_bytes)"

        self.satp_asid = {}
        "For each SFenceVMA or ImplicitOp, the value of satp.asid"

        self.meta = {}
        "For each Op, the parsing metadata that generated it"

        self.initial_values = {}
        "For each byte address, a preset initial value for that address."

        self.label = {}
        "For each Op, a string label"

        # Builder internal variables

        self._initial_value = z3.Function(
            "initial_value",
            z3.BitVecSort(self._physical_address_bits),
            z3.BitVecSort(8),
        )
        "For each byte address, the initial value of that address"

        self._latest_cache = {}
        """For each (read, address), a Relation representing the Write from
        which the Read reads its value.  Used as a cache for
        self._latest_write; not intended to be accessed directly.
        """

    ############################################################################

    def _addresses(self, op_name):
        "The list of addresses touched by MemoryOp `op_name`"

        # Get the base address
        address = self.base_address[op_name]
        if address is None:
            raise TypeError

        # Get the width
        width_bytes = self.width_bytes[op_name]
        if width_bytes > 8:
            raise TypeError(f"cannot calculate addresses for {op_name}")

        # Generate all (base + offset) byte addresses.  Addresses wrap around
        # if they overflow
        addresses = []
        for offset in range(width_bytes):
            addresses.append(simplify(address + offset))
        return addresses

    def condition(self, op_name):
        "The condition that indicates whether `op_name` actually executes"

        try:
            return z3.Or(self.Op[(op_name,)])
        except KeyError:
            raise KeyError(f"Unknown op {op_name}")

    def _byte(self, op_name, address, value):
        """Get the byte at `address`, if `value` is the value at its associated
        base address.  Intended to be called from _read_byte or _write_byte.
        """

        if self._big_endian:
            raise Exception("big-endian mode not supported")

        offset_bits = (address - self.base_address[op_name]) << 3
        offset_bits = resize(offset_bits, value.size())
        return z3.Extract(7, 0, value >> offset_bits)

    def _read_byte(self, op_name, address):
        "The byte that Op `op_name` reads for address `address`"

        return self._byte(op_name, address, self.return_value[op_name])

    def _write_byte(self, op_name, address):
        "The byte that Op `op_name` writes to address `address`"

        return self._byte(op_name, address, self.write_value[op_name])

    def _touches_address(self, op_name, address):
        "Return a boolean indicating whether `op_name` touches `address`"

        # Check for overflow, which is defined to wrap around
        upper_bound = self.base_address[op_name] + self.width_bytes[op_name]
        overflow = z3.ULT(upper_bound, self.base_address[op_name])

        return z3.If(
            overflow,
            # If overflow, account for wraparound
            z3.Or(
                z3.UGE(address, self.base_address[op_name]),
                z3.ULT(address, upper_bound),
            ),
            # If no overflow, the address should be in [base, base + offset)
            z3.And(
                z3.UGE(address, self.base_address[op_name]),
                z3.ULT(address, upper_bound),
            ),
        )

    def _overlap(self, a, b):
        "Return a boolean indicating that `a` and `b` overlap"

        base_a = self.base_address[a]
        base_b = self.base_address[b]
        max_a = base_a + self.width_bytes[a]
        max_b = base_b + self.width_bytes[b]

        overflow_a = z3.ULT(max_a, base_a)
        overflow_b = z3.ULT(max_b, base_b)

        # The non-overflow case is:
        # z3.And(z3.ULT(base_a, max_b), z3.ULT(base_b, max_a))
        #
        # If max_b overflows, then base_a < max_b is effectively true, and vice
        # versa.  Therefore, logical-or the overflow conditions in too.

        return z3.And(
            z3.Or(overflow_b, z3.ULT(base_a, max_b)),
            z3.Or(overflow_a, z3.ULT(base_b, max_a)),
        )

    def _loc(self, s):
        "Return the subset of 's->s' relating Ops with overlapping footprints"

        r = RelationBuilder()
        for k, v in s.cross(s).items():
            if len(k) != 2:
                raise TypeError(k)
            if (
                k[0] not in self.base_address
                or k[1] not in self.base_address
                or self.base_address[k[1]] is None
                or self.base_address[k[0]] is None
            ):
                continue
            r.update(k, z3.And(v, self._overlap(k[0], k[1])))
        return r.relation()

    def _filter_by_address(self, s, address):
        "Return the set of Ops in `s` that touch `address"

        r = RelationBuilder()
        for k, v in s.items():
            if len(k) != 1:
                raise TypeError
            if k[0] not in self.base_address:
                continue
            if self.base_address[k[0]] is None:
                continue
            r.update(k, z3.And(v, self._touches_address(k[0], address)))
        return r.relation()

    ############################################################################

    def _latest_write(self, m, read, address):
        "Calculate the write that feeds byte `address` of `read`."

        try:
            return self._latest_cache[read, address]
        except KeyError:
            pass

        # The latest write to the same address prior to r in program order or
        # global memory order
        r = Singleton(read, self.condition(read))
        candidates = r.join(m["~po + ~gmo"]).intersect(m["Write"])
        candidates = self._filter_by_address(candidates, address)
        latest = candidates - candidates.join(m["~gmo"])

        # Cache the result, just to speed up future lookups
        self._latest_cache[read, address] = latest
        return latest

    def _rf(self, m):
        """Calculate the relation `rf`.

        `(w, r) in rf` implies that `r` returns the value for at least one byte
        written by `w`.  In mixed-size settings, there may be more than one
        such `w` for a given `r`.
        """

        rf = RelationBuilder()
        for (r,) in m["Read"]:
            for address in self._addresses(r):
                latest = self._latest_write(m, r, address)
                for k, v in latest.items():
                    assert len(k) == 1
                    rf.update((k[0], r), v)
        return rf.relation()

    def _latest_byte_value(self, m, r, address):
        "Calculate the value that `r` returns for `address`."

        # Start by assuming the initial value for that address...
        value = self._initial_value(address)

        # ...then overwrite it with the byte value of the latest write, if
        # applicable
        r = self._latest_write(m, r, address)
        for k, v in r.items():
            assert len(k) == 1
            if k[0] not in self.write_value:
                assert false(v)
                continue
            byte = self._write_byte(k[0], address)
            value = z3.If(v, byte, value)

        return simplify(value)

    ############################################################################

    def _initial_values(self, m):
        # 'a' is a quantifier
        a = z3.BitVec("a", self._physical_address_bits)

        # First track the specified initial values
        disjuncts = []
        for k, v in self.initial_values.items():
            m.fact(self._initial_value(k) == v, f"Initial value {k}")
            disjuncts.append(a == k)

        # Then constrain the remaining initial values to be zero
        if not self._arbitrary_initial_values:
            m.fact(
                z3.ForAll(
                    a,
                    z3.Implies(
                        z3.Not(z3.Or(disjuncts)), self._initial_value(a) == 0
                    ),
                ),
                "Other initial values",
            )

    ############################################################################

    def _interleave_threads(self, m):
        thread_ops = self.thread_ops.relation()
        explicit = self.Op.relation() - self.ImplicitOp.relation()
        for hart, thread in self.threads.relation().keys():
            if thread == hart:
                continue
            h_events = Singleton(hart).join(thread_ops).intersect(explicit)
            t_events = Singleton(thread).join(thread_ops).intersect(explicit)
            b = z3.Bool(f"_interleave_{hart}_{thread}")
            for k1, v1 in h_events.items():
                for k2, v2 in t_events.items():
                    self.po.update(k1 + k2, z3.And(v1, v2, b))
                    self.po.update(k2 + k1, z3.And(v1, v2, z3.Not(b)))

    ############################################################################

    def _ppo(self, m):
        """Calculate RVWMO preserved program order"""

        m["ppo1"] = m["MemoryOp <: po_loc :> Write"]
        m["ppo2"] = m["rdw"]
        m["ppo3"] = m["(AMO + StoreConditional) <: (rf & po)"]
        m["ppo4rr"] = m["(Read  <: po :> (FencePR & FenceSR)).(po :> Read)"]
        m["ppo4rw"] = m["(Read  <: po :> (FencePR & FenceSW)).(po :> Write)"]
        m["ppo4wr"] = m["(Write <: po :> (FencePW & FenceSR)).(po :> Read)"]
        m["ppo4ww"] = m["(Write <: po :> (FencePW & FenceSW)).(po :> Write)"]
        m["ppo4tso1"] = m["(Read <: po :> (FenceTSO)).(po :> MemoryOp)"]
        m["ppo4tso2"] = m["(MemoryOp <: po :> (FenceTSO)).(po :> Write)"]
        m["ppo4"] = m["ppo4rr + ppo4rw + ppo4wr + ppo4ww + ppo4tso1 + ppo4tso2"]
        m["ppo5"] = m["Aq <: po :> MemoryOp"]
        m["ppo6"] = m["MemoryOp <: po :> Rl"]
        m["ppo7"] = m["RCsc <: po :> RCsc"]
        m["ppo8"] = m["pair"]
        m["ppo9"] = m["addrdep"]
        m["ppo10"] = m["datadep"]
        m["ppo11"] = m["ctrldep.po :> Write"]
        m["ppo12"] = m["(addrdep + datadep).(rf & po)"]
        m["ppo13"] = m["addrdep.po :> Write"]

        return m[
            "ppo1 + ppo2 + ppo3 + ppo4 + ppo5 + ppo6 + ppo7 + "
            "ppo8 + ppo9 + ppo10 + ppo11 + ppo12 + ppo13"
        ]

    def _rdw(self, m):
        "Cacluate the 'read-different-writes' subset of po_loc"

        rdw = RelationBuilder()
        for k, v in m["Read <: po_loc :> Read"].items():
            assert len(k) == 2
            k0 = Singleton(k[0])
            k1 = Singleton(k[1])

            for address in self._addresses(k[1]):
                # See if there is a write to the same address between k0 and k1
                # in program order
                write_in_between = self._filter_by_address(
                    m["Write"]
                    .intersect(k0.join(m["po"]))
                    .intersect(k1.join(m["~po"])),
                    address,
                ).some()

                rdw.update(
                    k,
                    z3.And(
                        # k0 and k1 both execute,
                        v,
                        # k0 and k1 both execute,
                        z3.Not(write_in_between),
                        # k0 touches address
                        self._touches_address(k[0], address),
                        # the reads return values for this byte address from
                        # different stores
                        self._latest_write(m, k[0], address)
                        != self._latest_write(m, k[1], address),
                    ),
                )
        return rdw.relation()

    def _pair(self, m):
        """Calculate LR/SC pairs

        Note that address-translation A/D bit updates are handled via
        implicit_pair instead of via this calculation.

        Implicit in that is the fact that a hardware update of a D bit does
        *not* kill the reservation.
        """

        pair = RelationBuilder()
        for (sc,), v in m["StoreConditional - ImplicitOp"].items():
            # Calculate the most recent LR or SC prior to the SC
            w = Singleton(sc, v)
            candidates = w.join(m["~po"]).intersect(
                m["LoadReserve + StoreConditional"]
            )
            r = candidates - candidates.join(m["~po"]) - m["StoreConditional"]

            # If the SC succeeds, the most recent LR or SC prior to w is indeed
            # an LR
            m.fact(
                z3.Implies(v, self.condition(sc) == r.some()), f"{sc} is paired"
            )

            # Update the "pair" relation
            for k, v in r.cross(w).items():
                pair.update(k, v)

                # If the SC succeeds, the LR and SC must have matching addresses
                if self._sc_address_must_match:
                    m.fact(
                        z3.Implies(
                            v,
                            self.base_address[k[0]] == self.base_address[k[1]],
                        )
                    )

        return pair.relation().union(m["implicit_pair"])

    def _gmo(self, m):
        "Generate the global memory order relation"

        memoryop = m["MemoryOp"]
        keys = list(m["MemoryOp"].keys())
        r = RelationBuilder()

        # For all pairs of distinct MemoryOps...
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = keys[i]
                b = keys[j]

                # ...either a->b or b->a is in gmo
                v = z3.Bool(f"_{a[0]}_gmo_{b[0]}")
                r.update(a + b, z3.And(memoryop[a], memoryop[b], v))
                r.update(b + a, z3.And(memoryop[a], memoryop[b], z3.Not(v)))

        # m is irreflexive by construction, but we need a fact to make it
        # transitive
        r = r.relation()
        m.fact(r.transitive(), "gmo transitive")
        return r

    ############################################################################

    def _load_value_axiom(self, m):
        "Calculate the RVWMO Load Value Axiom"

        conjuncts = []
        for (r,) in m["Read"]:
            for address in self._addresses(r):
                conjuncts.append(
                    self._read_byte(r, address)
                    == self._latest_byte_value(m, r, address)
                )
        return z3.And(conjuncts)

    ############################################################################

    def _atomicity_axiom(self, m):
        "Calculate the RVWMO Atomicity Axiom"

        if self._allow_misaligned_atomics:
            raise Exception("allow_misaligned_atomics not implemented")

        conjuncts = []

        for (sc,) in m["StoreConditional"]:
            w = Singleton(sc)
            r = w.join(m["~pair"])

            for k, v in r.items():
                for a in self._addresses(k[0]):
                    # If r and w are paired load and store operations generated
                    # by aligned LR and SC instructions in a hart h, s is a
                    # store to byte x, and r returns a value written by s,
                    s = self._latest_write(m, k[0], a)

                    # then s must precede w in the global memory order,
                    conjuncts.append(z3.Implies(v, s.in_(w.join(m["~gmo"]))))

                    # and there can be no store from a hart other than h to
                    # byte x
                    writes = self._filter_by_address(m["Write"], a)
                    writes -= w.join(m["~hart_ops.hart_ops"])
                    # following s
                    following_init = writes.if_(s.no())
                    following_s = s.join(m["gmo"]).union(following_init)
                    # and preceding w in the global memory order
                    preceding_w = w.join(m["~gmo"])
                    conjuncts.append(
                        z3.Implies(
                            v,
                            writes.intersect(following_s)
                            .intersect(preceding_w)
                            .no(),
                        )
                    )

        return z3.And(conjuncts)

    ############################################################################

    def _address_translation_axiom(self, m):
        # Generate the same_satp_asid relation
        same_satp_asid = RelationBuilder()
        for (f,) in m["SFenceVMA"]:
            if f not in self.satp_asid:
                # No ASID specified; applies to all MemoryOps
                for (k,), v in m["ImplicitOp"].items():
                    same_satp_asid.update((f, k), z3.And(v, self.condition(f)))
            else:
                # ASID specified; applies to MemoryOps generating implicit
                # accesses under the specified ASID
                for (k,), v in m["ImplicitOp"].items():
                    same_satp_asid.update(
                        (f, k),
                        z3.And(
                            v,
                            self.condition(f),
                            self.satp_asid[f] == self.satp_asid[k],
                        ),
                    )
        m["same_satp_asid"] = same_satp_asid.relation()

        # same_satp_asid is implicitly SFenceVMA <: ... :> ImplicitOp
        m["sfence_po"] = m["implicit_program_order & loc & same_satp_asid"]

        return z3.And(
            # If an implicit read is po-after an sfence.vma with a matching
            # address (including sfence.vma with rs1=x0), then the implicit
            # read follows (all ops prior to the sfence.vma in global memory
            # order) in global memory order.
            m["MemoryOp <: po.sfence_po in gmo"],
            # ImplicitOp reads precede their associated explicit ops in the
            # global memory order
            m["(translation_order :> (Op-ImplicitOp)).*po :> MemoryOp in gmo"],
        )

    ############################################################################

    def fact(self, f, name=None):
        self._model.fact(f, name)

    def model(self, verbose=False):
        """Return a `RelationalModel` representing the litmus test"""

        if verbose:
            print("Generating model...")

        m = self._model

        # Set up the initial values of memory

        self._initial_values(m)

        # Define the interleaving between threads

        self._interleave_threads(m)

        # Copy the `RelationBuilder`s in `self` into `Relation`s in `m`

        for k, v in vars(self).items():
            if isinstance(v, RelationBuilder):
                # Convert the RelationBuilder into a Relation
                r = v.relation()

                # Store `r` into `m`, and if verbose print it out
                m[k] = r
                if verbose:
                    for k1, v1 in r.items():
                        if const(v1):
                            print(f"{k}[{k1}] = {v1}")
                        else:
                            print(f"{k}[{k1}] = ...")

        # Calculate basic derived relations

        m.fact(m["^po"].irreflexive(), name="po acyclic")

        m["gmo"] = self._gmo(m)

        m["hart_ops"] = m["threads.thread_ops"]

        m["loc"] = self._loc(m["MemoryOp + SFenceVMA"])
        m["po_loc"] = m["po & loc"]

        m["pair"] = self._pair(m)

        m["rf"] = self._rf(m)
        m["rdw"] = self._rdw(m)

        m["ppo"] = self._ppo(m)

        # Hart ordering

        m["hart_force"] = m[
            "MemoryOp <: ~hart_ops.hart_order.hart_ops :> MemoryOp"
        ]
        m.fact(m["hart_force in gmo"], "forced hart order")

        # RVWMO

        m.fact("ppo in gmo")
        m.fact("force in gmo")
        m.fact(self._load_value_axiom(m), "load value axiom")
        m.fact(self._atomicity_axiom(m), "atomicity axiom")

        # Address translation axiom

        m.fact(self._address_translation_axiom(m), "address translation axiom")

        # Sanity checks

        m.assert_("MemoryOp in Op")
        m.assert_("Read in MemoryOp")
        m.assert_("Write in MemoryOp")
        m.assert_("DummyMemoryOp in MemoryOp")
        m.assert_("no DummyMemoryOp & (Read + Write)")
        m.assert_("no MemoryOp & Fence")
        m.assert_("ImplicitOp in Read + LoadReserve + StoreConditional")
        m.assert_("LoadReserve in Read")
        m.assert_("StoreConditional in Write")
        m.assert_("AMO = Read & Write")
        m.assert_("Aq & Rl in RCsc")
        m.assert_("FencePR + FencePW + FenceSR + FenceSW + FenceTSO in Fence")
        m.assert_("addrdep + ctrldep + datadep + pair in po + implicit_pair")
        m.assert_("no ImplicitOp <: po")
        m.assert_("no po :> ImplicitOp")
        m.assert_("force in DummyMemoryOp->DummyMemoryOp")

        # See the SBIRemoteSFenceVMA case in
        # execution.Thread.execute_instruction()
        m.assert_("Hart <: iden in threads")

        return m


class LitmusTestBuilder(RVWMOBuilder):
    """Helper class to build an RVWMOBuilder incrementally.

    Class methods manage all relations and dictionaries in the builder except:
    * addrdep
    * ctrldep
    * datadep
    * implicit_pair
    * translation_order
    * hart_order
    * initial_values
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._po_prev = {}
        "For each software thread, the preceding operations in program order"

        self._implicit_program_order_prev = {}
        """For each software thread, the preceding operations in implicit PO.

        This includes ImplicitOps into program order."""

    def thread(self, thread_name, hart_name):
        self.Hart.update((hart_name,), True)
        self.Thread.update((thread_name,), True)
        self.threads.update((hart_name, thread_name), True)

        # Keep track of multiple software threads on the same hart
        siblings = {}
        for hart, thread in self.threads.relation().keys():
            siblings.setdefault(hart, []).append(thread)

    def _operation(
        self, meta, thread_name, op_name, label, condition, implicit=False
    ):
        if (op_name,) in self.Op:
            raise Exception("Duplicate op {op_name}")

        self.Op.update((op_name,), condition)
        self.thread_ops.update((thread_name, op_name), condition)
        self.meta[op_name] = meta
        self.label[op_name] = label

        # Update implicit program order
        for k in self._implicit_program_order_prev.setdefault(thread_name, []):
            v = self.condition(k)
            self.implicit_program_order.update((k, op_name), z3.And(v, condition))
        self._implicit_program_order_prev[thread_name].append(op_name)

        # Implicit accesses are not entered into program order
        if implicit:
            return

        # Update program order for the same thread
        for k in self._po_prev.setdefault(thread_name, []):
            v = self.condition(k)
            self.po.update((k, op_name), z3.And(v, condition))

        self._po_prev[thread_name].append(op_name)

    def _memory_operation(
        self,
        meta,
        thread_name,
        op_name,
        label,
        condition,
        width_bytes,
        address,
        aq,
        rl,
        rcsc,
        implicit=False,
        satp_asid=None,
    ):
        self._operation(
            meta, thread_name, op_name, label, condition, implicit=implicit
        )
        self.width_bytes[op_name] = width_bytes
        self.base_address[op_name] = address

        if aq:
            self.Aq.update((op_name,), condition)
        if rl:
            self.Rl.update((op_name,), condition)
        if rcsc:
            self.RCsc.update((op_name,), condition)

        self.MemoryOp.update((op_name,), condition)
        if implicit:
            self.ImplicitOp.update((op_name,), condition)
            self.satp_asid[op_name] = satp_asid

    def read(
        self,
        meta,
        thread_name,
        op_name,
        label,
        condition,
        width_bytes,
        address,
        aq=False,
        rl=False,
        reserve=False,
        implicit=False,
        satp_asid=None,
    ):
        self._memory_operation(
            # FIXME Assumes l{b|h|w|d}.aq is RCpc
            meta,
            thread_name,
            op_name,
            label,
            condition,
            width_bytes,
            address,
            aq,
            rl,
            rcsc=(aq and rl),
            implicit=implicit,
        )
        self.Read.update((op_name,), condition)
        self.return_value[op_name] = z3.BitVec(f"{op_name}_rv", width_bytes * 8)
        if reserve:
            self.LoadReserve.update((op_name,), condition)

    def write(
        self,
        meta,
        thread_name,
        op_name,
        label,
        condition,
        width_bytes,
        address,
        value,
        aq=False,
        rl=False,
        conditional_dst=False,
        implicit=False,
        satp_asid=None,
    ):
        if conditional_dst:
            success = z3.Bool(f"{op_name}_success")
            condition = z3.And(condition, success)
            self.StoreConditional.update((op_name,), condition)

        self._memory_operation(
            # FIXME Assumes s{b|h|w|d}.rl is RCpc
            meta,
            thread_name,
            op_name,
            label,
            condition,
            width_bytes,
            address,
            aq,
            rl,
            rcsc=(aq and rl),
            implicit=implicit,
        )
        self.Write.update((op_name,), condition)

        if isinstance(value, int):
            self.write_value[op_name] = z3.BitVecVal(value, width_bytes * 8)
        else:
            if value.size() != width_bytes * 8:
                raise Exception(
                    f"store value of width {value.size()} "
                    f"does not match 8 * width_bytes={width_bytes}"
                )
            self.write_value[op_name] = value

    def amo(
        self,
        meta,
        thread_name,
        op_name,
        label,
        condition,
        width_bytes,
        address,
        op,
        value,
        aq=False,
        rl=False,
    ):
        self._memory_operation(
            meta,
            thread_name,
            op_name,
            label,
            condition,
            width_bytes,
            address,
            aq,
            rl,
            rcsc=(aq or rl),
        )
        self.return_value[op_name] = z3.BitVec(f"{op_name}_rv", width_bytes * 8)
        self.Read.update((op_name,), condition)
        self.Write.update((op_name,), condition)
        self.AMO.update((op_name,), condition)

        value = op(self.return_value[op_name], value)

        if value.size() != width_bytes * 8:
            raise Exception(
                f"AMO value of width {value.size()} "
                f"does not match 8 * width_bytes={width_bytes}"
            )
        self.write_value[op_name] = value

    def fence(
        self, meta, thread_name, op_name, label, condition, pr, pw, sr, sw, tso
    ):
        self._operation(meta, thread_name, op_name, label, condition)
        self.Fence.update((op_name,), condition)
        if pr:
            self.FencePR.update((op_name,), condition)
        if pw:
            self.FencePW.update((op_name,), condition)
        if sr:
            self.FenceSR.update((op_name,), condition)
        if sw:
            self.FenceSW.update((op_name,), condition)
        if tso:
            self.FenceTSO.update((op_name,), condition)

    def sfence_vma(
        self,
        meta,
        thread_name,
        op_name,
        label,
        condition,
        address=None,
        satp_asid=None,
    ):
        self._operation(meta, thread_name, op_name, label, condition)
        self.SFenceVMA.update((op_name,), condition)

        if address is not None:
            self.base_address[op_name] = address
            self.width_bytes[op_name] = 1
        else:
            # (0, -1) causes this SFence to cover the entire address space
            self.base_address[op_name] = resize(0, self._physical_address_bits)
            self.width_bytes[op_name] = -1

        if satp_asid is not None:
            self.satp_asid[op_name] = satp_asid

    def branch(self, meta, thread_name, op_name, label, condition):
        self._operation(meta, thread_name, op_name, label, condition)

    def dummy_memory_op(
        self,
        meta,
        thread_name,
        op_name,
        label,
        condition,
        aq=False,
        rl=False,
    ):
        self._memory_operation(
            meta,
            thread_name,
            op_name,
            label,
            condition,
            width_bytes=None,
            address=None,
            aq=aq,
            rl=rl,
            rcsc=(aq and rl),
            implicit=False,
        )
        self.DummyMemoryOp.update((op_name,), condition)
