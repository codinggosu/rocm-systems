## Motivation

On 16-NIC systems with `NCCL_IB_MERGE_NICS=0` (e.g. gfx910 1H16P),
`ncclCommInitRank()` doesn't return. The cause is `permuteNetIds()` at
`projects/rccl/src/graph/rome_models.cc:2414`, which enumerates NIC
permutations recursively and only checks the matching predicates at the
leaf. When no permutation satisfies the predicates (the realistic case
on any unmerged 16-NIC system not exactly matching `rome_model_68`),
the function walks every one of `N!` orderings before returning false.
At `N=16` that's `~1.7 * 16!` = 35 trillion function calls. At the rate
I measured (~100M calls/sec on x86), that's about 4 days of CPU work
inside a single init call.

This re-ups [ROCm/rccl#1566](https://github.com/ROCm/rccl/pull/1566),
closed during the `ROCm/rccl` to `ROCm/rocm-systems` migration, with a
correctness fix to the DP formulation that's described below.

## Technical Details

Recursion is replaced with a memoized bitmask DP. The non-obvious
detail: both predicates (NUMA, `gdrLevel`) are evaluated **inside the
per-step transition**, not at the leaf.

`dp[mask]` only records which topo NICs have been used. It doesn't
record which ref position each was paired with. But the `gdrLevel`
check is pairing-specific: it compares
`ref->gdrLevel[i*nGpus+j]` against `topo->gdrLevel[n[i]*nGpus+g[j]]`,
so its result at a given `mask` depends on the prefix `n[0..pos-1]`.
If you check `gdrLevel` only at the leaf (the formulation in #1566),
the first failing leaf caches `memo[full_mask] = false`. Every sibling
permutation reaching `full_mask` then hits the cached false and is
skipped without re-checking. That's a silent correctness bug on any
input with duplicate `nicNuma` entries, and every shipped
`rcclRomeModel` has those (see the all-1s pattern in `rome_model_68`
or the paired pattern in `rome_model_84`).

Hoisting the check fixes it. Once a choice `n[pos] = i` passes NUMA
and `gdrLevel` for that specific pairing across all `j`, the remaining
subproblem only depends on the set of NICs still available. `mask`
becomes a valid memo key.

Other small things in the same patch:

- Preserves the `PATH_PXB <-> PATH_PIX` equivalence rule at
  `rome_models.cc:2431-2433` (added after #1566 was authored).
- Uses `1u << N` for the `std::vector<int>` size, with an `N > 30`
  early return. The model table caps `N` at 16 today; the guard is
  defensive against future expansions where `1 << 31` would be signed
  UB.
- Wrapper signature is unchanged. The three call sites at
  `rome_models.cc:2594, 2783, 2958` keep working. The `int* time`
  parameter now counts DP node visits, which preserves its
  diagnostic meaning.

One observable difference: the DP returns the lexicographically
smallest valid permutation, while the recursive code returned whichever
permutation the `(m+2)%nnets` seed plus swap-exploration found first.
On inputs with multiple valid permutations the returned `n[]` may
differ. No caller depends on the specific value; `n[]` is just an
opaque remap table passed to `parseGraph(..., net_map, ...)`.

## JIRA ID

N/A (external contribution).

## Test Plan

Two things, both runnable in under a minute.

### 1. Standalone benchmark (no ROCm needed)

`permute_nic_bench.cpp` (attached) contains verbatim copies of the
recursive function body and the new DP body, plus the leaf-only-DP
formulation from #1566 for comparison. It compiles with plain `g++
-O2 -std=c++17`. Three scenarios:

- **Worst case (the actual hang):** synthetic ref/topo where no
  permutation satisfies the predicate. The recursive code must walk
  all `N!` orderings; the DP collapses at `pos=0`.
- **Find-then-stop:** identity-diagonal `gdrLevel`. The recursive code
  exits at the first valid permutation it stumbles into.
- **Correctness witness:** `N=4`, all-zero NUMA, `gdrLevel` whose
  unique valid permutation is `[0,1,3,2]`. Shows the leaf-only DP
  (#1566 formulation) returning false where it should return true.

### 2. Regression on existing models

Built baseline and patched `tools/topo_expl/` (ROCm 7.2.1, MI300X
host). Ran both binaries against all `topo_expl` models that exercise
`permuteNetIds`. Diffed full `NCCL_DEBUG=INFO` output.

## Test Result

### Regression on existing models

`diff` of `NCCL_DEBUG=INFO` output between baseline and patched is
empty on every model that reaches `permuteNetIds`. Sample:

```
$ diff <(./topo_expl_baseline -m 52 -n 1 2>&1) \
       <(./topo_expl          -m 52 -n 1 2>&1)
(no output)
```

Model 52 is `topo_16p1h.xml`, the 16-GPU rome system. Both binaries
emit 308 lines and pick the same NIC permutation.

### Worst-case scenario (no permutation satisfies the predicate)

This is the case that hangs on real hardware:

| N  | recursive (calls / wall)         | new DP (calls / wall) |
|---:|---------------------------------:|----------------------:|
| 8  |  69_281 / 0.62 ms                | 1 / ~0 ms             |
| 10 | 6_235_301 / 60.9 ms              | 1 / ~0 ms             |
| 11 | 68_588_312 / 706.7 ms            | 1 / 0.003 ms          |
| 16 | ~1.7 * 16! ~= 35e12, **~4 days** at the measured rate | 1 / ~0 ms |

The DP bails at `pos=0`: every choice for the first NIC fails the
gdrLevel check, so the search tree never expands.

### Find-then-stop (a valid permutation exists)

Identity-diagonal `gdrLevel` so the recursion exits as soon as it
stumbles into the identity:

| N  | recursive             | new DP   | speedup  |
|---:|----------------------:|---------:|---------:|
| 8  | 0.30 ms               | 0.001 ms | ~300x    |
| 10 | 26.4 ms               | 0.002 ms | ~13_000x |
| 12 | 3552 ms               | 0.012 ms | ~300_000x|
| 14 | 1_005_176 ms (16m45s) | 0.044 ms | ~23e6    |

### Correctness witness

| algorithm                                  | `found` | returned `n[]` |
|--------------------------------------------|:-------:|----------------|
| recursive (current upstream)               | true    | `[0,1,3,2]`    |
| naive leaf-only DP (formulation in #1566)  | **false** | (none)       |
| hoisted DP (this PR)                       | true    | `[0,1,3,2]`    |

The leaf-only DP returns false on a case that has a valid permutation.
Without the per-step hoist, this PR would have shipped a silent
correctness regression on production rome models since they all have
duplicate `nicNuma`.

### Reproducing

```bash
g++ -O2 -std=c++17 permute_nic_bench.cpp -o permute_nic_bench
./permute_nic_bench --maxN 12     # full output in seconds
./permute_nic_bench --maxN 14     # adds the 17-minute recursive baseline
```

The bench is attached, not added to the rccl tree.

## Submission Checklist

- [ ] Look over the contributing guidelines at
  https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.
