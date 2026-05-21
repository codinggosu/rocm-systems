// Standalone benchmark for the permuteNetIds optimization in
// projects/rccl/src/graph/rome_models.cc.
//
// Runs both the original O(N!) recursive enumeration and the bitmask DP
// replacement on synthetic rcclRomeModel pairs, comparing correctness and
// wall-clock time across a range of N.
//
// Build:   g++ -O2 -std=c++17 permute_nic_bench.cpp -o permute_nic_bench
// Usage:   ./permute_nic_bench [--maxN N]   (default maxN=14; old code hangs at 16)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---- Minimal stand-ins for rccl types ----------------------------------------
// These mirror the fields of struct rcclRomeModel and the PATH_* enum that
// permuteNetIds actually reads. Definitions are isolated so the bench links
// without any rccl headers.

enum {
    PATH_LOC = 0,
    PATH_NVL = 1,
    PATH_PIX = 2,
    PATH_PXB = 3,
    PATH_PXN = 4,
    PATH_PHB = 5,
    PATH_SYS = 6,
};

struct rcclRomeModel {
    int  nGpus;
    int  nNics;
    int* nicNuma;     // [nNics]
    int* gdrLevel;    // [nNics * nGpus]
};

// ---- The two algorithms under test -------------------------------------------
// Both bodies are verbatim copies of the production code at
// projects/rccl/src/graph/rome_models.cc:permuteNetIds, modulo using the
// stand-in struct above.

static bool permuteNetIds_recursive(int *n, int *g, int s, int last,
                                    rcclRomeModel* ref, rcclRomeModel* topo,
                                    int* time, bool ignore_numa) {
    (*time)++;
    if (s == last) {
        int i, j;
        if (!ignore_numa) {
            for (i = 0; i < ref->nNics; i++) {
                if (ref->nicNuma[i] != topo->nicNuma[n[i]]) break;
            }
            if (i < ref->nNics) return false;
        }
        for (i = 0; i < ref->nNics; i++) {
            for (j = 0; j < ref->nGpus; j++) {
                int topoLvl = topo->gdrLevel[n[i]*ref->nGpus+g[j]];
                int refLvl  = ref->gdrLevel[i*ref->nGpus+j];
                if (topoLvl == PATH_PXN) continue;
                if ((refLvl == PATH_PXB && topoLvl == PATH_PIX) ||
                    (refLvl == PATH_PIX && topoLvl == PATH_PXB)) continue;
                if (refLvl != topoLvl) break;
            }
            if (j < ref->nGpus) break;
        }
        if (i < ref->nNics) return false;
        return true;
    } else {
        for (int i = s; i <= last; i++) {
            std::swap(n[s], n[i]);
            if (permuteNetIds_recursive(n, g, s+1, last, ref, topo, time, ignore_numa)) return true;
            std::swap(n[s], n[i]);
        }
    }
    return false;
}

static bool permuteNetIdsDP(int mask, int pos, int N, int *n, int *g,
                            rcclRomeModel* ref, rcclRomeModel* topo,
                            int* time, bool ignore_numa, std::vector<int>& memo) {
    (*time)++;
    if (pos == N) return true;
    if (memo[mask] != -1) return memo[mask] == 1;

    for (int i = 0; i < N; i++) {
        if (mask & (1u << i)) continue;
        if (!ignore_numa && ref->nicNuma[pos] != topo->nicNuma[i]) continue;
        bool gdrOk = true;
        for (int j = 0; j < ref->nGpus; j++) {
            int topoLvl = topo->gdrLevel[i * ref->nGpus + g[j]];
            int refLvl  = ref->gdrLevel[pos * ref->nGpus + j];
            if (topoLvl == PATH_PXN) continue;
            if ((refLvl == PATH_PXB && topoLvl == PATH_PIX) ||
                (refLvl == PATH_PIX && topoLvl == PATH_PXB)) continue;
            if (refLvl != topoLvl) { gdrOk = false; break; }
        }
        if (!gdrOk) continue;

        n[pos] = i;
        if (permuteNetIdsDP(mask | (1u << i), pos + 1, N, n, g, ref, topo,
                            time, ignore_numa, memo)) {
            memo[mask] = 1;
            return true;
        }
    }
    memo[mask] = 0;
    return false;
}

static bool permuteNetIds_dp(int *n, int *g, int s, int last,
                             rcclRomeModel* ref, rcclRomeModel* topo,
                             int* time, bool ignore_numa) {
    int N = last - s + 1;
    if (N <= 0 || N > 30) return false;
    std::vector<int> memo(1u << N, -1);
    return permuteNetIdsDP(0, 0, N, n + s, g, ref, topo, time, ignore_numa, memo);
}

// Reference implementation of a "naive" bitmask DP: NUMA pruned per-step, but
// gdrLevel checked only at the leaf. This is the formulation in PR #1566 as
// originally submitted to ROCm/rccl. It is *unsound* on inputs where NUMA has
// duplicates (which every shipped rcclRomeModel does) because dp[mask] then
// fails to capture the order-dependent gdrLevel check at the leaf -- a sibling
// permutation that would have succeeded gets poisoned by a cached false from
// the first failing prefix. Included here only as a correctness comparator.
static bool permuteNetIdsDP_naive(int mask, int pos, int N, int *n, int *g,
                                  rcclRomeModel* ref, rcclRomeModel* topo,
                                  int* time, bool ignore_numa, std::vector<int>& memo) {
    (*time)++;
    if (memo[mask] != -1) return memo[mask] == 1;
    if (pos == N) {
        int i, j;
        for (i = 0; i < N; i++) {
            for (j = 0; j < ref->nGpus; j++) {
                int topoLvl = topo->gdrLevel[n[i] * ref->nGpus + g[j]];
                int refLvl  = ref->gdrLevel[i * ref->nGpus + j];
                if (topoLvl == PATH_PXN) continue;
                if ((refLvl == PATH_PXB && topoLvl == PATH_PIX) ||
                    (refLvl == PATH_PIX && topoLvl == PATH_PXB)) continue;
                if (refLvl != topoLvl) break;
            }
            if (j < ref->nGpus) break;
        }
        bool ok = (i == N);
        memo[mask] = ok ? 1 : 0;
        return ok;
    }
    for (int i = 0; i < N; i++) {
        if (mask & (1u << i)) continue;
        if (!ignore_numa && ref->nicNuma[pos] != topo->nicNuma[i]) continue;
        n[pos] = i;
        if (permuteNetIdsDP_naive(mask | (1u << i), pos + 1, N, n, g, ref, topo,
                                  time, ignore_numa, memo)) {
            memo[mask] = 1;
            return true;
        }
    }
    memo[mask] = 0;
    return false;
}

static bool permuteNetIds_dp_naive(int *n, int *g, int s, int last,
                                   rcclRomeModel* ref, rcclRomeModel* topo,
                                   int* time, bool ignore_numa) {
    int N = last - s + 1;
    if (N <= 0 || N > 30) return false;
    std::vector<int> memo(1u << N, -1);
    return permuteNetIdsDP_naive(0, 0, N, n + s, g, ref, topo, time, ignore_numa, memo);
}

// ---- Synthetic topology builder ----------------------------------------------
// Build a (ref, topo) pair at given N such that:
//   - Both are 16-GPU systems (nGpus = N for simplicity, so each NIC pairs with one GPU)
//   - NUMA is half-half (mirroring the shipped 16-NIC model {0,0,...,1,1,...})
//   - gdrLevel is a permutation-invariant pattern that admits at least one valid
//     permutation. We choose ref-NIC i to match topo-NIC i (identity is valid),
//     so the algorithms have at least one solution to find.

struct Scenario {
    int N;
    std::vector<int> refNuma;
    std::vector<int> topoNuma;
    std::vector<int> refGdr;
    std::vector<int> topoGdr;
    rcclRomeModel ref{};
    rcclRomeModel topo{};
};

static Scenario buildScenario(int N) {
    Scenario s;
    s.N = N;
    s.refNuma.resize(N);
    s.topoNuma.resize(N);
    s.refGdr.assign(N * N, PATH_PHB);
    s.topoGdr.assign(N * N, PATH_PHB);

    // NUMA: half-half (duplicates within each half -- the exact pattern that
    // triggers the latent memo-poisoning bug if predicates are checked only at the leaf).
    for (int i = 0; i < N; i++) {
        s.refNuma[i]  = (i < N/2) ? 0 : 1;
        s.topoNuma[i] = (i < N/2) ? 0 : 1;
    }
    // gdrLevel: identity-match plus a tighter diagonal (PIX) to make leaves
    // distinguishable. Choosing PIX so PIX/PXB equivalence is not triggered.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            s.refGdr [i*N + j] = (i == j) ? PATH_PIX : PATH_PHB;
            s.topoGdr[i*N + j] = (i == j) ? PATH_PIX : PATH_PHB;
        }
    }
    s.ref.nGpus    = N;
    s.ref.nNics    = N;
    s.ref.nicNuma  = s.refNuma.data();
    s.ref.gdrLevel = s.refGdr.data();
    s.topo.nGpus   = N;
    s.topo.nNics   = N;
    s.topo.nicNuma = s.topoNuma.data();
    s.topo.gdrLevel= s.topoGdr.data();
    return s;
}

// Worst-case scenario: no valid permutation exists. The recursive code is forced
// to walk ALL N! permutations before returning false. This is the practical hang
// case the original PR author hit on real 16-NIC hardware: when the system's
// gdrLevel layout does not satisfy rome_model_68's predicates for ANY permutation.
//
// Construction: NUMA matches (so NUMA pruning doesn't help), but ref->gdrLevel
// has a single position (0,0) where it is PIX and topo->gdrLevel is PHB
// everywhere. No (i, j=0) pairing can satisfy ref[0,0]=PIX == topo[i,0]=PHB,
// so the predicate fails at the leaf for every permutation.
static Scenario buildScenarioNoMatch(int N) {
    Scenario s;
    s.N = N;
    s.refNuma.assign(N, 0);
    s.topoNuma.assign(N, 0);
    s.refGdr.assign(N * N, PATH_PHB);
    s.topoGdr.assign(N * N, PATH_PHB);
    s.refGdr[0] = PATH_PIX;   // ref[0,0] is PIX; no topo[i,0] will ever match
    s.ref.nGpus    = N;
    s.ref.nNics    = N;
    s.ref.nicNuma  = s.refNuma.data();
    s.ref.gdrLevel = s.refGdr.data();
    s.topo.nGpus   = N;
    s.topo.nNics   = N;
    s.topo.nicNuma = s.topoNuma.data();
    s.topo.gdrLevel= s.topoGdr.data();
    return s;
}

// Memo-poisoning witness. N=4, NUMA=[0,0,0,0] everywhere (extreme duplication).
// Diagonal-PIX gdrLevel, but topo's positions 2 and 3 are swapped.
// The unique valid permutation is n=[0,1,3,2]; identity n=[0,1,2,3] fails.
//
// Behavior:
//   - recursive: tries every perm in swap-order, eventually finds [0,1,3,2] -> TRUE
//   - corrected DP (per-step gdrLevel): prunes the bad i=2 at pos=2, picks i=3 -> TRUE
//   - naive DP (leaf-only gdrLevel):    fills first leaf with [0,1,2,3], caches
//                                       memo[0b1111]=false, every sibling
//                                       permutation hits the cached false -> FALSE
static Scenario buildScenarioMemoPoison() {
    int N = 4;
    Scenario s;
    s.N = N;
    s.refNuma.assign(N, 0);
    s.topoNuma.assign(N, 0);
    s.refGdr.assign(N * N, PATH_PHB);
    s.topoGdr.assign(N * N, PATH_PHB);
    // ref: identity diagonal
    for (int i = 0; i < N; i++) s.refGdr[i*N + i] = PATH_PIX;
    // topo: swap PIX positions 2 and 3 -> only valid pairing is (2->3, 3->2)
    s.topoGdr[0*N + 0] = PATH_PIX;
    s.topoGdr[1*N + 1] = PATH_PIX;
    s.topoGdr[2*N + 3] = PATH_PIX;
    s.topoGdr[3*N + 2] = PATH_PIX;
    s.ref.nGpus    = N;
    s.ref.nNics    = N;
    s.ref.nicNuma  = s.refNuma.data();
    s.ref.gdrLevel = s.refGdr.data();
    s.topo.nGpus   = N;
    s.topo.nNics   = N;
    s.topo.nicNuma = s.topoNuma.data();
    s.topo.gdrLevel= s.topoGdr.data();
    return s;
}

// ---- Driver -------------------------------------------------------------------

struct RunResult {
    bool   found;
    double ms;
    int    workCounter;
    std::vector<int> permutation;
};

template <typename Fn>
static RunResult timeOne(Fn fn, Scenario& s) {
    int N = s.N;
    std::vector<int> n(N, 0), g(N, 0);
    for (int i = 0; i < N; i++) {
        n[i] = (i + 2) % N;   // mirrors the caller-supplied seed in rome_models.cc
        g[i] = i;
    }
    int counter = 0;
    auto t0 = std::chrono::steady_clock::now();
    bool ok = fn(n.data(), g.data(), 0, N - 1, &s.ref, &s.topo, &counter, false);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return RunResult{ok, ms, counter, n};
}

static void printRow(int N, const char* label, const RunResult& r) {
    std::printf("  N=%2d  %-12s found=%d  work=%-13d  time=%10.3f ms",
                N, label, (int)r.found, r.workCounter, r.ms);
    if (r.found) {
        std::printf("   n=[");
        for (int i = 0; i < N; i++) std::printf("%s%d", i?",":"", r.permutation[i]);
        std::printf("]");
    }
    std::printf("\n");
}

int main(int argc, char** argv) {
    int maxN = 14;
    bool runOldAtMax = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--maxN" && i + 1 < argc) { maxN = std::atoi(argv[++i]); }
        else if (a == "--force-old-at-max") { runOldAtMax = true; }
        else if (a == "-h" || a == "--help") {
            std::printf("Usage: %s [--maxN N] [--force-old-at-max]\n", argv[0]);
            std::printf("  --maxN N            Highest N to test (default 14)\n");
            std::printf("  --force-old-at-max  Run the recursive algorithm even at maxN >= 15 (warning: hours)\n");
            return 0;
        }
    }

    std::printf("=== Perf scenario ===\n");
    std::printf("nGpus=N, nNics=N, half-half NUMA, identity gdrLevel diagonal\n\n");

    int probes[] = {4, 6, 8, 10, 12, 14, 15, 16};
    for (int N : probes) {
        if (N > maxN) break;
        bool skipOld = (N >= 15 && !runOldAtMax);

        Scenario s = buildScenario(N);
        RunResult rDP = timeOne(permuteNetIds_dp, s);
        printRow(N, "[new DP]", rDP);

        if (skipOld) {
            std::printf("  N=%2d  %-12s skipped (would take > minutes; --force-old-at-max to run)\n",
                        N, "[old recur]");
        } else {
            RunResult rOld = timeOne(permuteNetIds_recursive, s);
            printRow(N, "[old recur]", rOld);
            if (rOld.found != rDP.found) {
                std::printf("  *** MISMATCH on found: old=%d new=%d ***\n", rOld.found, rDP.found);
            }
        }
        std::printf("\n");
    }

    std::printf("\n=== Worst-case (no-match) scenario: recursive walks ALL N! permutations ===\n");
    std::printf("Predicate fails for every permutation -> recursive returns false after exhausting.\n");
    std::printf("This is the practical hang case: 16 NICs, no rome_model match -> 16! = 21 trillion iters.\n\n");
    int worstProbes[] = {6, 8, 10, 11, 12};
    for (int N : worstProbes) {
        if (N > maxN) break;
        Scenario nm = buildScenarioNoMatch(N);
        bool skipOld = (N >= 13 && !runOldAtMax);

        RunResult rDP = timeOne(permuteNetIds_dp, nm);
        printRow(N, "[new DP]", rDP);
        if (skipOld) {
            std::printf("  N=%2d  %-12s skipped (--force-old-at-max to run)\n", N, "[old recur]");
        } else {
            RunResult rOld = timeOne(permuteNetIds_recursive, nm);
            printRow(N, "[old recur]", rOld);
            if (rOld.found || rDP.found) {
                std::printf("  *** UNEXPECTED FOUND=true on no-match scenario ***\n");
            }
        }
        std::printf("\n");
    }

    std::printf("\n=== Correctness witness: memo-poisoning scenario ===\n");
    std::printf("N=4, NUMA=[0,0,0,0], topo gdrLevel has positions 2<->3 swapped.\n");
    std::printf("Unique valid permutation is n=[0,1,3,2].\n\n");
    Scenario witness = buildScenarioMemoPoison();
    RunResult wOld   = timeOne(permuteNetIds_recursive, witness);
    RunResult wNaive = timeOne(permuteNetIds_dp_naive, witness);
    RunResult wNew   = timeOne(permuteNetIds_dp,       witness);
    printRow(4, "[old recur]",   wOld);
    printRow(4, "[naive DP]",    wNaive);
    printRow(4, "[new DP]",      wNew);
    std::printf("\n");
    bool naiveBuggy = (wOld.found && !wNaive.found);
    bool newAgrees  = (wOld.found == wNew.found);
    std::printf("  naive-DP-finds-it = %d (expected 0; naive is unsound)\n", (int)wNaive.found);
    std::printf("  new-DP-finds-it   = %d (expected 1; new is sound)\n",     (int)wNew.found);
    std::printf("  agrees-with-recursive = %d\n", (int)newAgrees);
    if (naiveBuggy && newAgrees) {
        std::printf("  -> Witness PASSES: new DP corrects the naive memo poisoning.\n");
    } else if (!naiveBuggy) {
        std::printf("  -> Witness DID NOT TRIGGER: naive DP found the solution; scenario not adversarial.\n");
    } else {
        std::printf("  -> Witness FAILS: new DP missed the valid permutation.\n");
    }
    return 0;
}
