# PR prep packet: replace permuteNetIds O(N!) recursion with sound bitmask DP

Everything in this directory is ready to copy-paste into a PR against
`ROCm/rocm-systems`, base branch `develop`.

## What to do when you're ready to submit

1. Fork `ROCm/rocm-systems` on github.com to `dongjoo-lee/rocm-systems`
   (browser, one click).
2. Create the branch and apply the patch:

   ```bash
   git clone --filter=blob:none --sparse \
       git@github.com:dongjoo-lee/rocm-systems.git
   cd rocm-systems
   git sparse-checkout add projects/rccl
   git checkout -b fix/permute-net-ids-bitmask-dp develop
   git apply /home/mangosw/GPU/private/dongjoo.lee/claude/rccl-pr-1566/prep/the_patch.diff
   git add projects/rccl/src/graph/rome_models.cc
   git commit -F /home/mangosw/GPU/private/dongjoo.lee/claude/rccl-pr-1566/prep/COMMIT_MSG.txt
   git push -u origin fix/permute-net-ids-bitmask-dp
   ```

   `git apply` will fail if `develop` has moved &mdash; in that case
   apply by hand from `rome_models.cc.patched` or rebase.

3. Open the PR via the GitHub UI:
   - **base**: `ROCm/rocm-systems:develop`
   - **head**: `dongjoo-lee/rocm-systems:fix/permute-net-ids-bitmask-dp`
   - **title**: paste from `PR_TITLE.txt`
   - **body**: paste the entire `PR_BODY.md` content into the
     description box (GitHub renders markdown live)
   - **attach `permute_nic_bench.cpp`**: drag-and-drop into the
     description, or just paste a link to it in your fork after the
     "Reproducing locally" section.

## What's in this directory

| file                      | purpose                                                    |
|---------------------------|------------------------------------------------------------|
| `COMMIT_MSG.txt`          | git commit subject + body, wrapped to ~72 columns          |
| `PR_TITLE.txt`            | PR title (same as commit subject)                          |
| `PR_BODY.md`              | full PR description, markdown                              |
| `the_patch.diff`          | unified diff against current `develop`                     |
| `rome_models.cc.patched`  | the full patched source file (drop-in replacement)         |
| `permute_nic_bench.cpp`   | standalone reproducer / benchmark (attach to PR)           |
| `bench_output.txt`        | sample bench output captured locally for reference         |

## Notes

- No `Signed-off-by`. The rocm-systems repo (per the latest merged
  commits to `projects/rccl/`, e.g.
  `efee65336...opt-in env variable RCCL_PXN_OPT_QP_USAGE...`) does not
  require it. Adding one is harmless but unnecessary.
- No `Co-Authored-By` line.
- The `(#NNNN)` suffix on the merged commit subject is appended by
  GitHub on squash-merge &mdash; your local commit subject should not
  include it.
- The "Submission Checklist" link points to
  `ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests`. That's what
  recent rocm-systems PRs use; the link sometimes 404s but reviewers
  expect the checklist item to be there.
- Do **not** push to `MangoBoost/` &mdash; this is github.com / ROCm
  upstream, not your usual MangoBoost workflow.
