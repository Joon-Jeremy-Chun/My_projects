# Project Memory Index

Purpose: a detailed, navigable, accurate computational mathematics portfolio.

- [Shared contract](../AGENTS.md)
- [Task status](project_task_status.md)
- [Session and decision log](research_direction_log.md)
- [Repository map](code_map.md)
- [Project purpose and decisions](project_philosophy.md)
- [Latest handoff](../docs/handoffs/HANDOFF_2026-10-20_portfolio_repairs.md)
- [Repair evidence](../docs/REPAIRS.md)
- [Handoffs](../docs/handoffs/)

## Active scope — 2026-10-20

The owner authorized detailed README organization, completion of unfinished
code, error fixes, and commit/push for later review. Work uses an isolated
worktree on `codex/portfolio-repair-20261020`, based on local `master` at
`ed71db0`, which was 14 commits ahead of origin/master (`58a5121`). Original
uncommitted work remains in the original My_projects directory.

Baseline: 1,238 tracked files; 287 Python, 14 R, five notebooks. Four former
comment-only files are completed; the focused suite has 21 passing cases.
Root plus collection/project READMEs total 60, supported by evidence,
reproduction, repairs, and a full 306-artifact catalog.

Publication is complete: commit `5b2833eb68a75399a9239b816ec6206b66f5e9be`
was pushed to GitHub master and verified with `git ls-remote`. The publication
worktree is `my_projects-publish-20261020`, on
`codex/portfolio-publish-20261020`. The repair commit is `d1dbfc7`; both commits
have the same reviewed tree, `4410cd1774f8fb26a245a34962a6776e521e0c42`.
Completion records follow in a separate documentation commit.

Publication preparation found historical unpushed CSV blobs up to 636,205,778
bytes, absent from the current final tree. The final-tree snapshot was added
on top of the previous origin/master without force pushing or rewriting
history. The original local 14-commit history and uncommitted work remain
preserved. Start future work from the verified remote/publication branch;
read the handoff before attempting to push original local master, whose
historical large blobs remain local.
