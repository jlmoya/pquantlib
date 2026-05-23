# PQuantLib Migration Documentation

Each phase has 3-4 docs:

- **`phase<N>-design.md`** — binding spec approved before any code is written. Sections: scope, approach, worktree topology, tolerance discipline, pause triggers, decision log.
- **`phase<N>-plan.md`** — executable, bite-sized task list with checkboxes. Per-task: exact file paths, code snippets, expected test deltas.
- **`phase<N>-progress.md`** — running log of cluster landings (optional, mainly for multi-day phases).
- **`phase<N>-completion.md`** — closure summary: tags, test counts, latent-bug fixes surfaced, deferred items, lessons learned.

## Sister-project equivalence

For every PQuantLib phase doc, there's a corresponding JQuantLib doc at `/Users/josemoya/eclipse-workspace/jquantlib/docs/migration/`. Cross-reference when designing a phase — the Java port already made most of the scope decisions and learned most of the lessons.

| PQuantLib | JQuantLib equivalent |
|---|---|
| Phase 0 (bootstrap) | n/a (JQuantLib inherited a 2007-era skeleton) |
| Phase 1 (L1 math) | jquantlib `phase2-l1-plan.md` + `phase2-l1-{A,B,C,D,E}-*-plan.md` |
| Phase 2 (L2 termstructures + indexes) | jquantlib `phase2-L2-termstructures-indexes-plan.md` |
| Phase 3 (L3 instruments + pricingengines) | jquantlib `phase2-L3-instruments-pricingengines-plan.md` |
| Phase 4 (L4 models) | jquantlib `phase2-L4-models-plan.md` |
| Phase 5 (L5 experimental + L6 test-suite parity) | jquantlib `phase2-L5-experimental-plan.md` + `phase2-L6-test-suite-parity-plan.md` |
| Phase 6 (Python 3.14 modernization) | jquantlib `jdk25-modernization-design.md` (analogue: dataclasses, match-case, PEP 695 generics, t-strings) |
| Phase 7 (final closure) | jquantlib `phase2-complete` + `truly-complete` + `final` tags |
