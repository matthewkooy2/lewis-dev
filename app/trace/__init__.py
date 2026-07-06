"""Read-only static code tracing and implementation review for Lewis.

The package builds a bounded, language-aware static index of a developer
workspace and answers three kinds of questions without executing any project
code: quick traces (definition, direct callers/callees, direct side effects),
deep traces (bounded transitive call paths, side effects, tests, error
handling, cycles), and implementation reviews (deterministic, evidence-backed
findings).

Every relationship is labeled with how it was established (confirmed,
inferred, framework-registered, ambiguous, external, dynamic, unresolved)
because static analysis cannot fully resolve dynamic behavior.

Import ``CodeTracer`` from ``app.trace.service`` (kept out of this package
namespace so per-module imports stay explicit and cheap).
"""
