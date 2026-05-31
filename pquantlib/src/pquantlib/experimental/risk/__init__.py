"""Experimental portfolio-risk models.

# C++ parity: ql/experimental/risk/* (v1.42.1).

NOTE — recovered-source provenance:
The C++ ``CreditRiskPlus`` and ``sensitivityanalysis`` were *removed* in
QuantLib version 1.36 (commit 6f379f4e9); the pinned tree (099987f0 /
v1.42.1) ships only empty deprecation-stub headers ("Out of scope; copy
this class in your codebase if needed").  PQuantLib's port therefore
follows the W8-B task brief (which requests these by name with probe
coverage) and ports from the *pre-removal* source recovered from
6f379f4e9~1 — the last commit at which the implementations existed.  The
CreditRisk+ probe vendors that recovered source to emit reference values.
"""
