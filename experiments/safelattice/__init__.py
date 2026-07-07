"""SafeLattice empirical evaluation package.

Modules
-------
subset       -- derive the safety-relevant task subset from the audit data
trace_corpus -- build a curated, ground-truth-labeled corpus of agent traces
dual_score   -- score traces under binary and SafeLattice; compute metrics
run_live     -- live harness that runs real models via the Claw-Eval CLI
analyze      -- generate comparison artifacts and a human-readable report
"""
