"""VE-1: final quantitative figures and tables.

VE-1 renders the figures and tables specified by the frozen VE-0 evidence
contract. It is a reporting task, not an experiment: it trains nothing,
evaluates nothing, loads no checkpoint, measures no latency and charges no GPU
hour. Every number it draws is read from `results/ve0/`, and every artefact it
writes records the evidence identifiers, source hashes and builder hash behind
it.
"""
