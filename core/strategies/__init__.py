"""Three-lane shadow trading strategies.

Each lane is an independent module with its own entry scanner and
monitor loop. All lanes share `core.execution.allocator` for capital,
`core.execution.shadow` for simulated entry/exit, and the same feed /
market / price infrastructure.
"""
