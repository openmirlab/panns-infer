"""Private inference implementation; use the package-qualified session facade.

The package keeps checkpoint geometry and model operators in one private owner.
Reads: model and audio modules only when the session loads or infers.
"""
