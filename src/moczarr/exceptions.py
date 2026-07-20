"""Exception types on moczarr's reader surface.

A tiny leaf module by design: exceptions are imported by both the store
layer and the opener, so they live below everything else — no moczarr
imports here, ever.
"""

from __future__ import annotations


class NoCoverageError(ValueError):
    """The store has no stamped coverage anywhere — no schema source exists.

    Raised by :func:`moczarr.open_hive` only for the store-wide condition
    (no root-MOC-listed leaf opens AND the discovery walk finds no
    commit-stamped leaf): with zero committed leaves there is no metadata
    from which to serve even a schema. An ``aoi`` or ``window`` that merely
    intersects none of an otherwise-covered store is a data answer, not an
    error — that case returns a schema-correct empty dataset with a
    ``UserWarning`` instead (issue #4).

    Subclasses ``ValueError`` so pre-existing ``except ValueError`` callers
    keep working; new callers can catch precisely without string-matching.
    """
