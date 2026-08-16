"""Exception and warning types on moczarr's reader surface.

A tiny leaf module by design: these are imported by the store layer, the
opener and the intersection helpers alike, so they live below everything
else — no moczarr imports here, ever. Warning categories belong here for
the same reason exceptions do: a caller filters or catches by class, so
the class has to be importable without dragging a reader module in.
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


class ConservativeCoverageWarning(UserWarning):
    """An answer degraded from exact occupancy to a conservative cover.

    Raised once per call by :func:`moczarr.iter_occupancy_and` /
    :func:`moczarr.occupancy_and` when a shared leaf cannot answer exactly
    at the harmonized cell order — a box-only envelope, a ``"bitmap"``
    envelope whose sidecar object is missing, a stamp with no usable
    envelope, or an envelope whose own ``cell_order`` sits below the
    harmonized order. The leaf then contributes its cover instead, so the
    result is a SUPERSET for cells under it: false positives possible,
    false negatives impossible.

    Its own category (rather than a bare ``UserWarning``) so a consumer can
    act on the degradation by class — ``warnings.simplefilter("error",
    ConservativeCoverageWarning)`` to refuse supersets in one place, or
    ``"ignore"`` to accept them without silencing everything else moczarr
    warns about. ``degrade="skip"`` / ``degrade="raise"`` are the
    per-call knobs for the same choice.

    Subclasses ``UserWarning`` so existing ``UserWarning`` filters and
    ``pytest.warns(UserWarning)`` callers keep matching.
    """
