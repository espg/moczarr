"""Multi-product store roots (zagg D19, mortie spec §6.5), read side.

A multi-product store is a **directory of stores**: each product lives under
its own human-readable root prefix ``{name}/``, and a product subtree is a
complete, unmodified morton-hive store (bare-named manifest, MOC, digit
tree). A bare single-product store — the manifest at the store root, today's
common form — remains fully valid; readers distinguish the two root forms
**by content** (spec §6.5): a manifest at the root ⇒ bare store; name-shaped
prefixes ⇒ product directory.

Product names are constrained by the §6.5 grammar (``[a-z0-9_-]{1,192}``,
URL-safe by construction) with the **base-component exclusion**: a name must
never match the morton base-component grammar ``-?[1-6]``, so the walker's
child classification stays unambiguous.

Discovery is one delimiter-LIST of the store root, one manifest GET per
name-shaped child, and one HEAD per product (the ``aggregation.yaml``
probe) — the enumeration recipe the design fixes for every viewer
("list ``{store_root}/`` and read each ``{name}/morton_hive.json`` directly —
no name↔hash translation layer") — so one product this reader is too old for
must never take down the enumeration: an unsupported-but-well-formed spec is
listed with ``manifest: None``, not raised on. The D19 ``semantic_hash``
(sha256 of the product's canonical semantic core) is surfaced from the
manifest, and the presence of the derived ``aggregation.yaml`` convenience
copy is probed —
the hash is truth; the YAML is a regenerable cache (D9 class).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from moczarr.convention import HIVE_SPEC, HIVE_SPEC_V2, MANIFEST_NAME, parse_manifest
from moczarr.store import _resolve_store, read_json

logger = logging.getLogger(__name__)

#: The §6.5 product-name grammar: 1–192 chars of ``[a-z0-9_-]`` (single-byte
#: ASCII, so chars == bytes; the 192 cap derives from the POSIX filename
#: budget of a D23 downloader's ``{name}+{catalog-hash}`` component).
PRODUCT_NAME_RE = re.compile(r"^[a-z0-9_-]{1,192}$")
#: The §2 morton base-component grammar a product name must NOT match.
_BASE_COMPONENT_RE = re.compile(r"^-?[1-6]$")
#: The D19 canonical-semantic-core convenience object at a product root.
AGGREGATION_CORE_NAME = "aggregation.yaml"
#: A well-formed ``morton-hive/N`` spec string. A product whose spec matches
#: this but is not one this reader opens (``/3`` and up) is not *malformed* —
#: it is a store this reader is too old for, so discovery lists it instead of
#: taking down its siblings (see :func:`list_products`).
_HIVE_SPEC_RE = re.compile(r"^morton-hive/\d+$")
_SUPPORTED_SPECS = (HIVE_SPEC, HIVE_SPEC_V2)


def is_product_name(name: str) -> bool:
    """Whether ``name`` is a legal §6.5 product name (grammar + exclusion)."""
    return bool(PRODUCT_NAME_RE.match(name)) and not _BASE_COMPONENT_RE.match(name)


def validate_product_name(name: str) -> str:
    """Validate a product name against the §6.5 grammar; returns it."""
    if not isinstance(name, str) or not is_product_name(name):
        raise ValueError(
            f"product name {name!r} does not match the §6.5 grammar "
            f"({PRODUCT_NAME_RE.pattern}, excluding morton base components "
            f"{_BASE_COMPONENT_RE.pattern}; mortie spec, espg/mortie#62)"
        )
    return name


def _has_object(store: Any, key: str) -> bool:
    """Whether ``key`` exists on ``store`` (one HEAD; absence is definitive)."""
    import obstore
    from obstore.exceptions import NotFoundError

    try:
        obstore.head(store, key)
    except (FileNotFoundError, NotFoundError):
        return False
    return True


def _unsupported_spec(payload: object) -> str | None:
    """A well-formed hive spec this reader cannot open, else ``None``."""
    spec = payload.get("spec") if isinstance(payload, dict) else None
    if isinstance(spec, str) and spec not in _SUPPORTED_SPECS and _HIVE_SPEC_RE.match(spec):
        return spec
    return None


def list_products(store_root: str, *, store: Any = None, **store_kwargs: Any) -> list[dict]:
    """Enumerate the named products of a multi-product store root (D19).

    One delimiter-LIST of the root, then one ``{name}/morton_hive.json`` GET
    per name-shaped child, then one HEAD per product for the
    ``has_aggregation_yaml`` probe — two sequential round-trips per product,
    measured on the committed fixture (1 LIST + 5 GET + 4 HEAD for its four
    products plus the non-product ``scratch/`` child). Bounded, but stated
    because ``list_products`` is a per-page-load call for a viewer. Returns
    one record per product, sorted by name::

        {
            "name": ...,               # the {name}/ prefix (§6.5 grammar)
            "spec": ...,               # manifest spec string
            "dataset": ...,            # manifest dataset block (or None)
            "semantic_hash": ...,      # D19 identity hash (or None, pre-D19)
            "has_aggregation_yaml": ., # derived semantic-core copy present?
            "manifest": ...,           # the full parsed manifest (None: the
                                       # spec is one this reader can't open)
        }

    A **bare single-product store** (a manifest at the root — the §6.5
    content discrimination) returns ``[]``: the degenerate form has no
    *named* products; open it directly. Name-shaped children **without** a
    manifest are skipped with a debug log (a prefix is a product iff it
    carries one); a present-but-malformed product manifest raises, the
    :func:`moczarr.store.read_manifest` posture — a manifest is a bootstrap,
    never a degradable cache.

    A product whose manifest carries a well-formed but **unsupported** hive
    spec (``morton-hive/3`` and up — a store this reader is too old for, not
    a broken one) is neither raised on nor hidden: it is listed with
    ``manifest: None`` (and ``dataset``/``semantic_hash`` ``None``) plus a
    warning, so a caller sees the product exists and why it can't be opened.
    Raising instead would make one newer sibling erase every readable
    product from an enumeration that D19 makes a viewer-facing contract.
    Opening it is still loud — ``open_hive(root, product=...)`` raises the
    spec error from the manifest bootstrap.
    """
    import obstore

    handle = _resolve_store(store_root, store, store_kwargs)
    listing = obstore.list_with_delimiter(handle)
    if any(meta["path"] == MANIFEST_NAME for meta in listing["objects"]):
        return []  # bare single-product store (manifest at the root)
    products = []
    for child in listing["common_prefixes"]:
        name = child.rstrip("/").split("/")[-1]
        if not is_product_name(name):
            logger.debug(f"{store_root}: skipping non-product-shaped child {name!r}")
            continue
        payload = read_json(handle, f"{name}/{MANIFEST_NAME}")
        if payload is None:
            logger.debug(f"{store_root}: name-shaped child {name!r} carries no manifest; skipping")
            continue
        unsupported = _unsupported_spec(payload)
        if unsupported is not None:
            logger.warning(
                f"{store_root}: product {name!r} declares spec {unsupported!r}, which this "
                f"reader cannot open; listing it with manifest=None"
            )
            products.append(
                {
                    "name": name,
                    "spec": unsupported,
                    "dataset": None,
                    "semantic_hash": None,
                    "has_aggregation_yaml": _has_object(handle, f"{name}/{AGGREGATION_CORE_NAME}"),
                    "manifest": None,
                }
            )
            continue
        manifest = parse_manifest(payload)
        products.append(
            {
                "name": name,
                "spec": manifest["spec"],
                "dataset": manifest.get("dataset"),
                "semantic_hash": manifest.get("semantic_hash"),
                "has_aggregation_yaml": _has_object(handle, f"{name}/{AGGREGATION_CORE_NAME}"),
                "manifest": manifest,
            }
        )
    return sorted(products, key=lambda p: p["name"])


__all__ = [
    "AGGREGATION_CORE_NAME",
    "PRODUCT_NAME_RE",
    "is_product_name",
    "list_products",
    "validate_product_name",
]
