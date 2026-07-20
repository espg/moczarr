# Development

```sh
git clone https://github.com/espg/moczarr && cd moczarr
uv sync --extra test --extra xdggs
uv run pytest -v
```

Lint/format is ruff, types are mypy, spelling is codespell — all wired
through `.pre-commit-config.yaml` (`pre-commit run --all-files` mirrors
CI). Docs build with `uv run mkdocs build --strict` (the `docs` dependency
group).

## The SERC fixture

Most reader tests run against the committed fixture store at
`tests/data/serc_hive`: six order-6 shards around the NEON SERC site,
ATL06-shaped synthetic (deterministic, seeded) data at cell order 8, plus
one deliberate debris leaf. Every byte is produced by **zagg's real write
path**, so writer↔reader drift fails this suite on whichever side moved.

Regenerate it from a zagg checkout's environment (zagg is deliberately not
a moczarr dependency):

```sh
cd ../zagg && uv run python ../moczarr/tools/generate_serc_fixture.py \
    --out ../moczarr/tests/data/serc_hive
```

The long-term home for the public example store is source.coop, once the
store spec stops drifting (tracked on
[issue #1](https://github.com/espg/moczarr/issues/1)).

## Golden-vector policy

The byte-level conventions are pinned twice, on purpose:

- `tests/conftest.py` builds hive stores **object-by-object** (raw JSON,
  raw zarr v3 metadata, hand-encoded bitmap bytes — no zagg, no zarr
  machinery), so the tests document the wire format explicitly.
- Golden vectors in `tests/test_coverage.py` and `tests/test_convention.py`
  pin moczarr's decoders against zagg's writer output (the fixture above).

A convention change must therefore show up as a *deliberate* edit to the
golden bytes here, never as a silent re-generation. When the spec moves,
regenerate the fixture with the new zagg, watch which goldens break, and
update them in the same commit as the reader change.

## Spec-alignment coordination

The leaf-naming convention (`{full_id}.zarr`) is slated to change with
zagg's template-hash leaf naming
([englacial/zagg#299](https://github.com/englacial/zagg/issues/299));
moczarr's side of that coordinated breaking change — `convention.py`
(`leaf_path`, `check_node_invariant`, `split_leaf_name`), the fixture, and
the golden vectors — is tracked in
[issue #11](https://github.com/espg/moczarr/issues/11). Tests deliberately
go through `open_hive` outputs (never raw leaf paths) so they don't deepen
the assumption.
