# schubki_2026_release — agent brief

Release-only sibling of the research checkout
`/home/igor/Projects/other/schubki_2026`. Igor's canonical global rules apply:
`~/.claude/CLAUDE.md` (canonical `~/research-os/AGENTS.md`). This file adds only
what is local to the release delivery.

## Purpose and boundary

- This directory is the **organizer delivery snapshot** only: standalone source,
  a pinned environment, versioned assets and reproduction evidence for one
  selected candidate. It is not a research workspace.
- **No access to the research checkout at run time.** The released code must run
  from this directory's own `src/` and asset bundle. Do not assume the sibling
  research tree, its `derived/`, `artifacts/` or `.pixi/` are reachable when the
  organizers re-run the code.

## Status

R02b ported the frozen D-091 final driver (`scripts/final_run.py run`), extended the exporter to the
`final-v1.yaml` `freeze.code` closure, pinned the environment to the research vision lock versions, built the
asset bundle (`tools/build_assets.py`, `tools/verify_assets.py`, `docs/ASSETS.md`) and ran the test bag end to
end. Parity with the research run is not bitwise (latency-dependent availability); see
`docs/R02B_ADAPTATIONS.md`. Packaging, scoring and the organizer quick start are later steps.

## Rules for work in this directory

- **The one Pixi environment is the root `pixi.toml` + `pixi.lock`.** Run it with
  `CONDA_OVERRIDE_CUDA=12.4 pixi run --manifest-path <sibling>/pixi.toml ...`
  (the CUDA override is a local detection workaround, not an organizer
  requirement). Never reference, modify or copy the research project's manifests,
  locks or `.pixi/`. `.pixi/` here is generated and git-ignored.
- Re-export source only through `tools/export_sources.py` with explicit
  `--source-root` / `--dest`; it refuses to overwrite a differing file and
  accepts byte-identical reruns.
- **No implicit training or downloads.** Inference reproduction must not train
  models or fetch data as a side effect. Asset acquisition is explicit, with a
  manifest entry and a verified hash.
- **No commits or pushes** from this directory until the release orchestrator
  coordinates and authorizes it. No remote is initialized by preparation steps.
- Preserve existing filename and path contracts carried over from the research
  code; do not rename working entry points.
- Keep internal preparation links (the research inventory, program receipts) out
  of the eventual organizer-facing quick start.

## Ownership

PRG-003 (release orchestrator, in the research checkout) owns orchestration
during preparation. Implementers work within a written brief and leave draft
receipts; reviewers are independent. This sibling does not own or edit
PRG-001/PRG-002 state.
