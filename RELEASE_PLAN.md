# Release plan — compact layout and export process

Working plan for turning a selected research candidate into this standalone
delivery. It points back to the internal inventory in the research checkout
(`reports/release/PRG003-R01/INVENTORY.md`, `SOURCE_MANIFEST.tsv`,
`ASSET_MANIFEST.tsv`) as preparation context; that inventory is not part of the
organizer delivery.

## Intended compact layout

```
schubki_2026_release/
  README.md                 # organizer-facing quick start (run flow filled by R02b/R04)
  AGENTS.md                 # this sibling's release-only agent rules
  RELEASE_PLAN.md           # this file
  pixi.toml · pixi.lock     # DONE (R02a): one pinned env (model + MCAP), no stream/jupyter
  .gitignore                # DONE (R02a): ignores .pixi/ and runtime outputs
  src/camloc/…              # DONE (R02a): runtime import closure only (no research-only modules)
  scripts/                  # DONE (R02a): run_release, export_test_input, export_telemetry,
                            #   package_release, verify_release
  tools/                    # DONE (R02a): export_sources.py (rerunnable), dep_probe.py
  configs/models/           # DONE (R02a): pinned filter params (e10g-g1.yaml)
  manifests/                # DONE (R02a): S2.9-constants.json + source-export.json
  external/                 # DONE (R02a): LightGlue subset (superpoint + unused backends excluded)
  tests/                    # DONE (R02a): test_runtime_foundation.py
  docs/RUNTIME_FOUNDATION.md# DONE (R02a): foundation + verified commands
  licenses/                 # DONE (R02a): external-resources + license table + lightglue-subset
  configs/release/ + model class   # R02b: the live D-086 adapter + one default config
  map/ · weights · calibration     # R02b: the asset bundle (delivered/verified, not in Git)
  reproduction/             # OPTIONAL map-building / training sources, clearly separate (R04)
  evidence/                 # final CSV identity, scorer stdout, runtime report (R03/R04)
```

## Candidate and compatibility

- **Candidate selection is PRG-003 gate G1**, reconciling the R01 recommendation
  with the scientific owner's choice. The scientific owner's current final model
  (D-086, 2026-10-18) is **mc7 + borders + poles + dynamic v_max**, excluding the
  learned/GRU layer. It has **no standalone live adapter yet** — today it runs
  only through the research runner (`scripts/v2_s42_lib.py`, config `mc7_bp_dyn`;
  dynamic v_max applied as an in-process patch in `scripts/v2_s43_vmax.py`).
  Building that adapter and making dynamic v_max shared code are the main R02
  tasks (gap detail in the internal inventory `reports/release/PRG003-R01/`).
- The reproduced older packaging `release-e04t-fb-v2` (option A′, clean-clone
  proven in S08b-prep) is the safe extraction baseline. `release-mc7-v1` is a
  packaged-but-untested runner for `mc7` (7-camera landmark fixes) only — it is
  **not** the D-086 borders/poles/dyn pipeline. No model-selection run is done in
  release work; GRU exclusion is per D-086.
- **Code/config/asset versioning is coupled.** The runtime already refuses to run
  unless the checkout's code equals the bundle's code snapshot by hash and
  `--config` is byte-identical. The release preserves this verification intent; it
  only changes where `camloc` is imported from (this directory's `src/`).

## Export process (from versioned research sources)

1. **Snapshot the source by content, not by parent commit.** Record the parent
   research `HEAD`, each file's exact sha256 and its tracked/untracked/dirty
   status (already captured in `SOURCE_MANIFEST.tsv`), copy the exact file
   content into this directory and version it **here**. Do not require committing
   the parent repository's files; parent clean-clone limits are real but parent
   Git history belongs to its existing owner.
2. **Copy the runtime closure** listed in `SOURCE_MANIFEST.tsv` into `src/` and
   `scripts/`; copy the selected `configs/` and `environment/` manifests. For the
   D-086 target this includes the borders/poles/segmentation modules currently
   marked `role=reproduction`, once a live adapter replaces the research scripts.
3. **Build the live D-086 adapter:** a release model class + config for
   mc7 + borders + poles, dynamic v_max folded into shared filter code, live
   segmentation per tick (not cached `S2.8-det` replays).
4. **Build and verify the asset bundle** with `package_release.py` then
   `verify_release.py` (stdlib-only; must pass with `python3 -I`). The D-086
   bundle adds the pole map, static masks and the road-segmentation weights.
5. **Resolve machine-specific assumptions**; keep the exporter/model env split.
6. **Smoke** (R03): standalone imports, one-command raw-MCAP run, causal/finite/
   reset/all-camera checks, parity vs the research pipeline, measured runtime.

## Upgrade and reproduction

- **Upgrade** the candidate only after the scientific owner identifies a newer
  model and release parity is re-checked; never silently change an earlier
  release. Each upgrade is a new versioned snapshot with its own asset bundle and
  verification evidence.
- **Reproduction** of training/maps is optional to running inference: the
  `reproduction/` sources and build commands are included for provenance but are
  not installed or executed by the run command.

## Open choices (for the orchestrator / G1)

- **Resolved (D-007/D-008): ship exactly one model** — the D-086 pipeline
  (mc7 + borders + poles + dynamic v_max). A′ is **not** shipped alongside it; it
  remains historical packaging evidence only and was not exported in R02a. The
  earlier "ship C alone or C + A′ fallback" question is closed.
- Asset delivery: the D-086 bundle adds the pole map, static masks and the
  road-segmentation weights to the ~0.86 GB map set; private GitHub release
  assets vs a separately shared archive; weights as bundled vs fetch-and-verify.
  (R02b/R04.)
- Confirm the road-segmentation weight path and licence for the D-086 target.
  (R02b.)
- Organizer access, submission channel and the confirmed deadline/timezone.
