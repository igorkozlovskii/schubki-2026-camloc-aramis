"""Filter parameter pins of a release config, enforced at validation (S08e, D-032 note): before packaging and before every run.

A release config whose model is a filter composite (E04Y / E04T families) must carry `filter_pins: {<slot>: {...}}` for each model slot (`model`,
`fallback_model`); the pin is recomputed here from the actual `params` block and a difference is a refusal, independent of the model's own
`expected_*` check at construction time (which stays as the second line of defence).

  kin2:  {kind: kin2, param_sha256}                       E04Y: sha256 of canonical {filter: Kin2Params(+defaults), latency_scale} (camloc.models.e04y.filter_params_sha256)
  tel:   {kind: tel, param_sha256, params_file_sha256, kin_param_sha256}
                                                            E04T: camloc.models.e04t.param_sha256 over the selected filter; params_file is hashed from the checkout;
                                                            its kin block must hash to kin_param_sha256 (the frozen Y1 block)

Imports numpy-level modules (kin2 filter, e04y/e04t, no torch): it is on the run/package side, never on the plain-python verify path.
"""
from __future__ import annotations

SLOTS = ("model", "fallback_model")
FILTER_CLASSES = {"camloc.models.e04y:E04YModel": "kin2", "camloc.models.e04y:E04YFbModel": "kin2",
                  "camloc.models.e04t:E04TModel": "tel", "camloc.models.e04t:E04TFbModel": "tel"}


class PinError(Exception):
    pass


def kind_of(model_block: dict):
    return FILTER_CLASSES.get(str(model_block.get("class")))


def compute(model_block: dict) -> dict:
    """The pin values of a model block, as the release would pin them."""
    kind = kind_of(model_block)
    p = model_block.get("params", {})
    if kind == "kin2":
        from camloc.models.e04y import filter_params_sha256
        return {"kind": "kin2", "param_sha256": filter_params_sha256(p.get("filter", {}), p.get("latency_scale", 0.0))}
    if kind == "tel":
        from camloc.models.e04t import param_sha256, resolve_filter
        from camloc.models.e04y import filter_params_sha256
        f = resolve_filter(p.get("filter"))
        return {"kind": "tel", "param_sha256": param_sha256(f, p.get("latency_scale", 0.0), p.get("fallback_age_s", 1.0)),
                "params_file_sha256": f["file_sha256"], "kin_param_sha256": filter_params_sha256(f["blocks"]["kin"], 0.0)}
    return {}


def check_pins(cfg: dict) -> dict:
    """Raise PinError on a missing or different pin; returns {slot: computed pin} for the slots that are filter composites."""
    out = {}
    pins = cfg.get("filter_pins") or {}
    for slot in SLOTS:
        mb = cfg.get(slot)
        if not mb or kind_of(mb) is None:
            if slot in pins:
                raise PinError(f"filter_pins.{slot} given but the {slot} is not a filter composite")
            continue
        want = pins.get(slot)
        if not want:
            raise PinError(f"release config lacks filter_pins.{slot}: a filter composite must carry its parameter pin")
        try:
            got = compute(mb)
        except Exception as e:   # noqa: BLE001  a params file that moved or changed is a pin failure too
            raise PinError(f"filter_pins.{slot}: cannot compute the filter parameter hash: {type(e).__name__}: {e}") from None
        if want.get("kind") != got["kind"]:
            raise PinError(f"filter_pins.{slot}.kind {want.get('kind')!r} != {got['kind']!r}")
        for k, v in got.items():
            if want.get(k) != v:
                raise PinError(f"filter parameters of {slot} differ from the pin: {k} {str(v)[:16]} != pinned {str(want.get(k))[:16]}")
        out[slot] = got
    extra = set(pins) - set(SLOTS)
    if extra:
        raise PinError(f"unknown filter_pins slots {sorted(extra)}")
    return out
