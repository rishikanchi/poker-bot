"""Turn a decision into a LEGAL SDK action. Never returns an illegal one.

Rules from the competition page:
  * only offered kinds are legal; check and call are never both offered
  * raise_to(total) must be an int within [min_amount, max_amount] (no clamping
    by the host: out of range = failed decision)
  * where neither check nor fold is offered, Action(first.kind, first.min_amount)
    takes the automatic choice
"""
from __future__ import annotations

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"


def _make(sdk, kind, amount=None):
    A = sdk.Action
    if kind == RAISE:
        f = getattr(A, "raise_to", None)
        return f(int(amount)) if f else A(kind, int(amount))
    f = getattr(A, kind, None)
    if callable(f):
        try:
            return f()
        except TypeError:
            pass
    return A(kind, amount)


def automatic(sdk, v):
    """Check if free, otherwise fold; else the host's automatic choice."""
    if CHECK in v.legal:
        return _make(sdk, CHECK)
    if FOLD in v.legal:
        return _make(sdk, FOLD)
    if v.legal_raw:
        first = v.legal_raw[0]
        kind = getattr(first, "kind", None) if not isinstance(first, dict) else first.get("kind")
        mn = getattr(first, "min_amount", None) if not isinstance(first, dict) else first.get("min_amount")
        return sdk.Action(kind, mn)
    return _make(sdk, FOLD)


def to_action(sdk, v, kind, amount=None):
    legal = v.legal
    if kind == RAISE:
        if RAISE in legal and amount is not None:
            lo, hi = legal[RAISE]
            if lo is not None and hi is not None:
                amt = int(min(max(int(amount), int(lo)), int(hi)))
                return _make(sdk, RAISE, amt)
        kind = CALL if CALL in legal else CHECK
    if kind == CALL and CALL not in legal:
        kind = CHECK if CHECK in legal else FOLD
    if kind == CHECK and CHECK not in legal:
        kind = FOLD if FOLD in legal else CALL
    if kind == FOLD and FOLD not in legal:
        kind = CHECK if CHECK in legal else kind
    if kind in legal and kind != RAISE:
        return _make(sdk, kind)
    return automatic(sdk, v)
