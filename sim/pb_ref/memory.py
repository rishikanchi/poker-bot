"""Compact binary memory carried between turns at one table.

The host gives back exactly the bytes we returned last turn (truncated at
``memory_bytes``; ~16 KB is free). Nothing else survives between calls.

Layout (little endian), version 1:
  b"PB" u8 version
  u8 n_slots, i16[n] seat ids
  i32[n] last known stacks (stack at the end of the previous hand)
  u16[n*K] opponent stat counters
  u16 n_pending, then pending events of the hand in progress:
      u8 kind, i8 seat (-1 = None), i32 amount, u8 ncards, u8[ncards] cards
  i32 cache_hand, u8 n_cached, then per cached street:
      u8 street, u16 length, u8[length] quantized strength scores

Any decoding problem returns a fresh Memory: a bad blob must never make
``act`` raise, because the host would then hand us the same bad blob on
every later turn at that table.
"""
from __future__ import annotations

import struct

import numpy as np

from .stats import K_STATS

MAGIC = b"PB"
VERSION = 3
KIND_CODE = {"hand": 1, "ante": 2, "blind": 3, "fold": 4, "check": 5, "call": 6, "raise": 7,
             "board": 8, "show": 9, "win": 10, "stack": 11}
CODE_KIND = {v: k for k, v in KIND_CODE.items()}
MAX_PENDING = 600


class Memory:
    def __init__(self):
        self.seats = []                      # seat ids, slot order
        self.last_stacks = np.zeros(0, np.int64)
        self.stats = np.zeros((0, K_STATS), np.int64)
        self.pending = []                    # (kind, seat, amount, cards) of the hand in progress
        self.cache_hand = -1
        self.cache = {}                      # street -> float scores array (or None)

    # seats ---------------------------------------------------------------
    def ensure_seats(self, seat_ids, starting_stack):
        if self.seats:
            missing = [s for s in seat_ids if s not in self.seats]
            if not missing:
                return
            n_old = len(self.seats)
            self.seats = self.seats + missing
            self.last_stacks = np.concatenate([self.last_stacks,
                                               np.full(len(missing), starting_stack, np.int64)])
            self.stats = np.vstack([self.stats, np.zeros((len(missing), K_STATS), np.int64)])
            assert self.stats.shape[0] == n_old + len(missing)
            return
        self.seats = list(seat_ids)[:16]
        n = len(self.seats)
        self.last_stacks = np.full(n, starting_stack, np.int64)
        self.stats = np.zeros((n, K_STATS), np.int64)

    def slot(self, seat):
        try:
            return self.seats.index(seat)
        except ValueError:
            return -1

    # codec ---------------------------------------------------------------
    def encode(self, limit=65536) -> bytes:
        n = len(self.seats)
        parts = [MAGIC, struct.pack("<BB", VERSION, n)]
        parts.append(struct.pack(f"<{n}h", *self.seats))
        parts.append(np.clip(self.last_stacks, -2**31, 2**31 - 1).astype("<i4").tobytes())
        parts.append(np.clip(self.stats, 0, 65535).astype("<u2").tobytes())
        pend = self.pending[-MAX_PENDING:]
        parts.append(struct.pack("<H", len(pend)))
        for kind, seat, amount, cards in pend:
            cs = bytes(int(c) & 0x3F for c in cards[:15])
            parts.append(struct.pack("<Bbi B", KIND_CODE.get(kind, 0),
                                     -1 if seat is None else max(-1, min(int(seat), 127)),
                                     max(-2**31, min(int(amount), 2**31 - 1)), len(cs)))
            parts.append(cs)
        body = b"".join(parts)
        cache_items = [(st, arr) for st, arr in sorted(self.cache.items()) if arr is not None]
        cache_parts = [struct.pack("<iB", self.cache_hand, 0)]
        used = len(body) + 5
        kept = 0
        for st, arr in cache_items:
            q = np.where(arr < 0, 255, np.clip(np.round(arr * 254), 0, 254)).astype(np.uint8)
            if used + 3 + q.size > min(limit, 14000):
                break
            cache_parts.append(struct.pack("<BH", st, q.size))
            cache_parts.append(q.tobytes())
            used += 3 + q.size
            kept += 1
        cache_parts[0] = struct.pack("<iB", self.cache_hand, kept)
        out = body + b"".join(cache_parts)
        if len(out) > limit:  # never let truncation corrupt the blob: drop cache, then events
            out = body + struct.pack("<iB", -1, 0)
            if len(out) > limit:
                self.pending = self.pending[-50:]
                self.cache = {}
                return self.encode(limit) if len(self.pending) > 0 and limit > 2000 else b""
        return out

    @classmethod
    def decode(cls, blob) -> "Memory":
        m = cls()
        if not blob or len(blob) < 4 or blob[:2] != MAGIC:
            return m
        try:
            off = 2
            ver, n = struct.unpack_from("<BB", blob, off)
            off += 2
            if ver != VERSION or n > 16:
                return cls()
            m.seats = list(struct.unpack_from(f"<{n}h", blob, off))
            off += 2 * n
            m.last_stacks = np.frombuffer(blob, "<i4", n, off).astype(np.int64)
            off += 4 * n
            m.stats = np.frombuffer(blob, "<u2", n * K_STATS, off).astype(np.int64).reshape(n, K_STATS)
            off += 2 * n * K_STATS
            (npend,) = struct.unpack_from("<H", blob, off)
            off += 2
            pend = []
            for _ in range(npend):
                code, seat, amount, nc = struct.unpack_from("<Bbi B", blob, off)
                off += 7
                cards = tuple(blob[off:off + nc])
                off += nc
                pend.append((CODE_KIND.get(code, ""), None if seat < 0 else seat, amount, cards))
            m.pending = pend
            if off + 5 <= len(blob):
                m.cache_hand, nc = struct.unpack_from("<iB", blob, off)
                off += 5
                for _ in range(nc):
                    st, ln = struct.unpack_from("<BH", blob, off)
                    off += 3
                    q = np.frombuffer(blob, np.uint8, ln, off)
                    off += ln
                    m.cache[st] = np.where(q == 255, -1.0, q / 254.0)
            return m
        except Exception:  # noqa: BLE001 - any corruption -> start fresh
            return cls()
