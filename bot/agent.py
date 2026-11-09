"""Competition entry point. Upload this folder (agent.py at the zip root).

Import-time work (inside the startup allowance): pin libraries to one thread,
build the variant plug-in, compile the Numba kernels and calibrate timing.
``act`` then only does lookups and short sampled calculations.
"""
import gc
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import poker_agent_sdk  # noqa: E402  (provided by the competition runtime)

from pb import brain  # noqa: E402

brain.init()
# Move everything built at import into the permanent GC generation so garbage
# collection pauses during act() stay short (forked copies inherit this).
gc.collect()
gc.freeze()


class Agent(poker_agent_sdk.Agent):
    def act(self, state, memory):
        t0 = time.perf_counter_ns()
        try:
            return brain.decide(state, memory, poker_agent_sdk, t0)
        except Exception:  # noqa: BLE001 - last line of defence: legal action, same memory
            return brain._fallback(poker_agent_sdk, state), memory
