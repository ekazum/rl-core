"""``DrawEvent`` — slots a buffer draw took, for the stats tally.

Best-effort: the buffer emits it with ``put_nowait`` and drops it on a full
queue; a lost event leaves a slot's count briefly off until its next overwrite.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DrawEvent:
    """Slots ``draw_block`` drew — the processor increments their draw tally."""

    indices: np.ndarray
