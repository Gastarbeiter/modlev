from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..block import Block


@dataclass
class AmplitudeModulator(Block[np.ndarray, np.ndarray]):
    """Амплитудная модуляция с несущей.

    Формула:
        s(t) = (1 + k * m(t)) * cos(2π f_c t)

    Вход:
        m(t): np.ndarray (сообщение)
    Выход:
        s(t): np.ndarray (вещественный passband сигнал)

    Требования к ctx:
        ctx["t"] должен быть установлен (обычно блоком Timebase).
    """

    fc: float = 50.0
    k: float = 0.7

    def process(self, m: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if ctx is None:
            ctx = {}
        t = ctx.get("t")
        if t is None:
            raise ValueError("AmplitudeModulator requires ctx['t'] (timebase)")

        carrier = np.cos(2.0 * np.pi * float(self.fc) * t)
        s = (1.0 + float(self.k) * m) * carrier
        ctx["carrier"] = carrier
        ctx["s"] = s
        return s
