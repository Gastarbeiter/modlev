from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
from ..block import Block


@dataclass
class DummyEqualizer(Block[np.ndarray, np.ndarray]):
    """Эквалайзер-заглушка."""
    def process(self, x: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        return x


@dataclass
class LMSLinearEqualizer(Block[np.ndarray, np.ndarray]):
    """Линейный LMS-эквалайзер символьной скорости.

    Вход: 1-D массив символов (float или complex).
    Выход: эквализированные символы той же длины.

    Параметры:
        num_taps: число отводов (нечётное).
        step: шаг адаптации (mu).
        preamble: известные символы преамбулы (если None — берётся из ctx['preamble']).
        dd: если True, после преамбулы переходит в режим жёстких решений.
    """
    num_taps: int = 15
    step: float = 0.01
    preamble: Optional[np.ndarray] = None
    dd: bool = True

    def __post_init__(self):
        # Инициализация будет выполнена в process при первом вызове
        self.weights = None
        self._initialized = False

    def process(self, x: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        real_mode = not np.iscomplexobj(x)
        dtype = x.dtype

        # Инициализация весов при первом вызове или смене типа
        if not self._initialized or self.weights.dtype != dtype:
            if real_mode:
                self.weights = np.zeros(self.num_taps, dtype=np.float64)
                self.weights[self.num_taps // 2] = 1.0
            else:
                self.weights = np.zeros(self.num_taps, dtype=np.complex128)
                self.weights[self.num_taps // 2] = 1.0 + 0j
            self._initialized = True

        # Преамбула из контекста, если не задана явно
        if self.preamble is None and ctx is not None and 'preamble' in ctx:
            self.preamble = np.asarray(ctx['preamble'], dtype=dtype)

        out = np.zeros_like(x)

        for i in range(len(x)):
            # Формируем блок из последних num_taps отсчётов: x[i], x[i-1], ..., x[i-num_taps+1]
            start = max(0, i - self.num_taps + 1)
            block = x[start:i+1]          # от старого к новому
            # Дополняем нулями слева, если не хватает отсчётов в начале
            if len(block) < self.num_taps:
                pad = np.zeros(self.num_taps - len(block), dtype=dtype)
                block = np.concatenate([pad, block])
            # Переворачиваем, чтобы block[0] был самым свежим (x[i]), block[-1] — самым старым
            block = block[::-1]

            y = np.dot(self.weights, block)
            out[i] = y

            # Ошибка
            if self.preamble is not None and i < len(self.preamble):
                err = self.preamble[i] - y
            elif self.dd and (self.preamble is None or i >= len(self.preamble)):
                if real_mode:
                    dec = np.sign(y)
                else:
                    dec = np.sign(y.real) + 0j
                err = dec - y
            else:
                continue

            self.weights += self.step * err * np.conj(block)

        return out