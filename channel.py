from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..block import Block


@dataclass
class AttenuationChannel(Block[np.ndarray, np.ndarray]):
    """Затухание: y = k * x."""

    k: float = 0.01

    def process(self, x: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        return self.k * x

@dataclass
class NoChannel(Block[np.ndarray, np.ndarray]):
    """Прямой канал без шума и искажений (заглушка)."""
    def process(self, x: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        return x

@dataclass
class AWGNChannel(Block[np.ndarray, np.ndarray]):
    """AWGN с заданным SNR (dB). Мощность сигнала измеряется как mean(|x|^2)."""

    snr_db: float = 10.0
    seed: int = 42

    def process(self, x: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        signal_power = np.mean(np.abs(x) ** 2)
        noise_power = signal_power * 10 ** (-self.snr_db / 10.0)
        if np.iscomplexobj(x):
            # комплексный шум с равной мощностью в I и Q
            noise = np.sqrt(noise_power / 2) * (
                rng.standard_normal(size=x.shape) + 1j * rng.standard_normal(size=x.shape)
            )
        else:
            noise = np.sqrt(noise_power) * rng.standard_normal(size=x.shape)
        return x + noise

@dataclass
class RayleighFlatFadingChannel(Block[np.ndarray, np.ndarray]):
    """Канал с рэлеевскими замираниями (комплексный коэффициент).

    Параметры:
        seed: для воспроизводимости.
        block_size: если >0, коэффициент постоянен на блоке из block_size отсчётов;
                   иначе новый коэффициент на каждый отсчёт (быстрые замирания).
    """
    seed: int = 123
    block_size: int = 50

    def process(self, x: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = len(x)
        if self.block_size <= 1:
            h = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        else:
            # Блочное постоянство
            n_blocks = int(np.ceil(n / self.block_size))
            h_blocks = (rng.standard_normal(n_blocks) + 1j * rng.standard_normal(n_blocks)) / np.sqrt(2)
            h = np.repeat(h_blocks, self.block_size)[:n]
        return x * h
    
from dataclasses import field

@dataclass
class MultipathChannel(Block[np.ndarray, np.ndarray]):
    """Многолучевой канал с заданными лучами."""
    taps: list[tuple[int, complex]] = field(default_factory=lambda: [(0, 1.0)])

    def process(self, x: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        max_delay = max(t[0] for t in self.taps) if self.taps else 0
        # Определяем тип выходного массива: комплексный, если есть комплексные коэффициенты или входной сигнал
        output_dtype = np.complex128 if (np.iscomplexobj(x) or any(np.iscomplex(c) for _, c in self.taps)) else np.float64
        out = np.zeros(len(x) + max_delay, dtype=output_dtype)
        for delay, coeff in self.taps:
            out[delay:delay+len(x)] += np.array(coeff) * x
        return out[:len(x)]