from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..block import Block


@dataclass
class BytesToBits(Block[bytes, np.ndarray]):
    """Байты → биты (MSB-first, uint8)."""

    def process(self, data: bytes, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if len(data) == 0:
            return np.array([], dtype=np.uint8)
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='big')


@dataclass
class BitsToBytes(Block[np.ndarray, bytes]):
    """Биты (MSB-first) → байты."""

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> bytes:
        # Дополняем нулями до кратности 8
        pad = (8 - len(bits) % 8) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        return np.packbits(bits, bitorder='big').tobytes()


@dataclass
class ZeroOrderHold(Block[np.ndarray, np.ndarray]):
    """Повышение частоты дискретизации прямоугольными импульсами."""

    samples_per_symbol: Optional[int] = None

    def process(self, symbols: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        sps = self.samples_per_symbol
        if sps is None and ctx is not None:
            sps = ctx.get("samples_per_symbol", 8)
        if sps is None:
            raise ValueError("samples_per_symbol must be provided in constructor or ctx")
        return np.repeat(symbols, sps)
    
@dataclass
class IntegrateAndDump(Block[np.ndarray, np.ndarray]):
    """Усреднение сигнала по символьным интервалам (комплексный вход)."""
    samples_per_symbol: int = 8

    def process(self, waveform: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        sps = self.samples_per_symbol
        n_symbols = len(waveform) // sps
        trimmed = waveform[:n_symbols * sps]
        # Усреднение комплексных отсчётов
        symbols = np.mean(trimmed.reshape(n_symbols, sps), axis=1)
        return symbols