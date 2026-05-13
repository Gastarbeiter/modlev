from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..block import Block


# --- Временная ось и сообщения (аналоговые) ---

@dataclass
class Timebase(Block[None, np.ndarray]):
    """Источник временной оси t (для passband-моделирования)."""

    n: int = 2000
    fs: float = 1000.0

    def process(self, x: None, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if ctx is None:
            ctx = {}
        t = np.arange(self.n, dtype=np.float64) / float(self.fs)
        ctx["fs"] = float(self.fs)
        ctx["t"] = t
        return t


@dataclass
class SineMessage(Block[np.ndarray, np.ndarray]):
    """Сообщение m(t) как синус."""

    f: float = 5.0
    a: float = 0.5
    phase: float = 0.0

    def process(self, t: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if ctx is None:
            ctx = {}
        m = self.a * np.sin(2.0 * np.pi * float(self.f) * t + float(self.phase))
        ctx["m"] = m
        return m


# --- Цифровые источники ---

@dataclass
class TextSource(Block[None, bytes]):
    """Источник: строка → байты UTF-8."""

    message: str = "Hello, BPSK!"

    def process(self, x: None, *, ctx: Optional[dict[str, Any]] = None) -> bytes:
        return self.message.encode("utf-8")


@dataclass
class RawBitSource(Block[None, np.ndarray]):
    """Источник бит из строки '0' и '1'."""

    bitstring: str = "0"

    def __post_init__(self):
        if not set(self.bitstring).issubset({'0', '1'}):
            raise ValueError("bitstring must contain only '0' and '1'")

    def process(self, x: None, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        return np.array([int(b) for b in self.bitstring], dtype=np.uint8)


@dataclass
class HexSource(Block[None, bytes]):
    """Источник байт из hex-строки (пробелы игнорируются)."""

    hexstring: str = "00"

    def process(self, x: None, *, ctx: Optional[dict[str, Any]] = None) -> bytes:
        clean = self.hexstring.replace(" ", "")
        return bytes.fromhex(clean)
    
# modem/blocks/sources.py (MessageSink)
@dataclass
class MessageSink(Block[bytes, str]):
    """Приёмник: байты → строка (UTF-8) с обработкой ошибок."""
    errors: str = "replace"  # 'strict', 'replace', 'ignore'
    
    def process(self, data: bytes, *, ctx: Optional[dict[str, Any]] = None) -> str:
        return data.decode("utf-8", errors=self.errors)