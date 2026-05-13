from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from ..block import Block



@dataclass
class HammingCoder(Block[np.ndarray, np.ndarray]):
    """Кодер Хэмминга (7,4) – систематический.

    Вход: биты (uint8) длиной кратной 4 (дополняется нулями).
    Выход: кодовые слова длиной 7 бит (всего 7*N блоков).
    """

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        # Дополнение нулями до кратности 4
        pad = (4 - len(bits) % 4) % 4
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        n_blocks = len(bits) // 4
        coded = np.zeros(n_blocks * 7, dtype=np.uint8)
        for i in range(n_blocks):
            d = bits[4*i:4*i+4]
            # Информационные биты (систематическая часть)
            coded[7*i + 2] = d[0]   # d1
            coded[7*i + 4] = d[1]   # d2
            coded[7*i + 5] = d[2]   # d3
            coded[7*i + 6] = d[3]   # d4
            # Проверочные биты
            coded[7*i + 0] = d[0] ^ d[1] ^ d[3]   # p1
            coded[7*i + 1] = d[0] ^ d[2] ^ d[3]   # p2
            coded[7*i + 3] = d[1] ^ d[2] ^ d[3]   # p3
        return coded


@dataclass
class HammingDecoder(Block[np.ndarray, np.ndarray]):
    """Декодер Хэмминга (7,4) с исправлением одиночной ошибки.

    Вход: кодовые слова длиной 7*N бит.
    Выход: восстановленные информационные биты (4*N).
    """

    def process(self, coded: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        n_blocks = len(coded) // 7
        decoded = np.zeros(n_blocks * 4, dtype=np.uint8)
        for i in range(n_blocks):
            c = coded[7*i:7*i+7]
            # Вычисление синдрома
            s1 = c[0] ^ c[2] ^ c[4] ^ c[6]
            s2 = c[1] ^ c[2] ^ c[5] ^ c[6]
            s3 = c[3] ^ c[4] ^ c[5] ^ c[6]
            error_pos = s1 * 1 + s2 * 2 + s3 * 4
            if 1 <= error_pos <= 7:
                c[error_pos - 1] ^= 1   # исправление ошибки
            # Извлечение информационных битов
            decoded[4*i + 0] = c[2]
            decoded[4*i + 1] = c[4]
            decoded[4*i + 2] = c[5]
            decoded[4*i + 3] = c[6]
        return decoded
    
@dataclass
class DummyCoder(Block[np.ndarray, np.ndarray]):
    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        return bits

@dataclass
class DummyDecoder(Block[np.ndarray, np.ndarray]):
    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        return bits