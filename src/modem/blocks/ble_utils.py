"""Блоки для анализа BLE: поиск преамбулы, извлечение данных, де‑whitening, CRC."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from ..block import Block

# ----------------------------------------------------------------
# Вспомогательные функции (точная копия из старого рабочего кода)
# ----------------------------------------------------------------
def bt_swap_bits(byte: int) -> int:
    """Переворачивает порядок битов в байте (как в BLE)."""
    result = 0
    for i in range(8):
        result |= ((byte >> i) & 1) << (7 - i)
    return result

def ll_packet_data_dewhitening(data: bytes, channel: int) -> bytes:
    """
    Вайтенинг / девайтенинг данных BLE (побайтовый алгоритм, проверенный временем).
    Операция симметрична – используется и для whitening, и для dewhitening.
    """
    length = len(data)
    lfsr = (bt_swap_bits(channel) | 2) & 0xFF
    output = bytearray(data)

    for i in range(length):
        d = bt_swap_bits(output[i])
        j = 128
        while j >= 1:
            if lfsr & 0x80:
                lfsr ^= 0x11
                d ^= j
            lfsr = (lfsr << 1) & 0xFF
            j >>= 1
        output[i] = bt_swap_bits(d)

    return bytes(output)

def crc24_ble(data: bytes) -> int:
    """24‑битный CRC для BLE."""
    poly = 0x100065b
    crc = 0x555555
    for byte in data:
        crc ^= (byte << 16)
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= poly
    return crc & 0xFFFFFF

# ----------------------------------------------------------------
# Блоки
# ----------------------------------------------------------------
@dataclass
class PreambleSearcher(Block[np.ndarray, dict]):
    preamble_bits: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))

    def process(self, rx_bits: np.ndarray, *, ctx: Optional[dict] = None) -> dict:
        rx_pm = 2 * rx_bits.astype(np.int16) - 1
        pre_pm = 2 * self.preamble_bits.astype(np.int16) - 1
        corr = np.correlate(rx_pm, pre_pm, mode='valid')
        peak_bit = int(np.argmax(np.abs(corr)))
        peak_val = np.max(np.abs(corr))
        errors = np.sum(rx_bits[peak_bit:peak_bit+len(self.preamble_bits)] != self.preamble_bits)
        return {
            'peak_bit': peak_bit,
            'peak_value': peak_val,
            'errors': errors,
            'preamble_match': errors <= 2
        }

@dataclass
class DataExtractor(Block[np.ndarray, np.ndarray]):
    preamble_len: int = 40
    data_len: int = 136

    def process(self, rx_bits: np.ndarray, *, ctx: Optional[dict] = None) -> np.ndarray:
        if ctx is None or 'peak_bit' not in ctx:
            raise ValueError("ctx['peak_bit'] требуется")
        peak = ctx['peak_bit']
        start = peak + self.preamble_len
        end = start + self.data_len
        return rx_bits[start:end]

@dataclass
class Dewhitener(Block[np.ndarray, np.ndarray]):
    """Де‑whitening BLE (использует проверенный побайтовый алгоритм)."""
    channel: int = 38

    def process(self, bits: np.ndarray, *, ctx: Optional[dict] = None) -> np.ndarray:
        # Упаковываем биты в байты
        if len(bits) % 8 != 0:
            pad = 8 - len(bits) % 8
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        raw_bytes = np.packbits(bits, bitorder='little')
        # Применяем девайтенинг
        dew_bytes = ll_packet_data_dewhitening(bytes(raw_bytes), self.channel)
        # Обратно в биты
        return np.unpackbits(np.frombuffer(dew_bytes, dtype=np.uint8), bitorder='little')

# ----------------------------------------------------------------
# Блоки
# ----------------------------------------------------------------
@dataclass
class BLE_GFSK_Modulator(Block[np.ndarray, np.ndarray]):
    """GFSK модулятор, совместимый с Bluetooth LE (1 Мбит/с, BT=0.5, h=0.5).

    Вход: массив бит uint8 (LSB first для каждого байта).
    Выход: комплексная огибающая, частота дискретизации fs (обычно 8 МГц → 8 сэмплов/бит).
    """
    fs: float = 8e6          # частота дискретизации
    bt: float = 0.5
    deviation_hz: float = 250e3   # пиковая девиация (половина разноса)
    samples_per_symbol: int = 8   # sps = fs / 1e6

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if bits.dtype != np.uint8:
            raise TypeError("Expected uint8 bits")

        sps = self.samples_per_symbol
        # Создаём гауссовский фильтр
        sigma_sym = np.sqrt(np.log(2)) / (2 * np.pi * self.bt)  # в символах
        sigma_samp = sigma_sym * sps
        filter_span = int(3 / self.bt)  # длительность в символах
        filter_len = filter_span * sps
        t_filt = np.arange(-filter_len//2, filter_len//2 + 1)
        g = np.exp(-0.5 * (t_filt / sigma_samp) ** 2)
        g /= g.sum()

        # Преобразуем биты в NRZ: 0 -> -1, 1 -> +1
        symbols = 2.0 * bits.astype(np.float64) - 1.0
        upsampled = np.repeat(symbols, sps)

        # Фильтрация
        filtered = np.convolve(upsampled, g, mode='same')

        # Масштабируем, чтобы в установившемся режиме частота менялась на ±deviation_hz
        # после нормализации фильтра максимальный отклик на одиночный символ ≈ 1
        freq = filtered * self.deviation_hz / np.max(np.abs(filtered))

        # Интегрируем частоту для фазы
        phase = 2 * np.pi * np.cumsum(freq) / self.fs
        return np.exp(1j * phase)
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np


@dataclass
class BLE_GFSK_Demodulator(Block[np.ndarray, np.ndarray]):

    sps: int = 8
    sync_bits: Optional[np.ndarray] = None
    invert: bool = False
    fixed_offset: Optional[int] = None

    def process(
        self,
        signal: np.ndarray,
        *,
        ctx: Optional[dict[str, Any]] = None
    ) -> np.ndarray:

        if len(signal) < self.sps:
            return np.array([], dtype=np.uint8)

        # ==========================================
        # FM discriminator
        # ==========================================

        freq = np.angle(
            signal[1:] * np.conj(signal[:-1])
        )

        # удаляем постоянную составляющую
        freq = freq - np.mean(freq)

        # ==========================================
        # Matched filter
        # ==========================================

        mf = np.ones(self.sps) / self.sps

        freq = np.convolve(
            freq,
            mf,
            mode="same"
        )

        # ==========================================
        # Выбор offset
        # ==========================================

        offsets = (
            [self.fixed_offset]
            if self.fixed_offset is not None
            else range(self.sps)
        )

        best_bits = None
        best_score = -np.inf
        best_offset = None

        for offset in offsets:

            x = freq[offset:]

            n_bits = len(x) // self.sps

            if n_bits < 20:
                continue

            x = x[:n_bits * self.sps]

            symbols = x.reshape(
                n_bits,
                self.sps
            )

            # интеграл по символу
            metric = np.sum(
                symbols,
                axis=1
            )

            # динамический порог
            threshold = np.mean(metric)

            bits = (
                metric > threshold
            ).astype(np.uint8)

            if self.invert:
                bits = 1 - bits

            # ----------------------------------
            # Оценка качества
            # ----------------------------------

            if self.sync_bits is not None:

                if len(bits) < len(self.sync_bits):
                    continue

                bits_pm = (
                    2 * bits.astype(np.int16) - 1
                )

                sync_pm = (
                    2 * self.sync_bits.astype(np.int16) - 1
                )

                corr = np.correlate(
                    bits_pm,
                    sync_pm,
                    mode="valid"
                )

                score = np.max(
                    np.abs(corr)
                )

                peak = np.argmax(
                    np.abs(corr)
                )

                print(
                    f"offset={offset} "
                    f"score={score} "
                    f"peak={peak}"
                )

            else:

                score = np.var(metric)

            if score > best_score:

                best_score = score
                best_bits = bits
                best_offset = offset

        if best_bits is None:
            return np.array([], dtype=np.uint8)

        print(
            f"\nВыбран offset = {best_offset}"
        )

        return best_bits
