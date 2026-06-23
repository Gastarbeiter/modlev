from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..block import Block


@dataclass
class BPSKModulator(Block[np.ndarray, np.ndarray]):
    """BPSK: биты 0/1 → символы -1/+1."""

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if bits.dtype != np.uint8:
            raise TypeError("Expected uint8 bits")
        return (-1.0) + 2.0 * bits.astype(np.float64)

@dataclass
class BPSKDemodulator(Block[np.ndarray, np.ndarray]):
    """BPSK демодулятор: жёсткое решение по знаку.

    Вход: np.ndarray символов (float/complex) – может быть с шумом.
    Выход: np.ndarray uint8 (0/1), где положительное значение -> 1, иначе 0.
    """
    def process(self, symbols: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        # Работает с вещественными BPSK (символы -1/+1)
        # Для надёжности берём действительную часть (если комплексный)
        real_part = np.real(symbols)
        bits = (real_part > 0).astype(np.uint8)
        return bits
    
@dataclass
class BFSKModulator(Block[np.ndarray, np.ndarray]):
    """BFSK модулятор – комплексная огибающая с двумя частотами.

    Вход: биты uint8 (0/1).
    Выход: комплексная огибающая (samples_per_symbol на бит).

    Параметры:
        deviation_hz: девиация частоты (разнос между тонами / 2).
        samples_per_symbol: число семплов на бит.
        fs: частота дискретизации (из ctx, если не задана).
    """
    deviation_hz: float = 10.0
    samples_per_symbol: int = 20
    fs: Optional[float] = None

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if ctx is None:
            ctx = {}
        fs = self.fs or ctx.get("fs")
        if fs is None:
            raise ValueError("fs must be provided via constructor or ctx")
        sps = self.samples_per_symbol
        n = len(bits)
        t = np.arange(n * sps) / fs
        # Частоты для каждого бита: 0 -> -deviation, 1 -> +deviation
        freq = np.where(np.repeat(bits, sps) == 1, self.deviation_hz, -self.deviation_hz)
        # Генерация комплексной экспоненты
        phase = 2 * np.pi * np.cumsum(freq) / fs  # интеграл частоты
        return np.exp(1j * phase)


@dataclass
class BFSKDemodulator(Block[np.ndarray, np.ndarray]):
    deviation_hz: float = 10.0
    samples_per_symbol: int = 20
    fs: Optional[float] = None
    invert: bool = False   # ← новый параметр

    def process(self, signal: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if ctx is None:
            ctx = {}
        fs = self.fs or ctx.get("fs")
        if fs is None:
            raise ValueError("fs must be provided")
        sps = self.samples_per_symbol
        n_bits = len(signal) // sps
        signal = signal[:n_bits * sps].reshape(n_bits, sps)
        t = np.arange(sps) / fs
        tone0_conj = np.exp(1j * 2 * np.pi * self.deviation_hz * t)
        tone1_conj = np.exp(-1j * 2 * np.pi * self.deviation_hz * t)
        corr0 = np.abs(np.sum(signal * tone0_conj, axis=1))
        corr1 = np.abs(np.sum(signal * tone1_conj, axis=1))
        bits = (corr1 > corr0).astype(np.uint8)
        if self.invert:
            bits = 1 - bits
        return bits


@dataclass
class DBFSKDemodulator(Block[np.ndarray, np.ndarray]):
    deviation_hz: float = 10.0
    samples_per_symbol: int = 20
    fs: Optional[float] = None
    invert: bool = False   # ← новый параметр

    def process(self, signal: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if ctx is None:
            ctx = {}
        fs = self.fs or ctx.get("fs")
        if fs is None:
            raise ValueError("fs must be provided")
        sps = self.samples_per_symbol
        n_symbols = len(signal) // sps
        signal = signal[:n_symbols * sps].reshape(n_symbols, sps)
        t = np.arange(sps) / fs
        tone0_conj = np.exp(1j * 2 * np.pi * self.deviation_hz * t)
        tone1_conj = np.exp(-1j * 2 * np.pi * self.deviation_hz * t)
        corr0 = np.sum(signal * tone0_conj, axis=1)
        corr1 = np.sum(signal * tone1_conj, axis=1)
        e0 = np.abs(corr0) ** 2
        e1 = np.abs(corr1) ** 2
        diff = e1 - e0
        bits = (diff[1:] * diff[:-1] < 0).astype(np.uint8)
        if self.invert:
            bits = 1 - bits
        return bits

@dataclass
class QPSKModulator(Block[np.ndarray, np.ndarray]):
    """QPSK модулятор с созвездием Грея.

    Маппинг: 00 -> 1+j, 01 -> -1+j, 11 -> -1-j, 10 -> 1-j
    (нормировка 1/√2)
    """
    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if bits.dtype != np.uint8:
            raise TypeError("Expected uint8 bits")
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
        pairs = bits.reshape(-1, 2)
        I = 1 - 2 * pairs[:, 0].astype(np.float64)   # 0 -> +1, 1 -> -1
        Q = 1 - 2 * pairs[:, 1].astype(np.float64)
        return (I + 1j * Q) / np.sqrt(2)


@dataclass
class QPSKDemodulator(Block[np.ndarray, np.ndarray]):
    """QPSK демодулятор: решение по знаку I и Q."""
    def process(self, symbols: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        bits = np.empty(len(symbols) * 2, dtype=np.uint8)
        bits[0::2] = (np.real(symbols) < 0).astype(np.uint8)   # I<0 -> бит0=1
        bits[1::2] = (np.imag(symbols) < 0).astype(np.uint8)   # Q<0 -> бит1=1
        return bits


@dataclass
class DQPSKModulator(Block[np.ndarray, np.ndarray]):
    """Дифференциальная QPSK модуляция (стандартный маппинг Грея).

    Кодируется *изменение* фазы:
      00 -> 0°, 01 -> +90°, 11 -> 180°, 10 -> -90°
    Начальный символ всегда 1+0j.
    """
    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
        pairs = bits.reshape(-1, 2)
        # Преобразуем пару в индекс, при котором 11 даёт 2 (фаза pi), а 10 – 3 (-pi/2)
        idx = pairs[:, 0] * 2 + (pairs[:, 1] ^ pairs[:, 0])
        # Таблица приращений: 0 -> +1, 1 -> +j, 2 -> -1, 3 -> -j
        table = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=np.complex128)
        d = table[idx]
        # Последовательное умножение
        symbols = np.empty(len(d) + 1, dtype=np.complex128)
        symbols[0] = 1 + 0j
        s = symbols[0]
        for i, val in enumerate(d):
            s = s * val
            symbols[i+1] = s
        return symbols


@dataclass
class DQPSKDemodulator(Block[np.ndarray, np.ndarray]):
    """DQPSK демодулятор: сравнивает соседние символы, извлекает биты."""
    def process(self, symbols: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        # Разность фаз: s[k] * conj(s[k-1])
        diff = symbols[1:] * np.conj(symbols[:-1])
        phase_diff = np.angle(diff)   # от -pi до pi
        # Квантуем к ближайшему значению 0, pi/2, pi, -pi/2
        idx = np.round(phase_diff / (np.pi/2)).astype(int) % 4
        # idx: 0 -> 00, 1 -> 01, 2 -> 11, 3 -> 10
        bits = np.empty(len(idx) * 2, dtype=np.uint8)
        bits[0::2] = (idx == 2) | (idx == 3)   # бит 0
        bits[1::2] = (idx == 1) | (idx == 2)   # бит 1
        return bits
    
@dataclass
class DifferentialEncoder(Block[np.ndarray, np.ndarray]):
    """Дифференциальный кодер для битового потока.

    Каждый входной бит сравнивается с предыдущим **выходным** битом:
        out[i] = in[i] XOR out[i-1]
    Начальное состояние: out[0] = 0 (или параметр initial_state).

    Вход: np.ndarray uint8 (0/1)
    Выход: np.ndarray uint8 (0/1)
    """
    initial_state: int = 0

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if len(bits) == 0:
            return np.array([], dtype=np.uint8)
        out = np.zeros_like(bits)
        prev = self.initial_state
        for i, b in enumerate(bits):
            out[i] = b ^ prev
            prev = out[i]
        return out


@dataclass
class DifferentialDecoder(Block[np.ndarray, np.ndarray]):
    """Дифференциальный декодер.

    Восстанавливает исходные биты:
        decoded[i] = rx[i] XOR rx[i-1]
    Начальное состояние: rx[-1] = initial_state.

    Вход: np.ndarray uint8 (0/1) – демодулированные биты.
    Выход: np.ndarray uint8 (0/1) – исправленные биты.
    """
    initial_state: int = 0

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if len(bits) == 0:
            return np.array([], dtype=np.uint8)
        # Сдвигаем, чтобы сравнивать соседей
        shifted = np.insert(bits[:-1], 0, self.initial_state)
        return bits ^ shifted


@dataclass
class DBPSKDemodulator(Block[np.ndarray, np.ndarray]):
    """DBPSK демодулятор (некогерентный).

    Принимает комплексные символы после интегрирования (или waveform),
    вычисляет разность фаз между соседними символами и принимает жёсткое решение.

    Алгоритм:
        r[i] = symbols[i] * conj(symbols[i-1])
        bit[i] = 1, если Re(r[i]) < 0, иначе 0

    Вход: np.ndarray complex, форма (N,)
    Выход: np.ndarray uint8, форма (N-1,) – биты
    """
    def process(self, symbols: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if len(symbols) < 2:
            return np.array([], dtype=np.uint8)
        # Произведение с предыдущим символом
        diff = symbols[1:] * np.conj(symbols[:-1])
        # Решение: Re < 0 -> 1 (изменение фазы), Re >= 0 -> 0 (нет изменения)
        bits = (np.real(diff) < 0).astype(np.uint8)
        return bits
    
@dataclass
class PrependBit(Block[np.ndarray, np.ndarray]):
    """Вставляет один фиксированный бит в начало массива (для DBPSK)."""
    bit: int = 0  # опорный бит, обычно 0

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        return np.concatenate([np.array([self.bit], dtype=bits.dtype), bits])
    
@dataclass
class DummyModulator(Block[np.ndarray, np.ndarray]):
    """Заглушка модулятора: биты 0/1 → символы -1/+1."""
    def process(self, bits: np.ndarray, *, ctx=None) -> np.ndarray:
        if bits.dtype != np.uint8:
            raise TypeError("Expected uint8 bits")
        return (-1.0) + 2.0 * bits.astype(np.float64)

@dataclass
class DummyDemodulator(Block[np.ndarray, np.ndarray]):
    """Заглушка демодулятора: жёсткое решение по знаку."""
    def process(self, symbols: np.ndarray, *, ctx=None) -> np.ndarray:
        real_part = np.real(symbols)
        return (real_part > 0).astype(np.uint8)
    
import scipy.signal as sps    # потребуется для filtfilt, но обойдёмся без него

@dataclass
class IQModulator(Block[np.ndarray, np.ndarray]):
    """Перенос комплексной огибающей на несущую частоту.

    Вход: комплексная огибающая (baseband), частота дискретизации fs из ctx.
    Выход: вещественный passband-сигнал.

    Параметры:
        fc: несущая частота (Гц). Если None – из ctx['fc'].
    """
    fc: Optional[float] = None

    def process(self, x: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if ctx is None:
            ctx = {}
        fc = self.fc if self.fc is not None else ctx.get("fc", None)
        if fc is None:
            raise ValueError("fc must be provided")
        fs = ctx.get("fs")
        if fs is None:
            raise ValueError("ctx['fs'] required")
        t = np.arange(len(x)) / fs
        return np.real(x * np.exp(1j * 2.0 * np.pi * fc * t))


@dataclass
class IQDemodulator(Block[np.ndarray, np.ndarray]):
    """Квадратурный демодулятор: passband → комплексная огибающая.

    Использует смеситель и ФНЧ для подавления удвоенной частоты.
    Фильтр – скользящее среднее с длиной, оптимизированной под несущую.

    Параметры:
        fc: несущая частота (Гц). Если None – из ctx['fc'].
        alpha: доля от fs, используемая для расчёта окна фильтра (по умолч. 0.5).
    """
    fc: Optional[float] = None
    alpha: float = 0.5   # окно будет fs/(4*fc) * alpha

    def process(self, s: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if ctx is None:
            ctx = {}
        fc = self.fc if self.fc is not None else ctx.get("fc", None)
        if fc is None:
            raise ValueError("fc must be provided")
        fs = ctx.get("fs")
        if fs is None:
            raise ValueError("ctx['fs'] required")

        t = np.arange(len(s)) / fs
        # Смеситель
        complex_baseband = s * np.exp(-1j * 2.0 * np.pi * fc * t)

        # Длина фильтра: примерно fs / (10*fc) для хорошего подавления
        # но не менее 1
        win_len = int(fs / (10.0 * fc))
        if win_len < 1:
            win_len = 1
        # Ограничиваем разумным числом, чтобы не смазать символы
        win_len = int(win_len * self.alpha)

        if win_len <= 1:
            return complex_baseband

        # Применяем скользящее среднее
        window = np.ones(win_len) / win_len
        # 'same' сохраняет длину, но края могут быть неточными
        filtered = np.convolve(complex_baseband, window, mode='same')
        return filtered

@dataclass
class GFSKModulator(Block[np.ndarray, np.ndarray]):
    """Универсальный GFSK модулятор (подходит для BLE и других систем).

    Параметры:
        bt: BT product (полоса фильтра * длительность бита)
        deviation_hz: пиковая девиация частоты (половина разноса тонов)
        samples_per_symbol: число семплов на бит
        fs: частота дискретизации (если None – берётся из ctx['fs'])
    """
    bt: float = 0.5
    deviation_hz: float = 250e3
    samples_per_symbol: int = 8
    fs: Optional[float] = None

    def __post_init__(self):
        sps = self.samples_per_symbol
        # Расчёт гауссовского фильтра
        sigma_sym = np.sqrt(np.log(2)) / (2 * np.pi * self.bt)   # стандартное отклонение в символах
        sigma_samp = sigma_sym * sps
        filter_span = int(3 / self.bt)  # длительность фильтра в символах
        filter_len = filter_span * sps
        t = np.arange(-filter_len//2, filter_len//2 + 1)
        g = np.exp(-0.5 * (t / sigma_samp) ** 2)
        self._gaussian_filter = g / np.sum(g)   # нормировка фильтра

    def process(self, bits: np.ndarray, *, ctx: Optional[dict[str, Any]] = None) -> np.ndarray:
        if bits.dtype != np.uint8:
            raise TypeError("Expected uint8 bits")

        # Частота дискретизации (из параметра или контекста)
        fs = self.fs or (ctx.get("fs") if ctx else None)
        if fs is None:
            raise ValueError("fs must be provided either as field or in ctx")

        sps = self.samples_per_symbol

        # Преобразуем биты в NRZ: 0 -> -1, 1 -> +1
        symbols = 2.0 * bits.astype(np.float64) - 1.0
        upsampled = np.repeat(symbols, sps)

        # Фильтрация гауссовым фильтром
        filtered = np.convolve(upsampled, self._gaussian_filter, mode='same')

        # Нормировка амплитуды фильтра: максимальный отклик на одиночный символ должен давать deviation_hz
        freq = filtered * self.deviation_hz / np.max(np.abs(filtered))

        # Интегрирование частоты для получения фазы
        phase = 2 * np.pi * np.cumsum(freq) / fs
        return np.exp(1j * phase)
    
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
class PreambleCorrelator(Block[np.ndarray, np.ndarray]):
    """Вычисляет взаимную корреляцию входного сигнала с идеальной преамбулой.

    Вход: комплексные отсчёты (возможно, зашумлённые).
    Выход: значения корреляции (1D массив той же длины).
    """
    preamble_signal: np.ndarray = None   # идеальный сигнал преамбулы (комплексный)

    def process(self, x: np.ndarray, *, ctx=None) -> np.ndarray:
        if self.preamble_signal is None:
            raise ValueError("preamble_signal must be set")
        return np.correlate(x, self.preamble_signal, mode='same')