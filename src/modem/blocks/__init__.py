"""Набор блоков для учебного стенда."""

from .sources import (
    TextSource, RawBitSource, HexSource,
    Timebase, SineMessage, MessageSink
)
from .am import AmplitudeModulator
from .transforms import BytesToBits, BitsToBytes, ZeroOrderHold, IntegrateAndDump
from .modulators import (
    BPSKModulator, BPSKDemodulator,
    BFSKModulator, BFSKDemodulator, DBFSKDemodulator,
    QPSKModulator, QPSKDemodulator, DQPSKModulator, DQPSKDemodulator,
    PrependBit, DifferentialEncoder, DifferentialDecoder, DBPSKDemodulator,
    DummyModulator, DummyDemodulator, IQModulator, IQDemodulator,  GFSKModulator, BLE_GFSK_Modulator, BLE_GFSK_Demodulator, PreambleCorrelator, 
)
from .channel import (AWGNChannel, AttenuationChannel, 
                      RayleighFlatFadingChannel, MultipathChannel, 
                      NoChannel, DopplerChannel, FrequencyCorrector
)
from .coding import HammingCoder, HammingDecoder, DummyCoder, DummyDecoder
from .equalizers import LMSLinearEqualizer

__all__ = [
    # Источники
    "TextSource", "RawBitSource", "HexSource",
    "Timebase", "SineMessage", "MessageSink",
    # Аналоговая модуляция
    "AmplitudeModulator",
    # Преобразования
    "BytesToBits", "BitsToBytes", "ZeroOrderHold", "IntegrateAndDump",
    # Цифровая модуляция
    "BPSKModulator", "BPSKDemodulator",
    "BFSKModulator", "BFSKDemodulator", "DBFSKDemodulator",
    "QPSKModulator", "QPSKDemodulator", "DQPSKModulator", "DQPSKDemodulator",
    "PrependBit", "DifferentialEncoder", "DifferentialDecoder", "DBPSKDemodulator",
    "DummyModulator", "DummyDemodulator",  "IQModulator", "IQDemodulator" , "GFSKModulator",
    "PreambleCorrelator","BLE_GFSK_Modulator","BLE_GFSK_Demodulator",
    # Каналы
    "AWGNChannel", "AttenuationChannel", "RayleighFlatFadingChannel", 
    "NoChannel","MultipathChannel","DopplerChannel", "FrequencyCorrector",
    # Кодеры
    "HammingCoder", "DummyCoder", "DummyDecoder", "HammingDecoder",
    # Эквалайзеры
    "LMSLinearEqualizer",
]