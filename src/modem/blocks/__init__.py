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
    DummyModulator, DummyDemodulator, IQModulator, IQDemodulator 
)
from .channel import AWGNChannel, AttenuationChannel, RayleighFlatFadingChannel, MultipathChannel, NoChannel
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
     "DummyModulator", "DummyDemodulator",  "IQModulator", "IQDemodulator" ,
    # Каналы
    "AWGNChannel", "AttenuationChannel", "RayleighFlatFadingChannel", 
    "NoChannel","MultipathChannel",
    # Кодеры
    "HammingCoder", "DummyCoder", "DummyDecoder", "HammingDecoder",
    # Эквалайзеры
    "LMSLinearEqualizer",
]