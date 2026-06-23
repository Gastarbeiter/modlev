"""Генерация синтетического BLE-пакета (использует проверенный вайтенинг)."""
import numpy as np
from .ble_utils import crc24_ble, ll_packet_data_dewhitening, BLE_GFSK_Modulator          # правильный импорт

def synthesize_ble_packet(
    channel: int = 38,
    access_addr: int = 0x8E89BED6,
    pdu_payload: bytes = b'\x11\x22\x33\x44\x55\x66',
    pdu_type: int = 0,
    tx_add: int = 0,
    rx_add: int = 0,
    offset_hz: float = 15000.0,
    snr_db: float = 35.0,
    fs: float = 8e6,
    sps: int = 8,
    deviation_hz: float = 250e3
) -> np.ndarray:
    """Создаёт IQ-сигнал для одного BLE-пакета с заданными параметрами."""
    payload_len = len(pdu_payload)
    if not (0 <= payload_len <= 37):
        raise ValueError("Длина payload должна быть 0..37 байт")

    # Заголовок PDU
    pdu_type_bits = [(pdu_type >> i) & 1 for i in range(4)]
    rfu_bit = [0]
    ch_sel_bit = [0]
    tx_add_bit = [tx_add]
    rx_add_bit = [rx_add]
    length_bits = [(payload_len >> i) & 1 for i in range(6)]

    header_bits = np.array(
        pdu_type_bits + rfu_bit + ch_sel_bit + tx_add_bit + rx_add_bit + length_bits,
        dtype=np.uint8
    )
    header_bytes = np.packbits(header_bits, bitorder='little').tobytes()

    # Сборка PDU и CRC
    pdu = header_bytes + pdu_payload
    crc_val = crc24_ble(pdu)
    crc_bytes = crc_val.to_bytes(3, byteorder='little')

    # Вайтенинг (тот же алгоритм, что в анализаторе)
    data_to_whiten = pdu + crc_bytes
    whitened_bytes = ll_packet_data_dewhitening(data_to_whiten, channel)

    # Полный пакет байтов
    preamble = b'\xAA'
    addr_bytes = access_addr.to_bytes(4, byteorder='little')
    packet_bytes = preamble + addr_bytes + whitened_bytes

    # Биты LSB first
    bits = np.unpackbits(np.frombuffer(packet_bytes, dtype=np.uint8), bitorder='little')

    # Модуляция
    mod = BLE_GFSK_Modulator(fs=fs, deviation_hz=deviation_hz, samples_per_symbol=sps)
    iq_signal = mod(bits)

    # Частотный сдвиг
    if offset_hz != 0.0:
        t = np.arange(len(iq_signal)) / fs
        iq_signal *= np.exp(1j * 2.0 * np.pi * offset_hz * t)

    # Шум
    if snr_db is not None:
        signal_power = np.mean(np.abs(iq_signal)**2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(iq_signal)) + 1j * np.random.randn(len(iq_signal))
        )
        iq_signal += noise

    return iq_signal.astype(np.complex64)