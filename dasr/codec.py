"""Versioned, checksummed boundary packets with sender-side reconstruction.

NumPy carries native BF16 as uint16 bit patterns (logical_dtype='bfloat16').
Sparse updates are absolute replacements, avoiding floating-point delta inversion
and double-counted error feedback. Residuals always target the reconstructed cache.
"""
import json
import struct
import zlib
from dataclasses import dataclass
import numpy as np

PREFIX = struct.Struct("!4sIII")
MODES = ("full", "zlib", "xor", "int8", "sparse", "sparse_int8")


class CacheMismatch(ValueError):
    pass


def numeric(a, logical_dtype):
    if logical_dtype == "bfloat16":
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def native(a, dtype, logical_dtype):
    if logical_dtype == "bfloat16":
        # Round to nearest even, matching torch BF16 for finite numbers.
        bits = np.asarray(a, dtype=np.float32).view(np.uint32)
        return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return np.asarray(a, dtype=dtype)


def pack(header, payload):
    meta = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    crc = zlib.crc32(meta + payload)
    return PREFIX.pack(b"DSR1", len(meta), len(payload), crc) + meta + payload


def unpack(packet):
    if len(packet) < PREFIX.size:
        raise CacheMismatch("Truncated boundary packet")
    magic, meta_size, size, crc = PREFIX.unpack_from(packet)
    if magic != b"DSR1" or len(packet) != PREFIX.size + meta_size + size:
        raise CacheMismatch("Malformed boundary packet")
    body = packet[PREFIX.size:]
    if zlib.crc32(body) != crc:
        raise CacheMismatch("Boundary checksum mismatch")
    return json.loads(body[:meta_size]), body[meta_size:]


def quantize_rows(x):
    scale = np.maximum(np.max(np.abs(x), axis=1), np.finfo(np.float32).tiny) / 127
    q = np.clip(np.rint(x / scale[:, None]), -127, 127).astype(np.int8)
    return scale.astype("<f4"), q


class BoundaryCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None
        self.version = -1
        self.split = None
        self.logical_dtype = None
        self.session = None

    def apply(self, packet):
        h, payload = unpack(packet)
        shape = tuple(h["shape"])
        dtype = np.dtype(h["dtype"])
        mode = h["mode"]
        anchor = h["anchor"]
        if len(shape) != 2 or min(shape) <= 0 or dtype.kind not in "fu":
            raise CacheMismatch("Unsupported boundary shape/dtype")
        if not anchor and (self.value is None or h["base"] != self.version
                           or h["split"] != self.split or h["session"] != self.session
                           or shape != self.value.shape or dtype != self.value.dtype
                           or h["logical_dtype"] != self.logical_dtype):
            raise CacheMismatch("Stale/wrong reference: send a new anchor")
        if h["version"] != h["base"] + 1:
            raise CacheMismatch("Invalid version transition")
        if mode in ("full", "zlib", "xor"):
            raw = payload if mode == "full" else zlib.decompress(payload)
            if len(raw) != int(np.prod(shape)) * dtype.itemsize:
                raise CacheMismatch("Incorrect dense payload size")
            if mode == "xor":
                if anchor:
                    raise CacheMismatch("XOR requires a reference")
                delta = np.frombuffer(raw, dtype=np.uint8)
                raw = np.bitwise_xor(delta, self.value.view(np.uint8).ravel()).tobytes()
            out = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        elif mode in ("int8", "sparse", "sparse_int8"):
            count = h["rows"]
            if not 0 <= count <= shape[0]:
                raise CacheMismatch("Invalid row count")
            if mode == "int8":
                indices = np.arange(shape[0])
                start = 0
                if count != shape[0]:
                    raise CacheMismatch("Dense quantization must send all rows")
            else:
                start = count * 4
                indices = np.frombuffer(payload[:start], dtype="<u4").astype(np.int64)
                if len(indices) != count or len(np.unique(indices)) != count or np.any(indices >= shape[0]):
                    raise CacheMismatch("Invalid sparse indices")
            if mode in ("int8", "sparse_int8"):
                end = start + count * 4
                scales = np.frombuffer(payload[start:end], dtype="<f4")
                q = np.frombuffer(payload[end:], dtype=np.int8).reshape(count, shape[1])
                rows = native(scales[:, None] * q, dtype, h["logical_dtype"])
            else:
                rows = np.frombuffer(payload[start:], dtype=dtype).reshape(count, shape[1])
            if anchor and count != shape[0]:
                raise CacheMismatch("Sparse anchor must be complete")
            out = np.zeros(shape, dtype=dtype) if anchor else self.value.copy()
            out[indices] = rows
        else:
            raise CacheMismatch("Unknown codec")
        # Commit only after all validation, never partially apply a bad packet.
        self.value, self.version, self.split = out, h["version"], h["split"]
        self.logical_dtype, self.session = h["logical_dtype"], h["session"]
        return out


@dataclass
class Encoded:
    packet: bytes
    mode: str
    anchor: bool
    active_ratio: float
    relative_error: float
    bitwise_equal: bool
    version: int


class BoundaryEncoder:
    def __init__(self, mode="xor", threshold=0., refresh=4, top_fraction=0.):
        if mode not in MODES or threshold < 0 or refresh < 1 or not 0 <= top_fraction <= 1:
            raise ValueError("Invalid codec configuration")
        self.mode, self.threshold, self.refresh = mode, threshold, refresh
        self.top_fraction = top_fraction
        self.cache = BoundaryCache()
        self.since_anchor = 0

    def encode(self, value, split, session, changed=(), logical_dtype=None, force_anchor=False):
        value = np.ascontiguousarray(value)
        logical_dtype = logical_dtype or str(value.dtype)
        old = self.cache.value
        anchor = (force_anchor or old is None or old.shape != value.shape or old.dtype != value.dtype
                  or self.cache.split != split or self.cache.session != session
                  or self.cache.logical_dtype != logical_dtype or self.since_anchor >= self.refresh)
        indices = np.arange(value.shape[0])
        # Full/int8 modes are independent states, not patches.
        if self.mode in ("full", "zlib", "int8"):
            anchor = True
        mode = self.mode
        if anchor and mode in ("xor", "sparse", "sparse_int8"):
            mode = "full"  # Exact native anchors in both correctness tiers.
        if mode in ("sparse", "sparse_int8"):
            residual = numeric(value, logical_dtype) - numeric(old, logical_dtype)
            drift = np.linalg.norm(residual, axis=1) / (np.linalg.norm(numeric(value, logical_dtype), axis=1) + 1e-12)
            selected = drift > self.threshold
            changed = np.asarray(changed, dtype=np.int64)
            if np.any(changed < 0) or np.any(changed >= len(selected)):
                raise ValueError("Changed-token row outside sequence")
            selected[changed] = True
            top = int(np.ceil(len(selected) * self.top_fraction))
            if top:
                selected[np.argsort(drift)[-top:]] = True
            indices = np.flatnonzero(selected)
        if mode == "full":
            payload = value.tobytes()
        elif mode == "zlib":
            payload = zlib.compress(value.tobytes(), level=1)
        elif mode == "xor":
            delta = np.bitwise_xor(value.view(np.uint8), old.view(np.uint8))
            payload = zlib.compress(delta.tobytes(), level=1)
            # Never send an expanded delta when a native anchor is cheaper.
            if len(payload) >= value.nbytes:
                mode, anchor, payload = "full", True, value.tobytes()
        elif mode in ("int8", "sparse_int8"):
            scales, q = quantize_rows(numeric(value[indices], logical_dtype))
            payload = (b"" if mode == "int8" else indices.astype("<u4").tobytes()) + scales.tobytes() + q.tobytes()
        else:
            payload = indices.astype("<u4").tobytes() + value[indices].tobytes()
        h = dict(mode=mode, anchor=anchor, base=self.cache.version, version=self.cache.version + 1,
                 split=split, session=session, shape=list(value.shape), dtype=value.dtype.str,
                 logical_dtype=logical_dtype, rows=len(indices))
        packet = pack(h, payload)
        reconstructed = self.cache.apply(packet)
        self.since_anchor = 1 if anchor else self.since_anchor + 1
        error = np.linalg.norm(numeric(value, logical_dtype) - numeric(reconstructed, logical_dtype))
        error /= np.linalg.norm(numeric(value, logical_dtype)) + 1e-12
        return Encoded(packet, mode, anchor, len(indices) / value.shape[0], float(error),
                       value.tobytes() == reconstructed.tobytes(), self.cache.version)
