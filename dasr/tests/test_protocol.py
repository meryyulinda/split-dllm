import copy
import unittest
import socket
import threading
import json
import tempfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
from dasr.codec import BoundaryCache, BoundaryEncoder, CacheMismatch, numeric, native, pack, unpack
from dasr.runtime import RunConfig, CloudEndpoint, run_query
from dasr.scheduler import Action, Scheduler
from dasr.toy import ToyModel
from dasr.transport import decode_message, encode_message, receive, TCPClient


class ProtocolTests(unittest.TestCase):
    def test_xor_is_bitwise_across_dtypes_and_sign_zero(self):
        rng = np.random.RandomState(42)
        for dtype in (np.float16, np.float32, np.uint16):
            sender, receiver = BoundaryEncoder("xor", refresh=16), BoundaryCache()
            x = rng.randn(12, 8).astype(dtype)
            x[0, 0] = 0
            for step in range(10):
                x[step % 12, 1] = rng.randn()
                if dtype != np.uint16:
                    x[0, 0] = -0. if step % 2 else 0.
                encoded = sender.encode(x, 2, "query-a", logical_dtype="bfloat16" if dtype == np.uint16 else None)
                restored = receiver.apply(encoded.packet)
                self.assertEqual(x.tobytes(), restored.tobytes())
                self.assertEqual(sender.cache.value.tobytes(), restored.tobytes())

    def test_corruption_does_not_mutate_cache(self):
        sender, receiver = BoundaryEncoder("xor"), BoundaryCache()
        packet = sender.encode(np.ones((3, 4), np.float32), 1, "q").packet
        receiver.apply(packet)
        before, version = receiver.value.copy(), receiver.version
        packet = bytearray(packet)
        packet[-1] ^= 1
        with self.assertRaises(CacheMismatch):
            receiver.apply(bytes(packet))
        np.testing.assert_array_equal(receiver.value, before)
        self.assertEqual(receiver.version, version)

    def test_lost_delta_requires_anchor(self):
        sender, receiver = BoundaryEncoder("sparse", threshold=0., refresh=20), BoundaryCache()
        x = np.zeros((4, 8), np.float32)
        receiver.apply(sender.encode(x, 1, "q").packet)
        x[0] = 1
        sender.encode(x, 1, "q")  # dropped packet
        x[1] = 2
        with self.assertRaises(CacheMismatch):
            receiver.apply(sender.encode(x, 1, "q").packet)
        restored = receiver.apply(sender.encode(x, 1, "q", force_anchor=True).packet)
        np.testing.assert_array_equal(restored, x)

    def test_sparse_quantization_uses_reconstructed_reference(self):
        sender, receiver = BoundaryEncoder("sparse_int8", threshold=.001, refresh=30), BoundaryCache()
        rng = np.random.RandomState(2)
        x = rng.randn(12, 16).astype(np.float32)
        for _ in range(20):
            x += rng.randn(*x.shape).astype(np.float32) * .01
            encoded = sender.encode(x, 2, "q", changed=[0])
            restored = receiver.apply(encoded.packet)
            self.assertEqual(sender.cache.value.tobytes(), restored.tobytes())
            self.assertLess(encoded.relative_error, .02)

    def test_empty_sparse_patch_and_mandatory_rows(self):
        sender, receiver = BoundaryEncoder("sparse", threshold=1.), BoundaryCache()
        x = np.ones((4, 6), np.float32)
        receiver.apply(sender.encode(x, 1, "q").packet)
        encoded = sender.encode(x, 1, "q")
        self.assertEqual(encoded.active_ratio, 0.)
        receiver.apply(encoded.packet)
        encoded = sender.encode(x, 1, "q", changed=[3])
        self.assertEqual(encoded.active_ratio, .25)

    def test_split_and_query_changes_force_anchor(self):
        sender, receiver = BoundaryEncoder("xor"), BoundaryCache()
        x = np.ones((4, 6), np.float32)
        for split, query in ((1, "a"), (2, "a"), (2, "b")):
            e = sender.encode(x, split, query)
            self.assertTrue(e.anchor)
            receiver.apply(e.packet)

    def test_bfloat16_roundtrip(self):
        bits = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        values = numeric(bits, "bfloat16")
        finite = np.isfinite(values)
        back = native(values[finite], np.uint16, "bfloat16")
        np.testing.assert_array_equal(bits[finite], back)

    def test_float_subtract_add_can_lose_bits(self):
        x = np.array([1.], dtype=np.float32)
        old = np.array([1e20], dtype=np.float32)
        self.assertNotEqual((old + (x - old)).tobytes(), x.tobytes())


class RuntimeTests(unittest.TestCase):
    def test_real_loopback_rpc_matches_local_split(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        errors = []
        def server():
            try:
                conn, _ = listener.accept()
                with conn:
                    endpoint = CloudEndpoint(ToyModel(), 2)
                    for _ in range(8):
                        conn.sendall(endpoint.exchange(receive(conn)))
            except Exception as error:
                errors.append(error)
            finally:
                listener.close()
        worker = threading.Thread(target=server, daemon=True)
        worker.start()
        client = TCPClient("127.0.0.1", port)
        cfg = RunConfig(split=2, codec="xor", gen_length=8, steps=8, block_length=8)
        try:
            remote = run_query(ToyModel(), [1, 2, 3], cfg, client)
        finally:
            client.close()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        local = run_query(ToyModel(), [1, 2, 3], cfg)
        self.assertEqual(remote["output_ids"], local["output_ids"])

    def test_live_scheduler_uses_profile_at_refreshes(self):
        actions = [Action("full", 2, "full", 0., 2, 1., 1., 0., 0., 5000, 64, 100, 0.),
                   Action("xor", 2, "xor", 0., 2, 1., 1., 10., 0., 200, 64, 100, 0.)]
        with tempfile.TemporaryDirectory() as folder:
            profile = Path(folder) / "profile.json"
            trace = Path(folder) / "trace.json"
            profile.write_text(json.dumps(dict(actions=[asdict(a) for a in actions])))
            trace.write_text(json.dumps([dict(bandwidth_mbps=1000 if i < 2 else 1, rtt_ms=5) for i in range(8)]))
            cfg = RunConfig(split=2, refresh=2, steps=8, gen_length=8, block_length=8,
                            adaptive_profile=str(profile), network_trace=str(trace), memory_budget_bytes=1000)
            result = run_query(ToyModel(), [1, 2, 3], cfg)
        self.assertEqual(result["steps"][0]["scheduler_action"], "full")
        self.assertEqual(result["steps"][2]["scheduler_action"], "xor")
        dense = run_query(ToyModel(), [1, 2, 3], RunConfig(split=2, codec="full", steps=8, gen_length=8, block_length=8))
        self.assertEqual(result["output_ids"], dense["output_ids"])

    def test_exact_codecs_reproduce_dense_generation_all_splits(self):
        outputs = []
        for split in range(5):
            for codec in ("full", "zlib", "xor"):
                cfg = RunConfig(split=split, codec=codec, gen_length=8, steps=8, block_length=4)
                result = run_query(ToyModel(), [1, 2, 3], cfg)
                outputs.append(result["output_ids"])
                self.assertEqual(result["totals"]["output_mask_count"], 0)
                if split == 4:
                    self.assertEqual(result["totals"]["application_bytes"], 0)
        self.assertTrue(all(output == outputs[0] for output in outputs))

    def test_unchanged_prompt_hidden_states_drift(self):
        model = ToyModel()
        x = np.array([1, 2, 0, 0])
        before = model.prefix(x, 2)
        x[3] = 3
        after = model.prefix(x, 2)
        self.assertFalse(np.array_equal(before[0], after[0]))

    def test_selective_ffn_executes_fewer_rows(self):
        cfg = RunConfig(split=2, selective=True, refresh=8, active_fraction=.1,
                        gen_length=16, steps=16, block_length=16)
        result = run_query(ToyModel(), [1, 2, 3], cfg)
        self.assertLess(result["totals"]["ffn_selected_rows"], result["totals"]["ffn_total_rows"])

    def test_dense_boundary_packets_do_not_disable_suffix_ffn_reuse(self):
        cfg = RunConfig(split=2, codec="full", selective=True, refresh=8, active_fraction=.1,
                        gen_length=16, steps=16, block_length=16)
        model = ToyModel()
        run_query(model, [1, 2, 3], cfg)
        for layer in range(model.layers):
            self.assertLess(model.selected_by_layer[layer], 19 * 16)

    def test_sampler_cloud_only_uses_one_request(self):
        class CountEndpoint(CloudEndpoint):
            calls = 0
            def exchange(self, message):
                self.calls += 1
                return super().exchange(message)
        model = ToyModel()
        endpoint = CountEndpoint(model, 0)
        run_query(model, [1, 2], RunConfig(split=0, steps=8, gen_length=8, block_length=8), endpoint)
        self.assertEqual(endpoint.calls, 1)

    def test_stale_receiver_retries_full_anchor(self):
        class ForgetEndpoint(CloudEndpoint):
            calls = 0
            def exchange(self, message):
                self.calls += 1
                if self.calls == 2:
                    self.cache.reset()
                return super().exchange(message)
        model = ToyModel()
        endpoint = ForgetEndpoint(model, 2, reset_model=False)
        cfg = RunConfig(split=2, codec="sparse", gen_length=8, steps=8, block_length=8)
        result = run_query(model, [1, 2], cfg, endpoint)
        self.assertEqual(sum(s["retries"] for s in result["steps"]), 1)

    def test_invalid_step_schedule_rejected(self):
        with self.assertRaises(ValueError):
            RunConfig(gen_length=10, block_length=4).validate("llada")


class SchedulerTests(unittest.TestCase):
    def action(self, name, split, edge_ms, volume, risk=0):
        return Action(name, split, "full", 0., 4, edge_ms, 1., 0., 0., volume, 64, 1000, risk, 10.)

    def test_switch_only_at_anchor_and_reject_risk(self):
        small = self.action("small", 1, 30, 100)
        large = self.action("large", 2, 1, 100000)
        unsafe = self.action("unsafe", 3, 0, 0, risk=1.)
        scheduler = Scheduler([small, large, unsafe], 2000, .01)
        self.assertEqual(scheduler.choose(1000, 5, anchor=True).name, "large")
        self.assertEqual(scheduler.choose(1, 5, anchor=False).name, "large")
        self.assertEqual(scheduler.choose(1, 5, anchor=True).name, "small")


if __name__ == "__main__":
    unittest.main()
