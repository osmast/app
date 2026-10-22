"""Acquisition protocols: the named capture + QA presets."""

import unittest

from daphnia import protocols


class ProtocolTests(unittest.TestCase):
    def test_default_protocol_exists(self):
        self.assertIn(protocols.DEFAULT_PROTOCOL, protocols.PROTOCOLS)

    def test_labels_are_unique_and_map_back_to_their_key(self):
        labels = protocols.labels()
        self.assertEqual(len(labels), len(set(labels)))
        for key, protocol in protocols.PROTOCOLS.items():
            self.assertEqual(protocols.key_for_label(protocol.label), key)

    def test_every_protocol_is_usable_and_states_its_trade_off(self):
        for protocol in protocols.PROTOCOLS.values():
            self.assertGreater(protocol.burst_size, 0, protocol.key)
            self.assertGreaterEqual(protocol.min_sharpness, 0.0, protocol.key)
            self.assertTrue(protocol.tradeoff, protocol.key)

    def test_unknown_key_and_label_fall_back_to_the_default(self):
        self.assertEqual(protocols.get("nonsense").key, protocols.DEFAULT_PROTOCOL)
        self.assertEqual(protocols.key_for_label("nonsense"), protocols.DEFAULT_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
