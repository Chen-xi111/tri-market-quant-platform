import unittest
from unittest.mock import patch

from execution.ibkr_broker import IBKRBroker, PAPER_PORTS


class IBKRSafetyTests(unittest.TestCase):
    def test_only_known_paper_ports_are_allowed_by_default(self):
        self.assertEqual(PAPER_PORTS, frozenset({7497, 4002}))
        broker = IBKRBroker(port=7496, paper_only=True)
        self.assertFalse(broker.connect())
        self.assertIn("纸账户安全模式拒绝", broker.last_error)

    def test_live_port_cannot_be_used_by_accident(self):
        broker = IBKRBroker(port=4001, paper_only=True)
        self.assertFalse(broker.connect())
        self.assertFalse(broker.is_connected)

    def test_closed_paper_port_returns_actionable_socket_api_hint(self):
        broker = IBKRBroker(port=7497, paper_only=True, timeout=5)
        with patch(
            "execution.ibkr_broker.socket.create_connection",
            side_effect=ConnectionRefusedError,
        ):
            self.assertFalse(broker.connect())
        self.assertIn("127.0.0.1:7497 未监听", broker.last_error)
        self.assertIn("启用 ActiveX 和 Socket 客户端", broker.last_error)


if __name__ == "__main__":
    unittest.main()
