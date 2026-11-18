import io
import json
import unittest
from unittest.mock import patch

from bioagent_bench.gateway import GatewayClient, GatewayError


class GatewayTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_key_fails_clearly(self):
        with self.assertRaisesRegex(GatewayError, "AI_GATEWAY_API_KEY"):
            GatewayClient()

    @patch("bioagent_bench.gateway.urllib.request.urlopen")
    def test_openai_compatible_request(self, urlopen):
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        urlopen.return_value = Response(json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
            "usage": {"total_tokens": 3},
        }).encode("utf-8"))
        client = GatewayClient(api_key="test-key", base_url="https://gateway.example/v1")
        result = client.chat("anthropic/claude-haiku-5.5", [{"role": "user", "content": "hello"}], [])
        self.assertEqual(result["choices"][0]["message"]["content"], "done")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://gateway.example/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "anthropic/claude-haiku-5.5")
        self.assertEqual(payload["tool_choice"], "auto")


if __name__ == "__main__":
    unittest.main()
