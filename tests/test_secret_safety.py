import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from secret_safety import redact
from function_agent import run


class SecretSafetyTests(unittest.TestCase):
    def test_nested_output_redacts_connection_and_configured_secret(self):
        fake_url = ''.join(('postgresql', '://', 'example-user:example-pass@localhost:5432/example'))
        fake_secret = 'sample-key-value-not-real'
        with patch.dict(os.environ, {'DATABASE_URL': fake_url, 'EXAMPLE_API_KEY': fake_secret}):
            output = redact({'error': f'connection failed: {fake_url}', 'nested': [fake_secret]})
        self.assertNotIn(fake_url, str(output))
        self.assertNotIn(fake_secret, str(output))

    def test_unconfigured_connection_string_is_redacted(self):
        scheme = ''.join(('postgresql', '://'))
        value = 'error with ' + scheme + 'u:p@localhost/db'
        self.assertNotIn(scheme, redact(value))

    def test_agent_error_and_flow_do_not_expose_connection_value(self):
        fake_url = ''.join(('postgresql', '://', 'sample-user:sample-password@localhost/sample'))
        with patch.dict(os.environ, {'DATABASE_URL': fake_url}):
            with patch('function_agent.call_model', side_effect=RuntimeError('failed ' + fake_url)):
                trace = run('question', 'test', object())
        self.assertEqual(trace['status'], 'error')
        self.assertNotIn(fake_url, str(trace))

    def test_quoted_secret_and_dictionary_keys_are_redacted(self):
        secret = 'example-private-value'
        with patch.dict(os.environ, {'EXAMPLE_API_KEY': secret}):
            result = redact({secret: 'password="quoted value"'})
        self.assertNotIn(secret, str(result))
        self.assertNotIn('quoted value', str(result))


if __name__ == '__main__':
    unittest.main()
