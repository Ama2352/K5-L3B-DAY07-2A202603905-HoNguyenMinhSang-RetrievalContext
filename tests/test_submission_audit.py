import unittest

from src.common.submission_audit import environment_secrets, inspect_text


class SubmissionAuditTests(unittest.TestCase):
    def test_findings_never_expose_values(self):
        value = ''.join(['synthetic', '-credential-for-test'])
        source = 'API_KEY=' + value + '\n' + 'https://' + 'user:' + value + '@example.invalid'
        findings = inspect_text(source, 'config.txt', [value])
        self.assertEqual({f['kind'] for f in findings}, {
            'literal_credential_assignment', 'credential_bearing_url', 'environment_secret_value'})
        self.assertNotIn(value, repr(findings))

    def test_placeholder_and_source_patterns_are_allowed(self):
        self.assertEqual(inspect_text('API_KEY=<provided-at-runtime>\npassword=os.environ.get(\"PASSWORD\")', 'config.txt'), [])
        self.assertEqual(inspect_text('TOKEN = re.compile(r\"sk-[A-Za-z0-9]+\")', 'scanner.py'), [])

    def test_environment_matching_only_uses_sensitive_keys(self):
        sensitive = ''.join(['synthetic', '-environment-value'])
        self.assertEqual(environment_secrets({'EXAMPLE_API_KEY': sensitive,
                                              'PATH': 'arbitrary-path',
                                              'EXAMPLE_TOKEN': '[REDACTED]'}), {sensitive})


if __name__ == '__main__':
    unittest.main()
