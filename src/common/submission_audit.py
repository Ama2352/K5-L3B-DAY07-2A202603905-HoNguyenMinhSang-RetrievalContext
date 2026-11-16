"""Scan submission candidates without disclosing matched credential values."""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

TOKEN = re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{30,}|AIza[A-Za-z0-9_-]{30,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b')
PRIVATE_KEY = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----')
URL = re.compile(r'\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"\'<>]+')
ASSIGNMENT = re.compile(r'''^\s*(?:export\s+|\$env:|\$)?["']?(?P<key>(?:(?:[A-Za-z0-9]+[_-])*(?:api[_-]?key|password|passwd|secret|token|authorization|access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key)))["']?\s*[:=]\s*(?P<value>"[^"\n]*"|'[^'\n]*'|[^\s,;#]+)''', re.I)


def placeholder(value):
    value = value.strip().strip('\"\'').strip()
    return (not value or value.lower() in {'none', 'null', 'false', 'true'}
            or value.startswith(('[REDACTED', '<', '${', '{{')))


def environment_secrets(environment):
    """Return only potential values in memory; they never enter audit output."""
    return {v for k, v in environment.items() if len(v) >= 8 and
            (k.upper() == 'DATABASE_URL' or k.upper().endswith(
                ('_API_KEY', '_TOKEN', '_SECRET', '_PASSWORD', '_PRIVATE_KEY')))
            and not placeholder(v)}


def inspect_text(text, path, known_secrets=()):
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        kinds = set()
        if TOKEN.search(line):
            kinds.add('credential_token')
        if PRIVATE_KEY.search(line):
            kinds.add('private_key')
        if any(secret in line for secret in known_secrets):
            kinds.add('environment_secret_value')
        for match in URL.finditer(line):
            try:
                parsed = urlsplit(match.group())
                if parsed.password and not placeholder(parsed.password):
                    kinds.add('credential_bearing_url')
            except ValueError:
                pass
        match = ASSIGNMENT.match(line)
        if match and not placeholder(match['value']):
            value = match['value']
            # Only literal assignments; expressions such as os.environ.get(...)
            # are code, not embedded credentials.
            if value.startswith(('"', "'")) or re.fullmatch(r'[A-Za-z0-9_./+@!$%-]+', value):
                kinds.add('literal_credential_assignment')
        findings.extend({'path': path, 'line': number, 'kind': kind} for kind in sorted(kinds))
    return findings


def candidate_paths(root):
    result = subprocess.run(['git', '-c', f'safe.directory={root.as_posix()}',
                             'ls-files', '-co', '--exclude-standard', '-z'],
                            cwd=root, capture_output=True, check=True)
    return sorted(set(part.decode('utf8') for part in result.stdout.split(b'\0') if part))


def audit(root, environment=None):
    root = Path(root).resolve()
    secrets = environment_secrets(os.environ if environment is None else environment)
    findings = []
    scanned = 0
    for relative in candidate_paths(root):
        path = root / relative
        if not path.is_file():
            continue
        scanned += 1
        name = path.name.lower()
        if name == '.env' or name.startswith('.env.') or name.endswith('.env'):
            findings.append({'path': relative, 'line': 0, 'kind': 'environment_file'})
        raw = path.read_bytes()
        if b'\0' in raw:
            continue
        text = raw.decode('utf-8-sig', errors='replace')
        findings.extend(inspect_text(text, relative, secrets))
    return {'scope': 'Existing tracked and nonignored untracked files; excludes Git history and ignored local files.',
            'scanned_files': scanned, 'passed': not findings, 'findings': findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('tmp/security-audit.json'))
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
