#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_secrets.py -- fail if credential-like strings or key files are tracked."""
import os, re, subprocess, sys

PATTERNS = {
    'OpenSSH private key': r'BEGIN OPENSSH PRIVATE KEY',
    'RSA private key':     r'BEGIN RSA (PRIVATE|PUBLIC) KEY',
    'EC private key':      r'BEGIN EC PRIVATE KEY',
    'GitHub PAT':          r'ghp_[A-Za-z0-9]{36}',
    'GitHub fine PAT':     r'github_pat_[A-Za-z0-9_]{20,}',
    'AWS access key id':   r'AKIA[0-9A-Z]{16}',
    'Generic token':       r'(?i)(api[_-]?key|access[_-]?token)\s*[=:]\s*["\'][A-Za-z0-9_\-]{16,}',
    'Slack token':         r'xox[baprs]-[0-9A-Za-z\-]{10,}',
}
KEYFILE_RE = re.compile(r'(^|/)(id_rsa|id_ed25519|id_ecdsa|id_dsa)(\.pub)?$|\.pem$|\.pfx$|\.p12$|\.key$')

def run(*args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, errors='ignore')

def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    have_git = os.path.isdir(os.path.join(repo, '.git'))
    if have_git:
        g = run('git', 'ls-files', cwd=repo)
        files = g.stdout.splitlines() if g.returncode == 0 else []
    else:
        files = []
        for dp, dns, fns in os.walk(repo):
            dns[:] = [d for d in dns if d not in ('.git', '.venv', '__pycache__')]
            for fn in fns:
                files.append(os.path.relpath(os.path.join(dp, fn), repo).replace(os.sep, '/'))
    print('[repo] %s  mode=%s  files=%d' % (repo, 'git' if have_git else 'walk', len(files)))
    if not files:
        print('[FAIL] no files scanned')
        return 1
    bad = []
    for f in files:
        if KEYFILE_RE.search(f):
            bad.append('key file tracked: ' + f); continue
        if os.path.basename(f).startswith('check_secrets.py'):
            continue
        p = os.path.join(repo, f)
        try:
            if os.path.getsize(p) > 5 * 1024 * 1024: continue
            txt = open(p, 'rb').read().decode('utf-8', 'ignore')
        except OSError:
            continue
        for name, pat in PATTERNS.items():
            if re.search(pat, txt):
                bad.append('%s : %s' % (f, name))
    if have_git:
        hist = run('git', 'log', '-p', '--all', cwd=repo).stdout
        for name, pat in PATTERNS.items():
            n = len(re.findall(pat, hist))
            if n: bad.append('history diff : %s x%d' % (name, n))
        hist_files = set(run('git', 'log', '--all', '--pretty=format:', '--name-only', cwd=repo).stdout.splitlines())
        bad += ['key file in history: ' + f for f in hist_files if f and KEYFILE_RE.search(f)]
    else:
        print('[WARN] no .git: history scan skipped; this is a file snapshot, not proof of clean git history')
    if bad:
        print('[FAIL] %d finding(s):' % len(bad))
        for b in bad: print('    ' + b)
        return 1
    print('[PASS] no credential-like strings or key files in tree or history')
    return 0

if __name__ == '__main__':
    sys.exit(main())