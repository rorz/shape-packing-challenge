"""Run a bounded experiment and retain command, source hashes, output and status.

Example: python scripts/research_run.py --label probe --hypothesis '...' -- \
    .venv/bin/python scripts/experiment.py --seconds 60
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import platform
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'experiments/2026-11-04/runs')
    parser.add_argument('--label', required=True)
    parser.add_argument('--hypothesis', required=True)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide a command after --')
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sources = {}
    snapshots = directory.parent/'source-snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    for folder in ('src', 'scripts'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.suffix in ('.py', '.cpp'):
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                sources[str(path.relative_to(ROOT))] = digest
                snapshot = snapshots/(digest+path.suffix)
                if not snapshot.exists():
                    snapshot.write_bytes(content)
    inputs = {}
    for argument in [*command, 'solution.json', 'experiments/best.json',
                     'experiments/branch-3.141616-3.141592.json']:
        if argument.startswith('-') and '=' in argument:
            argument = argument.split('=',1)[1]
        path = ROOT/argument
        if path.is_file() and path.suffix in ('.json', '.npz'):
            inputs[argument] = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps([command, sources, inputs], sort_keys=True).encode()).hexdigest()
    ledger = directory/'ledger.jsonl'
    if ledger.exists() and not args.repeat:
        for line in ledger.read_text().splitlines():
            previous = json.loads(line)
            if previous['fingerprint'] == fingerprint and previous.get('exit_code') == 0:
                raise SystemExit('Identical successful run already recorded: '+previous['record']+' (use --repeat intentionally)')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    label = ''.join(c if c.isalnum() or c in '-_' else '_' for c in args.label)
    prefix = directory/(stamp+'-'+label)
    record = dict(started_utc=stamp, hypothesis=args.hypothesis, command=command,
                  cwd=str(ROOT), sources=sources, inputs=inputs, fingerprint=fingerprint,
                  python=sys.version, platform=platform.platform(),
                  packages={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions()},
                  record=str(prefix.with_suffix('.json').relative_to(ROOT)),
                  log=str(prefix.with_suffix('.log').relative_to(ROOT)))
    prefix.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
    started = time.monotonic()
    env = os.environ.copy()
    for key in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '1'
    with prefix.with_suffix('.log').open('w') as output:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record['exit_code'] = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            record['exit_code'] = 124
            record['timed_out'] = True
    record['seconds'] = time.monotonic()-started
    record['sources_changed_during_run'] = [name for name,digest in sources.items()
        if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    record['output_sha256'] = hashlib.sha256(prefix.with_suffix('.log').read_bytes()).hexdigest()
    prefix.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
    with ledger.open('a') as out:
        out.write(json.dumps({k:v for k,v in record.items() if k not in ('sources','inputs')})+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('sources','inputs','packages')}), flush=True)
    sys.exit(record['exit_code'])


if __name__ == '__main__':
    main()
