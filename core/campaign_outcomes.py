"""Project campaign execution states without fabricating benchmark quality."""

def index_outcomes(status):
    rows = []
    for target, entry in status.get('targets', {}).items():
        state = entry.get('state', 'planned')
        started, finished = entry.get('started_at'), entry.get('finished_at')
        rows.append({'target': target, 'run_id': entry.get('run_id'), 'status': state,
                     'quality_evaluation': 'pending' if state == 'index_complete' else 'unavailable',
                     'quality_score': None,
                     'attempt_wall_seconds': max(0, finished-started) if started is not None and finished is not None else None,
                     'attempt_phase': 'index' if 'index_exit_code' in entry else 'smoke',
                     'exit_code': entry.get('index_exit_code', entry.get('smoke_exit_code')),
                     'failure': entry.get('failure'),
                     'logs': {phase: {stream: f'{phase}-{target.replace(chr(47), chr(45))}.{stream}.log' for stream in ('stdout', 'stderr')} for phase in ('smoke', 'index')},
                     'cost_telemetry': 'see attempt logs and index stats; unavailable values are not zero'})
    return {'version': 1, 'scope': 'index_execution_outcomes', 'benchmark_admitted': False, 'targets': rows}


def outcome_markdown(report):
    lines = ['# Index execution outcomes', '', 'Execution report only; no benchmark quality is inferred.', '',
             '| Target | Status | Attempt phase | Wall seconds | Quality |', '|---|---|---|---:|---|']
    for row in report['targets']:
        wall = row['attempt_wall_seconds']
        seconds = f'{wall:.3f}' if wall is not None else 'N/A'
        lines.append(f"| {row['target']} | {row['status']} | {row['attempt_phase']} | {seconds} | {row['quality_evaluation']} |")
    return '\n'.join(lines) + '\n'
