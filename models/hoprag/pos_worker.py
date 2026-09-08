"""JSON-lines transport to PaddleNLP's native CPU POS task."""
import contextlib,hashlib,json,os,shutil,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
runtime=root/'data/runtime_envs/hoprag-paper-20260908'
output=Path(os.environ['RAG_HOP_OUTPUT_ROOT']);output.mkdir(parents=True,exist_ok=True)
task_dir=Path(tempfile.mkdtemp(prefix='native-pos-',dir=output))
for name in json.loads((root/'configs/hoprag_pos_model.json').read_text()):
    shutil.copy2(runtime/'pos-model'/name,task_dir/name)
with contextlib.redirect_stdout(sys.stderr):
    from paddlenlp import Taskflow
    task=Taskflow('pos_tagging',device_id=-1,task_path=str(task_dir))
(task_dir/'realized_files.json').write_text(json.dumps({str(p.relative_to(task_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in task_dir.rglob('*') if p.is_file()},sort_keys=True))
for line in sys.stdin:
    try:
        with contextlib.redirect_stdout(sys.stderr):result=task(json.loads(line))
        value={'result':result}
    except Exception as exc:
        value={'error':f'{type(exc).__name__}: {exc}'}
    print(json.dumps(value,ensure_ascii=False),flush=True)
