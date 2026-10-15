import importlib.util
from pathlib import Path
import hashlib
import pytest

spec = importlib.util.spec_from_file_location('idle_watch', Path(__file__).parents[1]/'scripts/watch_gpu_idle.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_twenty_continuous_minutes_and_busy_unknown_or_gaps_reset():
    w = module.IdleWindow()
    for t in range(0, 1200, 30):
        assert not w.observe(t, True)[0]
    assert w.observe(1200, True)[0]
    assert w.observe(1230, False) == (False, 0)
    assert w.observe(1260, True) == (False, 0)
    assert w.observe(1400, True) == (False, 0)
    assert w.observe(100, True) == (False, 0)


def test_source_changes_block_launch(tmp_path):
    p = tmp_path/'runner.py'
    p.write_text('original')
    cfg = {'cwd': str(tmp_path), 'command': ['python', 'runner.py'],
           'source_sha256': {'runner.py': hashlib.sha256(p.read_bytes()).hexdigest()}}
    assert module.verify_spec(cfg) == tmp_path
    p.write_text('changed')
    with pytest.raises(ValueError, match='Source changed'):
        module.verify_spec(cfg)


def test_monitor_launches_exactly_once_and_records_child(tmp_path, monkeypatch):
    import json
    import sys
    script = tmp_path/'runner.py'
    script.write_text("from pathlib import Path\np=Path('count'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\n")
    cfg = {'cwd': str(tmp_path), 'command': [sys.executable, str(script)],
           'source_sha256': {'runner.py': hashlib.sha256(script.read_bytes()).hexdigest()},
           'gpu_uuid': 'GPU-test', 'min_free_gib': 32}
    config = tmp_path/'spec.json'; config.write_text(json.dumps(cfg))
    state = tmp_path/'state'
    monkeypatch.setattr(sys, 'argv', ['watch', '--spec', str(config), '--state-dir', str(state)])
    real_window = module.IdleWindow
    monkeypatch.setattr(module, 'IdleWindow', lambda: real_window(seconds=0))
    monkeypatch.setattr(module, 'sample_gpu', lambda uuid: dict(uuid=uuid, utilization=0, memory_utilization=0, free_mib=90000))
    old_term = module.signal.getsignal(module.signal.SIGTERM)
    old_int = module.signal.getsignal(module.signal.SIGINT)
    try:
        module.main()
        assert (tmp_path/'count').read_text() == '1'
        assert json.loads((state/'status.json').read_text())['phase'] == 'completed'
        with pytest.raises(ValueError, match='already claimed'):
            module.main()
        assert (tmp_path/'count').read_text() == '1'
    finally:
        module.signal.signal(module.signal.SIGTERM, old_term)
        module.signal.signal(module.signal.SIGINT, old_int)
