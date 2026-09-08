"""Publication byte parity only; no experiment or historical payload access.

Suggested commit: test(f1-live): preserve exact compact evidence publication
"""
import json
from pathlib import Path

import pytest

from . import publish as p


def test_copies_preserve_exact_original_json_and_source_bytes(tmp_path):
    original = tmp_path/'original.json'; original.write_bytes(b'{ "unicode": "caf\xc3\xa9", "x": 1 }\n')
    source = tmp_path/'run.py'; source.write_bytes(b'# exact historical source\r\nprint("x")\r\n')
    copied = p.copy_compact([(original,'execution_v2/result.json'),
        (source,'prior_freeze/freeze_source_before_guard/run.py')],tmp_path/'published',root=tmp_path)
    check = p.verify_copies(copied,root=tmp_path)
    assert check['all_copies_byte_identical'] and check['files_checked']==2
    assert original.read_bytes()==(tmp_path/'published/execution_v2/result.json').read_bytes()
    assert source.read_bytes()==(tmp_path/'published/prior_freeze/freeze_source_before_guard/run.py').read_bytes()
    with pytest.raises(FileExistsError): p.copy_compact([],tmp_path/'published',root=tmp_path)


def test_altered_published_payload_is_detected(tmp_path):
    original=tmp_path/'a.json'; original.write_text('{"ok":true}\n')
    copied=p.copy_compact([(original,'a.json')],tmp_path/'published',root=tmp_path)
    (tmp_path/'published/a.json').write_text('{"ok":false}\n')
    with pytest.raises(AssertionError): p.verify_copies(copied,root=tmp_path)


@pytest.mark.parametrize('suffix',['.jsonl','.npz','.bin','.pkl','.pt','.httpbody'])
def test_raw_rows_arrays_and_models_are_refused_before_output(tmp_path,suffix):
    source=tmp_path/('source'+suffix);source.write_bytes(b'not a compact receipt')
    with pytest.raises(AssertionError): p.copy_compact([(source,'copied'+suffix)],tmp_path/'published',root=tmp_path)
    assert not (tmp_path/'published').exists()


def test_nonstandard_json_and_path_escape_rejected_before_output(tmp_path):
    source=tmp_path/'a.json';source.write_text('{"x":NaN}')
    with pytest.raises(ValueError):p.copy_compact([(source,'a.json')],tmp_path/'published',root=tmp_path)
    assert not (tmp_path/'published').exists()
    source.write_text('{}')
    with pytest.raises(AssertionError):p.copy_compact([(source,'../a.json')],tmp_path/'published',root=tmp_path)
    assert not (tmp_path/'published').exists()
