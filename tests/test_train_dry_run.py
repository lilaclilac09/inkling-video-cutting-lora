import json

import pytest

import prepare
import train


@pytest.mark.parametrize("accuracy", ["1.0", "0.0"])
def test_dry_run(tmp_path, monkeypatch, capsys, accuracy):
    real_ledger = prepare.RESULTS_PATH
    before = real_ledger.read_bytes()
    ledger = tmp_path / "results.tsv"
    ledger.write_text("sentinel\n")
    monkeypatch.setattr(prepare, "RESULTS_PATH", ledger)
    monkeypatch.setattr(train, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(prepare, "record_result", lambda *args: pytest.fail("dry run wrote ledger"))
    assert train.main(["--dry-run", "--run-id", "test", "--fake-accuracy", accuracy]) == 0
    predictions = (tmp_path / "test/predictions.jsonl").read_text().splitlines()
    assert len(predictions) == len(prepare.split(prepare.load_rows())[1])
    assert all(json.loads(line)["answer"] for line in predictions)
    assert "dry run: ledger not written" in capsys.readouterr().out
    assert ledger.read_text() == "sentinel\n"
    assert real_ledger.read_bytes() == before
