import json

from jev_mcts.pilot import CASES, run_pilot


def test_pilot_cases_cover_two_apartments_and_four_task_types():
    assert len(CASES) == 8
    assert len({case.name for case in CASES}) == 8
    for case in CASES:
        assert case.unseen_apartment == case.name.startswith("unseen_")
        assert case.unseen_item == ("novel_" in case.name)


def test_pilot_resumes_without_repeating_completed_episode(monkeypatch, tmp_path):
    import jev_mcts.pilot as pilot

    generated = []
    executed = []

    def fake_generate(executable, dataset, **kwargs):
        generated.append(dataset.name)
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_bytes(b"task")

    def fake_run(dataset, executable, priors, output, **kwargs):
        executed.append(output.name)
        result = {
            "status": "success", "success": True, "steps": [{"action": "walk"}],
            "jev_usage": {"calls": 1, "input_tokens": 100, "output_tokens": 0, "models": ["jev-1.13.0"]},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result))
        return result

    monkeypatch.setattr(pilot, "generate_one", fake_generate)
    monkeypatch.setattr(pilot, "run_one", fake_run)
    first = run_pilot(tmp_path / "unity", tmp_path / "priors", tmp_path / "run")
    second = run_pilot(tmp_path / "unity", tmp_path / "priors", tmp_path / "run")
    assert first["status"] == second["status"] == "complete"
    assert len(generated) == len(executed) == 8
    assert second["jev_usage"]["calls"] == 8
