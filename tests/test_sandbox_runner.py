"""The sandbox's containment guarantees, exercised with hostile test files —
this is what makes generated code safe to run at all."""

from pathlib import Path

from agent_maker import sandbox_runner


def write_agent(tmp_path: Path, test_body: str) -> Path:
    (tmp_path / "test_agent.py").write_text(test_body)
    return tmp_path


def test_passing_tests_pass(tmp_path):
    agent_dir = write_agent(tmp_path, "def test_ok():\n    assert 1 + 1 == 2\n")
    passed, output = sandbox_runner.run_tests(agent_dir)
    assert passed, output


def test_failing_tests_fail_with_output(tmp_path):
    agent_dir = write_agent(tmp_path, "def test_bad():\n    assert 1 == 2\n")
    passed, output = sandbox_runner.run_tests(agent_dir)
    assert not passed
    assert "test_bad" in output


def test_network_is_denied(tmp_path):
    agent_dir = write_agent(
        tmp_path,
        "import urllib.request\n"
        "def test_phone_home():\n"
        "    urllib.request.urlopen('https://example.com', timeout=5)\n",
    )
    passed, _ = sandbox_runner.run_tests(agent_dir)
    assert not passed


def test_write_outside_agent_dir_is_denied(tmp_path):
    escape_target = Path.home() / "sandbox_escape_should_not_exist.txt"
    agent_dir = write_agent(
        tmp_path,
        "def test_escape():\n"
        f"    open({str(escape_target)!r}, 'w').write('escaped')\n",
    )
    passed, _ = sandbox_runner.run_tests(agent_dir)
    assert not passed
    assert not escape_target.exists()


def test_env_is_scrubbed(tmp_path):
    agent_dir = write_agent(
        tmp_path,
        "import os\n"
        "def test_no_secrets():\n"
        "    assert os.environ.get('ANTHROPIC_API_KEY') is None\n"
        "    assert os.environ.get('DATABASE_URL') is None\n",
    )
    passed, output = sandbox_runner.run_tests(agent_dir)
    assert passed, output
