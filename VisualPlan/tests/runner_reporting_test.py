"""The standalone runner must distinguish passed, skipped, and failed checks."""

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest import SkipTest
from unittest.mock import patch

import run_tests


def test_runner_reports_skips_and_preserves_failure_exit_status():
    def passing():
        pass

    def skipped():
        raise SkipTest("optional dependency unavailable")

    def failing():
        raise AssertionError("intentional regression fixture")

    fixture = ModuleType("fixture_test")
    fixture.test_pass = passing
    fixture.test_skip = skipped
    fixture.test_fail = failing
    with TemporaryDirectory() as td:
        here = Path(td)
        (here / "fixture_test.py").touch()
        for include_failure, expected_status, expected_summary in (
            (True, 1, "1 passed, 1 skipped, 1 failed across 1 modules"),
            (False, 0, "1 passed, 1 skipped, 0 failed across 1 modules"),
        ):
            if not include_failure:
                del fixture.test_fail
            output = io.StringIO()
            with patch.object(run_tests, "HERE", here), patch.object(
                run_tests.importlib, "import_module", return_value=fixture
            ), redirect_stdout(output), redirect_stderr(io.StringIO()):
                status = run_tests.main()
            assert status == expected_status
            assert expected_summary in output.getvalue(), output.getvalue()
            assert "SKIP fixture_test.test_skip: optional dependency unavailable" in output.getvalue()
