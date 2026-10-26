"""Run the retained fallback evidence and corruption checks in correctness CI."""
import importlib.util
from pathlib import Path


_source = (
    Path(__file__).resolve().parents[1]
    / "results/scheduled_fallback_v1/test_verify.py"
)
_spec = importlib.util.spec_from_file_location("scheduled_fallback_release_tests", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ReleaseVerification = _module.ReleaseVerification
NumericalContracts = _module.NumericalContracts
