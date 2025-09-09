import importlib

def test_imports_ok():
    """Basic import check for main dependencies (Flask stack)."""
    for pkg in ["flask", "pydantic", "joblib", "numpy", "pandas"]:
        importlib.import_module(pkg)

def test_dummy_math():
    """Trivial test to always pass (ensures pytest is wired)."""
    assert 2 + 2 == 4
