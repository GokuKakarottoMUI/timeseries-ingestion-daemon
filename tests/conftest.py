"""Shared test setup.

The data root is redirected to a throwaway directory **before** anything from
``ingestion`` is imported, because ``ingestion.config_fetch_data`` resolves
``DATABASE_ROOT_PATH`` once at module load. This guarantees the suite can never
touch a real database, and lets the TileDB round-trip test write for real.
"""
import os
import shutil
import tempfile

_TMP_DATA_ROOT = tempfile.mkdtemp(prefix="tsd-tests-")
os.environ["TSD_DATA_ROOT"] = _TMP_DATA_ROOT


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP_DATA_ROOT, ignore_errors=True)
