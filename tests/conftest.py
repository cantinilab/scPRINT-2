import shutil
import sys

import bionty as bt
import lamindb as ln
import pytest
from lamindb_setup.errors import InstanceNotFoundError


def pytest_sessionstart():
    ln.setup.init(storage="./test-scprintdb", name="test-scprint", modules="bionty")


def pytest_sessionfinish(session):
    shutil.rmtree("./test-scprintdb", ignore_errors=True)
    try:
        ln.setup.delete("test-scprint", force=True)
    except InstanceNotFoundError:
        pass


# each test runs on cwd to its temp dir
@pytest.fixture(autouse=True)
def go_to_tmpdir(request):
    # Get the fixture dynamically by its name.
    tmpdir = request.getfixturevalue("tmpdir")
    # ensure local test created packages can be imported
    sys.path.insert(0, str(tmpdir))
    # Chdir only for the duration of the test.
    with tmpdir.as_cwd():
        yield
