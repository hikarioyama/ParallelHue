from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture
def socket_dir():
    # Unix socket addresses include the run ID and must fit sockaddr_un;
    # pytest's descriptive temporary paths can exceed that bound.
    with TemporaryDirectory(prefix=".ph-", dir=Path.home()) as directory:
        yield Path(directory)
