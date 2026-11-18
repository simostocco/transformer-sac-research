import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.mark.skipif(not os.environ.get('RESEARCH_NATIVE_BUILD'), reason='requires optional native CPU build')
def test_numpy_float_log_on_owned_synthetic_corpus():
    values = np.geomspace(1e-6, 1e6, 4096, dtype=np.float32)
    probe = Path(os.environ['RESEARCH_NATIVE_BUILD']) / 'numeric_probe'
    output = subprocess.check_output([str(probe)], input='\n'.join(format(float(v), '.9g') for v in values), text=True)
    native = np.fromstring(output, sep='\n', dtype=np.float32)
    assert native.shape == values.shape
    np.testing.assert_allclose(native, np.log(values), rtol=0, atol=2e-6)
