"""Guard immutable raw runs and reject invalid numerical summaries."""
import json
from pathlib import Path

import numpy as np
import pytest

from noether_optics.artifacts import write_json,load_config
from noether_optics.latency import statistics


def test_json_rejects_nonfinite_and_does_not_replace_previous(tmp_path):
    p=tmp_path/'data.json'; write_json(p,{'value':1})
    with pytest.raises(ValueError): write_json(p,{'value':float('nan')})
    assert json.loads(p.read_text())=={'value':1}


@pytest.mark.parametrize('values',[[],[-1],[np.nan],[np.inf],[[1]]])
def test_bad_timing_vectors_are_rejected(values):
    with pytest.raises(ValueError): statistics(np.asarray(values))


def test_config_must_be_mapping(tmp_path):
    p=tmp_path/'bad.yaml';p.write_text('- a\n- b\n')
    with pytest.raises(ValueError):load_config(p)
