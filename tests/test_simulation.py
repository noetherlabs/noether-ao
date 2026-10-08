"""Small deterministic checks of physical and causal invariants, never timing."""
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from noether_optics.artifacts import load_config
from noether_optics.atmosphere import phase_sequence
from noether_optics.simulation import (modal_basis,pyramid_sensor,calibration,loop,
    initialize_weights,predict_fit,paired_interval)


@pytest.fixture
def cfg():
    c=load_config(Path(__file__).parents[1]/'configs/simulation.yaml')
    c.update(n=16,modes=8,detector_n=32,frames=16,burn_in=4,sensor='linear',screen_factor=2)
    c['faults'].update(onset=8,end=16)
    return c


def test_modal_basis_is_orthonormal_and_piston_free(cfg):
    b=modal_basis(cfg['n'],cfg['modes']); mask=np.any(b!=0,axis=0)
    np.testing.assert_allclose(b[:,mask].mean(axis=1),0,atol=1e-13)
    np.testing.assert_allclose(b[:,mask]@b[:,mask].T/mask.sum(),np.eye(cfg['modes']),atol=1e-12)


def test_frozen_flow_and_r0_scaling(cfg):
    c=deepcopy(cfg)
    for layer in c['layers']: layer['velocity_m_s']=[0,0]
    a=np.asarray(phase_sequence(c,4)); b=np.asarray(phase_sequence(c|{'r0_m':2*c['r0_m']},4))
    np.testing.assert_allclose(a[0],a[-1],atol=1e-12)
    np.testing.assert_allclose(b,a*2**(-5/6),rtol=1e-12,atol=1e-12)
    c['layers']=c['layers'][:1]
    with pytest.raises(ValueError): phase_sequence(c,4)


def test_pyramid_flux_and_derivative(cfg):
    sensor=pyramid_sensor(16,(650e-9,750e-9),750e-9,32)
    phase=jnp.asarray(modal_basis(16,2)[0])*.05
    y=sensor(phase)
    assert float(y.min())>=0
    np.testing.assert_allclose(y.sum(),1,atol=1e-12)
    direction=jnp.asarray(modal_basis(16,2)[1])
    derivative=jax.jvp(sensor,(phase,),(direction,))[1]
    eps=1e-5
    finite=(sensor(phase+eps*direction)-sensor(phase-eps*direction))/(2*eps)
    np.testing.assert_allclose(derivative,finite,rtol=1e-5,atol=1e-10)


def test_delay_fifo_causality_and_unavailable_fallback(cfg):
    cal=calibration(cfg); p=jnp.asarray(modal_basis(16,8)[0])[None]*jnp.ones((16,1,1))*.1
    pred=jnp.stack([jnp.ones(8)*2,-jnp.ones(8)],axis=1)
    run=jax.jit(lambda p:loop(cfg,cal,p,8,'forward',pred,threshold=1e30,fallback=True,fault='missing_twin'))
    a=run(p); b=run(p.at[12:].multiply(3))
    np.testing.assert_array_equal(a['applied'][:2],0)
    np.testing.assert_allclose(a['applied'][2:],a['command'][:-2])
    np.testing.assert_allclose(a['prediction'][:13],b['prediction'][:13])
    np.testing.assert_array_equal(a['fallback'][:8],False)
    np.testing.assert_array_equal(a['fallback'][8:],True)
    assert a['variance'][-1]<a['variance'][0]


def test_training_gradients_are_finite_and_not_zero(cfg):
    cfg=cfg|{'sensor':'pyramid'}; cal=calibration(cfg)
    p=phase_sequence(cfg,4); pred=jnp.stack([jnp.ones(8)*2,-jnp.ones(8)],axis=1)
    params=initialize_weights(7)
    def loss(params):
        out=loop(cfg,cal,p,4,'learned',pred,params=params,differentiable_noise=True)
        return out['variance'][4:].mean()
    lossvalue,g=jax.jit(jax.value_and_grad(loss))(params)
    assert np.isfinite(lossvalue)
    assert all(np.isfinite(x).all() for x in g)
    assert sum(float(jnp.sum(x*x)) for x in g)>0


def test_predictor_alignment_and_paired_interval():
    a=np.arange(100,dtype=float)[:,None]
    p=predict_fit(a,2,1e-8)
    np.testing.assert_allclose(p,[[3,-2]],atol=1e-6)
    assert paired_interval([2,3,4,5,6],[1,2,3,4,5])==(1.,1.,1.)


def test_experiment_split_and_flux_contract():
    from noether_optics.config import validate_simulation
    cfg=load_config(Path(__file__).parents[1]/'configs/simulation.yaml')
    validate_simulation(cfg)
    with pytest.raises(ValueError,match='overlap'):
        validate_simulation(cfg|{'seeds':[11,102,103,104,105]})
    with pytest.raises(ValueError,match='five distinct'):
        validate_simulation(cfg|{'seeds':[101,101,103,104,105]})


def test_shifted_pyramid_conserves_full_pupil_flux(cfg):
    nominal=pyramid_sensor(16,(750e-9,),750e-9,32)
    shifted=pyramid_sensor(16,(750e-9,),750e-9,32,pupil_shift_pixels=2)
    flat=jnp.zeros((16,16))
    np.testing.assert_allclose(shifted(flat).sum(),nominal(flat).sum(),atol=1e-12)
    assert np.linalg.norm(shifted(flat)-nominal(flat))>1e-4


def test_missing_twin_holds_advanced_command_without_fallback(cfg):
    cal=calibration(cfg);phase=phase_sequence(cfg,4)
    predictor=jnp.stack((jnp.ones(8)*2,-jnp.ones(8)),axis=1)
    out=loop(cfg,cal,phase,4,'forward',predictor,fault='missing_twin')
    np.testing.assert_allclose(out['command'][8:],jnp.broadcast_to(out['command'][7],(8,8)))
