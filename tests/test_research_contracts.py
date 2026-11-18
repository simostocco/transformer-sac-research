import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from streaming_research.features.trades import BasicSecondAggConfig, FeatureCfg, aggregate_trades_1s, build_features_1s, transformer_feature_cols
from streaming_research.forecasting.transformer import HFTQuantileCPCTransformer
from streaming_research.policy.sac import SAC, SacCfg, NetworkArchCfg
from streaming_research.runtime.transformer_runner import LocalTransformerRunner
from streaming_research.runtime.transformer_fast import FastLocalTransformerRunner


def model(length=32):
    torch.manual_seed(7)
    return HFTQuantileCPCTransformer(x_dim=3, n_kmeans1=4, n_kmeans2=3,
        windows=[10], neutral_id=4, d_model=32, n_heads=4, n_layers=1,
        dropout=0, max_len=length, horizons=[10], quantiles=[0.5],
        use_cpc=False, relpos_features=False, use_rel_bias=True, use_ret_delta_bias=True).eval()


def metadata(length=32):
    return dict(feat_cols=['f0', 'f1'], mean=[0, 0], std=[1, 1], seq_len=length,
                quantiles=[0.5], target_spec=[['mfe', 10]], neutral_id=4, neutral_ret_id=3)


def test_trade_features_do_not_read_future_rows():
    seconds = np.arange(500, dtype=np.int64)
    trades = pd.DataFrame(dict(ts_event=1_767_225_600_000_000_000 + seconds * 10**9,
        price=10 + seconds * 0.001 + np.sin(seconds) * 0.01,
        size=10 + seconds % 7, side=np.where(seconds % 2, 'B', 'A')))
    bars = aggregate_trades_1s(trades, BasicSecondAggConfig())
    cfg = FeatureCfg(W_entropy=10, W_persist=10, W_slope=10, W_z=180)
    prefix = build_features_1s(bars.iloc[:400], cfg)
    full = build_features_1s(bars, cfg)
    cols = transformer_feature_cols(cfg)
    np.testing.assert_array_equal(prefix[cols].to_numpy(), full[cols].iloc[:400].to_numpy())
    assert bars['trades'].sum() == len(trades)


def test_transformer_attention_is_causal():
    net = model()
    x = torch.randn(1, 32, 3)
    k = torch.zeros(1, 32, dtype=torch.int64)
    r = torch.zeros(1, 32, 1)
    changed = x.clone()
    changed[:, 16:] += 10
    with torch.inference_mode():
        a, b = net(x, k, k, r), net(changed, k, k, r)
    torch.testing.assert_close(a['h'][:, :16], b['h'][:, :16], rtol=0, atol=1e-6)
    assert torch.isfinite(a['q']).all()


def test_original_and_fast_inference_paths_agree_on_cpu():
    net = model()
    frame = pd.DataFrame(dict(f0=np.arange(32) * 0.01, f1=np.arange(32) * 0.02,
        close=10 + np.arange(32) * 0.001, kmeans_k4=0, kmeans_ret_k3=0))
    baseline = LocalTransformerRunner(net, metadata(), device='cpu', emb_pool='mean16')
    fast = FastLocalTransformerRunner(net, metadata(), device='cpu', emb_pool='mean16', precision='fp32')
    a, _ = baseline.infer_batch([frame])
    b, _ = fast.infer_batch([frame])
    assert a[0].keys() == b[0].keys()
    for key in a[0]:
        assert a[0][key] == pytest.approx(b[0][key], abs=1e-6)


@pytest.mark.parametrize('bad', [dict(std=[0, 1]), dict(mean=[float('nan'), 0]), dict(seq_len=4),
    dict(quantiles=[0.9]), dict(target_spec=[]), dict(neutral_ret_id=0)])
def test_model_binding_rejects_invalid_metadata(bad):
    with pytest.raises(ValueError):
        LocalTransformerRunner(model(), metadata() | bad)


def test_signed_beta_inventory_mapping_and_tiny_cpu_update():
    arch = dataclasses.replace(NetworkArchCfg(), actor_hidden_dims=(16,), critic_state_hidden_dims=(16,),
        critic_action_hidden_dims=(16,), critic_hidden_dims=(16,), critic_k=8)
    agent = SAC(stack_dim=4, emb_dim=0, prev_u_idx_in_obs=3, cfg=SacCfg(device='cpu'), arch=arch)
    state = torch.zeros(4, 4)
    state[:, 1] = torch.tensor([0.0, 0.25, 1.0, 0.5])
    latent = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])
    action, _ = agent._latent_to_executable_action(state, latent)
    torch.testing.assert_close(action[:, 0], torch.tensor([0.0, -0.25, 0.0, 0.5]))
    batch = dict(s=state, sp=state.clone(), a=action, r=torch.zeros(4, 1), d=torch.zeros(4, 1))
    first = agent.update(batch)
    assert np.isfinite(first['q_loss']) and np.isnan(first['pi_loss'])
    stats = agent.update(batch)
    assert stats and all(np.isfinite(value) for value in stats.values())


@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize('field', ['forecast', 'embedding'])
def test_nonfinite_forecasts_never_become_valid(fast, field):
    runner = (FastLocalTransformerRunner(model(), metadata(), device='cpu', emb_pool='mean16')
              if fast else LocalTransformerRunner(model(), metadata(), device='cpu'))
    q, embedding = np.zeros((1, 1)), np.zeros(32)
    if field == 'forecast':
        q[0, 0] = np.nan
    else:
        embedding[0] = np.inf
    with pytest.raises(ValueError, match='nonfinite'):
        runner._materialize_output(q, embedding)


def test_equal_timestamp_trades_preserve_source_order():
    from types import SimpleNamespace
    from streaming_research.features.trades import trades_records_to_dataframe
    offsets = np.array([0, 2, 2, 0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 0, 1, 0, 0, 1, 1])
    trades = pd.DataFrame(dict(ts_event=1_767_225_600_000_000_000 + offsets * 100,
                              price=np.arange(1, 21), size=1, side='B'))
    converted = trades_records_to_dataframe([SimpleNamespace(**r) for r in trades.to_dict('records')], 'SYNTH')
    for frame in [trades, converted]:
        bars = aggregate_trades_1s(frame, BasicSecondAggConfig())
        assert bars.iloc[0]['close'] == 14
        assert bars.iloc[0]['trades'] == 20


def test_one_second_aggregator_rejects_lossy_cadence():
    trades = pd.DataFrame(dict(ts_event=1_767_225_600_000_000_000 + np.arange(5) * 10**9,
                              price=10, size=1, side='B'))
    with pytest.raises(ValueError, match='cadence_s=1'):
        aggregate_trades_1s(trades, BasicSecondAggConfig(cadence_s=2))
