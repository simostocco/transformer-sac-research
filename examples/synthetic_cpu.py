"""Owned synthetic trades and random weights: no data, model files or network."""
import json

import numpy as np
import pandas as pd
import torch

from streaming_research.features.trades import BasicSecondAggConfig, FeatureCfg, aggregate_trades_1s, build_features_1s
from streaming_research.forecasting.transformer import HFTQuantileCPCTransformer
from streaming_research.runtime.transformer_fast import FastLocalTransformerRunner


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    t = np.arange(240, dtype=np.int64)
    trades = pd.DataFrame(dict(ts_event=1_767_225_600_000_000_000 + t * 10**9,
        price=10 + 0.001 * t, size=10 + t % 3, side=np.where(t % 2, 'B', 'A')))
    bars = aggregate_trades_1s(trades, BasicSecondAggConfig())
    features = build_features_1s(bars, FeatureCfg(W_entropy=10, W_persist=10, W_slope=10, W_z=180))
    features['kmeans_k4'] = 0
    features['kmeans_ret_k3'] = 0
    net = HFTQuantileCPCTransformer(x_dim=3, n_kmeans1=4, n_kmeans2=3,
        windows=[10], neutral_id=4, d_model=32, n_heads=4, n_layers=1, dropout=0,
        max_len=32, horizons=[10], quantiles=[0.5], use_cpc=False)
    metadata = dict(feat_cols=['dlogret_1', 'vol_log1p'], mean=[0, 0], std=[1, 1], seq_len=32,
                    quantiles=[0.5], target_spec=[['mfe', 10]], neutral_id=4, neutral_ret_id=3)
    runner = FastLocalTransformerRunner(net, metadata, device='cpu', emb_pool='mean16', precision='fp32')
    values, _ = runner.infer_batch([features.iloc[-32:]])
    print(json.dumps(dict(mode='SYNTHETIC_RANDOM_WEIGHTS_CPU', bars=len(bars),
        output_fields=len(values[0]), finite=bool(np.isfinite(list(values[0].values())).all()),
        qualification='not a trained-model or live-performance reproduction')))


if __name__ == '__main__':
    main()
