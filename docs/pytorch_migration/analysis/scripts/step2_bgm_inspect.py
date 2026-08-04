import pickle
import numpy as np
import json

PT = "model_artifacts/resources/housing-pt/vae/checkpoints/latent_model.pkl"
TF = "model_artifacts/resources/housing-tf/vae/checkpoints/latent_model.pkl"

np.random.seed(0)
out = {}


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def inspect(name, bgm):
    w = np.asarray(bgm.weights_)
    means = np.asarray(bgm.means_)              # (n_comp, latent_dim)
    # covariances_ shape depends on covariance_type (default 'full'):
    covs = np.asarray(bgm.covariances_)
    # effective components: weight above a small threshold
    eff = int((w > 0.01).sum())
    # average per-dimension within-component std (sqrt of diag of cov)
    if covs.ndim == 3:        # full: (n_comp, d, d)
        comp_std = np.sqrt(np.diagonal(covs, axis1=1, axis2=2))   # (n_comp, d)
    elif covs.ndim == 2:      # diag: (n_comp, d)
        comp_std = np.sqrt(covs)
    else:
        comp_std = np.sqrt(covs)[:, None]
    # weighted mean within-component std per dim, then avg over dims
    within_std = (w[:, None] * comp_std).sum(axis=0)        # per-dim
    # spread of component means (between-component) per dim
    between_std = means.std(axis=0)
    # total latent std the BGM actually produces (sample it)
    samp = bgm.sample(20000)[0]
    samp_std = samp.std(axis=0)

    print(f"\n===== {name} BGM =====")
    print(f"n_components (config) : {len(w)}")
    print(f"effective components  : {eff}   (weight > 0.01)")
    print(f"latent_dim            : {means.shape[1]}")
    print(f"covariance_type       : {bgm.covariance_type}")
    print(f"top-8 weights         : "
          f"{np.round(np.sort(w)[::-1][:8], 4).tolist()}")
    print(f"weight entropy / log(K): "
          f"{(-(w*np.log(w+1e-12)).sum())/np.log(len(w)):.3f}  "
          f"(1.0 = all components used equally)")
    print(f"per-dim within-comp std (weighted) : "
          f"{np.round(within_std, 3).tolist()}")
    print(f"per-dim between-comp std (means)   : "
          f"{np.round(between_std, 3).tolist()}")
    print(f"per-dim SAMPLED latent std         : "
          f"{np.round(samp_std, 3).tolist()}")
    print(f"mean sampled latent std (all dims) : {samp_std.mean():.4f}")
    out[name] = {
        "n_components_config": len(w),
        "effective_components": eff,
        "latent_dim": int(means.shape[1]),
        "covariance_type": bgm.covariance_type,
        "top8_weights": np.round(np.sort(w)[::-1][:8], 4).tolist(),
        "weight_entropy_ratio": round(float((-(w*np.log(w+1e-12)).sum())/np.log(len(w))), 3),
        "within_comp_std_per_dim": np.round(within_std, 4).tolist(),
        "between_comp_std_per_dim": np.round(between_std, 4).tolist(),
        "sampled_latent_std_per_dim": np.round(samp_std, 4).tolist(),
        "sampled_latent_std_mean": round(float(samp_std.mean()), 4),
    }


inspect("PyTorch", load(PT))
inspect("TensorFlow", load(TF))

with open("/tmp/latent_analysis/step2_bgm_inspect.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved -> /tmp/latent_analysis/step2_bgm_inspect.json")
