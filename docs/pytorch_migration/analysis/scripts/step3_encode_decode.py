import pickle
import json
import numpy as np
import pandas as pd
import torch

from syngen.ml.vae.models.model import CVAE, reparameterize, _to_tensors

CKPT = "model_artifacts/resources/housing-pt/vae/checkpoints"
ORIG = "examples/example-data/housing.csv"
COLS = ["latitude", "longitude"]

with open(f"{CKPT}/model_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)
with open(f"{CKPT}/train_config.pkl", "rb") as f:
    tcfg = pickle.load(f)

batch_size = tcfg.batch_size
print(f"batch_size={batch_size}")

# Rebuild the CVAE exactly like VAEWrapper (latent_dim=10, intermediate_dim=128,
# latent_components=min(30, latent_dim*2)).
vae = CVAE(dataset, batch_size=batch_size, latent_dim=10,
           latent_components=min(30, 10 * 2), intermediate_dim=128)
vae.build_model()
vae.load_state(CKPT)
vae.model.eval()

# Load real data and transform exactly as fit_sampler does.
raw = pd.read_csv(ORIG)
# Replicate the pipeline preprocessing: add the null-indicator column and
# fill total_bedrooms NaNs with the mean (dataset._preprocess_nan_cols).
proc = raw.copy()
isnull = pd.isnull(proc["total_bedrooms"])
proc["total_bedrooms_null"] = isnull.astype(int)
proc["total_bedrooms"] = proc["total_bedrooms"].fillna(proc["total_bedrooms"].mean())
transformed = dataset.transform(proc)

with torch.no_grad():
    mu, log_sigma = vae.model.encode(_to_tensors(transformed))
mu_np = mu.cpu().numpy()
sigma_np = np.exp(0.5 * log_sigma.cpu().numpy())  # posterior std per point/dim

print("\n--- LATENT (per-dim) ---")
print("encoded mu std per dim   :", np.round(mu_np.std(axis=0), 3).tolist())
print("mean posterior sigma/dim :", np.round(sigma_np.mean(axis=0), 3).tolist())
print(f"mean mu std (all dims)   : {mu_np.std(axis=0).mean():.4f}")
print(f"mean posterior sigma     : {sigma_np.mean():.4f}")

# BGM sampled latent std (fit on mu)
bgm_samp = vae.latent_model.sample(20000)[0]
print(f"BGM sampled latent std   : {bgm_samp.std(axis=0).mean():.4f}")


def lat_lon_std(df):
    return {c: round(float(df[c].dropna().astype(float).std()), 4) for c in COLS}


orig_std = lat_lon_std(raw)

# (A) RECONSTRUCTION: decode(mu) for the real data -> decoder under REAL latents
with torch.no_grad():
    recon = vae.model.decode(mu)
recon_df = dataset.inverse_transform([r.cpu().numpy() for r in recon])
recon_std = lat_lon_std(recon_df)

# (B) RECONSTRUCTION with reparam sampling: decode(mu + sigma*eps)
torch.manual_seed(0)
with torch.no_grad():
    z = reparameterize(mu, log_sigma)
    recon_z = vae.model.decode(z)
recon_z_df = dataset.inverse_transform([r.cpu().numpy() for r in recon_z])
recon_z_std = lat_lon_std(recon_z_df)

# (C) GENERATION: BGM sample -> decode  (what infer actually does)
np.random.seed(0)
torch.manual_seed(0)
gen = vae.sample(len(raw))
gen_std = lat_lon_std(gen)

print("\n--- LAT/LON STD: where is the spread lost? ---")
hdr = f"{'stage':<34}{'latitude':>12}{'longitude':>12}"
print(hdr)
print("-" * len(hdr))
def row(label, d, base):
    lat = f"{d['latitude']:.4f} ({100*d['latitude']/base['latitude']:.0f}%)"
    lon = f"{d['longitude']:.4f} ({100*d['longitude']/base['longitude']:.0f}%)"
    print(f"{label:<34}{lat:>12}{lon:>12}")
print(f"{'ORIGINAL':<34}{orig_std['latitude']:>12.4f}{orig_std['longitude']:>12.4f}")
row("(A) decode(mu)  reconstruction", recon_std, orig_std)
row("(B) decode(mu+sig*eps) reconstr", recon_z_std, orig_std)
row("(C) BGM-sample -> decode  (gen)", gen_std, orig_std)

out = {
    "latent": {
        "mu_std_per_dim": np.round(mu_np.std(axis=0), 4).tolist(),
        "mu_std_mean": round(float(mu_np.std(axis=0).mean()), 4),
        "posterior_sigma_mean": round(float(sigma_np.mean()), 4),
        "bgm_sampled_std_mean": round(float(bgm_samp.std(axis=0).mean()), 4),
    },
    "latlon_std": {
        "original": orig_std,
        "A_reconstruct_mu": recon_std,
        "B_reconstruct_reparam": recon_z_std,
        "C_generation_bgm": gen_std,
    },
}
with open("/tmp/latent_analysis/step3_encode_decode.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved -> /tmp/latent_analysis/step3_encode_decode.json")
