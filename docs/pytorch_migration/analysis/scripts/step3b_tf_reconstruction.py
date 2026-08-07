import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import pickle
import json
import numpy as np
import pandas as pd

from syngen.ml.vae.models.model import CVAE

CKPT = "model_artifacts/resources/housing-tf/vae/checkpoints"
ORIG = "examples/example-data/housing.csv"
COLS = ["latitude", "longitude"]

with open(f"{CKPT}/model_dataset.pkl", "rb") as f:
    dataset = pickle.load(f)
with open(f"{CKPT}/train_config.pkl", "rb") as f:
    tcfg = pickle.load(f)
batch_size = tcfg.batch_size
print(f"batch_size={batch_size}")

vae = CVAE(dataset, batch_size=batch_size, latent_dim=10,
           latent_components=min(30, 10 * 2), intermediate_dim=128)
vae.build_model()
vae.load_state(CKPT)

# Same preprocessing as the pipeline (add null indicator, mean-fill).
raw = pd.read_csv(ORIG)
proc = raw.copy()
isnull = pd.isnull(proc["total_bedrooms"])
proc["total_bedrooms_null"] = isnull.astype(int)
proc["total_bedrooms"] = proc["total_bedrooms"].fillna(proc["total_bedrooms"].mean())
transformed = dataset.transform(proc)

# encoder_model -> mu (latent points the BGM is fit on)
mu = vae.encoder_model.predict(transformed, batch_size=batch_size, verbose=0)
mu = np.asarray(mu)
print("\n--- LATENT ---")
print("encoded mu std per dim :", np.round(mu.std(axis=0), 3).tolist())
print(f"mean mu std (all dims) : {mu.std(axis=0).mean():.4f}")
bgm_samp = vae.latent_model.sample(20000)[0]
print(f"BGM sampled latent std : {bgm_samp.std(axis=0).mean():.4f}")


def lat_lon_std(df):
    return {c: round(float(df[c].dropna().astype(float).std()), 4) for c in COLS}


orig_std = lat_lon_std(raw)

# (A) decode(mu): generator_model on the encoded means
recon_mu = vae.generator_model.predict(mu, batch_size=batch_size, verbose=0)
recon_mu_df = dataset.inverse_transform(recon_mu)
A = lat_lon_std(recon_mu_df)

# (B) full model reconstruction (input -> encode -> sample_z(reparam) -> decode)
recon_full = vae.model.predict(transformed, batch_size=batch_size, verbose=0)
recon_full_df = dataset.inverse_transform(recon_full)
B = lat_lon_std(recon_full_df)

# (C) generation: BGM sample -> generator_model
np.random.seed(0)
gen = vae.latent_model.sample(len(raw))[0]
np.random.shuffle(gen)
gen_out = vae.generator_model.predict(gen, batch_size=batch_size, verbose=0)
gen_df = dataset.inverse_transform(gen_out)
C = lat_lon_std(gen_df)

print("\n--- TF LAT/LON STD ---")
hdr = f"{'stage':<34}{'latitude':>14}{'longitude':>14}"
print(hdr); print("-" * len(hdr))
def row(label, d):
    lat = f"{d['latitude']:.4f} ({100*d['latitude']/orig_std['latitude']:.0f}%)"
    lon = f"{d['longitude']:.4f} ({100*d['longitude']/orig_std['longitude']:.0f}%)"
    print(f"{label:<34}{lat:>14}{lon:>14}")
print(f"{'ORIGINAL':<34}{orig_std['latitude']:>14.4f}{orig_std['longitude']:>14.4f}")
row("(A) decode(mu)  reconstruction", A)
row("(B) full model  reconstruction", B)
row("(C) BGM-sample -> decode  (gen)", C)

out = {
    "latent": {
        "mu_std_mean": round(float(mu.std(axis=0).mean()), 4),
        "bgm_sampled_std_mean": round(float(bgm_samp.std(axis=0).mean()), 4),
    },
    "latlon_std": {"original": orig_std, "A_decode_mu": A,
                   "B_full_reconstruction": B, "C_generation_bgm": C},
}
with open("/tmp/latent_analysis/step3b_tf_reconstruction.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved -> /tmp/latent_analysis/step3b_tf_reconstruction.json")
