import sys
import pickle
import numpy as np
import pandas as pd
import torch
from syngen.ml.vae.models.model import CVAE, _to_tensors

CKPT = sys.argv[1]
ORIG = "examples/example-data/housing.csv"
COLS = ["latitude", "longitude"]

dataset = pickle.load(open(f"{CKPT}/model_dataset.pkl", "rb"))
tcfg = pickle.load(open(f"{CKPT}/train_config.pkl", "rb"))
vae = CVAE(dataset, batch_size=tcfg.batch_size, latent_dim=10,
           latent_components=min(30, 20), intermediate_dim=128)
vae.build_model()
vae.load_state(CKPT)
vae.model.eval()

raw = pd.read_csv(ORIG)
proc = raw.copy()
isnull = pd.isnull(proc["total_bedrooms"])
proc["total_bedrooms_null"] = isnull.astype(int)
proc["total_bedrooms"] = proc["total_bedrooms"].fillna(proc["total_bedrooms"].mean())
transformed = dataset.transform(proc)

with torch.no_grad():
    mu, _ = vae.model.encode(_to_tensors(transformed))
    recon = vae.model.decode(mu)
recon_df = dataset.inverse_transform([r.cpu().numpy() for r in recon])

np.random.seed(0); torch.manual_seed(0)
gen = vae.sample(len(raw))


def std(df, c):
    return float(df[c].dropna().astype(float).std())


tag = CKPT.split("/")[-3] if "resources" in CKPT else CKPT
print(f"\n[{tag}]  mu std/dim mean = {mu.cpu().numpy().std(axis=0).mean():.3f}")
print(f"{'col':<11}{'orig':>9}{'recon(A)':>16}{'gen(C)':>16}")
for c in COLS:
    o = std(raw, c)
    r = std(recon_df, c)
    g = std(gen, c)
    print(f"{c:<11}{o:>9.3f}{r:>9.3f} ({100*r/o:>3.0f}%){g:>9.3f} ({100*g/o:>3.0f}%)")
