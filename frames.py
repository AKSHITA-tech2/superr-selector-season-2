"""Build (and cache) the three delivery-level frames: validation fold, test fold, retrospective."""
import json
import pickle
from src import config as C
from src.data.state_reconstructor import build_deliveries
from src.validation.temporal_split import forward_fold, retrospective
from src.ranking.candidates import add_raw_outcomes


def build_all(force=False):
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = C.CACHE_DIR / "frames.pkl"
    if f.exists() and not force:
        fr = pickle.load(open(f, "rb"))
        for k in ("valid", "test", "retro"):
            add_raw_outcomes(fr[k][0])
        return fr
    d = build_deliveries()
    xv, mv, _ = forward_fold(d, C.TRAIN_SEASONS, C.VALID_SEASON)          # selection fold
    xt, mt, _ = forward_fold(d, C.TRAIN_SEASONS + [C.VALID_SEASON], C.TEST_SEASON)  # final test fold
    xr, mr = retrospective(d)                                               # final descriptive ranking
    frames = dict(valid=(add_raw_outcomes(xv), mv), test=(add_raw_outcomes(xt), mt),
                  retro=(add_raw_outcomes(xr), mr), deliveries=d)
    pickle.dump(frames, open(f, "wb"))
    return frames
