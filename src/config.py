"""Central configuration. Every tunable choice lives here so it is documented and reproducible."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data Pack"
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
EVIDENCE_DIR = ROOT / "evidence"
CACHE_DIR = OUT_DIR / "cache"

SEED = 2026
N_BOOT = 1000

# ---- temporal design (forward-only) ----
TRAIN_SEASONS = [2022, 2023, 2024]
VALID_SEASON = 2025
TEST_SEASON = 2026
ALL_SEASONS = [2022, 2023, 2024, 2025, 2026]

# ---- cricket attribution ----
ILLEGAL_EXTRAS = {"wide", "Overstep", "other"}          # 'Overstep' and 'other' are no-balls
NOBALL_EXTRAS = {"Overstep", "other"}
BOWLER_CHARGED_EXTRAS = {"wide", "Overstep", "other"}  # byes / leg-byes / penalty are NOT charged to the bowler
BOWLER_WICKETS = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}  # run-outs, obstruction, retirements excluded
NON_DISMISSALS = {"Retire Out"}                        # retirements are not dismissals

# ---- phases (legal balls already bowled in the innings) ----
POWERPLAY_END = 36   # balls 0..35  -> overs 1-6
DEATH_START = 96     # balls 96..119 -> overs 17-20

# ---- pressure ----
HIGH_P = 0.75        # percentile of training-period leverage defining "high pressure" for counting opportunities
LIFT_HI, LIFT_LO = 0.90, 0.30   # pressure-lift contrast
P_CENTER = 0.5       # pressure is centred so alpha = performance at median pressure
P_HIGH = 0.875       # mean pressure percentile of the top-quartile ("high-pressure") deliveries: E[P | P>=0.75]

# ---- eligibility (fixed before looking at rankings) ----
ELIG = {
    "bat": dict(min_balls=250, min_hp_balls=60, min_matches=12),
    "bowl": dict(min_balls=300, min_hp_balls=60, min_matches=12),
}
# wicket value used by the fixed-WV variant (training-period mean of the state model, see wicket_value.py)
HGB_PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=80,
                  l2_regularization=1.0, random_state=SEED)
