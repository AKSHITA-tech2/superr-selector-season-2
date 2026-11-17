"""Value Above Average (runs-equivalent). HIGHER = BETTER for both batters and bowlers.

Batter (deliveries faced):  VAA_bat  = Runs - xR_bat - WV * (Out - xW_bat)
Bowler (all deliveries):    VAA_bowl = (xR_bowl - RunsConceded) + WV * (Wicket - xW_bowl)
"""
import numpy as np


def vaa_bat(runs, xr, out, xw, wv):
    return np.asarray(runs) - np.asarray(xr) - np.asarray(wv) * (np.asarray(out) - np.asarray(xw))


def vaa_bowl(conceded, xr, wicket, xw, wv):
    return (np.asarray(xr) - np.asarray(conceded)) + np.asarray(wv) * (np.asarray(wicket) - np.asarray(xw))
