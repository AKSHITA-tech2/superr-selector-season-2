"""Qualitative, publicly sourced context for the final Top-10 players.

QUALITATIVE ONLY: these links are NOT inputs to any model or ranking. Each entry ties a data-derived high-pressure
match (evidence/player_match_evidence.csv) to a public report or video, recorded on 2026-11-17 from web searches.
Only URLs returned by those searches are recorded; observations describe what the source reports, not opinions.
"""
import pandas as pd
from src.config import EVIDENCE_DIR

ROWS = [
    ("Jos Buttler", "bat", "KKR v RR, IPL 2024 (16 Apr 2024)",
     "Chasing 224 (data: 101/4 after 9 overs, needing 123 off 66, win prob. 18%), Buttler made 107* off 60 and RR won by 2 wickets in the final over - chase acceleration while wickets fell at the other end.",
     "IPL 2024: Jos Buttler's Brilliant Century Ensures Historic 2-Wicket Win for RR Against KKR", "https://www.youtube.com/watch?v=oW-zgZWL7yE", "Video (highlights)"),
    ("Suryakumar Yadav", "bat", "England v India, 3rd T20I, Trent Bridge (10 Jul 2022)",
     "117 off 55 chasing 216 from an early hole (data: 13/2 after 2.4 overs, RRR 11.7): sustained boundary hitting at a required rate above 11; India still lost (our data: high VAA, team result 'Lost').",
     "Suryakumar Yadav's Incredible 117 off 55 Balls at Trent Bridge! (official ECB channel)", "https://www.youtube.com/watch?v=8S5lApwP3yo", "Video (official)"),
    ("Rinku Singh", "bat", "KKR v LSG, IPL 2023 (20 May 2023)",
     "67* off 33 from 108/4 needing 69 off 36; KKR lost by 1 run - a near-miss finish that our VAA credits even though the team lost.",
     "KKR vs LSG Highlights, IPL 2023: Rinku Singh's Fifty in Vain as Lucknow Clinches Narrow 1-Run Win", "https://www.youtube.com/watch?v=OfX5BTU0j2A", "Video (highlights)"),
    ("Tim David", "bat", "RCB v PBKS, IPL 2025 (18 Apr 2025, rain-shortened)",
     "Came in during a collapse (data: 33/5 after 6.1 overs) and made 50* in a 14-over game, lifting RCB to 95/9 - recovery batting when resources were depleted.",
     "IPL: RCB Lose Third Straight Home Game As Punjab Kings Dominate In Rain Clash", "https://www.youtube.com/watch?v=vzFlE5xRZrI", "Video (news)"),
    ("Shashank Singh", "bat", "GT v PBKS, IPL 2024 (4 Apr 2024)",
     "61* off 29 (6x4, 4x6) chasing 200 after the top order fell (data: 70/4 after 8.4 overs, RRR 11.5) - PBKS won by 3 wickets with a ball to spare.",
     "IPL 2024 Match Recap: Shashank Guides PBKS to Thrilling Win over GT", "https://www.youtube.com/watch?v=kfyG0SrIsdU", "Video (recap)"),
    ("Tristan Stubbs", "bat", "MI v DC, IPL 2024 (7 Apr 2024)",
     "71* off 25 chasing 235 from a near-hopeless position (data: win prob. 2%); DC lost by 29 - high individual value in a low-probability chase.",
     "MI vs DC Highlights: Stubbs' 71 In Vain, Mumbai Indians Beat Delhi Capitals By 29 Runs", "https://www.youtube.com/watch?v=LP55jzoUaAs", "Video (highlights)"),
    ("Dhruv Jurel", "bat", "SRH v RR, Qualifier 2, IPL 2024 (24 May 2024)",
     "Knockout chase of 176: 56* from 82/5 needing 94 off 45 (data); RR fell short - match-level pressure response, not a match-winning outcome.",
     "SRH vs RR - IPL 2024 Qualifier 2 Match Highlights", "https://www.youtube.com/watch?v=7TC0FX5HO1k", "Video (highlights)"),
    ("Jasprit Bumrah", "bowl", "MI v KKR, IPL 2022 (9 May 2022)",
     "5/10 in 4 overs at the death of the first innings (data: 139/5 after 14.5) - wickets when the batting side tried to accelerate.",
     "Jasprit Bumrah 5/10 (4) vs Kolkata Knight Riders IPL 2022 (Ball-by-Ball)", "https://www.youtube.com/watch?v=lFW8_4PlD1I", "Video (ball-by-ball)"),
    ("Mohammad Nawaz", "bowl", "Pakistan v Australia, 3rd T20I, Lahore (1 Feb 2026)",
     "Career-best 5/18 as Australia, chasing 208, were bowled out for 96. Note: the composite index flags this as pressure (high RRR) although leverage says the match was largely settled - an honest limitation of the selected pressure definition.",
     "PCB: Nawaz's 5-18, Babar, Saim fifties help Pakistan complete T20I series sweep over Australia", "https://www.pcb.com.pk/news-detail/nawaz-s-5-18-babar-saim-fifties-help-pakistan-complete-t20i-series-sweep-over-australia.html", "Report (official board)"),
    ("Abrar Ahmed", "bowl", "Pakistan v UAE, T20I tri-series, Sharjah (4 Sep 2025)",
     "4/9 in 4 overs with 14 dot balls defending 172; took two wickets in the 13th over with UAE 84/4 - wickets in the middle overs of a live chase.",
     "ESPNcricinfo: Pakistan seal final berth as Fakhar and Abrar headline commanding win", "https://www.espncricinfo.com/ci/content/story/1501585.html", "Report"),
    ("Mitchell Santner", "bowl", "MI v DC, IPL 2025 (21 May 2025)",
     "3/11 in 4 overs in a must-win playoff-qualification game; DC 121 all out chasing 181 - control on a gripping pitch while the chase was still live.",
     "ESPNcricinfo: Suryakumar, Santner and Bumrah lead MI into playoffs", "https://www.espncricinfo.com/ci/content/story/1486873.html", "Report"),
]


def write():
    df = pd.DataFrame(ROWS, columns=["player", "role", "match", "observation", "source_title", "url", "source_type"])
    df["evidence_type"] = "qualitative - not a ranking input"
    df["retrieved"] = "2026-11-17"
    df.to_csv(EVIDENCE_DIR / "player_video_evidence.csv", index=False)
    return df


if __name__ == "__main__":
    write()
