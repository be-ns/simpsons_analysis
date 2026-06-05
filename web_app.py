"""Minimal Flask demo for the rebuilt recommender + rating model.

Run with:  python web_app.py   then open http://localhost:8080

This replaces the original app, which relied on a pre-computed hash table and
``sklearn.externals.joblib`` (removed from scikit-learn years ago). The
recommender now scores episodes live and transparently, so there is no opaque
cache to keep in sync.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template_string, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from simpsons import data as D  # noqa: E402
from simpsons.recommender import Preferences, build_episode_profiles, recommend  # noqa: E402

app = Flask(__name__)

PROFILES = build_episode_profiles()
_artifact = joblib.load(ROOT / "models" / "rating_model.joblib") if (
    ROOT / "models" / "rating_model.joblib"
).exists() else None

PAGE = """
<!doctype html><html><head><meta charset="utf-8"><title>the Simpsonian</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{--gold:#FED41D;--ink:#2b2b2b;--blue:#3267a8;}
  *{box-sizing:border-box} body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
    margin:0;color:var(--ink);background:#fafafa}
  header{background:var(--ink);color:var(--gold);padding:1.1rem 1.6rem;font-size:1.4rem;font-weight:700}
  .wrap{max-width:860px;margin:1.6rem auto;padding:0 1.2rem}
  form{background:#fff;border:1px solid #e6e6e6;border-radius:10px;padding:1.2rem 1.4rem;
    display:grid;grid-template-columns:1fr 1fr;gap:.9rem 1.4rem}
  label{font-size:.85rem;font-weight:600;display:block;margin-bottom:.25rem}
  input,select{width:100%;padding:.5rem;border:1px solid #ccc;border-radius:6px;font-size:.95rem}
  button{grid-column:1/3;background:var(--gold);border:none;padding:.7rem;border-radius:8px;
    font-weight:700;font-size:1rem;cursor:pointer}
  table{width:100%;border-collapse:collapse;background:#fff;margin-top:1.4rem;border-radius:10px;overflow:hidden;
    box-shadow:0 1px 4px rgba(0,0,0,.06)}
  th,td{padding:.6rem .8rem;text-align:left;border-bottom:1px solid #eee;font-size:.92rem}
  th{background:#f3f3f3} .muted{color:#888;font-size:.85rem;margin:.6rem 0 0}
  .badge{background:var(--blue);color:#fff;border-radius:5px;padding:.1rem .45rem;font-size:.8rem}
</style></head><body>
<header>the Simpsonian · episode recommender</header>
<div class="wrap">
  <p class="muted">Cold-start, content-based ranking over {{ n }} episodes. Every score is a
  transparent weighted blend of how well an episode matches your preferences (plus a small
  nudge toward higher-rated episodes). No hidden cut-offs.</p>
  <form method="post" action="/recommend">
    <div><label>Favorite character</label><input name="character" value="{{ f.character or '' }}" placeholder="e.g. Lisa"></div>
    <div><label>Favorite location</label><input name="location" value="{{ f.location or '' }}" placeholder="e.g. Moe's"></div>
    <div><label>Songs?</label><select name="song">
      <option value="">no preference</option><option value="1" {{ 'selected' if f.wants_song }}>more songs</option>
      <option value="0" {{ 'selected' if f.wants_song is sameas false }}>fewer songs</option></select></div>
    <div><label>Politics?</label><select name="politics">
      <option value="">no preference</option><option value="1" {{ 'selected' if f.wants_politics }}>more political</option>
      <option value="0" {{ 'selected' if f.wants_politics is sameas false }}>less political</option></select></div>
    <button type="submit">Recommend episodes</button>
  </form>
  {% if recs is not none %}
  <table><tr><th>#</th><th>Episode</th><th>Season</th><th>IMDb</th><th>Match</th></tr>
    {% for r in recs %}
    <tr><td>{{ loop.index }}</td><td>{{ r.title }}</td><td>S{{ r.season }}E{{ r.number_in_season }}</td>
        <td>{{ '%.1f'|format(r.imdb_rating) }}</td><td><span class="badge">{{ '%.2f'|format(r.match_score) }}</span></td></tr>
    {% endfor %}
  </table>{% endif %}
</div></body></html>
"""


def _bool(v: str) -> bool | None:
    return None if v == "" else v == "1"


@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE, n=len(PROFILES), f=Preferences(), recs=None)


@app.route("/recommend", methods=["POST"])
def recommend_route():
    prefs = Preferences(
        character=request.form.get("character") or None,
        location=request.form.get("location") or None,
        wants_song=_bool(request.form.get("song", "")),
        wants_politics=_bool(request.form.get("politics", "")),
    )
    recs = recommend(prefs, PROFILES, top_n=5)
    return render_template_string(
        PAGE, n=len(PROFILES), f=prefs, recs=list(recs.itertuples())
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
