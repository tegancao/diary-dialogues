"""Build the blind human-rating page for the flow rubric, to be hosted as a private claude.ai page.

    uv run python experiments/05-flow-rubric/make_rating_page.py      # writes human_rating.html

Every rater gets 15 replies: 5 anchor items that everyone rates (human-to-human agreement) and 10 drawn at random
for them when they first start (balanced over turns, at most one per app × profile). Each item shows the person's
life situation, their latest diary entry and the reply to rate with a reading-time estimate; earlier exchanges are
optional. No app, schema or judge answer is shown. The pool is every app reply of the main run (not the baseline);
the judge has rated all of them in the same view (grade.py --source human).

Where answers go: on the hosted page, each rater's session and answers are saved automatically to the page's
database at ratings/<their user id> (private to them; the owner can read all). Without the database (e.g. opened
as a local file), answers stay in the browser and "Save a copy" downloads them as JSON.
"""
import json
import random
from pathlib import Path

from diary_dialogues.benchmark.flow_rubric import about_person, questions
from diary_dialogues.design import BACKGROUND_SUMMARIES
from diary_dialogues.harness.conversation import DATES
from diary_dialogues.paths import ROOT

HERE = Path(__file__).resolve().parent
APPS = ["gemma-3-12b", "llama-3.1-8b", "phi-4"]
PROFILES = ["control", "Social_Isolation", "Entitlement", "Unrelenting_Standards"]
ANCHORS, RANDOM_ITEMS = 5, 10
SHARE = "I've been keeping a diary. Here's what I wrote today, I just wanted to share it with someone:\n\n"

# The page's database rules: each rater writes only their own ratings/<id> document and can't read others';
# the owner reads all.
DB_RULES = [{"path": "ratings", "read": "owner", "write": "owner"},
            {"path": "ratings/{self}", "read": "interact", "write": "interact"}]


def anchor_keys(convs: list[dict], seed: int = 0) -> list[str]:
    """Five anchor replies: different app × profile cells, turns spread out, seeded."""
    rng = random.Random(seed)
    cells = [(a, p) for a in APPS for p in PROFILES]
    rng.shuffle(cells)
    out = []
    for k, (a, p) in enumerate(cells[:ANCHORS]):
        c = rng.choice([c for c in convs if c["responder"] == a and c["profile"] == p])
        out.append(f"{c['id']}#{k % 4}")
    return out


def main():
    convs = json.loads((ROOT / "experiments/02-dialogues/main.json").read_text())["conversations"]
    groups = {(a, p): g for g, (a, p) in enumerate((a, p) for a in APPS for p in PROFILES)}
    pool, conversations = {}, {}
    for n, c in enumerate(cv for cv in convs if cv["responder"] in APPS):
        bg = c.get("background") or {}
        conversations[n] = {"entries": [SHARE + e for e in c["entries"]], "replies": c["replies"],
                            "dates": DATES[:len(c["entries"])], "about": BACKGROUND_SUMMARIES.get(bg.get("name"), "")}
        for t in range(len(c["replies"])):            # the page sees only numbers, never app or profile names
            pool[f"{c['id']}#{t}"] = {"c": n, "t": t, "g": groups[(c["responder"], c["profile"])]}
    keys = list(pool)
    anchors = anchor_keys([c for c in convs if c["responder"] in APPS])
    qs = [{"dim": d, "q": ("About the person, in this entry (ignore the reply): " if about_person(q) else "") + q}
          for d, q in questions()]
    data = {"anchors": [keys.index(a) for a in anchors], "random_items": RANDOM_ITEMS, "questions": qs,
            "pool": [pool[k] for k in keys], "conversations": conversations}
    (HERE / "human_pool_keys.json").write_text(json.dumps(keys, indent=0))     # page item number -> reply key
    (HERE / "human_sample.json").write_text(json.dumps({
        "view": "latest exchange (earlier ones optional)", "anchors": anchors, "random_items_per_rater": RANDOM_ITEMS,
        "pool": "all app replies of the main run (baseline excluded)", "pool_size": len(pool),
        "db_rules": DB_RULES}, indent=1))
    for old in HERE.glob("human_rating*.html"):
        old.unlink()
    (HERE / "human_rating.html").write_text(PAGE.replace("/*DATA*/", json.dumps(data)))
    print(f"human_rating.html: {len(anchors)} anchors + {RANDOM_ITEMS} random of {len(pool)} replies per rater")


PAGE = r"""<title>Diary Reply Rating</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400&display=swap">
<style>
/* A quiet reading tool: one column, the conversation as a card, the questions as a checklist below it. */
:root{--bg:#f6f6f3;--card:#ffffff;--ink:#1f2a44;--mute:#5f6673;--line:#e3e4e8;--accent:#6d4ee0;--accent-ink:#ffffff;
  --graded:#f3f0ff;--yes:#b4471a;--no:#0f766e;--about:#eef2f6;
  --sans:"Source Sans 3","Segoe UI",system-ui,-apple-system,sans-serif;--serif:"Source Serif 4",Georgia,serif}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15171c;--card:#1d2027;--ink:#e6e7ea;--mute:#a2a8b3;
  --line:#2f333d;--accent:#9c87ff;--accent-ink:#15171c;--graded:#2a2540;--yes:#f08a5d;--no:#4fc4b4;--about:#252a33;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15171c;--card:#1d2027;--ink:#e6e7ea;--mute:#a2a8b3;--line:#2f333d;--accent:#9c87ff;
  --accent-ink:#15171c;--graded:#2a2540;--yes:#f08a5d;--no:#4fc4b4;--about:#252a33;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:16px/1.6 var(--sans)}
main{max-width:820px;margin:0 auto;padding-block:12px 48px;padding-inline:16px}
header{position:sticky;top:env(safe-area-inset-top,0px);z-index:2;background:var(--bg);padding-block:10px;
  border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center;flex-wrap:wrap}
header h1{font:600 17px var(--sans);margin:0;flex:1 1 auto}
.status{color:var(--mute);font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-block:14px}
.meta{font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.05em}
.about{background:var(--about);border-radius:8px;padding:8px 12px;margin-block:10px}
.turn{border-left:3px solid var(--line);padding:4px 12px;margin-block:10px;white-space:pre-wrap;font-family:var(--serif);font-size:16.5px}
.turn.assistant{border-color:var(--mute)}.turn.graded{border-color:var(--accent);background:var(--graded);border-radius:0 6px 6px 0}
.turn .meta{font-family:var(--sans)}
details summary{cursor:pointer;color:var(--mute);margin-block:6px}
.q{display:flex;gap:10px;align-items:flex-start;padding-block:9px;border-top:1px solid var(--line)}
.q p{margin:0;flex:1;min-width:0}
button{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:7px;padding:7px 12px;cursor:pointer}
button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.q button{min-width:56px}
.q button.on[data-v=yes]{background:var(--yes);border-color:var(--yes);color:#fff}
.q button.on[data-v=no]{background:var(--no);border-color:var(--no);color:#fff}
.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
input{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:7px;padding:8px 10px}
.intro p{margin-block:8px}
.intro h2{font:600 16px var(--sans);margin-block:16px 4px}
.fine{color:var(--mute);font-size:14px}
.form{display:grid;grid-template-columns:minmax(0,1fr);gap:6px;max-width:520px}
.form label{margin-top:6px}
select,textarea{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:7px;padding:8px 10px;width:100%;box-sizing:border-box}
.q textarea{margin-top:6px}
</style>
<main>
<header><h1>Diary reply rating</h1><span class="status" id="status"></span>
<button id="prev">← Previous</button><button id="next">Next →</button><button id="copy" hidden>Save a copy</button></header>
<div id="item"></div>
</main>
<script>
const D = /*DATA*/;
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const minutes = ts => Math.max(1, Math.round(ts.reduce((s, t) => s + t.text.split(/\s+/).length, 0) / 230));
function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function shuffle(a, r) { for (let k = a.length - 1; k > 0; k--) { const j = Math.floor(r() * (k + 1)); [a[k], a[j]] = [a[j], a[k]]; } return a; }
function draw(seed) {   // random items: balanced over turns, at most one per app × profile group, plus the anchors
  const r = rng(seed), cap = Math.ceil(D.random_items / 4), byTurn = [0, 0, 0, 0], groups = new Set(), out = [];
  for (const k of shuffle(D.pool.map((_, k) => k).filter(k => !D.anchors.includes(k)), r)) {
    const it = D.pool[k];
    if (out.length < D.random_items && byTurn[it.t] < cap && !groups.has(it.g)) { out.push(k); byTurn[it.t]++; groups.add(it.g); }
  }
  return shuffle([...D.anchors, ...out], r);
}

// Storage: the page's database when available (each rater's own ratings/<id> document), else this browser.
let S = null, i = 0, ref = null, saving = Promise.resolve(), mode = "browser";
const LKEY = "diary_rating_session";
const local = { load() { try { return JSON.parse(localStorage.getItem(LKEY) || "null"); } catch { return null; } },
                save(s) { try { localStorage.setItem(LKEY, JSON.stringify(s)); } catch {} } };
async function persist() {
  local.save(S);
  if (!ref) return;
  const body = JSON.parse(JSON.stringify(S));
  saving = saving.then(() => ref.set(body)).then(() => setStatus("saved"))
    .catch(e => { setStatus("not saved: " + (e && e.code || "error")); });
}
function setStatus(word) {
  const done = S ? S.items.filter(x => Object.keys(S.answers[x] || {}).length === D.questions.length).length : 0;
  document.getElementById("status").textContent = S ? `${S.rater}: ${done} of ${S.items.length} rated · ${mode === "db" ? word : "kept in this browser"}` : "";
}

function turnsOf(k) {
  const it = D.pool[k], c = D.conversations[it.c], turns = [];
  for (let n = 0; n <= it.t; n++) {
    turns.push({role: "person", text: c.entries[n], date: c.dates[n]});
    turns.push({role: "assistant", text: c.replies[n], graded: n === it.t});
  }
  return {turns, about: c.about};
}
function intro() {
  const el = document.getElementById("item");
  el.innerHTML = `<div class="card intro">
    <p>Thank you for helping. You will read <strong>${D.anchors.length + D.random_items}</strong> short exchanges: a person shares an entry from their diary with an AI assistant, and you answer 15 yes/no questions about the assistant's reply. Most items take 3–5 minutes.</p>
    <p>Your items are chosen at random for you; ${D.anchors.length} of them are the same for every helper. ${mode === "db" ? "Every answer is saved as you go; you can stop and come back." : "Your answers stay in this browser; use “Save a copy” when you are done."}</p>
    <p class="fine">This is a research study on how AI assistants respond to people. It is for adults (18+). Your answers are used only to check an automated rating method; only the study's organizer can read them, and other helpers cannot.</p>
    <h2>About you</h2>
    <p class="fine">Only your initials are required.</p>
    <div class="form">
      <label for="who">Your initials *</label><input id="who" autocomplete="off" maxlength="12">
      <label for="age">Age</label><select id="age"><option value="">Prefer not to say</option>
        ${["18–24", "25–34", "35–44", "45–54", "55–64", "65+"].map(a => `<option>${a}</option>`).join("")}</select>
      <label for="job">Occupation</label><input id="job" maxlength="80">
      <label for="clin">Are you a clinician or trained in mental health?</label><select id="clin">
        ${["No", "Yes: psychologist", "Yes: psychiatrist", "Yes: therapist or counselor", "Yes: social worker",
           "Yes: nurse", "Yes: other clinician", "In clinical training", "Prefer not to say"].map(c => `<option>${c}</option>`).join("")}</select>
      <label for="years" id="yearsLabel" hidden>Years of clinical experience</label><input id="years" type="number" min="0" max="70" hidden>
      <label for="more">Anything else you'd like us to know? (optional: e.g. relevant experience)</label><textarea id="more" rows="3" maxlength="1000"></textarea>
    </div>
    <p><button class="primary" id="go">Start</button></p></div>`;
  const clin = document.getElementById("clin");
  clin.onchange = () => { const show = clin.value.startsWith("Yes") || clin.value.startsWith("In clinical");
    document.getElementById("years").hidden = !show; document.getElementById("yearsLabel").hidden = !show; };
  document.getElementById("go").onclick = () => {
    const v = id => document.getElementById(id).value.trim();
    const who = v("who"); if (!who) { document.getElementById("who").focus(); return; }
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    S = {rater: who, profile: {age: v("age"), occupation: v("job"), clinician: v("clin"),
                               clinical_years: v("years") === "" ? null : Number(v("years")), other: v("more")},
         seed, started: new Date().toISOString(), items: draw(seed), answers: {}, notes: {}, version: 2};
    persist(); i = 0; render();
  };
}
let noteTimer = null;
function render() {
  if (!S) return intro();
  const k = S.items[i], {turns, about} = turnsOf(k), ans = S.answers[k] || {};
  const earlier = turns.slice(0, -2), latest = turns.slice(-2);
  const turn = (t, j) => { const n = Math.floor(j / 2) + 1;
    const who = t.role === "person"
      ? `Entry ${n} · ${t.date}${n > 1 ? " (3 days later)" : ""} · the person shares their diary`
      : (t.graded ? `Reply to rate · the assistant's reply to entry ${n}` : `Assistant · reply to entry ${n}`);
    return `<div class="turn ${t.role}${t.graded ? " graded" : ""}"><div class="meta">${who}</div>${esc(t.text)}</div>`; };
  document.getElementById("item").innerHTML = `<div class="card"><div class="meta">Item ${i + 1} of ${S.items.length} · about ${minutes(latest)} min read</div>
    ${about ? `<p class="about"><strong>The person:</strong> ${esc(about)}</p>` : ""}
    ${earlier.length ? `<details><summary>Optional: the ${earlier.length / 2} earlier exchange${earlier.length > 2 ? "s" : ""} in this conversation (about ${minutes(earlier)} min read)</summary>${earlier.map(turn).join("")}</details>` : ""}
    ${latest.map((t, j) => turn(t, earlier.length + j)).join("")}</div>
    <div class="card">${D.questions.map((q, j) => `<div class="q"><p>${esc(q.q)}</p>
      <button data-k="${j}" data-v="yes" class="${ans[j] === "yes" ? "on" : ""}" aria-pressed="${ans[j] === "yes"}">Yes</button>
      <button data-k="${j}" data-v="no" class="${ans[j] === "no" ? "on" : ""}" aria-pressed="${ans[j] === "no"}">No</button></div>`).join("")}
      <div class="q"><p><label for="note">Optional note on this reply (anything the questions missed, or an answer you were unsure about)</label>
        <textarea id="note" rows="2" maxlength="1500"></textarea></p></div></div>`;
  const note = document.getElementById("note");
  note.value = (S.notes || {})[k] || "";
  note.oninput = () => { clearTimeout(noteTimer); setStatus("saving…");
    noteTimer = setTimeout(() => { S.notes = {...(S.notes || {}), [k]: note.value}; persist(); }, 800); };
  setStatus("saved");
  document.querySelectorAll(".q button").forEach(b => b.onclick = () => {
    clearTimeout(noteTimer); S.notes = {...(S.notes || {}), [k]: document.getElementById("note").value};   // keep an unsaved note
    S.answers[k] = {...(S.answers[k] || {}), [b.dataset.k]: b.dataset.v}; render(); setStatus("saving…"); persist(); });
}
document.getElementById("prev").onclick = () => { if (!S) return; i = Math.max(0, i - 1); render(); scrollTo(0, 0); };
document.getElementById("next").onclick = () => { if (!S) return; i = Math.min(S.items.length - 1, i + 1); render(); scrollTo(0, 0); };
document.getElementById("copy").onclick = async () => {
  if (!S) return;
  const out = JSON.stringify({...S, anchors: D.anchors, questions: D.questions.map(q => q.q)}, null, 1);
  const filename = `ratings_${S.rater.replace(/\W+/g, "")}_${S.seed}.json`;
  const dl = window.claude ? await window.claude.use("downloads").catch(() => null) : null;
  if (dl) { try { await dl.save({filename, data: out}); } catch {} return; }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([out], {type: "application/json"})); a.download = filename; a.click();
};

(async () => {
  S = local.load(); render();                      // usable at once; the database lights up when it answers
  if (!window.claude) { document.getElementById("copy").hidden = false; return; }
  const [db, user] = await Promise.all([window.claude.use("db"), window.claude.use("user")]);
  const id = user ? await user.id() : null;
  if (!db || !id) { document.getElementById("copy").hidden = false; return; }
  ref = db.doc("ratings/" + id); mode = "db";
  try {
    const snap = await ref.get();
    if (snap.exists) { S = JSON.parse(JSON.stringify(snap.data())); local.save(S); }
    else if (S) { await ref.set(JSON.parse(JSON.stringify(S))); }
  } catch { mode = "browser"; ref = null; document.getElementById("copy").hidden = false; }
  render();
})();
</script>
"""

if __name__ == "__main__":
    main()
