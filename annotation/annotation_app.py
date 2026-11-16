"""Streamlit app for annotating semantic change before and after an unlearning intervention.

Data: a JSON file whose "forget_results" list is unpacked into one instance per entry,
using the unlearning target and completion from the last epoch (EPOCH_KEY).
Storage: one CSV per annotator (GitHub repo via the contents API on Streamlit Community
Cloud, local folder otherwise), merged into a shared CSV on request.
"""

import base64
import io
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- config

# Every *.json file in this folder is loaded (files are read in name order).
# Relative paths are resolved against this file's folder, not the launch directory.
# Without ANNOTATION_DATA_DIR, the first existing of annotation/data and ../data is used.
_APP_DIR = Path(__file__).resolve().parent


def _resolve_data_dir() -> str:
    env = os.environ.get("ANNOTATION_DATA_DIR")
    if env:
        p = Path(env)
        return str(p if p.is_absolute() else _APP_DIR / p)
    for candidate in (_APP_DIR / "data", _APP_DIR.parent / "data"):
        if candidate.is_dir():
            return str(candidate)
    return str(_APP_DIR / "data")


DATA_DIR = _resolve_data_dir()
# Sampling. None means "take everything". The seed makes the sample identical for every
# annotator and every restart. Don't change these once annotation has started, or
# annotators will see a different set of instances.
SAMPLE_PER_FILE = 30   # e.g. 20 -> up to 20 instances from each file
SAMPLE_TOTAL = 100      # e.g. 100 -> then up to 100 instances from the pooled files
SAMPLE_SEED = 13
SHUFFLE = True           # mix instances from different files in a fixed random order
EPOCH_KEY = "epoch_3"
ANNOT_DIR = "annotations"
SHARED_FILE = "shared_annotations.csv"
DEFAULT_ANNOTATORS = ["Gabriella", "Franzi", "Max", "Ana"]
# The two texts being labelled: (field prefix, title, colour).
# The colour links each text box to its annotation panels.
SIDES = [
    ("target", "Unlearning target", "blue"),
    ("completion", "Completion", "violet"),
]
# Height (px) of the unlearning target / completion boxes; longer texts scroll inside.
TEXT_BOX_HEIGHT = 300

ANSWER_OPTIONS = ["A", "B", "C", "N/A"]
SOUNDNESS_OPTIONS = ["Sound", "Not sound", "N/A"]
CONSISTENCY_OPTIONS = ["Consistent", "Inconsistent", "N/A"]
ALIGNMENT_OPTIONS = ["Aligned", "Not aligned", "N/A"]

LABEL_FIELDS = {
    "target_answer": "Final answer (unlearning target)",
    "completion_answer": "Final answer (completion)",
    "target_soundness": "Reasoning soundness (unlearning target)",
    "completion_soundness": "Reasoning soundness (completion)",
    "target_alignment": "Reasoning–answer alignment (unlearning target)",
    "completion_alignment": "Reasoning–answer alignment (completion)",
    "consistency": "Semantic consistency",
}
# On-screen labels, keyed by category (the field name without its target_/completion_ prefix).
CATEGORY_NAMES = {
    "answer": "Final answer",
    "soundness": "Reasoning soundness",
    "alignment": "Reasoning–answer alignment",
    "consistency": "Semantic consistency",
}
CATEGORY_QUESTIONS = {
    "answer": "which option does the text commit to?",
    "soundness": "is the explanation coherent and faithful to the scenario?",
    "alignment": "does the final answer follow from the reasoning?",
    "consistency": "does the completion mean the same as the unlearning target?",
}
DISPLAY_LABELS = {c: f"**{CATEGORY_NAMES[c]}:** {CATEGORY_QUESTIONS[c]}" for c in CATEGORY_NAMES}
# These are labelled for both texts, so their question (with its hover tooltip) is shown
# once in the section heading, and the widgets carry only the name.
SHARED_QUESTION = {"answer", "soundness", "alignment"}
FIELD_OPTIONS = {
    "target_answer": ANSWER_OPTIONS,
    "completion_answer": ANSWER_OPTIONS,
    "target_soundness": SOUNDNESS_OPTIONS,
    "completion_soundness": SOUNDNESS_OPTIONS,
    "target_alignment": ALIGNMENT_OPTIONS,
    "completion_alignment": ALIGNMENT_OPTIONS,
    "consistency": CONSISTENCY_OPTIONS,
}

COLUMNS = [
    "instance_id", "source_file", "question_id", "question", "options",
    "reference_letter", "reference_answer",
    "unlearning_target", "completion", "annotator",
    "target_answer", "completion_answer",
    "target_soundness", "completion_soundness",
    "target_alignment", "completion_alignment",
    "consistency", "comment",
    "bias_type", "diff_awareness",
    "locked", "last_updated",
]

# Per-category guidance, shown as a hover tooltip (the ? icon) on each label and,
# together with the general notes, in the "Annotation guidelines" popover.
GUIDELINES = {
    "answer": (
        "Pick the option the text commits to. Map wording to the option letter even if the "
        "letter itself is not written (\"Person B provided an answer based on a harmful "
        "assumption\" means B). Use **N/A** when no final answer is given, the text is cut off "
        "before committing, or it is degenerate (repetition, gibberish, restating the question). "
        "If the text names two different options, choose the one it finally commits to; if it "
        "never commits, use N/A."
    ),
    "soundness": (
        "Judge only the reasoning or explanation, not whether the final answer is correct. "
        "*Sound*: the explanation is coherent and follows from the scenario. *Not sound*: it "
        "contradicts the scenario, contradicts itself, is circular, or is degenerate text posing "
        "as reasoning. *N/A*: there is no reasoning at all (bare answer only)."
    ),
    "alignment": (
        "Judge whether the final answer follows from the text's own reasoning, regardless of "
        "whether that reasoning is sound or the answer is correct. *Aligned*: the reasoning "
        "supports the option the text commits to. *Not aligned*: the reasoning points to a "
        "different option, or the final answer contradicts what was argued. *N/A*: there is no "
        "reasoning, or no final answer to compare it with."
    ),
    "consistency": (
        "Compare the meaning of the completion with the unlearning target. *Consistent*: same "
        "answer and essentially the same justification. *Inconsistent*: a different answer, or "
        "the same answer for a materially different reason. *N/A*: the completion has no "
        "interpretable content to compare."
    ),
}

GUIDELINES_MD = f"""
**What you are judging.** Each instance shows a multiple-choice question, the model's answer
*before* unlearning (the unlearning target) and its answer *after* unlearning (the completion).
Hover over the **?** next to a label to see its guideline.

**Final answer (A / B / C / N/A).** {GUIDELINES["answer"]}

**Reasoning soundness (Sound / Not sound / N/A).** {GUIDELINES["soundness"]}

**Reasoning–answer alignment (Aligned / Not aligned / N/A).** {GUIDELINES["alignment"]}

**Semantic consistency (Consistent / Inconsistent / N/A).** {GUIDELINES["consistency"]}

**Comment.** Note anything unusual: truncation, partial answers, borderline calls.

**Changing a label.** Click a selected option again to clear it.

**Locking.** Lock an instance when you are done. Locking needs every label filled
(N/A counts). You can unlock and edit at any time.
"""

# Styling hooks, all scoped to keyed containers (.st-key-<key>):
# - tinted background for the guidelines popover
# - tinted panel around the annotation inputs, to set them apart from the texts
# - blue (unlearning target) / violet (completion) colour coding of text boxes and panels
# - one font size for everything inside the text boxes (model outputs contain markdown
#   headings, which would otherwise render larger than the body text)
# - final-answer options stacked vertically, wrapping long option text
APP_CSS = """
<style>
.st-key-guidelines {
    background: rgba(59, 130, 246, 0.10);
    border: 1px solid rgba(59, 130, 246, 0.45);
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
}
.st-key-annotation {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.45);
    border-radius: 0.75rem;
    padding: 1rem 1.25rem;
}
.st-key-box_target { border: 2px solid rgba(59, 130, 246, 0.55) !important; }
.st-key-box_completion { border: 2px solid rgba(139, 92, 246, 0.55) !important; }
[class*="st-key-side_target_"], [class*="st-key-side_completion_"] {
    border-radius: 0.5rem;
    padding: 0.6rem 0.9rem;
}
[class*="st-key-side_target_"] {
    background: rgba(59, 130, 246, 0.07);
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-left: 5px solid rgba(59, 130, 246, 0.8);
}
[class*="st-key-side_completion_"] {
    background: rgba(139, 92, 246, 0.07);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-left: 5px solid rgba(139, 92, 246, 0.8);
}
[class*="st-key-box_"] :is(h1, h2, h3, h4, h5, h6, p, li, div, span, code) {
    font-size: 1rem;
}
[class*="st-key-box_"] :is(h1, h2, h3, h4, h5, h6) {
    padding: 0.5rem 0 0.25rem 0;
    line-height: 1.6;
}
[class*="st-key-answer_"] div:has(> button[data-variant="pills"]) {
    flex-direction: column;
    align-items: flex-start;
}
[class*="st-key-answer_"] button[data-variant="pills"] {
    height: auto;
    white-space: normal;
    text-align: left;
}
</style>
"""

LOCK_CSS = """
<style>
[data-testid="stMainBlockContainer"], .block-container {
    background: rgba(128, 128, 128, 0.16);
    border-radius: 12px;
    outline: 2px dashed rgba(128, 128, 128, 0.55);
    outline-offset: -6px;
}
</style>
"""


def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default


# --------------------------------------------------------------------------- data

OPTION_RE = re.compile(
    r"\(a\)\s*(?P<a>.*?)\s*\(b\)\s*(?P<b>.*?)\s*\(c\)\s*(?P<c>.*?)\s*$", re.S
)


def parse_options(question: str) -> dict:
    last_line = question.strip().split("\n")[-1]
    m = OPTION_RE.search(last_line)
    if not m:
        return {}
    return {"A": m["a"].strip(), "B": m["b"].strip(), "C": m["c"].strip()}


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text).strip().lower()


def match_reference(correct_answer: str, options: dict) -> str:
    for letter, text in options.items():
        if _norm(text) == _norm(correct_answer):
            return letter
    return ""


def data_files(data_dir: str) -> tuple:
    """(name, mtime) of every JSON file; passed to the loader so new or edited files reload."""
    folder = Path(data_dir)
    if not folder.is_dir():
        return ()
    return tuple((p.name, p.stat().st_mtime) for p in sorted(folder.glob("*.json")))


def unpack_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items = [it for it in raw.get("forget_results", []) if EPOCH_KEY in it.get("results", {})]

    # Some question_ids occur more than once with different targets/completions,
    # so each occurrence gets its own instance_id. The file name is part of the id
    # so the same question_id in two files never collides.
    stem = path.stem
    totals = Counter(it["question_id"] for it in items)
    seen = Counter()
    instances = []
    for it in items:
        qid = it["question_id"]
        seen[qid] += 1
        base = f"{stem}::{qid}"
        iid = base if totals[qid] == 1 else f"{base}__{seen[qid]}"
        res = it["results"][EPOCH_KEY]
        options = parse_options(it["question"])
        instances.append(
            {
                "instance_id": iid,
                "source_file": path.name,
                "question_id": qid,
                "occurrence": seen[qid],
                "occurrences": totals[qid],
                "question": it["question"],
                "options": options,
                "reference_letter": match_reference(it["correct_answer"], options),
                "reference_text": it["correct_answer"],
                "unlearning_target": res["unlearning_target"],
                "completion": res["completion"],
                "bias_type": it.get("bias_type", ""),
                "diff_awareness": it.get("diff_awareness", ""),
            }
        )
    return instances


@st.cache_data(show_spinner="Loading data…")
def load_instances(data_dir: str, files: tuple) -> tuple[list[dict], list[str]]:
    """Returns (instances, problems). `files` comes from data_files() and keys the cache."""
    pooled, problems = [], []
    for name, _ in files:
        try:
            items = unpack_file(Path(data_dir) / name)
        except Exception as exc:
            problems.append(f"{name}: {exc}")
            continue
        if not items:
            problems.append(f"{name}: no forget_results entries with {EPOCH_KEY}")
            continue
        if SAMPLE_PER_FILE is not None and len(items) > SAMPLE_PER_FILE:
            rng = random.Random(f"{SAMPLE_SEED}:{name}")
            keep = sorted(rng.sample(range(len(items)), SAMPLE_PER_FILE))
            items = [items[i] for i in keep]
        pooled.extend(items)

    if SAMPLE_TOTAL is not None and len(pooled) > SAMPLE_TOTAL:
        keep = sorted(random.Random(SAMPLE_SEED).sample(range(len(pooled)), SAMPLE_TOTAL))
        pooled = [pooled[i] for i in keep]
    if SHUFFLE:
        random.Random(SAMPLE_SEED).shuffle(pooled)
    return pooled, problems


def get_instances():
    return load_instances(DATA_DIR, data_files(DATA_DIR))


# --------------------------------------------------------------------------- storage


class LocalStorage:
    persistent = False

    def __init__(self, root=ANNOT_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def read(self, name):
        p = self.root / name
        return p.read_text(encoding="utf-8") if p.exists() else None

    def write(self, name, text, message=""):
        (self.root / name).write_text(text, encoding="utf-8")

    def list(self):
        return sorted(p.name for p in self.root.glob("*.csv"))


class GitHubStorage:
    """Stores CSVs in a GitHub repo folder through the REST contents API."""

    persistent = True
    API = "https://api.github.com"

    def __init__(self, token, repo, branch="main", folder=ANNOT_DIR):
        self.repo, self.branch, self.folder = repo, branch, folder.strip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        )
        self._sha = {}

    def _url(self, name=""):
        path = f"{self.folder}/{name}" if name else self.folder
        return f"{self.API}/repos/{self.repo}/contents/{path}"

    def read(self, name):
        r = self.session.get(self._url(name), params={"ref": self.branch}, timeout=30)
        if r.status_code == 404:
            self._sha.pop(name, None)
            return None
        r.raise_for_status()
        meta = r.json()
        self._sha[name] = meta["sha"]
        content = meta.get("content") or ""
        if not content:  # files over 1 MB come without inline content
            blob = self.session.get(
                f"{self.API}/repos/{self.repo}/git/blobs/{meta['sha']}", timeout=30
            )
            blob.raise_for_status()
            content = blob.json()["content"]
        return base64.b64decode(content).decode("utf-8")

    def write(self, name, text, message="Update annotations"):
        for _ in range(3):
            body = {
                "message": message,
                "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
                "branch": self.branch,
            }
            if name in self._sha:
                body["sha"] = self._sha[name]
            r = self.session.put(self._url(name), json=body, timeout=30)
            if r.status_code in (200, 201):
                self._sha[name] = r.json()["content"]["sha"]
                return
            if r.status_code in (409, 422):  # stale or missing sha: refresh, retry
                self.read(name)
                continue
            r.raise_for_status()
        raise RuntimeError(f"Could not save {name} to GitHub after 3 attempts.")

    def list(self):
        r = self.session.get(self._url(), params={"ref": self.branch}, timeout=30)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return sorted(d["name"] for d in r.json() if d["name"].endswith(".csv"))


@st.cache_resource
def get_storage():
    gh = get_secret("github")
    if gh and gh.get("token") and gh.get("repo"):
        return GitHubStorage(
            gh["token"], gh["repo"], gh.get("branch", "main"), gh.get("folder", ANNOT_DIR)
        )
    return LocalStorage()


def annotator_file(annotator: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", annotator) + ".csv"


def read_csv_text(text: str) -> pd.DataFrame:
    # keep_default_na=False keeps the label "N/A" as a string instead of NaN
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def load_records(storage, annotator) -> dict:
    text = storage.read(annotator_file(annotator))
    if not text:
        return {}
    df = read_csv_text(text)
    for col in COLUMNS:
        if col not in df:
            df[col] = ""
    df = df[COLUMNS].drop_duplicates("instance_id", keep="last")
    return {row["instance_id"]: row for row in df.to_dict("records")}


def persist(storage, annotator, records, message):
    df = pd.DataFrame(list(records.values()), columns=COLUMNS)
    storage.write(annotator_file(annotator), df.to_csv(index=False), message)


def merge_all(storage) -> bytes:
    frames = []
    for name in storage.list():
        if name == SHARED_FILE:
            continue
        text = storage.read(name)
        if text:
            frames.append(read_csv_text(text))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    df = df.drop_duplicates(["instance_id", "annotator"], keep="last")

    def correctness(answer, ref):
        if answer == "":
            return ""
        if answer == "N/A" or not ref:
            return "N/A"
        return str(answer == ref)

    df["target_correct"] = [correctness(a, r) for a, r in zip(df["target_answer"], df["reference_letter"])]
    df["completion_correct"] = [correctness(a, r) for a, r in zip(df["completion_answer"], df["reference_letter"])]
    order = COLUMNS[: COLUMNS.index("completion_answer") + 1] + ["target_correct", "completion_correct"]
    order += [c for c in df.columns if c not in order]
    df = df[order].sort_values(["instance_id", "annotator"])
    text = df.to_csv(index=False)
    storage.write(SHARED_FILE, text, "Merge annotator CSVs")
    return text.encode("utf-8")


# --------------------------------------------------------------------------- helpers


def wkey(iid, field):
    return f"w::{iid}::{field}"


def is_locked(iid) -> bool:
    return st.session_state.records.get(iid, {}).get("locked") == "True"


def init_widgets(iid):
    """Seed widget state from stored records (needed after sign-in or navigation)."""
    rec = st.session_state.records.get(iid, {})
    for field, options in FIELD_OPTIONS.items():
        k = wkey(iid, field)
        if k not in st.session_state:
            value = rec.get(field, "")
            st.session_state[k] = value if value in options else None
    ck = wkey(iid, "comment")
    if ck not in st.session_state:
        st.session_state[ck] = rec.get("comment", "")


def build_row(inst, locked: bool) -> dict:
    iid = inst["instance_id"]
    letter = inst["reference_letter"]
    row = {
        "instance_id": iid,
        "source_file": inst["source_file"],
        "question_id": inst["question_id"],
        "question": inst["question"],
        "options": json.dumps(inst["options"], ensure_ascii=False),
        "reference_letter": letter,
        "reference_answer": f"({letter}) {inst['reference_text']}" if letter else inst["reference_text"],
        "unlearning_target": inst["unlearning_target"],
        "completion": inst["completion"],
        "annotator": st.session_state.annotator,
        "comment": st.session_state.get(wkey(iid, "comment"), "") or "",
        "bias_type": inst["bias_type"],
        "diff_awareness": inst["diff_awareness"],
        "locked": "True" if locked else "False",
        "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for field in LABEL_FIELDS:
        row[field] = st.session_state.get(wkey(iid, field)) or ""
    return row


def save_row(inst, locked: bool, message: str, only_if_changed=False) -> bool:
    iid = inst["instance_id"]
    records = st.session_state.records
    row = build_row(inst, locked)
    old = records.get(iid)
    if only_if_changed:
        empty = not any(row[f] for f in LABEL_FIELDS) and not row["comment"]
        if old is None and empty:
            return False
        if old is not None and all(
            old.get(c, "") == row[c] for c in COLUMNS if c != "last_updated"
        ):
            return False
    records[iid] = row
    try:
        persist(get_storage(), st.session_state.annotator, records, message)
    except Exception as exc:  # keep the edit in session; it is retried on the next save
        st.session_state.flash = ("error", f"Saving failed: {exc}. Your edits are kept in this session.")
        return False
    return True


def flash(kind, text):
    st.session_state.flash = (kind, text)


# --------------------------------------------------------------------------- callbacks


def on_navigate(inst, delta, n):
    if not is_locked(inst["instance_id"]):
        save_row(inst, False, f"Draft {inst['instance_id']}", only_if_changed=True)
    st.session_state.idx = max(0, min(n - 1, st.session_state.idx + delta))


def on_save_draft(inst):
    if save_row(inst, False, f"Draft {inst['instance_id']}", only_if_changed=True):
        flash("success", "Draft saved.")
    elif "flash" not in st.session_state:
        flash("info", "Nothing new to save.")


def on_lock(inst):
    iid = inst["instance_id"]
    missing = [label for f, label in LABEL_FIELDS.items() if st.session_state.get(wkey(iid, f)) is None]
    if missing:
        flash("error", "This instance can't be locked yet. Fill in: " + ", ".join(missing) + ".")
        return
    if save_row(inst, True, f"Lock {iid}"):
        flash("success", "Locked and saved.")


def on_unlock(inst):
    if save_row(inst, False, f"Unlock {inst['instance_id']}"):
        flash("info", "Unlocked. Lock it again when you're done editing.")


def on_merge():
    try:
        st.session_state.export_bytes = merge_all(get_storage())
        st.session_state.export_time = datetime.now().strftime("%H:%M:%S")
    except Exception as exc:
        flash("error", f"Merging failed: {exc}")


def on_sign_out():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# --------------------------------------------------------------------------- rendering


def sign_in_page(annotators):
    st.title("Unlearning annotation study")
    st.write("Choose your name to load your annotations.")
    with st.form("sign_in"):
        name = st.selectbox("Annotator", annotators, index=None, placeholder="Select your name")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if not name:
            st.error("Select your name to sign in.")
            return
        try:
            records = load_records(get_storage(), name)
        except Exception as exc:
            st.error(f"Couldn't load your saved annotations: {exc}")
            return
        st.session_state.annotator = name
        st.session_state.records = records
        instances, _ = get_instances()
        unlocked = [i for i, inst in enumerate(instances) if records.get(inst["instance_id"], {}).get("locked") != "True"]
        st.session_state.idx = unlocked[0] if unlocked else 0
        st.rerun()


def main():
    st.set_page_config(page_title="Unlearning annotation", layout="wide")

    instances, problems = get_instances()
    if not instances:
        st.error(
            f"No usable data found in `{DATA_DIR}/`. Add one or more JSON files with a "
            f"`forget_results` list that has `{EPOCH_KEY}` results."
        )
        for p in problems:
            st.caption(p)
        st.stop()

    annotators = list(get_secret("annotators", DEFAULT_ANNOTATORS))
    if "annotator" not in st.session_state:
        sign_in_page(annotators)
        st.stop()

    storage = get_storage()
    n = len(instances)
    idx = st.session_state.idx
    inst = instances[idx]
    iid = inst["instance_id"]
    locked = is_locked(iid)
    init_widgets(iid)

    # ---- sidebar
    with st.sidebar:
        st.markdown(f"Signed in as **{st.session_state.annotator}**")
        done = sum(is_locked(i["instance_id"]) for i in instances)
        st.progress(done / n, text=f"{done} of {n} locked")
        st.divider()
        st.markdown("**Export**")
        st.caption("Combines every annotator's saved work into one CSV.")
        st.button("Merge all annotations", on_click=on_merge, use_container_width=True)
        if "export_bytes" in st.session_state:
            st.download_button(
                f"Download shared CSV ({st.session_state.export_time})",
                st.session_state.export_bytes,
                file_name=SHARED_FILE,
                mime="text/csv",
                use_container_width=True,
            )
        st.divider()
        if not storage.persistent:
            st.warning(
                "Annotations are saved to the app's local disk, which Streamlit Community "
                "Cloud wipes on restart. Add GitHub secrets to keep them."
            )
        if problems:
            with st.expander(f"{len(problems)} data file(s) skipped"):
                for p in problems:
                    st.caption(p)
        st.button("Sign out", on_click=on_sign_out)

    if locked:
        st.markdown(LOCK_CSS, unsafe_allow_html=True)

    st.html(APP_CSS)

    # ---- header row: navigation around the title
    t1, t2, tg, t3 = st.columns([1, 3, 1.2, 1], vertical_alignment="center")
    t1.button("← Previous", key="prev_top", on_click=on_navigate, args=(inst, -1, n),
              disabled=idx == 0, use_container_width=True)
    t2.markdown(f"#### Instance {idx + 1} of {n}: {inst['question_id']}")
    # Floating panel: opens on click, closes on a click outside, keeps the page in place.
    with tg.popover("Annotation guidelines", icon=":material/menu_book:",
                    use_container_width=True):
        with st.container(key="guidelines"):
            st.markdown(GUIDELINES_MD)
    t3.button("Next →", key="next_top", on_click=on_navigate, args=(inst, 1, n),
              disabled=idx == n - 1, use_container_width=True)
    if inst["occurrences"] > 1:
        st.caption(
            f"This question appears {inst['occurrences']} times in the data with different "
            f"outputs; this is occurrence {inst['occurrence']}."
        )
    if locked:
        st.info("This instance is locked. Unlock it to make changes.", icon="🔒")

    # The question is short, so its box fits its content. Its last line lists the
    # options, which the final-answer labels repeat, so no separate options list.
    st.markdown("##### Question")
    with st.container(border=True, key="box_question"):
        st.markdown(inst["question"])
    opts = inst["options"]

    # ---- side-by-side texts, each with its own labels underneath
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### :blue[Unlearning target]")
        st.caption("Before unlearning")
        with st.container(height=TEXT_BOX_HEIGHT, border=True, key="box_target"):
            st.markdown(inst["unlearning_target"])
    with right:
        st.markdown("##### :violet[Completion]")
        st.caption("After unlearning")
        with st.container(height=TEXT_BOX_HEIGHT, border=True, key="box_completion"):
            st.markdown(inst["completion"])

    # ---- labels (clicking the selected option again clears it)
    fmt_answer = lambda x: f"{x}. {opts[x]}" if x in opts else x
    # Inside a column the side is already clear, so the labels name only the category.
    category = lambda field: field.split("_", 1)[-1]
    def display(field):
        c = category(field)
        return f"**{CATEGORY_NAMES[c]}**" if c in SHARED_QUESTION else DISPLAY_LABELS[c]

    def label(field, collapsed=False):
        # The question and its tooltip are shown in the section heading above.
        st.segmented_control(display(field), FIELD_OPTIONS[field], key=wkey(iid, field),
                             disabled=locked,
                             label_visibility="collapsed" if collapsed else "visible")

    def answer_box(field):
        # Pills inside a keyed container, stacked vertically by APP_CSS.
        with st.container(key=f"answer_{field}"):
            st.pills(display(field), FIELD_OPTIONS[field], key=wkey(iid, field),
                     format_func=fmt_answer, disabled=locked)

    def section(title, categories=()):
        """Numbered section heading, then each category's question with its tooltip."""
        st.markdown(f"#### {title}")
        for c in categories:
            # With one category the heading already names it, so show only its question.
            q = CATEGORY_QUESTIONS[c]
            text = q[0].upper() + q[1:] if len(categories) == 1 else DISPLAY_LABELS[c]
            st.markdown(text, help=GUIDELINES[c])

    # Each side's inputs sit in a colour-coded panel matching its text box above
    # (blue = unlearning target, violet = completion; see APP_CSS and SIDES).
    def side_panels(part):
        for col, (side, title, color) in zip(st.columns(2, gap="large"), SIDES):
            with col:
                with st.container(key=f"side_{side}_{part}"):
                    st.markdown(f":{color}[**{title}**]")
                    yield side

    letter = inst["reference_letter"]
    ref = f"{letter}. {inst['reference_text']}" if letter else inst["reference_text"]

    # Everything the annotator fills in sits in one tinted panel (APP_CSS), set apart
    # from the texts above, which are only to be read.
    st.markdown("### :material/edit_note: Annotation task")
    st.caption("Read the question and both texts above, then label each text below.")
    with st.container(key="annotation"):
        section("1. Final answer", ["answer"])
        for side in side_panels("answer"):
            answer_box(f"{side}_answer")
        st.caption(f":material/info: Reference answer: **{ref}**")

        st.divider()
        section("2. Reasoning", ["soundness", "alignment"])
        for side in side_panels("reasoning"):
            a, b = st.columns(2)
            with a:
                label(f"{side}_soundness")
            with b:
                label(f"{side}_alignment")

        st.divider()
        bl, br = st.columns(2, gap="large")
        with bl:
            section("3. Semantic consistency", ["consistency"])
            label("consistency", collapsed=True)
        with br:
            st.markdown("#### Comment")
            st.caption("Optional: truncation, partial answers, borderline calls.")
            st.text_area("Comment", key=wkey(iid, "comment"), height=110, disabled=locked,
                         label_visibility="collapsed")

    # ---- actions
    if "flash" in st.session_state:
        kind, text = st.session_state.pop("flash")
        getattr(st, kind)(text)

    c1, c2, c3, c4 = st.columns(4)
    c1.button("← Previous", key="prev_bottom", on_click=on_navigate, args=(inst, -1, n),
              disabled=idx == 0, use_container_width=True)
    c2.button("Save draft", on_click=on_save_draft, args=(inst,),
              disabled=locked, use_container_width=True)
    if locked:
        c3.button("🔓 Unlock", on_click=on_unlock, args=(inst,), use_container_width=True)
    else:
        c3.button("🔒 Lock", on_click=on_lock, args=(inst,), type="primary", use_container_width=True)
    c4.button("Next →", key="next_bottom", on_click=on_navigate, args=(inst, 1, n),
              disabled=idx == n - 1, use_container_width=True)


main()