"""Download the scenario corpus and build (or rebuild) the ChromaDB index.

Run once after cloning:  python add_scenarios.py

The corpus is distributed as a Hugging Face dataset rather than tracked in this
repository, which keeps the clone small. On first run the scenario XMLs are
downloaded into CollectedScenarios/ (the path every other module expects) and
then indexed into the persistent chroma/ store used by the Selection module.
Re-runs reuse whatever is already on disk.
"""
import shutil
import sys
from pathlib import Path

from db_wrapper import ScenarioDBWrapper

#: Hugging Face dataset holding the CommonRoad scenario corpus.
HF_DATASET = "TUM-AVS/PlannerForge-Scenarios"
#: Path inside the dataset that holds the scenario XMLs.
HF_SCENARIO_DIR = "scenarios"
#: Where the app expects the corpus. Do not change — the planner batch presets,
#: the selection ground truth and the modification pipeline all resolve scenario
#: names against this directory.
CORPUS_DIR = Path(__file__).resolve().parent / "CollectedScenarios"


def download_corpus(target: Path = CORPUS_DIR) -> int:
    """Fetch the scenario XMLs from Hugging Face into `target`.

    Returns the number of scenario files available afterwards. Existing files
    are left alone, so this is safe to re-run and cheap when the corpus is
    already present.
    """
    existing = sorted(target.glob("*.xml"))
    if existing:
        print(f"✅ {len(existing)} scenarios already in {target.name}/ — skipping download")
        return len(existing)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "❌ huggingface_hub is not installed.\n"
            "   pip install -r requirements.txt"
        )

    print(f"⬇️  Downloading the scenario corpus from {HF_DATASET} …")
    try:
        snapshot = snapshot_download(
            repo_id=HF_DATASET,
            repo_type="dataset",
            allow_patterns=[f"{HF_SCENARIO_DIR}/*.xml"],
        )
    except Exception as e:  # auth, network, or a renamed dataset
        sys.exit(
            f"❌ Could not download {HF_DATASET}: {type(e).__name__}: {e}\n"
            f"   If the dataset is private, log in first:  huggingface-cli login\n"
            f"   You can also drop the scenario XMLs into {target}/ by hand."
        )

    src = Path(snapshot) / HF_SCENARIO_DIR
    target.mkdir(parents=True, exist_ok=True)
    n = 0
    for xml in sorted(src.glob("*.xml")):
        shutil.copy2(xml, target / xml.name)
        n += 1
    if n == 0:
        sys.exit(f"❌ No scenario XMLs found under {HF_DATASET}:{HF_SCENARIO_DIR}/")
    print(f"✅ {n} scenarios written to {target.name}/")
    return n


def main() -> None:
    download_corpus()
    db = ScenarioDBWrapper("chroma")
    db.save_folder(str(CORPUS_DIR))


if __name__ == "__main__":
    main()
