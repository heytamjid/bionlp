import json
import os
import sys
import subprocess


def extract_item(input_json, item_id, out_path):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if isinstance(item, dict) and item.get("id") == item_id:
            with open(out_path, "w", encoding="utf-8") as o:
                json.dump(item, o, ensure_ascii=False, indent=2)
            return True
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: run_single_on_id.py <item_id> [out_prefix]")
        sys.exit(1)

    item_id = sys.argv[1]
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else f"single_{item_id}"

    repo_root = os.path.dirname(os.path.dirname(__file__))
    annot_path = os.path.join(
        repo_root, "gemini direct prediction", "annotated_train_data.json"
    )
    tmp_item = os.path.join(repo_root, f"tmp_item_{item_id}.json")

    ok = extract_item(annot_path, item_id, tmp_item)
    if not ok:
        print(f"Item {item_id} not found in {annot_path}")
        sys.exit(2)

    # Call the single-case predictor
    single_script = os.path.join(
        repo_root, "gemini direct prediction", "gemini_predict_legacy_single.py"
    )
    cmd = [
        sys.executable,
        single_script,
        "--item",
        tmp_item,
        "--out_prefix",
        out_prefix,
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)
    print("Exit code:", proc.returncode)


if __name__ == "__main__":
    main()
