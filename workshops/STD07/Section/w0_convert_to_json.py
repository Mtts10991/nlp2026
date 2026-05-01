import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from data_set import THAI_IP_DATASET, LABEL_NAMES

OUTPUT_PATH = Path(__file__).parent / "data" / "processed" / "thai_ip_dataset.json"


def convert_to_json(dataset, label_names, output_path):
    label_counts = Counter(item["label"] for item in dataset)
    samples = [
        {"id": i, "text": item["text"], "label": item["label"], "label_name": label_names[item["label"]]}
        for i, item in enumerate(dataset)
    ]
    payload = {
        "metadata": {
            "version": "v1.0",
            "labeling_method": "manual",
            "labeling_confidence_level": "gold_standard",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "label_names": label_names,
            "total": len(dataset),
            "dataset_stats": {
                "label_distribution": {label_names[k]: v for k, v in label_counts.items()},
            },
        },
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    payload = convert_to_json(THAI_IP_DATASET, LABEL_NAMES, OUTPUT_PATH)
    print(f"Wrote {payload['metadata']['total']} samples to {OUTPUT_PATH}")
