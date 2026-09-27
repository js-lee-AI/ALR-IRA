import argparse
import json
from pathlib import Path
import statistics

from alr.metrics import geometric_mean


def aggregate(paths):
    records = [json.loads(Path(path).read_text()) for path in paths]
    expected = set(records[0]["cells"])
    if any(set(record["cells"]) != expected for record in records):
        raise ValueError(
            "Training replicates must cover the same tasks and temperatures"
        )
    output = {}
    for temperature in sorted({key.split("/")[-1] for key in expected}):
        keys = sorted(key for key in expected if key.endswith("/" + temperature))
        values = {}
        for metric in ("tau", "speedup"):
            replicas = [
                geometric_mean(
                    [record["cells"][key]["summary"][metric] for key in keys]
                )
                for record in records
            ]
            values[metric] = {
                "mean": statistics.mean(replicas),
                "sd": statistics.stdev(replicas) if len(replicas) > 1 else None,
                "replicates": len(replicas),
            }
        output[temperature] = values
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate tasks within each training replicate"
    )
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.results), indent=2))
