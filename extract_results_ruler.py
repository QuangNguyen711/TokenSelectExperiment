import os
import csv
import json
from collections import defaultdict

ROOT = "result/ruler/qwen2-7b-inst/synthetic"

# method -> context -> list(values)
score_dict = defaultdict(lambda: defaultdict(list))
latency_dict = defaultdict(lambda: defaultdict(list))


def parse_summary_csv(path):
    with open(path, "r") as f:
        reader = list(csv.reader(f))

    # Row 1 = Tasks
    tasks = reader[1][1:]
    # Row 2 = Score
    scores = list(map(float, reader[2][1:]))

    return dict(zip(tasks, scores))


def parse_latency_json(path):
    with open(path, "r") as f:
        return json.load(f)


for context_len in os.listdir(ROOT):
    context_path = os.path.join(ROOT, context_len)

    if not os.path.isdir(context_path):
        continue

    for method in os.listdir(context_path):
        pred_path = os.path.join(context_path, method, "pred")

        if not os.path.exists(pred_path):
            continue

        summary_path = os.path.join(pred_path, "summary.csv")
        timing_path = os.path.join(pred_path, "dataset_timing.json")

        # --- SCORE ---
        if os.path.exists(summary_path):
            try:
                scores = parse_summary_csv(summary_path)
                avg_score = sum(scores.values()) / len(scores)
                score_dict[method][context_len].append(avg_score)
            except Exception as e:
                print(f"[WARN] Failed parsing summary: {summary_path} -> {e}")

        # --- LATENCY ---
        if os.path.exists(timing_path):
            try:
                timings = parse_latency_json(timing_path)
                avg_latency = sum(timings.values()) / (len(timings) * 100)
                latency_dict[method][context_len].append(avg_latency)
            except Exception as e:
                print(f"[WARN] Failed parsing timing: {timing_path} -> {e}")


def build_markdown_table(data_dict, title):
    # collect all contexts
    all_contexts = sorted(
        {int(ctx) for m in data_dict.values() for ctx in m.keys()}
    )

    all_contexts_str = [str(c) for c in all_contexts]

    md = []
    md.append(f"## {title}\n")

    # header
    header = "| Method | " + " | ".join(all_contexts_str) + " |"
    sep = "|---" * (len(all_contexts_str) + 1) + "|"

    md.append(header)
    md.append(sep)

    # rows
    for method in sorted(data_dict.keys()):
        row = [method]

        for ctx in all_contexts_str:
            values = data_dict[method].get(ctx, [])
            if values:
                avg = sum(values) / len(values)
                row.append(f"{avg:.2f}")
            else:
                row.append("-")

        md.append("| " + " | ".join(row) + " |")

    return "\n".join(md)


score_md = build_markdown_table(score_dict, "Average Score")
latency_md = build_markdown_table(latency_dict, "Average Latency")

print(score_md)
print()
print(latency_md)