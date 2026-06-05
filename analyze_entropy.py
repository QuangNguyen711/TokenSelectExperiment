import json
from collections import defaultdict
from statistics import mean, median
import numpy as np

stats = defaultdict(list)

with open("adaptive_prob_mass_log.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        stats[data["layer"]].append(
            (data["k_dynamic"], data["prob_mass"])
        )

print("# K Dynamic Statistics\n")
print(
    "| Layer | Mean | Median | P25 | P75 | P90 | P95 | P99 | Min | Max | Samples |"
)
print(
    "|-------|------|--------|-----|-----|-----|-----|-----|-----|-----|---------|"
)

for layer in sorted(stats.keys()):
    k_values = np.array([x[0] for x in stats[layer]])

    print(
        f"| {layer} "
        f"| {mean(k_values):.2f} "
        f"| {median(k_values):.2f} "
        f"| {np.percentile(k_values, 25):.2f} "
        f"| {np.percentile(k_values, 75):.2f} "
        f"| {np.percentile(k_values, 90):.2f} "
        f"| {np.percentile(k_values, 95):.2f} "
        f"| {np.percentile(k_values, 99):.2f} "
        f"| {np.min(k_values)} "
        f"| {np.max(k_values)} "
        f"| {len(k_values)} |"
    )

print("\n# Probability Mass Statistics\n")
print(
    "| Layer | Mean (%) | Median (%) | P25 (%) | P75 (%) | P90 (%) | P95 (%) | P99 (%) | Min (%) | Max (%) |"
)
print(
    "|-------|----------|------------|---------|---------|---------|---------|---------|---------|---------|"
)

for layer in sorted(stats.keys()):
    mass_values = np.array([x[1] for x in stats[layer]]) * 100

    print(
        f"| {layer} "
        f"| {mean(mass_values):.2f} "
        f"| {median(mass_values):.2f} "
        f"| {np.percentile(mass_values, 25):.2f} "
        f"| {np.percentile(mass_values, 75):.2f} "
        f"| {np.percentile(mass_values, 90):.2f} "
        f"| {np.percentile(mass_values, 95):.2f} "
        f"| {np.percentile(mass_values, 99):.2f} "
        f"| {np.min(mass_values):.2f} "
        f"| {np.max(mass_values):.2f} |"
    )