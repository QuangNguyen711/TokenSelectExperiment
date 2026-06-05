import json
from collections import defaultdict

stats = defaultdict(list)
with open("adaptive_prob_mass_log.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        stats[data["layer"]].append((data["k_dynamic"], data["prob_mass"]))

print(f"{'Layer':<10} | {'Avg K_dynamic':<15} | {'Avg Prob Mass':<15}")
print("-" * 45)
for layer in sorted(stats.keys()):
    layer_data = stats[layer]
    avg_k = sum(x[0] for x in layer_data) / len(layer_data)
    avg_mass = sum(x[1] for x in layer_data) / len(layer_data)
    print(f"Layer {layer:<4} | {avg_k:<15.0f} | {avg_mass * 100:.2f}%")