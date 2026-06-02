import os
import sys
import json
import argparse
import sys

# Add benchmark to path to import infinitebench_eval
sys.path.append(os.path.join(os.path.dirname(__file__), 'benchmark'))
try:
    from infinitebench_eval import get_score_one
except ImportError as e:
    print("Could not import get_score_one:", e)
    sys.exit(1)

def compare_folders(dir1, dir2):
    files1 = set([f for f in os.listdir(dir1) if f.endswith('.jsonl')])
    files2 = set([f for f in os.listdir(dir2) if f.endswith('.jsonl')])
    common_files = sorted(list(files1.intersection(files2)))

    if not common_files:
        print("No common .jsonl files found in the given directories.")
        return

    results = {}

    for file in common_files:
        task_name = file.replace('.jsonl', '')
        path1 = os.path.join(dir1, file)
        path2 = os.path.join(dir2, file)

        both_wrong_same = 0
        both_wrong_diff = 0
        both_right = 0
        only_1_right = 0
        only_2_right = 0

        with open(path1, 'r', encoding='utf-8') as f1, open(path2, 'r', encoding='utf-8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()

            if len(lines1) != len(lines2):
                print(f"Warning: {file} has different number of lines in both folders. Skipping.")
                continue

            for l1, l2 in zip(lines1, lines2):
                data1 = json.loads(l1)
                data2 = json.loads(l2)

                pred1 = str(data1.get('pred', ''))
                pred2 = str(data2.get('pred', ''))
                
                # Assume answers are the same
                label = data1.get('answers', [])
                
                score1 = get_score_one(pred1, label, task_name)
                score2 = get_score_one(pred2, label, task_name)

                right1 = (score1 > 0)
                right2 = (score2 > 0)

                pred_same = (pred1 == pred2)

                if not right1 and not right2:
                    if pred_same:
                        both_wrong_same += 1
                    else:
                        both_wrong_diff += 1
                elif right1 and right2:
                    both_right += 1
                else: # One right, one wrong
                    if right1:
                        only_1_right += 1
                    else:
                        only_2_right += 1

        results[task_name] = {
            'total': len(lines1),
            'both_wrong_same': both_wrong_same,
            'both_wrong_diff': both_wrong_diff,
            'both_right': both_right,
            'folder1_right_folder2_wrong': only_1_right,
            'folder1_wrong_folder2_right': only_2_right
        }

    # Print results
    print(f"{'Task':<25} | {'BothW_Same':<11} | {'BothW_Diff':<11} | {'Both_Right':<11} | {'F1_R_F2_W':<11} | {'F1_W_F2_R':<11}")
    print("-" * 90)
    for task_name, stats in results.items():
        print(f"{task_name:<25} | {stats['both_wrong_same']:<11} | {stats['both_wrong_diff']:<11} | {stats['both_right']:<11} | {stats['folder1_right_folder2_wrong']:<11} | {stats['folder1_wrong_folder2_right']:<11}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare prediction results between two folders.')
    parser.add_argument('dir1', type=str, help='First directory (e.g. result_release_ttft/infinitbench/qwen-token-retrieval)')
    parser.add_argument('dir2', type=str, help='Second directory (e.g. result_release_ttft/infinitbench/qwen-sim-0.97-max-4096-anchor-no-balance)')
    args = parser.parse_args()

    compare_folders(args.dir1, args.dir2)
