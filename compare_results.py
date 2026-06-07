import os
import sys
import json
import argparse
from collections import OrderedDict

# Add benchmark to path to import infinitebench_eval
sys.path.append(os.path.join(os.path.dirname(__file__), "benchmark"))

try:
    from infinitebench_eval import get_score_one
except ImportError as e:
    print("Could not import get_score_one:", e)
    sys.exit(1)


def compare_folders(dir1, dir2):
    files1 = {f for f in os.listdir(dir1) if f.endswith(".jsonl")}
    files2 = {f for f in os.listdir(dir2) if f.endswith(".jsonl")}
    common_files = sorted(files1.intersection(files2))

    if not common_files:
        print("No common .jsonl files found in the given directories.")
        return

    results = OrderedDict()

    for file in common_files:
        task_name = file.replace(".jsonl", "")
        path1 = os.path.join(dir1, file)
        path2 = os.path.join(dir2, file)

        correct_same = 0
        correct_diff = 0
        wrong_same = 0
        wrong_diff = 0
        only_1_right = 0
        only_2_right = 0

        with open(path1, "r", encoding="utf-8") as f1, open(
            path2, "r", encoding="utf-8"
        ) as f2:

            lines1 = f1.readlines()
            lines2 = f2.readlines()

            if len(lines1) != len(lines2):
                print(
                    f"Warning: {file} has different number of lines in both folders. Skipping."
                )
                continue

            for idx, (l1, l2) in enumerate(zip(lines1, lines2)):
                data1 = json.loads(l1)
                data2 = json.loads(l2)

                pred1 = str(data1.get("pred", "")).strip()
                pred2 = str(data2.get("pred", "")).strip()

                label = data1.get("answers", [])

                score1 = get_score_one(pred1, label, task_name)
                score2 = get_score_one(pred2, label, task_name)

                right1 = score1 > 0
                right2 = score2 > 0

                pred_same = pred1 == pred2

                # Both correct
                if right1 and right2:
                    if pred_same:
                        correct_same += 1
                    else:
                        correct_diff += 1

                # Both wrong
                elif not right1 and not right2:
                    if pred_same:
                        wrong_same += 1
                    else:
                        wrong_diff += 1

                # One correct, one wrong
                else:
                    if right1:
                        only_1_right += 1
                    else:
                        only_2_right += 1

        total = len(lines1)

        results[task_name] = {
            "total": total,
            "correct_same": correct_same,
            "correct_diff": correct_diff,
            "wrong_same": wrong_same,
            "wrong_diff": wrong_diff,
            "folder1_right_folder2_wrong": only_1_right,
            "folder1_wrong_folder2_right": only_2_right,
        }

    # Print table
    print(
        f"{'Task':<25} | "
        f"{'Total':<6} | "
        f"{'C_Same':<8} | "
        f"{'C_Diff':<8} | "
        f"{'W_Same':<8} | "
        f"{'W_Diff':<8} | "
        f"{'F1_R_F2_W':<11} | "
        f"{'F1_W_F2_R':<11}"
    )

    print("-" * 110)

    grand_total = {
        "total": 0,
        "correct_same": 0,
        "correct_diff": 0,
        "wrong_same": 0,
        "wrong_diff": 0,
        "folder1_right_folder2_wrong": 0,
        "folder1_wrong_folder2_right": 0,
    }

    for task_name, stats in results.items():
        print(
            f"{task_name:<25} | "
            f"{stats['total']:<6} | "
            f"{stats['correct_same']:<8} | "
            f"{stats['correct_diff']:<8} | "
            f"{stats['wrong_same']:<8} | "
            f"{stats['wrong_diff']:<8} | "
            f"{stats['folder1_right_folder2_wrong']:<11} | "
            f"{stats['folder1_wrong_folder2_right']:<11}"
        )

        for k in grand_total:
            grand_total[k] += stats[k]

    print("-" * 110)

    print(
        f"{'ALL':<25} | "
        f"{grand_total['total']:<6} | "
        f"{grand_total['correct_same']:<8} | "
        f"{grand_total['correct_diff']:<8} | "
        f"{grand_total['wrong_same']:<8} | "
        f"{grand_total['wrong_diff']:<8} | "
        f"{grand_total['folder1_right_folder2_wrong']:<11} | "
        f"{grand_total['folder1_wrong_folder2_right']:<11}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare prediction results between two folders."
    )

    parser.add_argument(
        "dir1",
        type=str,
        help="First directory",
    )

    parser.add_argument(
        "dir2",
        type=str,
        help="Second directory",
    )

    args = parser.parse_args()

    compare_folders(args.dir1, args.dir2)