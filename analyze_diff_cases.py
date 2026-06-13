import os
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

def analyze_volatility(base_dir, methods_to_compare, out_dir="volatility_analysis"):
    os.makedirs(out_dir, exist_ok=True)
    
    # Tìm các file dataset chung giữa tất cả các phương pháp
    dataset_files = None
    for m in methods_to_compare:
        folder_path = os.path.join(base_dir, m)
        if not os.path.isdir(folder_path):
            continue
        files = set([f for f in os.listdir(folder_path) if f.endswith('.jsonl')])
        if dataset_files is None:
            dataset_files = files
        else:
            dataset_files = dataset_files.intersection(files)
            
    if not dataset_files:
        print("Không tìm thấy dataset chung nào!")
        return

    dataset_files = sorted(list(dataset_files))
    
    print(f"\n{'='*95}")
    print(f"📊 PHÂN TÍCH ĐỘ BIẾN ĐỘNG (VOLATILITY) TRÊN {len(methods_to_compare)} PHƯƠNG PHÁP")
    print(f"Các phương pháp đưa vào: {', '.join(methods_to_compare)}")
    print(f"{'='*95}\n")
    
    print(f"{'Dataset':<20} | {'Tổng câu':<10} | {'Luôn ĐÚNG':<12} | {'Luôn SAI':<10} | {'Biến động':<10}")
    print("-" * 75)
    
    for ds_file in dataset_files:
        task_name = ds_file.replace('.jsonl', '')
        
        # Đọc dữ liệu của toàn bộ phương pháp
        lines_by_method = {}
        for m in methods_to_compare:
            with open(os.path.join(base_dir, m, ds_file), 'r', encoding='utf-8') as f:
                lines_by_method[m] = f.readlines()
                
        num_lines = len(lines_by_method[methods_to_compare[0]])
        
        always_right = 0
        always_wrong = 0
        volatile_cases = []
        
        for i in range(num_lines):
            scores = {}
            preds = {}
            gt = None
            
            for m in methods_to_compare:
                try:
                    data = json.loads(lines_by_method[m][i])
                    pred = str(data.get('pred', '')).strip()
                    gt = data.get('answers', [])
                    score = get_score_one(pred, gt, task_name)
                    
                    scores[m] = 1 if score > 0 else 0
                    preds[m] = pred
                except Exception:
                    scores[m] = 0
                    preds[m] = "ERROR/EMPTY"
                
            total_correct = sum(scores.values())
            
            if total_correct == len(methods_to_compare):
                always_right += 1
            elif total_correct == 0:
                always_wrong += 1
            else:
                volatile_cases.append({
                    'line_idx': i + 1,
                    'correct_count': total_correct,
                    'scores': scores,
                    'preds': preds,
                    'gt': gt
                })
                
        # In ra terminal
        print(f"{task_name:<20} | {num_lines:<10} | {always_right:<12} | {always_wrong:<10} | {len(volatile_cases):<10}")
        
        # Ghi log chi tiết các câu biến động ra file
        if volatile_cases:
            # Sắp xếp các câu "gây tranh cãi nhất" lên đầu (số phương pháp đúng/sai gần bằng nhau nhất)
            ideal_half = len(methods_to_compare) / 2.0
            volatile_cases.sort(key=lambda x: abs(x['correct_count'] - ideal_half))
            
            report_path = os.path.join(out_dir, f"{task_name}_volatility.txt")
            with open(report_path, 'w', encoding='utf-8') as f_out:
                f_out.write(f"=== DANH SÁCH CÁC CÂU BỊ BIẾN ĐỘNG (DATASET: {task_name}) ===\n")
                f_out.write(f"Tổng số: {len(volatile_cases)} câu. Sắp xếp các câu gây tranh cãi nhất lên đầu.\n\n")
                
                for case in volatile_cases:
                    f_out.write(f"🔹 Dòng {case['line_idx']} | Số phương pháp trả lời ĐÚNG: {case['correct_count']}/{len(methods_to_compare)}\n")
                    f_out.write(f"   Đáp án chuẩn: {case['gt']}\n")
                    f_out.write("   Kết quả từng phương pháp:\n")
                    
                    # In theo thứ tự truyền vào cho dễ nhìn
                    for m in methods_to_compare:
                        status = "✅" if case['scores'][m] == 1 else "❌"
                        f_out.write(f"      [{status}] {m}: {case['preds'][m]}\n")
                    f_out.write("-" * 80 + "\n")
                    
    print(f"\nChi tiết các câu trả lời khác biệt đã được lưu vào thư mục: '{out_dir}/'")

def main():
    parser = argparse.ArgumentParser(description='Phân tích độ biến động của các câu hỏi qua nhiều phương pháp.')
    parser.add_argument('--base_dir', type=str, default="result_release_ttft/infinitbench", help='Thư mục gốc')
    parser.add_argument('--baseline', type=str, default="qwen-token-retrieval", help='Thư mục baseline (sẽ được in ra đối chiếu đầu tiên)')
    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Lỗi: Không tìm thấy {args.base_dir}")
        return

    methods = sorted([d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))])
    
    # Gom tất cả các phương pháp có chữ 'true' và ghép thêm baseline vào đầu danh sách
    true_methods = [m for m in methods if "true" in m.lower() and m != args.baseline]
    methods_to_compare = [args.baseline] + true_methods

    if len(methods_to_compare) < 2:
        print("Không đủ phương pháp để so sánh biến động.")
        return

    analyze_volatility(args.base_dir, methods_to_compare)

if __name__ == "__main__":
    main()