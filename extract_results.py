import os
import json

def extract_and_analyze(base_dir="result_release/infinitbench"):
    if not os.path.exists(base_dir):
        print(f"Thư mục {base_dir} không tồn tại.")
        return

    acc_data = {}  # {dataset: {method: score}}
    lat_data = {}  # {dataset: {method: time}}
    
    methods = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    temp = []
    for method in methods:
        if method.startswith("qwen-chunk-"):
            temp.append(method)

    methods = temp
    print(f"Phương pháp được phân tích: {methods}")

    # Đọc dữ liệu
    for method in methods:
        exp_dir = os.path.join(base_dir, method)
        res_file = os.path.join(exp_dir, "result.txt")
        time_file = os.path.join(exp_dir, "dataset_timing.json")

        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                for line in f:
                    if ',' in line:
                        parts = line.strip().split(',')
                        ds = parts[0].strip()
                        score = float(parts[1].strip())
                        if ds not in acc_data: acc_data[ds] = {}
                        acc_data[ds][method] = score

        if os.path.exists(time_file):
            with open(time_file, 'r') as f:
                try:
                    timing = json.load(f)
                    for ds, t in timing.items():
                        if ds not in lat_data: lat_data[ds] = {}
                        lat_data[ds][method] = float(t/50)
                except json.JSONDecodeError:
                    pass

    # Hàm in bảng
    def print_table(title, data_dict, is_accuracy=True):
        if not data_dict: 
            return
            
        datasets = sorted(list(data_dict.keys()))
        method_list = sorted(methods)

        print(f"\n### {title}")
        
        header = "| Dataset | " + " | ".join(method_list) + " |"
        separator = "|---|" + "|".join(["---" for _ in method_list]) + "|"
        
        print(header)
        print(separator)

        for ds in datasets:
            row = [f"**{ds}**"]
            
            for method in method_list:
                val = data_dict[ds].get(method, None)

                if val is None:
                    row.append("-")
                else:
                    best_val = (
                        max(data_dict[ds].values())
                        if is_accuracy
                        else min(data_dict[ds].values())
                    )

                    if val == best_val:
                        row.append(f"**{val:.1f}**")
                    else:
                        row.append(f"{val:.1f}")

            print("| " + " | ".join(row) + " |")

    print_table("BẢNG 1: ĐỘ CHÍNH XÁC (ACCURACY %)", acc_data, is_accuracy=True)
    print_table("BẢNG 2: THỜI GIAN HOÀN THÀNH (LATENCY - Giây)", lat_data, is_accuracy=False)

    # In phân tích cực trị
    print("\n### TỔNG HỢP CÁC PHƯƠNG PHÁP TỐI ƯU CỤC BỘ THEO BENCHMARK")
    
    print("\n#### 1. Theo Độ chính xác (Max Accuracy):")
    for ds in sorted(acc_data.keys()):
        best_score = max(acc_data[ds].values())
        best_methods = [m for m, s in acc_data[ds].items() if s == best_score]
        print(f"- **{ds}**: {best_score:.1f}% -> Giữ bởi: {', '.join(best_methods)}")
        
    print("\n#### 2. Theo Thời gian (Min Latency):")
    for ds in sorted(lat_data.keys()):
        best_time = min(lat_data[ds].values())
        best_methods = [m for m, t in lat_data[ds].items() if t == best_time]
        print(f"- **{ds}**: {best_time:.1f}s -> Giữ bởi: {', '.join(best_methods)}")

if __name__ == "__main__":
    extract_and_analyze()