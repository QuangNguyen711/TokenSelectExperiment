import pandas as pd
import os

def print_summary_table(csv_path="attention_profiling_qwen-union-of-sets.csv"):
    if not os.path.exists(csv_path):
        print(f"[LỖI] Không tìm thấy file {csv_path}.")
        return

    df = pd.read_csv(csv_path)
    
    # Ép kiểu an toàn - Đã thêm fragmentation_ratio
    numeric_cols = [
        'num_tokens', 'min_entropy', 'mean_entropy', 
        'diversity_ratio', 'fragmentation_ratio', 
        'min_entropy_l2', 'mean_l2'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()

    print(f"\n=== BẢNG TỔNG HỢP ATTENTION PROFILING (Từ {len(df)} mẫu đo lường) ===\n")

    # 1. BẢNG TỔNG QUAN THEO TỪNG DATASET (Thêm Frag_Ratio)
    print("[1] CHỈ SỐ TRUNG BÌNH TOÀN CỤC THEO TỪNG TASK:")
    agg_dict = {
        'diversity_ratio': 'mean',
        'min_entropy': 'mean',
        'min_entropy_l2': 'mean',
        'mean_l2': 'mean'
    }
    # Chỉ thêm fragmentation_ratio nếu nó tồn tại trong CSV mới
    if 'fragmentation_ratio' in df.columns:
        agg_dict['fragmentation_ratio'] = 'mean'

    summary = df.groupby('dataset').agg(agg_dict).round(4)
    
    # Đổi tên cột cho dễ đọc
    cols = ['Diversity', 'Min_Entropy', 'L2_Min_Ent', 'Mean_L2']
    if 'fragmentation_ratio' in df.columns:
        cols.append('Frag_Ratio')
    summary.columns = cols
    
    print(summary.to_markdown())
    print("\n" + "-"*85 + "\n")

    # 2. PHÂN TÍCH ĐỘ ĐỨT GÃY THEO CHIỀU DÀI (Nếu có dữ liệu)
    if 'fragmentation_ratio' in df.columns:
        print("[2] TỶ LỆ ĐỨT GÃY (FRAG_RATIO) THEO CHIỀU DÀI NGỮ CẢNH:")
        print("(Cao > 0.8: Token rời rạc | Thấp < 0.5: Token đi theo cụm/Span)")
        bins = [0, 50000, 100000, 250000, 500000, 1000000, float('inf')]
        labels = ['0-50k', '50k-100k', '100k-250k', '250k-500k', '500k-1M', '>1M']
        df['length_bin'] = pd.cut(df['num_tokens'], bins=bins, labels=labels)
        
        frag_summary = df.groupby(['dataset', 'length_bin'], observed=False)['fragmentation_ratio'].mean().unstack().round(3)
        print(frag_summary.to_markdown())
        print("\n" + "-"*85 + "\n")

    # 3. ĐIỂM CHẠM CỦA CÁC LỚP (LAYER)
    print("[3] TOP 5 LAYERS CÓ ĐỘ ĐA DẠNG CAO NHẤT (Diversity):")
    for dataset in df['dataset'].unique():
        ds_data = df[df['dataset'] == dataset]
        top_layers = ds_data.groupby('layer_id')['diversity_ratio'].mean().sort_values(ascending=False).head(5)
        print(f"\n  * Task: {dataset}")
        layers_str = " | ".join([f"L{int(idx)}: {val:.3f}" for idx, val in top_layers.items()])
        print(f"    {layers_str}")
    print("\n")

if __name__ == "__main__":
    # Tự động tìm file CSV mới nhất nếu không chỉ định
    import glob
    csv_files = glob.glob("attention_profiling_*.csv")
    if csv_files:
        latest_csv = max(csv_files, key=os.path.getmtime)
        print(f"Đang phân tích file mới nhất: {latest_csv}")
        print_summary_table(latest_csv)
    else:
        print_summary_table("attention_profiling_qwen-union-of-sets.csv")