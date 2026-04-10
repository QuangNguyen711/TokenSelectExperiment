import os
import glob
import pandas as pd

def print_md_header(title, level=2):
    hashes = "#" * level
    print(f"\n{hashes} {title}\n")

def analyze_context_topology():
    print_md_header("BÁO CÁO: ĐỊA HÌNH NGỮ CẢNH VÀ SỰ PHÂN MẢNH (CONTEXT TOPOLOGY)", 1)
    
    files = glob.glob("context_topology_*.csv")
    if not files:
        print("> **Lỗi:** Không tìm thấy file `context_topology_*.csv`. Vui lòng kiểm tra lại thư mục.\n")
        return

    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)

    # Lọc các đoạn prompt đủ dài để có ý nghĩa phân tích
    if 'num_tokens' in df.columns:
        df = df[df['num_tokens'] > 50000]

    # Tính trung bình các chỉ số
    summary = df.groupby(['dataset', 'method'])[['max_span_len', 'mean_span_len', 'num_spans', 'spread_ratio']].mean().reset_index()

    # --- BẢNG 1: SỐ LƯỢNG MẢNH VỠ (NUM SPANS) ---
    print_md_header("Bảng 1: Số Lượng Mảnh Vỡ (Num Spans)", 3)
    print("> Văn bản bị băm thành bao nhiêu mảnh vụn? (**Số CÀNG CAO = CÀNG NÁT**)\n")
    pivot_num_spans = summary.pivot(index='dataset', columns='method', values='num_spans')
    print(pivot_num_spans.round(1).fillna('-').to_markdown())
    print("\n")

    # --- BẢNG 2: ĐỘ DÀI TRUNG BÌNH MỖI MẢNH (MEAN SPAN LEN) ---
    print_md_header("Bảng 2: Độ Dài Trung Bình Mỗi Mảnh (Mean Span Length)", 3)
    print("> Mỗi cụm nhặt về dài bao nhiêu token? (**< 3 token nghĩa là nhặt từng chữ rời rạc**)\n")
    pivot_mean_span = summary.pivot(index='dataset', columns='method', values='mean_span_len')
    print(pivot_mean_span.round(2).fillna('-').to_markdown())
    print("\n")

    # --- BẢNG 3: MẢNG LIỀN KỀ LỚN NHẤT (MAX SPAN LEN) ---
    print_md_header("Bảng 3: Sự Toàn Vẹn Cấu Trúc (Max Span Length)", 3)
    print("> Mảng liên tục dài nhất nhặt được (**Chỉ số sống còn với JSON/KV**)\n")
    pivot_max_span = summary.pivot(index='dataset', columns='method', values='max_span_len')
    print(pivot_max_span.round(1).fillna('-').to_markdown())
    print("\n")

    # --- BẢNG 4: ĐỘ BAO PHỦ (SPREAD RATIO) ---
    print_md_header("Bảng 4: Độ Bao Phủ Ngữ Cảnh (Spread Ratio)", 3)
    print("> Tỉ lệ bao phủ từ đầu đến cuối văn bản (**0.0 -> 1.0**)\n")
    pivot_spread = summary.pivot(index='dataset', columns='method', values='spread_ratio')
    print(pivot_spread.round(4).fillna('-').to_markdown())
    print("\n---")

if __name__ == "__main__":
    analyze_context_topology()
    print_md_header("HOÀN THÀNH XUẤT BÁO CÁO", 3)