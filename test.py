import torch
import torch.nn.functional as F

def pooled(scores, K):
    total_pad = K - 1
    pl, pr = total_pad // 2, total_pad - total_pad // 2
    x = F.pad(scores.view(1,1,-1), (pl, pr), value=float("-inf"))
    return F.max_pool1d(x, kernel_size=K, stride=1).view(-1)

s = torch.arange(10).float()  # [0,1,...,9], đỉnh rõ ràng ở cuối
for K in [3, 4, 5, 12, 128]:
    out = pooled(s, K)
    assert out.shape[-1] == s.shape[-1], f"K={K} lệch độ dài!"
    print(f"K={K}: len OK = {out.shape[-1]}, argmax = {out.argmax().item()}")