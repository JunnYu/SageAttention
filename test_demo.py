import time
import paddle
import paddle.nn.functional as F
from ppsageattention import sageattn
import matplotlib.pyplot as plt
 
times = {
    "sageattn": [],
    "scaled_dot_product_attention": []
}
n_iter = 20  # 迭代次数
shape = (4, 32, 1536, 128)
dtype_list= ["float16", "bfloat16"]
paddle.set_grad_enabled(False)

for dtype in dtype_list:
    paddle.seed(42)
    paddle.set_default_dtype(dtype)
    q = paddle.randn(shape)
    k = paddle.randn(shape)
    v = paddle.randn(shape)
    
    # 测量 sageattn 的运行时间
    sageattn_times = []
    for _ in range(n_iter):
        paddle.device.synchronize()  # 确保CUDA操作完成
        start_time = time.time()
        out1 = sageattn(q, k, v, is_causal=True)
        paddle.device.synchronize()  # 确保CUDA操作完成
        end_time = time.time()
        if _ < 5: continue
        sageattn_times.append(end_time - start_time)
    times["sageattn"].append(sum(sageattn_times) / len(sageattn_times))
    
    # 测量 scaled_dot_product_attention 的运行时间
    qq, kk, vv = q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]), v.transpose([0, 2, 1, 3])
    sdpa_times = []
    for _ in range(n_iter):
        paddle.device.synchronize()  # 确保CUDA操作完成
        start_time = time.time()
        out2 = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
        paddle.device.synchronize()  # 确保CUDA操作完成
        end_time = time.time()
        if _ < 5: continue
        sdpa_times.append(end_time - start_time)
    times["scaled_dot_product_attention"].append(sum(sdpa_times) / len(sdpa_times))

# 绘制性能对比图
bar_width = 0.35
index = range(len(times["sageattn"]))
bar_labels = dtype_list
 
fig, ax = plt.subplots()
rects1 = ax.bar(index, times["sageattn"], bar_width, label='sageattn')
rects2 = ax.bar([i + bar_width for i in index], times["scaled_dot_product_attention"], bar_width, label='scaled_dot_product_attention')
 
ax.set_xlabel('Data Type')
ax.set_ylabel('Mean Time per Iteration (seconds)')
ax.set_title('Performance Comparison of Attention Mechanisms')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(bar_labels)
ax.legend()
 
fig.tight_layout()
plt.savefig("performance_comparison.png")
plt.show()