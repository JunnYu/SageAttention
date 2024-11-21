import time
import paddle
import paddle.nn.functional as F
from ppsageattention import sageattn
import matplotlib.pyplot as plt
 
# 初始化一个字典来存储每种数据类型的运行时间
times = {
    "sageattn": [],
    "scaled_dot_product_attention": []
}
 

# 为 q, k, v 生成随机数据，并测量运行时间
for dtype in ["float16", "bfloat16"]:
    paddle.seed(42)
    q = paddle.randn([2, 32, 8192, 64]).cast(dtype)
    k = paddle.randn([2, 32, 8192, 64]).cast(dtype)
    v = paddle.randn([2, 32, 8192, 64]).cast(dtype)
    
    # 测量 sageattn 运行 100 次的时间
    start_time = time.time()
    for _ in range(100):
        out1 = sageattn(q, k, v, is_causal=True)
    end_time = time.time()
    times["sageattn"].append(end_time - start_time)
    
    # 测量 scaled_dot_product_attention 运行 100 次的时间
    start_time = time.time()
    for _ in range(100):
        out2 = F.scaled_dot_product_attention(q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]), v.transpose([0, 2, 1, 3]), is_causal=True)
    end_time = time.time()
    times["scaled_dot_product_attention"].append(end_time - start_time)
    diff = paddle.abs(out1 - out2.transpose([0, 2, 1, 3]))
    print(diff.mean(), diff.max(), diff.min())
    
# 计算每种方法的平均时间
average_times = {key: sum(values) / len(values) for key, values in times.items()}
 
# 绘制性能对比图
bar_width = 0.35
index = range(len(times["sageattn"]))
bar_labels = ["float16", "bfloat16"]
 
fig, ax = plt.subplots()
rects1 = ax.bar(index, [average_times["sageattn"]] * len(index), bar_width, label='sageattn')
rects2 = ax.bar([i + bar_width for i in index], [average_times["scaled_dot_product_attention"]] * len(index), bar_width, label='scaled_dot_product_attention')
 
ax.set_xlabel('Data Type')
ax.set_ylabel('Mean Time per Iteration (seconds)')
ax.set_title('Performance Comparison of Attention Mechanisms')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(bar_labels)
ax.legend()
 
fig.tight_layout()
plt.savefig('performance_comparison.png')
plt.show()