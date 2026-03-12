import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# 定义要执行的函数
def run_command(i, start, stop):
    """执行单个run.py命令"""
    print(f"{i} Start the {stop} run and the time is: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 使用f-string构建命令字符串，将参数值插入到命令中
    cmd = f"python run.py ./instrument_info/instrument_info6.0-10.5A_12.75m_8mm.txt ./sample_info/sample_info6.0-10.5A_12.75m_8mm_PS_spheres_SANS.txt {start} {stop} 'PS_spheres_SANS'"
    print(f"Executing: {cmd}")
    os.system(cmd)
    return stop

if __name__ == '__main__':
    # 设置参数
    num = 0
    num2 = 50  # 执行完整的batchrun任务
    start_time = time.time()

    # 创建任务列表
    tasks = []
    for i in range(num + 1):
        for j in range(num2 + 1):
            start = i * num2 + j
            stop = start + 1
            tasks.append((i, start, stop))

    # 使用ProcessPoolExecutor并行执行任务
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 提交所有任务
        futures = [executor.submit(run_command, i, start, stop) for i, start, stop in tasks]

        # 等待所有任务完成
        for future in futures:
            future.result()

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time} seconds")
