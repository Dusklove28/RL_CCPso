import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 设备数量:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("可用 GPU:")
    for index in range(torch.cuda.device_count()):
        print(f"  - cuda:{index} | {torch.cuda.get_device_name(index)}")
else:
    print("未检测到 GPU，将使用 CPU 运行")
