import torch

print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")

# print gpu device
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))


# Check allocated memory in bytes
allocated_memory = torch.cuda.memory_allocated()

# Check cached memory (includes allocated memory)
cached_memory = torch.cuda.memory_reserved()

print(f"Allocated Memory: {allocated_memory / (1024 ** 3)} GB")
print(f"Cached Memory: {cached_memory / (1024 ** 3)} GB")
