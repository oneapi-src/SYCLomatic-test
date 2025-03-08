import numpy as np
import torch
import pytest
from custom_gguf import *  # Make sure to import your custom module

# Type table mapping GGML quantization types to their corresponding data types
GGML_DATA_TYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "Q8_0": torch.float32,  # Adjust as needed
    "Q2_K": torch.float32,  # Adjust as needed
    "Q3_K": torch.float32,  # Adjust as needed
    "Q4_K": torch.float32,  # Adjust as needed
    "Q5_K": torch.float32,  # Adjust as needed
    "Q6_K": torch.float32,  # Adjust as needed
    "IQ4_XS": torch.float32,  # Adjust as needed
    # Add other mappings as needed
}

@pytest.mark.parametrize("ggml_name", GGML_DATA_TYPES.keys())
def test_dequant_function(ggml_name):
    num_blocks = 4
    device = "xpu"  # or "cpu" if you're not using a GPU
    elements_per_block = GGML_ELEMENTS_PER_BLOCK[ggml_name]
    block_size = GGML_BLOCK_SIZES[ggml_name]
    size = block_size * elements_per_block * num_blocks
    target_dtype = GGML_DATA_TYPES[ggml_name]

    # Initialize the 1D np.ndarray with random data
    data = np.random.randint(1, 256, size, dtype=np.uint8)

    # Get the CPU and GPU dequantization functions
    dequant_cpu_func = globals()[f"dequantize_{ggml_name.lower()}"]
    dequant_gpu_func = globals()[f"dequantize_{ggml_name.lower()}_gpu"]

    # Perform dequantization on CPU
    res_cpu = dequant_cpu_func(data)
    res_cpu = torch.from_numpy(res_cpu)

    # Perform dequantization on GPU
    res_gpu = dequant_gpu_func(data, device=device, target_dtype=target_dtype)
    res_gpu = res_gpu.cpu().view(res_cpu.shape)

    # Check if all elements are either close or NaN in both tensors
    close_or_nan_both = torch.isclose(res_cpu, res_gpu) | (torch.isnan(res_cpu) & torch.isnan(res_gpu))
    all_elements_close = close_or_nan_both.all()

    # Print "Pass" or "Fail" based on the comparison result
    if all_elements_close:
        print(f"Pass for {ggml_name}")
    else:
        print(f"Fail for {ggml_name}")
        # Print the indices and the values from both tensors
        not_close_and_not_nan_both = ~close_or_nan_both
        differing_indices = not_close_and_not_nan_both.nonzero(as_tuple=False)
        cpu_values = res_cpu[not_close_and_not_nan_both]
        gpu_values = res_gpu[not_close_and_not_nan_both]
        for idx, cpu_val, gpu_val in zip(differing_indices, cpu_values, gpu_values):
            print(f"Index: {idx}, CPU value: {cpu_val}, GPU value: {gpu_val}")

    assert all_elements_close, f"Dequantization failed for {ggml_name}"

