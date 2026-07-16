import os
import torch
from safetensors import safe_open
import folder_paths
import json

class AdvancedModelInspector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_file": (folder_paths.get_filename_list("diffusion_models"), ),
                "inspection_depth": (["summary", "detailed", "full"], ),
                "show_layer_ranges": ("BOOLEAN", {"default": True}),
                "calculate_memory": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_info", "layer_structure")
    FUNCTION = "inspect"
    CATEGORY = "experimental/analysis"

    def inspect(self, model_file, inspection_depth, show_layer_ranges, calculate_memory):
        model_path = folder_paths.get_full_path("diffusion_models", model_file)
        
        if not os.path.exists(model_path):
            return ("Error: Model file not found", "")
        
        # Open model safely without loading into RAM
        with safe_open(model_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            
        total_params = 0
        layer_stats = {}
        tensor_details = []
        
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                num_params = tensor.numel()
                total_params += num_params
                
                # Extract layer info
                parts = key.split('.')
                if len(parts) > 1:
                    layer_type = parts[0]
                    if layer_type not in layer_stats:
                        layer_stats[layer_type] = {'count': 0, 'params': 0, 'tensors': []}
                    layer_stats[layer_type]['count'] += 1
                    layer_stats[layer_type]['params'] += num_params
                    layer_stats[layer_type]['tensors'].append({
                        'key': key,
                        'shape': list(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'params': num_params
                    })
        
        # Calculate memory
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        param_memory_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
        
        # Build summary
        info_text = f"=== MODEL INSPECTION REPORT ===\n\n"
        info_text += f"File: {model_file}\n"
        info_text += f"Size: {file_size_mb:.2f} MB\n"
        info_text += f"Total Parameters: {total_params:,} ({total_params/1e9:.2f}B)\n"
        info_text += f"Parameter Memory: {param_memory_mb:.2f} MB (float32)\n"
        info_text += f"Total Tensors: {len(keys)}\n\n"
        
        if calculate_memory:
            info_text += "=== MEMORY REQUIREMENTS ===\n"
            info_text += f"Disk Space: {file_size_mb:.2f} MB\n"
            info_text += f"RAM for Loading (BF16): {param_memory_mb * 0.5:.2f} MB\n"
            info_text += f"RAM for Loading (FP16): {param_memory_mb:.2f} MB\n"
            info_text += f"RAM for Loading (FP32): {param_memory_mb * 2:.2f} MB\n"
            info_text += f"Recommended System RAM: {param_memory_mb * 3 / 1024:.2f} GB+\n\n"
        
        # Layer structure
        info_text += "=== LAYER DISTRIBUTION ===\n"
        for layer_type, stats in sorted(layer_stats.items()):
            info_text += f"{layer_type}: {stats['count']} tensors, {stats['params']:,} params\n"
        
        layer_text = ""
        if inspection_depth in ["detailed", "full"] and show_layer_ranges:
            layer_text += "=== TENSOR DETAILS ===\n\n"
            
            # Group by transformer blocks
            block_ranges = {}
            for key in keys:
                if 'transformer_blocks.' in key or 'blocks.' in key:
                    # Extract block number
                    import re
                    match = re.search(r'(?:transformer_)?blocks\.(\d+)', key)
                    if match:
                        block_num = int(match.group(1))
                        if block_num not in block_ranges:
                            block_ranges[block_num] = []
                        block_ranges[block_num].append(key)
            
            if block_ranges:
                layer_text += f"Transformer Blocks: {min(block_ranges.keys())} to {max(block_ranges.keys())}\n"
                layer_text += f"Total Blocks: {len(block_ranges)}\n\n"
                
                if inspection_depth == "full":
                    # Show first and last block details
                    for block_num in sorted(block_ranges.keys())[:2]:
                        layer_text += f"\n--- Block {block_num} ---\n"
                        for key in sorted(block_ranges[block_num]):
                            with safe_open(model_path, framework="pt", device="cpu") as f:
                                if key in f.keys():
                                    tensor = f.get_tensor(key)
                                    layer_text += f"  {key}\n"
                                    layer_text += f"    Shape: {list(tensor.shape)}\n"
                                    layer_text += f"    Dtype: {tensor.dtype}\n"
                                    layer_text += f"    Params: {tensor.numel():,}\n"
                    
                    layer_text += f"\n... ({len(block_ranges) - 4} blocks omitted) ...\n"
                    
                    for block_num in sorted(block_ranges.keys())[-2:]:
                        layer_text += f"\n--- Block {block_num} ---\n"
                        for key in sorted(block_ranges[block_num]):
                            with safe_open(model_path, framework="pt", device="cpu") as f:
                                if key in f.keys():
                                    tensor = f.get_tensor(key)
                                    layer_text += f"  {key}\n"
                                    layer_text += f"    Shape: {list(tensor.shape)}\n"
                                    layer_text += f"    Dtype: {tensor.dtype}\n"
                                    layer_text += f"    Params: {tensor.numel():,}\n"
        
        return (info_text, layer_text)

NODE_CLASS_MAPPINGS = {
    "AdvancedModelInspector": AdvancedModelInspector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedModelInspector": "Advanced Model Inspector"
}