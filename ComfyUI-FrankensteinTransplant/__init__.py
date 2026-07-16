import os
import torch
import json
from safetensors.torch import load_file, save_file
import folder_paths

# --- Internal Mapping Configurations ---
COMPONENT_MAPS = {
    "krea_to_qwen": {
        "attn_to_q": ("transformer_blocks.{t}.attn.to_q.weight", "blocks.{d}.attn.wq.weight"),
        "attn_to_k": ("transformer_blocks.{t}.attn.to_k.weight", "blocks.{d}.attn.wk.weight"),
        "attn_to_v": ("transformer_blocks.{t}.attn.to_v.weight", "blocks.{d}.attn.wv.weight"),
        "attn_out": ("transformer_blocks.{t}.attn.to_out.0.weight", "blocks.{d}.attn.wo.weight"),
        "mlp_up": ("transformer_blocks.{t}.img_mlp.net.0.proj.weight", "blocks.{d}.mlp.up.weight"),
        "mlp_down": ("transformer_blocks.{t}.img_mlp.net.2.weight", "blocks.{d}.mlp.down.weight"),
    },
    "qwen_to_krea": {
        "attn_wq": ("blocks.{t}.attn.wq.weight", "transformer_blocks.{d}.attn.to_q.weight"),
        "attn_wk": ("blocks.{t}.attn.wk.weight", "transformer_blocks.{d}.attn.to_k.weight"),
        "attn_wv": ("blocks.{t}.attn.wv.weight", "transformer_blocks.{d}.attn.to_v.weight"),
        "attn_wo": ("blocks.{t}.attn.wo.weight", "transformer_blocks.{d}.attn.to_out.0.weight"),
        "mlp_up": ("blocks.{t}.mlp.up.weight", "transformer_blocks.{d}.img_mlp.net.0.proj.weight"),
        "mlp_down": ("blocks.{t}.mlp.down.weight", "transformer_blocks.{d}.img_mlp.net.2.weight"),
    },
    "flux_to_flux": {
        "img_attn_qkv": ("double_blocks.{t}.img_attn.qkv.weight", "double_blocks.{d}.img_attn.qkv.weight"),
        "img_attn_proj": ("double_blocks.{t}.img_attn.proj.weight", "double_blocks.{d}.img_attn.proj.weight"),
        "img_mlp_up": ("double_blocks.{t}.img_mlp.0.weight", "double_blocks.{d}.img_mlp.0.weight"),
        "img_mlp_down": ("double_blocks.{t}.img_mlp.2.weight", "double_blocks.{d}.img_mlp.2.weight"),
        "single_lin1": ("single_blocks.{t}.linear1.weight", "single_blocks.{d}.linear1.weight"),
        "single_lin2": ("single_blocks.{t}.linear2.weight", "single_blocks.{d}.linear2.weight"),
    }
}

class FrankensteinPresetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (
                    "Krea_to_Qwen_Mid_Realism (Blocks 40-55)",
                    "Krea_to_Qwen_Late_Refinement (Blocks 50-59)",
                    "Qwen_to_Krea_Coherence (Blocks 20-27)",
                    "Flux_to_Flux_Double_Blocks (Image Attention & MLP)",
                    "Flux_to_Flux_Single_Blocks (Linear 1 & 2)",
                    "Flux_to_Flux_Full (Double & Single)",
                    "Custom_Template (Edit JSON manually)"
                ),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mapping_config_json",)
    FUNCTION = "load_preset"
    CATEGORY = "Frankenstein"

    def load_preset(self, preset):
        if preset == "Custom_Template (Edit JSON manually)":
            config = {
                "_instructions": "Set 'mode' to 'krea_to_qwen', 'qwen_to_krea', or 'flux_to_flux'. For Flux, use keys like 'double_0' or 'single_0'.",
                "mode": "flux_to_flux",
                "blocks": {
                    "double_0": {"donor": 0, "enabled": True, "ratio": 0.15},
                    "single_0": {"donor": 0, "enabled": True, "ratio": 0.15}
                }
            }
            return (json.dumps(config, indent=2),)

        blocks = {}
        if "Krea_to_Qwen_Mid" in preset:
            mode = "krea_to_qwen"
            for t in range(40, 56):
                d = 18 + int((t - 40) / 15 * 9)
                blocks[str(t)] = {"donor": d, "enabled": True, "ratio": 0.12}
        elif "Krea_to_Qwen_Late" in preset:
            mode = "krea_to_qwen"
            for t in range(50, 60):
                d = 24 + int((t - 50) / 9 * 3)
                blocks[str(t)] = {"donor": d, "enabled": True, "ratio": 0.10}
        elif "Qwen_to_Krea" in preset:
            mode = "qwen_to_krea"
            for t in range(20, 28):
                d = 45 + int((t - 20) / 7 * 14)
                blocks[str(t)] = {"donor": d, "enabled": True, "ratio": 0.10}
        elif "Flux_to_Flux_Double" in preset:
            mode = "flux_to_flux"
            for t in range(0, 38): # Covers Flux 1 Dev
                blocks[f"double_{t}"] = {"donor": t, "enabled": True, "ratio": 0.15}
        elif "Flux_to_Flux_Single" in preset:
            mode = "flux_to_flux"
            for t in range(0, 38):
                blocks[f"single_{t}"] = {"donor": t, "enabled": True, "ratio": 0.15}
        elif "Flux_to_Flux_Full" in preset:
            mode = "flux_to_flux"
            for t in range(0, 38):
                blocks[f"double_{t}"] = {"donor": t, "enabled": True, "ratio": 0.12}
                blocks[f"single_{t}"] = {"donor": t, "enabled": True, "ratio": 0.12}

        config = {
            "mode": mode,
            "global_note": "Ratio > 0.20 may cause artifacts. Start low.",
            "blocks": blocks
        }
        return (json.dumps(config, indent=2),)


class FrankensteinTransplantUltimate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The main model that keeps its architecture"}),
                "donor_model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The model to steal weights from"}),
                "mapping_config_json": ("STRING", {"default": "", "multiline": True, "tooltip": "JSON config from Preset Loader"}),
                "global_ratio_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "If > 0, overrides all individual block ratios"}),
                "output_filename": ("STRING", {"default": "frankenstein_hybrid.safetensors"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "transplant"
    CATEGORY = "Frankenstein"

    def transplant(self, target_model, donor_model, mapping_config_json, global_ratio_override, output_filename):
        try:
            config = json.loads(mapping_config_json)
        except json.JSONDecodeError as e:
            return (f"ERROR: Invalid JSON in mapping_config. Details: {e}",)

        mode = config.get("mode", "krea_to_qwen")
        if mode not in COMPONENT_MAPS:
            return (f"ERROR: Invalid mode '{mode}'.",)

        target_path = folder_paths.get_full_path("diffusion_models", target_model)
        donor_path = folder_paths.get_full_path("diffusion_models", donor_model)

        if not os.path.exists(target_path) or not os.path.exists(donor_path):
            return ("ERROR: One or both model files not found.",)

        print(f"Frankenstein: Loading Target to CPU: {target_model}", flush=True)
        print(f"Frankenstein: Loading Donor to CPU: {donor_model}", flush=True)
        
        target_weights = load_file(target_path, device="cpu")
        donor_weights = load_file(donor_path, device="cpu")
        
        new_weights = dict(target_weights)
        merged_count = 0
        skipped_count = 0
        
        component_map = COMPONENT_MAPS[mode]
        blocks_config = config.get("blocks", {})

        print(f"Frankenstein: Starting merge process for {len(blocks_config)} configured blocks...", flush=True)

        for t_str, block_info in blocks_config.items():
            if not block_info.get("enabled", False):
                continue
                
            # Parse block type and index (e.g., "double_0" -> "double", 0)
            if "_" in t_str:
                t_type, t_idx_str = t_str.split("_", 1)
                t_idx = int(t_idx_str)
            else:
                t_type = "standard"
                t_idx = int(t_str)
                
            d_idx = block_info.get("donor", t_idx)
            ratio = global_ratio_override if global_ratio_override > 0.0 else block_info.get("ratio", 0.15)
            
            for comp_name, (target_pattern, donor_pattern) in component_map.items():
                t_key = target_pattern.format(t=t_idx)
                d_key = donor_pattern.format(d=d_idx)
                
                if t_key in new_weights and d_key in donor_weights:
                    t_tensor = new_weights[t_key]
                    d_tensor = donor_weights[d_key].to(t_tensor.dtype)
                    
                    if t_tensor.dim() != d_tensor.dim():
                        skipped_count += 1
                        continue
                        
                    min_shape = tuple(min(a, b) for a, b in zip(t_tensor.shape, d_tensor.shape))
                    d_sliced = d_tensor[tuple(slice(0, dim) for dim in min_shape)]
                    
                    if d_sliced.shape != t_tensor.shape:
                        padded = torch.zeros_like(t_tensor)
                        padded[tuple(slice(0, dim) for dim in d_sliced.shape)] = d_sliced
                        d_sliced = padded
                    
                    new_weights[t_key] = (1.0 - ratio) * t_tensor + ratio * d_sliced
                    merged_count += 1
                    
                    # CRITICAL FIX: flush=True forces immediate console output
                    print(f"  Merged: {comp_name} ({t_type.capitalize()} Block {t_idx} <- {t_type.capitalize()} Block {d_idx}) @ {ratio:.2f}", flush=True)
                else:
                    skipped_count += 1

        output_dir = os.path.dirname(target_path)
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Frankenstein: Saving hybrid model to: {output_path}", flush=True)
        save_file(new_weights, output_path)
        
        del target_weights, donor_weights, new_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_msg = f"SUCCESS: Transplanted {merged_count} tensors. Skipped {skipped_count}. Saved as: {output_filename}"
        print(f"Frankenstein: {final_msg}", flush=True)
        return (final_msg,)

NODE_CLASS_MAPPINGS = {
    "FrankensteinPresetLoader": FrankensteinPresetLoader,
    "FrankensteinTransplantUltimate": FrankensteinTransplantUltimate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrankensteinPresetLoader": "🧟 Frankenstein Preset Loader",
    "FrankensteinTransplantUltimate": " Frankenstein Transplant Ultimate"
}