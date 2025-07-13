#!/usr/bin/env python3
"""
Test script to compare Gemma3n config.json structure with Swift implementation
This helps identify missing or misnamed fields that cause parsing errors.
"""

import json
import requests
from typing import Dict, Any, Set

def download_config(model_id: str) -> Dict[str, Any]:
    """Download config.json from HuggingFace model repository"""
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error downloading config: {e}")
        return {}

def extract_swift_fields() -> Dict[str, Set[str]]:
    """Extract expected field names from our Swift implementation"""
    
    # Fields from Gemma3nTextConfiguration
    text_config_fields = {
        "model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
        "num_attention_heads", "head_dim", "rms_norm_eps", "vocab_size",
        "vocab_size_per_layer_input", "num_key_value_heads", "laurel_rank",
        "frac_shared_layers", "altup_active_idx", "pad_token_id", "altup_num_inputs",
        "altup_coef_clip", "altup_correct_scale", "hidden_size_per_layer_input",
        "rope_local_base_freq", "rope_traditional", "rope_theta", "query_pre_attn_scalar",
        "sliding_window", "rope_scaling", "mm_tokens_per_image", "sliding_window_pattern",
        "activation_sparsity_pattern", "final_logit_softcapping", "query_rescale_scalar",
        "num_kv_shared_layers", "max_position_embeddings", "attn_logit_softcapping",
        "layer_types"
    }
    
    # Fields from Gemma3nVisionConfiguration
    vision_config_fields = {
        "model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "patch_size", "image_size", "num_channels",
        "rms_norm_eps", "vocab_size", "vocab_offset"
    }
    
    # Fields from Gemma3nAudioConfiguration
    audio_config_fields = {
        "input_feat_size", "hidden_size", "conf_attention_chunk_size",
        "conf_attention_context_left", "conf_attention_context_right",
        "conf_attention_logit_cap", "conf_num_attention_heads", "conf_num_hidden_layers",
        "conf_conv_kernel_size", "conf_positional_bias_size", "conf_reduction_factor",
        "conf_residual_weight", "vocab_size", "rms_norm_eps", "gradient_clipping",
        "vocab_offset"
    }
    
    # Fields from Gemma3nConfiguration (main config)
    main_config_fields = {
        "text_config", "vision_config", "audio_config", "model_type", "vocab_size",
        "ignore_index", "image_token_index", "audio_token_id", "image_token_id",
        "hidden_size", "pad_token_id", "vision_soft_tokens_per_image",
        "audio_soft_tokens_per_image", "eos_token_id"
    }
    
    return {
        "main": main_config_fields,
        "text_config": text_config_fields,
        "vision_config": vision_config_fields,
        "audio_config": audio_config_fields
    }

def analyze_config_structure(config: Dict[str, Any], prefix: str = "") -> Set[str]:
    """Recursively extract all field names from config"""
    fields = set()
    
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        fields.add(full_key)
        
        if isinstance(value, dict):
            nested_fields = analyze_config_structure(value, full_key)
            fields.update(nested_fields)
    
    return fields

def compare_configs():
    """Main comparison function"""
    model_id = "mlx-community/gemma-3n-E2B-it-4bit"
    
    print(f"Downloading config.json from {model_id}...")
    actual_config = download_config(model_id)
    
    if not actual_config:
        print("Failed to download config.json")
        return
    
    print("\n" + "="*80)
    print("ACTUAL CONFIG.JSON STRUCTURE")
    print("="*80)
    print(json.dumps(actual_config, indent=2))
    
    # Extract all field paths from actual config
    actual_fields = analyze_config_structure(actual_config)
    
    # Get expected fields from Swift implementation
    expected_fields = extract_swift_fields()
    
    print("\n" + "="*80)
    print("FIELD COMPARISON ANALYSIS")
    print("="*80)
    
    # Check main config fields
    print("\n📋 MAIN CONFIG FIELDS:")
    main_actual = {k for k in actual_config.keys()}
    main_expected = expected_fields["main"]
    
    print(f"✅ Present in both: {main_actual & main_expected}")
    print(f"❌ Missing from Swift: {main_actual - main_expected}")
    print(f"🔍 Expected but not in actual: {main_expected - main_actual}")
    
    # Check text_config if present
    if "text_config" in actual_config:
        print("\n📝 TEXT_CONFIG FIELDS:")
        text_actual = {k for k in actual_config["text_config"].keys()}
        text_expected = expected_fields["text_config"]
        
        print(f"✅ Present in both: {text_actual & text_expected}")
        print(f"❌ Missing from Swift: {text_actual - text_expected}")
        print(f"🔍 Expected but not in actual: {text_expected - text_actual}")
    
    # Check vision_config if present
    if "vision_config" in actual_config:
        print("\n🖼️  VISION_CONFIG FIELDS:")
        vision_actual = {k for k in actual_config["vision_config"].keys()}
        vision_expected = expected_fields["vision_config"]
        
        print(f"✅ Present in both: {vision_actual & vision_expected}")
        print(f"❌ Missing from Swift: {vision_actual - vision_expected}")
        print(f"🔍 Expected but not in actual: {vision_expected - vision_actual}")
    
    # Check audio_config if present
    if "audio_config" in actual_config:
        print("\n🔊 AUDIO_CONFIG FIELDS:")
        audio_actual = {k for k in actual_config["audio_config"].keys()}
        audio_expected = expected_fields["audio_config"]
        
        print(f"✅ Present in both: {audio_actual & audio_expected}")
        print(f"❌ Missing from Swift: {audio_actual - audio_expected}")
        print(f"🔍 Expected but not in actual: {audio_expected - audio_actual}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Generate Swift code suggestions
    if "text_config" in actual_config:
        print("\n🔧 Suggested Swift text_config updates:")
        text_actual = actual_config["text_config"]
        for key, value in text_actual.items():
            if key not in expected_fields["text_config"]:
                value_type = type(value).__name__
                swift_type = {
                    'int': 'Int',
                    'float': 'Float', 
                    'str': 'String',
                    'bool': 'Bool',
                    'list': '[Int]',  # Assuming most lists are Int
                    'NoneType': 'String?'
                }
                swift_var_type = swift_type.get(value_type, 'Any')
                camel_case = ''.join(word.capitalize() for word in key.split('_'))
                camel_case = camel_case[0].lower() + camel_case[1:]
                
                print(f"public let {camel_case}: {swift_var_type}")
                print(f"case {camel_case} = \"{key}\"")
    
    print("\n✨ Analysis complete! Update your Swift configuration based on the missing fields above.")

if __name__ == "__main__":
    compare_configs()