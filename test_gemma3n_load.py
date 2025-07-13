#!/usr/bin/env python3
"""
Test script to load the actual Gemma3n model and examine its weight structure.
This helps identify potential parameter naming mismatches.
"""

import json
from pathlib import Path
import tempfile
from huggingface_hub import snapshot_download

def examine_model_weights():
    """Download and examine the actual model weights structure"""
    print("🔍 Examining Gemma3n model weights structure...")
    
    model_id = "mlx-community/gemma-3n-E2B-it-4bit"
    
    try:
        # Download model to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📥 Downloading model {model_id}...")
            local_dir = snapshot_download(
                repo_id=model_id,
                local_dir=temp_dir,
                allow_patterns=["*.json", "*.safetensors"],
            )
            
            print(f"📁 Model downloaded to: {local_dir}")
            
            # List all files
            model_path = Path(local_dir)
            print("\n📋 Model files:")
            for file in sorted(model_path.iterdir()):
                if file.is_file():
                    print(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Examine config.json
            config_path = model_path / "config.json"
            if config_path.exists():
                print("\n⚙️  Configuration structure:")
                with open(config_path) as f:
                    config = json.load(f)
                
                # Print key configuration sections
                print(f"  Model type: {config.get('model_type', 'Unknown')}")
                print(f"  Architecture: {config.get('architectures', ['Unknown'])}")
                
                if 'audio_config' in config:
                    audio_config = config['audio_config']
                    print(f"  Audio model type: {audio_config.get('model_type', 'Unknown')}")
                    print(f"  Audio hidden size: {audio_config.get('hidden_size', 'Unknown')}")
                
                # Look for quantization info
                if 'quantization' in config:
                    quant = config['quantization']
                    print(f"  Quantization bits: {quant.get('bits', 'Unknown')}")
                    print(f"  Group size: {quant.get('group_size', 'Unknown')}")
                    
                    # Show which layers are quantized vs not quantized
                    quantized_layers = []
                    non_quantized_layers = []
                    
                    for key, value in quant.items():
                        if isinstance(value, bool):
                            if value:
                                quantized_layers.append(key)
                            else:
                                non_quantized_layers.append(key)
                    
                    print(f"\n  📊 Quantization analysis:")
                    print(f"    Quantized layers: {len(quantized_layers)}")
                    print(f"    Non-quantized layers: {len(non_quantized_layers)}")
                    
                    # Show some examples of each
                    if quantized_layers:
                        print(f"    Example quantized: {quantized_layers[:3]}")
                    if non_quantized_layers:
                        print(f"    Example non-quantized: {non_quantized_layers[:3]}")
                        
                    # Look for bias patterns
                    bias_patterns = [key for key in non_quantized_layers if 'bias' in key.lower()]
                    if bias_patterns:
                        print(f"    🚨 Bias parameters found: {bias_patterns}")
                    else:
                        print(f"    ✅ No bias parameters found - all layers should use bias=false")
            
            # Try to load with MLX Python to see if it works
            try:
                import mlx.core as mx
                import mlx.nn as nn
                from pathlib import Path
                
                print(f"\n🧪 Testing MLX Python loading...")
                
                # Try to load one of the safetensors files
                safetensor_files = list(model_path.glob("*.safetensors"))
                if safetensor_files:
                    print(f"  Found {len(safetensor_files)} safetensors files")
                    
                    # Just check if they can be read
                    for sf in safetensor_files[:1]:  # Just check first one
                        try:
                            weights = mx.load(str(sf))
                            print(f"  ✅ Successfully loaded {sf.name} with {len(weights)} parameters")
                            
                            # Show some parameter names
                            param_names = list(weights.keys())[:10]
                            print(f"  Sample parameters: {param_names}")
                            
                            # Look for bias parameters
                            bias_params = [name for name in weights.keys() if 'bias' in name.lower()]
                            if bias_params:
                                print(f"  🚨 Found bias parameters: {bias_params[:5]}")
                            else:
                                print(f"  ✅ No bias parameters found in weights")
                                
                        except Exception as e:
                            print(f"  ❌ Failed to load {sf.name}: {e}")
                else:
                    print(f"  ❌ No safetensors files found")
                    
            except ImportError:
                print(f"  ⚠️  MLX Python not available, skipping weight inspection")
                
    except Exception as e:
        print(f"❌ Error examining model: {e}")

if __name__ == "__main__":
    examine_model_weights()