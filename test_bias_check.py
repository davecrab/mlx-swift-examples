#!/usr/bin/env python3
"""
Simple script to check if the model has bias parameters
"""

import requests
import json

def check_bias_in_quantization():
    """Check the quantization config to see which parameters exist"""
    print("🔍 Checking bias parameters in Gemma3n quantization config...")
    
    url = "https://huggingface.co/mlx-community/gemma-3n-E2B-it-4bit/raw/main/config.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        config = response.json()
        
        if 'quantization' in config:
            quant = config['quantization']
            print(f"✅ Found quantization config with {len(quant)} entries")
            
            # Look for bias-related parameters
            bias_params = {}
            conv_params = {}
            
            for key, value in quant.items():
                if 'bias' in key.lower():
                    bias_params[key] = value
                elif 'conv' in key.lower():
                    conv_params[key] = value
            
            print(f"\n🎯 Bias parameters in quantization config:")
            if bias_params:
                for key, value in bias_params.items():
                    print(f"  {key}: {value}")
                print(f"  🚨 FOUND {len(bias_params)} bias parameters!")
            else:
                print(f"  ✅ No bias parameters found - all Conv2d should use bias=false")
            
            print(f"\n🔧 Conv parameters in quantization config:")
            if conv_params:
                for key, value in list(conv_params.items())[:5]:  # Show first 5
                    print(f"  {key}: {value}")
                if len(conv_params) > 5:
                    print(f"  ... and {len(conv_params) - 5} more")
            else:
                print(f"  ❌ No conv parameters found")
                
        else:
            print(f"❌ No quantization config found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_bias_in_quantization()