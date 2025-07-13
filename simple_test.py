#!/usr/bin/env python3

# Simple test to verify our understanding of the model structure
import tempfile
import json
import requests

def simple_config_test():
    print("🧪 Testing configuration download and parsing...")
    
    url = "https://huggingface.co/mlx-community/gemma-3n-E2B-it-4bit/raw/main/config.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        config = response.json()
        
        print("✅ Successfully downloaded config.json")
        print(f"📊 Config size: {len(json.dumps(config))} characters")
        print(f"🏷️  Model type: {config.get('model_type')}")
        
        # Check if there's a vision_config
        has_vision_config = 'vision_config' in config
        has_text_config = 'text_config' in config
        has_audio_config = 'audio_config' in config
        
        print(f"📋 Configuration structure:")
        print(f"  - Has vision_config: {has_vision_config}")
        print(f"  - Has text_config: {has_text_config}")
        print(f"  - Has audio_config: {has_audio_config}")
        
        if has_audio_config:
            audio = config['audio_config']
            print(f"  🔊 Audio config type: {audio.get('model_type')}")
            print(f"  🔊 Audio hidden size: {audio.get('hidden_size')}")
        
        # Check quantization structure for vision clues
        if 'quantization' in config:
            quant = config['quantization']
            vision_keys = [k for k in quant.keys() if 'vision' in k.lower()]
            print(f"  👁️  Vision-related quantization keys: {len(vision_keys)}")
            if vision_keys:
                print(f"     Examples: {vision_keys[:3]}")
        
        print("\n✅ Test completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    simple_config_test()