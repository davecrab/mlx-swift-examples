#!/usr/bin/env swift

import Foundation
import MLX
import MLXVLM
import MLXLMCommon

// Simple test to check if Gemma3n configuration parsing works
@main
struct TestGemma3n {
    static func main() async {
        print("🧪 Testing Gemma3n VLM configuration parsing...")
        
        do {
            let factory = VLMModelFactory.shared
            let configuration = VLMRegistry.gemma3n_2B_4bit
            
            print("📥 Attempting to load model: \(configuration.id)")
            
            // Try to load the model - this will test config.json parsing
            let modelContainer = try await factory.loadContainer(
                hub: .default,
                configuration: configuration
            ) { progress in
                print("📊 Download progress: \(progress.fractionCompleted * 100)%")
            }
            
            print("✅ Successfully loaded Gemma3n model!")
            print("🏷️  Model type: \(type(of: modelContainer.model))")
            print("🎯 Vocabulary size: \(modelContainer.model.vocabularySize)")
            
        } catch {
            print("❌ Failed to load Gemma3n model:")
            print("   Error: \(error)")
            
            if let error = error as? LocalizedError {
                print("   Description: \(error.errorDescription ?? "No description")")
            }
        }
    }
}