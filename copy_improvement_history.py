#!/usr/bin/env python3
"""
Script to copy all files ending in '-improvement_history.jsonl' from 
final-results/<model>/attack to replay/<model>/attack directories.
"""

import os
import shutil
from pathlib import Path

def copy_improvement_history_files():
    """Copy improvement history files from final-results to replay directories."""
    
    # Base directories
    base_dir = Path(__file__).parent
    final_results_dir = base_dir / "final-results"
    replay_dir = base_dir / "data" /"replay"
    
    # Ensure replay directory exists
    replay_dir.mkdir(exist_ok=True)
    
    # Track statistics
    total_files_copied = 0
    models_processed = []
    
    # Process each model directory in final-results
    for model_dir in final_results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        attack_dir = model_dir / "attack"
        
        # Skip if attack directory doesn't exist
        if not attack_dir.exists():
            print(f"⚠️  Skipping {model_name}: no attack directory found")
            continue
            
        # Create corresponding directories in replay
        replay_model_dir = replay_dir / model_name / "attack"
        replay_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and copy improvement history files
        files_copied = 0
        for file_path in attack_dir.glob("*-improvement_history.jsonl"):
            dest_path = replay_model_dir / file_path.name
            
            try:
                shutil.copy2(file_path, dest_path)
                files_copied += 1
                print(f"✅ Copied: {file_path.name}")
            except Exception as e:
                print(f"❌ Error copying {file_path.name}: {e}")
        
        if files_copied > 0:
            models_processed.append(model_name)
            total_files_copied += files_copied
            print(f"📁 {model_name}: {files_copied} files copied")
        else:
            print(f"⚠️  {model_name}: no improvement history files found")
    
    # Summary
    print("\n" + "="*50)
    print("📊 COPY SUMMARY")
    print("="*50)
    print(f"Models processed: {len(models_processed)}")
    print(f"Total files copied: {total_files_copied}")
    
    if models_processed:
        print(f"Models: {', '.join(models_processed)}")
    
    print(f"Destination directory: {replay_dir}")

if __name__ == "__main__":
    copy_improvement_history_files()
