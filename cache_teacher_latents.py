import os
import argparse
import importlib
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.datasets import get_dataset

def main():
    parser = argparse.ArgumentParser(description='Cache SUTrack Latent Representations for Distillation')
    parser.add_argument('--tracker_name', type=str, default='sutrack', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='sutrack_ti_base', help='Name of parameter file.')
    parser.add_argument('--dataset_name', type=str, default='uav', help='Name of dataset (e.g., uav).')
    parser.add_argument('--output_dir', type=str, default='teacher_latents', help='Directory to save latents.')
    args = parser.parse_args()

    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    dataset_out_dir = os.path.join(output_dir, args.dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)

    # Initialize tracker wrapper
    tracker = Tracker(args.tracker_name, args.tracker_param, args.dataset_name)
    
    # Get parameters and create actual tracker instance
    params = tracker.get_parameters()
    # Ensure network only does forward pass without grad
    params.debug = 0 
    
    tracker_instance = tracker.create_tracker(params)
    network = tracker_instance.network
    network.eval()
    for param in network.parameters():
        param.requires_grad = False
        
    print(f"Loaded SUTrack model: {args.tracker_name} / {args.tracker_param}")

    # Setup Forward Hook on the final encoder block
    latent_cache = {}
    
    def get_latents_hook(module, input, output):
        # The output of encoder blocks is typically a tensor of shape [B, N, C]
        # For one-stream trackers, we need to locate where the search tokens start.
        # usually xz = torch.cat([z, x], dim=1)
        # We will dynamically calculate template length based on num_patch_z
        # For SUTrack base: template uses 128x128 with 16x16 patch -> 64 tokens + cls token if present
        
        num_patch_z = network.num_patch_z
        
        # Determine offset for search tokens
        # If cls token exists (e.g. at index 0), template is [1:num_patch_z+1], search is [num_patch_z+1:]
        # If no cls token, template is [0:num_patch_z], search is [num_patch_z:]
        
        start_idx = num_patch_z
        if getattr(network, 'class_token', False):
             start_idx = num_patch_z + 1
             
        # Extract only search region tokens
        search_features = output[:, start_idx:, :].detach().cpu()
        
        # Store in cache
        latent_cache['search_features'] = search_features

    # Register hook on the last block of the encoder
    # SUTrack uses a ViT body, blocks are usually in network.encoder.body.blocks
    target_layer = network.encoder.body.blocks[-1]
    hook_handle = target_layer.register_forward_hook(get_latents_hook)
    
    print("Registered forward hook on the final encoder block.")

    # Get dataset
    dataset = get_dataset(args.dataset_name)
    print(f"Loaded dataset: {args.dataset_name} with {len(dataset)} sequences.")

    # Process each sequence
    for seq in dataset:
        seq_name = seq.name
        seq_out_dir = os.path.join(dataset_out_dir, seq_name)
        os.makedirs(seq_out_dir, exist_ok=True)
        
        print(f"Processing sequence: {seq_name} ({len(seq.frames)} frames)")
        
        init_info = seq.init_info()
        image = tracker._read_image(seq.frames[0])
        init_info['seq_name'] = seq.name
        
        # Initialize tracking (this sets up the template)
        tracker_instance.initialize(image, init_info)
        
        # Save template latent just in case it's needed later (optional)
        # Note: hook will be called during initialize as well
        if 'search_features' in latent_cache:
            init_latent_path = os.path.join(seq_out_dir, f"{0:06d}.pt")
            torch.save(latent_cache['search_features'], init_latent_path)

        # Track rest of sequence
        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = tracker._read_image(frame_path)
            info = seq.frame_info(frame_num)
            
            # Forward pass happens here
            # tracker_instance.track returns dict with bounding box, but hook captures latent
            _ = tracker_instance.track(image, info)
            
            # Save latent
            if 'search_features' in latent_cache:
                latent_path = os.path.join(seq_out_dir, f"{frame_num:06d}.pt")
                torch.save(latent_cache['search_features'], latent_path)
            else:
                print(f"Warning: No latent captured for frame {frame_num} in {seq_name}")
                
    hook_handle.remove()
    print("Finished caching latents.")

if __name__ == '__main__':
    main()
