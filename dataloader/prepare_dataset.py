"""Split SWC files into train, validation, and test directories."""

import argparse
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict


def find_all_swc_files(root_dir):
    swc_files_by_subfolder = defaultdict(list)
    root_path = Path(root_dir)
    

    for subfolder in root_path.iterdir():
        if not subfolder.is_dir():
            continue
            
        subfolder_name = subfolder.name
        

        for swc_file in subfolder.rglob('*.swc'):

            relative_path = swc_file.relative_to(subfolder)
            swc_files_by_subfolder[subfolder_name].append({
                'absolute': swc_file,
                'relative': relative_path,
                'subfolder': subfolder_name
            })
    
    return swc_files_by_subfolder


def split_files(files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    

    shuffled = files.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = shuffled[:train_end]
    val_files = shuffled[train_end:val_end]
    test_files = shuffled[val_end:]
    
    return train_files, val_files, test_files


def copy_files_to_split(file_infos, source_root, target_root, split_name):
    target_split_dir = Path(target_root) / split_name
    
    for file_info in file_infos:

        target_file = target_split_dir / file_info['subfolder'] / file_info['relative']
        

        target_file.parent.mkdir(parents=True, exist_ok=True)
        

        shutil.copy2(file_info['absolute'], target_file)


def prepare_dataset(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    print('=' * 60)
    print('RamiGlyph Dataset Preparation')
    print('=' * 60)
    

    random.seed(seed)
    
    root_path = Path(root_dir)
    if not root_path.exists():
        raise ValueError(f"Root directory not found: {root_dir}")
    
    print(f'\nScanning directory: {root_dir}')
    

    swc_files_by_subfolder = find_all_swc_files(root_dir)
    
    if not swc_files_by_subfolder:
        raise ValueError("No SWC files found!")
    
    print(f'\nFound {len(swc_files_by_subfolder)} subfolders:')
    for subfolder, files in swc_files_by_subfolder.items():
        print(f'  - {subfolder}: {len(files)} files')
    

    all_train_files = []
    all_val_files = []
    all_test_files = []
    
    print(f'\nSplitting files (train:{train_ratio:.0%} val:{val_ratio:.0%} test:{test_ratio:.0%}):')
    
    for subfolder, files in swc_files_by_subfolder.items():
        train_files, val_files, test_files = split_files(
            files, train_ratio, val_ratio, test_ratio
        )
        
        all_train_files.extend(train_files)
        all_val_files.extend(val_files)
        all_test_files.extend(test_files)
        
        print(f'  {subfolder}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test')
    
    total_files = len(all_train_files) + len(all_val_files) + len(all_test_files)
    print(f'\nTotal: {total_files} files')
    print(f'  Train: {len(all_train_files)} ({len(all_train_files)/total_files:.1%})')
    print(f'  Val:   {len(all_val_files)} ({len(all_val_files)/total_files:.1%})')
    print(f'  Test:  {len(all_test_files)} ({len(all_test_files)/total_files:.1%})')
    

    train_dir = root_path / 'train'
    val_dir = root_path / 'val'
    test_dir = root_path / 'test'
    
    if train_dir.exists() or val_dir.exists() or test_dir.exists():
        response = input('\nWarning: train/val/test directories already exist. Overwrite? (y/n): ')
        if response.lower() != 'y':
            print('Cancelled.')
            return
        

        for dir_path in [train_dir, val_dir, test_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
    

    print('\nCopying files...')
    
    print('  Copying train files...')
    copy_files_to_split(all_train_files, root_dir, root_dir, 'train')
    
    print('  Copying val files...')
    copy_files_to_split(all_val_files, root_dir, root_dir, 'val')
    
    print('  Copying test files...')
    copy_files_to_split(all_test_files, root_dir, root_dir, 'test')
    
    print('\n' + '=' * 60)
    print('Dataset Preparation Completed!')
    print('=' * 60)
    print(f'\nDataset structure created in: {root_dir}')
    print('  ├── train/')
    print('  ├── val/')
    print('  └── test/')
    print('\nEach split directory maintains the original subfolder structure.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root_dir', help='Directory containing grouped SWC files')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(
        root_dir=args.root_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
