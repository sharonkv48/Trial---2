import os
import shutil
import random

def prepare_balanced_dataset(base_path, output_path):
    # Mapping your folder names
    folders = {
        'bank': 'Input_bank_documents',
        'invoice': 'Input_invoice_documents',
        'loan': 'Input_loan_documents'
    }
    
    # 1. Create Output Structure
    for split in ['train', 'test']:
        for cat in folders.keys():
            os.makedirs(os.path.join(output_path, split, cat), exist_ok=True)

    # 2. Split and Move
    for cat, folder_name in folders.items():
        src_path = os.path.join(base_path, folder_name)
        files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        random.shuffle(files)
        
        # Calculate 60% for training
        split_idx = int(len(files) * 0.6)
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        # Copy originals to train/test
        for f in train_files:
            shutil.copy(os.path.join(src_path, f), os.path.join(output_path, 'train', cat, f))
        for f in test_files:
            shutil.copy(os.path.join(src_path, f), os.path.join(output_path, 'test', cat, f))

    # 3. Balancing (Duplication)
    # We find the category with the most training files (Loan has 16)
    train_root = os.path.join(output_path, 'train')
    cat_counts = {cat: len(os.listdir(os.path.join(train_root, cat))) for cat in folders.keys()}
    target_count = max(cat_counts.values())
    
    print(f"Targeting {target_count} files per category for training...")

    for cat in folders.keys():
        cat_train_path = os.path.join(train_root, cat)
        current_files = os.listdir(cat_train_path)
        current_count = len(current_files)
        
        if current_count < target_count:
            needed = target_count - current_count
            print(f"Adding {needed} duplicates to {cat}")
            for i in range(needed):
                file_to_copy = current_files[i % current_count]
                name, ext = os.path.splitext(file_to_copy)
                shutil.copy(
                    os.path.join(cat_train_path, file_to_copy),
                    os.path.join(cat_train_path, f"{name}_bal_{i}{ext}")
                )

    print("\nProcess Complete!")
    print(f"Train set: {target_count * 3} files (Balanced)")
    print(f"Test set: {50 - sum(cat_counts.values())} files")

# Execution
prepare_balanced_dataset('input_documents', 'azure_training_ready')
