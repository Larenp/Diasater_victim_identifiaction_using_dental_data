# labels.py
import os
import random
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# === CONFIG ===
# IMPORTANT: Adjust these paths for your environment
# Use relative paths from your project root (where labels.py is located)
input_base_dir = "cropped_parts" # Where extraction.py saved cropped images
output_csv_train = "train_online_mining.csv" # For training (online mining)
output_csv_val_pairs = "val_balanced_pairs.csv" # For validation metrics

regions = ['gum', 'upper', 'lower', 'full']
# Target number of pairs for the BALANCED validation set
num_val_positive_pairs = 1000 # Adjust based on your data size
num_val_negative_pairs = 1000 # Adjust based on your data size

# === 🧠 Helper: Get image paths and person IDs ===
def get_all_image_data(base_dir, regions):
    all_image_data = [] # List of (relative_path, person_id)
    for region in regions:
        region_path = os.path.join(base_dir, region)
        if not os.path.isdir(region_path):
            print(f"⚠️ Skipping missing region: {region_path}")
            continue
        for filename in os.listdir(region_path):
            if filename.endswith('.png'):
                person_id = os.path.splitext(filename)[0]
                relative_path = os.path.join(region, filename)
                full_path = os.path.join(base_dir, relative_path)
                if os.path.exists(full_path): # Double check existence
                    all_image_data.append({'path': relative_path, 'person_id': person_id})
    return pd.DataFrame(all_image_data)

# === Main Execution ===
if __name__ == "__main__":
    print(f"Collecting image data from: {os.path.join(os.getcwd(), input_base_dir)}")
    all_image_df = get_all_image_data(input_base_dir, regions)
    
    if all_image_df.empty:
        raise Exception("❌ No image data found. Please ensure 'extraction.py' ran correctly and paths are valid.")
    
    print(f"Found {len(all_image_df)} total images from {len(all_image_df['person_id'].unique())} unique persons.")

    # --- Step 1: Create train_online_mining.csv (All images for online mining) ---
    # This CSV simply lists all images and their associated person_ids.
    # It will be read by SiameseDataset for the training DataLoader.
    all_image_df.to_csv(output_csv_train, index=False)
    print(f"✅ All image data saved to {os.path.join(os.getcwd(), output_csv_train)} for training.")

    # --- Step 2: Create val_balanced_pairs.csv (Balanced pairs for validation metrics) ---
    # We need to ensure we have enough people to split for validation without overlap.
    unique_pids = all_image_df['person_id'].unique()
    if len(unique_pids) < 5: # Need at least a few for reasonable split
        print("Warning: Very few unique persons found. Validation split might be challenging.")

    # GroupShuffleSplit to split persons, ensuring no person appears in both train_pids and val_pids
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) # 20% of persons for validation
    
    # We apply the split on the unique person IDs to ensure disjoint sets of persons
    train_person_indices, val_person_indices = next(gss.split(unique_pids, groups=unique_pids))
    train_pids_for_val_gen = unique_pids[train_person_indices] # Use some train pids to generate val data
    val_pids_for_val_gen = unique_pids[val_person_indices]

    if len(val_pids_for_val_gen) < 2:
        raise Exception("❌ Not enough unique persons for a meaningful validation set after splitting. Consider reducing 'test_size' in GroupShuffleSplit or adding more data.")

    # Filter image data based on these PIDs for validation pair generation
    val_image_df = all_image_df[all_image_df['person_id'].isin(val_pids_for_val_gen)]
    
    # --- Generate Positive Pairs for Validation ---
    positive_val_pairs = []
    # Create a dict from val_image_df to quickly get all paths for a person
    person_val_paths = val_image_df.groupby('person_id')['path'].apply(list).to_dict()

    for pid in val_pids_for_val_gen:
        paths_for_pid = person_val_paths.get(pid, [])
        if len(paths_for_pid) >= 2:
            for i in range(len(paths_for_pid)):
                for j in range(i + 1, len(paths_for_pid)):
                    positive_val_pairs.append((paths_for_pid[i], paths_for_pid[j], 1))

    random.shuffle(positive_val_pairs)
    positive_val_pairs = positive_val_pairs[:num_val_positive_pairs]
    print(f"Generated {len(positive_val_pairs)} positive validation pairs.")

    # --- Generate Negative Pairs for Validation ---
    negative_val_pairs = set()
    all_val_paths = val_image_df['path'].tolist()

    if len(all_val_paths) < 2:
        raise Exception("❌ Not enough images in validation split to form negative pairs.")

    while len(negative_val_pairs) < num_val_negative_pairs:
        path1_rel, path2_rel = random.sample(all_val_paths, 2)
        pid1 = os.path.basename(path1_rel).split('.')[0]
        pid2 = os.path.basename(path2_rel).split('.')[0]

        if pid1 != pid2:
            # Use sorted tuple to avoid duplicate (pathA, pathB) and (pathB, pathA)
            pair_key = tuple(sorted([path1_rel, path2_rel]))
            negative_val_pairs.add((pair_key[0], pair_key[1], 0))

    print(f"Generated {len(negative_val_pairs)} negative validation pairs.")

    # Combine and save balanced validation pairs
    all_val_pairs = positive_val_pairs + list(negative_val_pairs)
    random.shuffle(all_val_pairs)

    val_df = pd.DataFrame(all_val_pairs, columns=['path1', 'path2', 'label'])
    val_df.to_csv(output_csv_val_pairs, index=False)
    print(f"✅ Balanced validation pairs saved to {os.path.join(os.getcwd(), output_csv_val_pairs)}.")

    print("All CSV generation complete.")