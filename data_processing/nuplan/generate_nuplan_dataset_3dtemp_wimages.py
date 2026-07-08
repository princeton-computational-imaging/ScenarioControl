from pathlib import Path
import pickle
import hydra
import os
from cfgs.config import CONFIG_PATH
import yaml
import shutil
from tqdm import tqdm
import random
import gzip

def find_temporal_feature_paths(root_path, feature_name):
    """Find all paths to temporal sequence feature files based on actual structure."""  
    sequence_paths = []  
      
    for log_path in root_path.iterdir():  
        if not log_path.is_dir() or log_path.name == "metadata":
            continue
        for scenario_type_path in log_path.iterdir():  
            for sequence_id_path in scenario_type_path.iterdir():  
                # Look for frame files in the sequence directory  
                frame_files = list(sequence_id_path.glob(f"{feature_name}_frame_*.gz"))  
                  
                if frame_files:  
                    # Sort by frame number to maintain order  
                    frame_files.sort(key=lambda x: int(x.stem.split('_')[-1]))  
                      
                    sequence_paths.append({  
                        'sequence_id': sequence_id_path.name,  # unique identifier for the sequence (token)
                        'log_name': log_path.name,  
                        'scenario_type': scenario_type_path.name,  
                        'frame_files': frame_files,  
                        'base_path': sequence_id_path,  
                        'num_frames': len(frame_files)  
                    })  
    
    return sequence_paths


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config3dtemp")
def main(cfg):
    gcfg = cfg.generate_nuplan_dataset_temp
    cam_order = list(gcfg.cam_order)
    image_root = gcfg.image_root
    subsample_ratio = gcfg.subsample_ratio
    subsample = gcfg.subsample
    filter_cam = gcfg.filter_cam
    mode = f"{len(cam_order)}cam_{subsample_ratio}xsubsampled"  # "nofilter", "8cam"
    out_data_folder = f"scenario_control_nuplan_temp_{mode}"

    print("Creating train/val/test splits for temporal SLEDGE autoencoder cache files...")
    autoencoder_cache_path = os.path.join(gcfg.sledge_cache_root, gcfg.sledge_preprocessed_data_folder)
    path = Path(autoencoder_cache_path)
      
    print(f"Searching for temporal sequence files in {path}...")  
    sequence_paths = find_temporal_feature_paths(path, 'sledge_raw')  
    print(f"Number of sequences found: {len(sequence_paths)}")  
      
    # Load train/val/test splits  
    yaml_file_path = os.path.join(cfg.project_root, 'metadata/sledge_files/nuplan.yaml')  
    with open(yaml_file_path, 'r') as yaml_file:  
        yaml_data = yaml.safe_load(yaml_file)  
      
    log_splits = yaml_data.get('log_splits', {})  
    train_dirs = log_splits.get('train', [])  
    val_dirs = log_splits.get('val', [])  
    test_dirs = log_splits.get('test', [])  
  
    # Create destination directories  
    train_dest_sledge_raw = os.path.join(cfg.dataset_root, f'{out_data_folder}/sledge_raw/train')  
    val_dest_sledge_raw = os.path.join(cfg.dataset_root, f'{out_data_folder}/sledge_raw/val')  
    test_dest_sledge_raw = os.path.join(cfg.dataset_root, f'{out_data_folder}/sledge_raw/test')  
    train_dest_map_id = os.path.join(cfg.dataset_root, f'{out_data_folder}/map_id/train')  
    val_dest_map_id = os.path.join(cfg.dataset_root, f'{out_data_folder}/map_id/val')  
    test_dest_map_id = os.path.join(cfg.dataset_root, f'{out_data_folder}/map_id/test')  
  
    # Create directories  
    for dest_dir in [train_dest_sledge_raw, val_dest_sledge_raw, test_dest_sledge_raw,   
                     train_dest_map_id, val_dest_map_id, test_dest_map_id]:  
        os.makedirs(dest_dir, exist_ok=True)  
  
    train_sequences = []  
    val_sequences = []  
    test_sequences = []  
    skip_count = 0
    left_sequences = [] 
      
    for sequence_info in tqdm(sequence_paths):  
        log_name = sequence_info['log_name']  
        sequence_id = sequence_info['sequence_id']  
        frame_files = sequence_info['frame_files']  
          
        # Determine split based on log name  
        if log_name in train_dirs:  
            split_list = train_sequences  
        elif log_name in val_dirs:  
            split_list = val_sequences  
        elif log_name in test_dirs:  
            split_list = test_sequences  
        else:  
            split_list = left_sequences
          
        # Validate sequence by checking first frame  
        if frame_files:  
            try:  
                with gzip.open(frame_files[0], 'rb') as f:  
                    data = pickle.load(f)  
                  
                # Check camera availability  
                if filter_cam:
                    cam_infos = data.get('cam_infos', {})
                    file_exists = []
                    for value in cam_infos.values():
                        filename_jpg = value.get('filename_jpg', '')
                        file_exists.append(os.path.exists(os.path.join(image_root, filename_jpg)))
                    
                    if False in file_exists or 'CAM_F0' not in cam_infos:
                        #print(f"Skipping sequence {sequence_id} due to missing camera files or incorrect number of cameras.")
                        skip_count += 1  
                        continue  

                split_list.append(sequence_info)  
                  
            except Exception as e:  
                print(f"Error validating sequence {sequence_id}: {e}")  
                continue  
  
    print(f"Train sequences: {len(train_sequences)}")  
    print(f"Val sequences: {len(val_sequences)}")  
    print(f"Test sequences: {len(test_sequences)}")
    print(f"Left sequences: {len(left_sequences)}")
    print(f"Skipped sequences due to missing camera files or errors: {skip_count}")

    def copy_sequence_files(sequence_list, sledge_raw_destination, map_id_destination):  
        """Copy temporal sequence files to destination."""  
        for sequence_info in tqdm(sequence_list):  
            sequence_id = sequence_info['sequence_id']  
            frame_files = sequence_info['frame_files']  
            base_path = sequence_info['base_path']  
            
            # Create sequence-specific destination directory  
            sequence_dest_dir = os.path.join(sledge_raw_destination, sequence_id)  
            os.makedirs(sequence_dest_dir, exist_ok=True)  
            
            if subsample:
                frame_files = frame_files[::subsample_ratio]
            # Copy all sledge_raw frame files  
            for frame_file in frame_files:  
                dest_path = os.path.join(sequence_dest_dir, frame_file.name)  
                shutil.copy(str(frame_file), dest_path)  
            
            # Copy map_id.gz file if it exists  
            map_id_file = base_path / "map_id.gz"  
            if map_id_file.exists():  
                map_id_dest_dir = os.path.join(map_id_destination, sequence_id)  
                os.makedirs(map_id_dest_dir, exist_ok=True)  
                dest_path = os.path.join(map_id_dest_dir, "map_id.gz")  
                shutil.copy(str(map_id_file), dest_path)  
  
    # Copy files for each split  
    copy_sequence_files(train_sequences, train_dest_sledge_raw, train_dest_map_id)  
    copy_sequence_files(val_sequences, val_dest_sledge_raw, val_dest_map_id)  
    copy_sequence_files(test_sequences, test_dest_sledge_raw, test_dest_map_id)

    # Copy metadata json file
    metadata_json_path = os.path.join(path, 'sequence_token_mapping.json')
    if os.path.exists(metadata_json_path):
        shutil.copy(metadata_json_path, os.path.join(cfg.dataset_root, f'{out_data_folder}/metadata.json'))

    # Create dataset dictionary with sequence information (replaces the nuplan_dataset_dict section)  
    nuplan_dataset_dict = {  
        'train_sequences': [seq['sequence_id'] for seq in train_sequences],  
        'val_sequences': [seq['sequence_id'] for seq in val_sequences],  
        'test_sequences': [seq['sequence_id'] for seq in test_sequences],  
        'len_train_sequences': len(train_sequences),  
        'len_val_sequences': len(val_sequences),  
        'len_test_sequences': len(test_sequences),  
        'total_train_frames': sum(seq['num_frames'] for seq in train_sequences),  
        'total_val_frames': sum(seq['num_frames'] for seq in val_sequences),  
        'total_test_frames': sum(seq['num_frames'] for seq in test_sequences),  
    }  
    
    with open(os.path.join(cfg.project_root, 'metadata', f'nuplan_temp_{mode}_dataset.pkl'), 'wb') as f:  
        pickle.dump(nuplan_dataset_dict, f)  
    
    print("Done.")

    # Create nuplan eval set for temporal sequences  
    print("Creating nuplan eval set for temporal sequences (for computing metrics)...")  
    random.seed(42)  
    
    # Updated path to match your new temporal structure  
    map_id_path = os.path.join(cfg.dataset_root, f'{out_data_folder}', 'map_id', 'test')
    sequence_dirs = os.listdir(map_id_path)

    num_map_ids = cfg.ae.dataset.num_map_ids  # number of cities in the nuPlan dataset
    test_sequences_dict = {i: [] for i in range(num_map_ids)}

    for sequence_dir in tqdm(sequence_dirs):
        sequence_dir_path = os.path.join(map_id_path, sequence_dir)  
        if os.path.isdir(sequence_dir_path):  
            # Look for map_id.gz file in the sequence directory  
            map_id_file = os.path.join(sequence_dir_path, 'map_id.gz')  
            if os.path.exists(map_id_file):  
                try:  
                    with gzip.open(map_id_file, 'rb') as f:  
                        map_id_dict = pickle.load(f)  
                    test_sequences_dict[map_id_dict['id'].item()].append(sequence_dir)  
                except Exception as e:  
                    print(f"Error reading map_id for sequence {sequence_dir}: {e}")  
    
    for i in range(num_map_ids):
        random.shuffle(test_sequences_dict[i])

    print("Number of sequences in each city:",
        [len(test_sequences_dict[i]) for i in range(num_map_ids)])

    # Adjust the number based on available sequences (you may not have 12500 sequences per city)
    sequences_per_city = min(12500, min(len(test_sequences_dict[i]) for i in range(num_map_ids)))
    nuplan_test_sequences = []
    for i in range(num_map_ids):
        nuplan_test_sequences += test_sequences_dict[i][:sequences_per_city]

    # Shuffle to ensure sequences are not ordered by city
    random.shuffle(nuplan_test_sequences)  
    
    nuplan_test_dict = {  
        'sequences': nuplan_test_sequences,  
        'sequences_per_city': sequences_per_city,  
        'total_sequences': len(nuplan_test_sequences)  
    }  
    
    with open(os.path.join(cfg.project_root, 'metadata', f'nuplan_temp_{mode}_eval_set.pkl'), 'wb') as f:  
        pickle.dump(nuplan_test_dict, f)  
    
    print(f"Created evaluation set with {len(nuplan_test_sequences)} temporal sequences.")  
    print("Done.")


if __name__ == "__main__":
    main()