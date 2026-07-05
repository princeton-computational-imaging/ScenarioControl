CODE_DIR="$PROJECT_ROOT/data_processing/nuplan"

cd "$CODE_DIR"
python preprocess_dataset_nuplan_3dtemp_wimages.py dataset_name=nuplan preprocess_nuplan.mode=train ae.dataset.load_images=True
python preprocess_dataset_nuplan_3dtemp_wimages.py dataset_name=nuplan preprocess_nuplan.mode=val ae.dataset.load_images=True
python preprocess_dataset_nuplan_3dtemp_wimages.py dataset_name=nuplan preprocess_nuplan.mode=test ae.dataset.load_images=True