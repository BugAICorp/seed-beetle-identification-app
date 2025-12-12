""" globals.py 

When adding new files to the project in the categories of: Models, Transformations, Model Trackers;
Add a variable here with the full path of the file so that the code will only show a simple variable
rather than joining the path in the code.
(Please check spelling to ensure easy access as well)
"""
training_database = "training.db"
class_list = "src/models/class_list.txt"
img_height = "src/models/height.txt"

# Species Files
spec_caud_model = "src/models/spec_caud.pth"
spec_dors_model = "src/models/spec_dors.pth"
spec_fron_model = "src/models/spec_fron.pth"
spec_late_model = "src/models/spec_late.pth"
spec_class_dictionary = "src/models/spec_dict.json"
spec_accuracy_list = "src/models/spec_accuracies.json"

# Species Files with Other
spec_caud_model_with_other = "src/models/spec_caud_other.pth"
spec_dors_model_with_other = "src/models/spec_dors_other.pth"
spec_fron_model_with_other = "src/models/spec_fron_other.pth"
spec_late_model_with_other = "src/models/spec_late_other.pth"
spec_class_dictionary_with_other = "src/models/spec_dict_other.json"
spec_accuracy_list_with_other = "src/models/spec_accuracies_other.json"

# Genus Files
gen_caud_model = "src/models/gen_caud.pth"
gen_dors_model = "src/models/gen_dors.pth"
gen_fron_model = "src/models/gen_fron.pth"
gen_late_model = "src/models/gen_late.pth"
gen_class_dictionary = "src/models/gen_dict.json"
gen_accuracy_list = "src/models/gen_accuracies.json"

# Genus Files with Other
gen_caud_model_with_other = "src/models/gen_caud_other.pth"
gen_dors_model_with_other = "src/models/gen_dors_other.pth"
gen_fron_model_with_other = "src/models/gen_fron_other.pth"
gen_late_model_with_other = "src/models/gen_late_other.pth"
gen_class_dictionary_with_other = "src/models/gen_dict_other.json"
gen_accuracy_list_with_other = "src/models/gen_accuracies_other.json"

# Transformation Files
caud_transformation = "src/models/caud_transformation.pth"
dors_transformation = "src/models/dors_transformation.pth"
fron_transformation = "src/models/fron_transformation.pth"
late_transformation = "src/models/late_transformation.pth"
gs_transformation = "src/models/all_transformation.pth"

# CAM Species Files
cam_spec_caud_model = "src/models/cam_spec_caud.pth"
cam_spec_dors_model = "src/models/cam_spec_dors.pth"
cam_spec_fron_model = "src/models/cam_spec_fron.pth"
cam_spec_late_model = "src/models/cam_spec_late.pth"
cam_spec_class_dictionary = "src/models/cam_spec_dict.json"
cam_spec_accuracy_list = "src/models/cam_spec_accuracies.json"

# CAM Species Files with Other
cam_spec_caud_model_with_other = "src/models/cam_spec_caud_other.pth"
cam_spec_dors_model_with_other = "src/models/cam_spec_dors_other.pth"
cam_spec_fron_model_with_other = "src/models/cam_spec_fron_other.pth"
cam_spec_late_model_with_other = "src/models/cam_spec_late_other.pth"
cam_spec_class_dictionary_with_other = "src/models/cam_spec_dict_other.json"
cam_spec_accuracy_list_with_other = "src/models/cam_spec_accuracies_other.json"

# CAM Genus Files
cam_gen_caud_model = "src/models/cam_gen_caud.pth"
cam_gen_dors_model = "src/models/cam_gen_dors.pth"
cam_gen_fron_model = "src/models/cam_gen_fron.pth"
cam_gen_late_model = "src/models/cam_gen_late.pth"
cam_gen_class_dictionary = "src/models/cam_gen_dict.json"
cam_gen_accuracy_list = "src/models/cam_gen_accuracies.json"

# CAM Genus Files with Other
cam_gen_caud_model_with_other = "src/models/cam_gen_caud_other.pth"
cam_gen_dors_model_with_other = "src/models/cam_gen_dors_other.pth"
cam_gen_fron_model_with_other = "src/models/cam_gen_fron_other.pth"
cam_gen_late_model_with_other = "src/models/cam_gen_late_other.pth"
cam_gen_class_dictionary_with_other = "src/models/cam_gen_dict_other.json"
cam_gen_accuracy_list_with_other = "src/models/cam_gen_accuracies_other.json"

# CAM Transformation Files
cam_caud_transformation = "src/models/cam_caud_transformation.pth"
cam_dors_transformation = "src/models/cam_dors_transformation.pth"
cam_fron_transformation = "src/models/cam_fron_transformation.pth"
cam_late_transformation = "src/models/cam_late_transformation.pth"

# Mask Directory
mask_directory = "mask_dataset"

# Alternate Species Files
alt_img_height = "src/models/alt_height.txt"
spec_dors_caud_model = "src/models/alt_spec_dors_caud.pth"
spec_all_model = "src/models/alt_spec_all.pth"
spec_dors_late_model = "src/models/alt_spec_dors_late.pth"
alt_spec_class_dictionary = "src/models/alt_spec_dict.json"
alt_spec_accuracy_list =  "src/models/alt_spec_accuracies.json"

# Alternate Genus Files
gen_dors_caud_model = "src/models/alt_gen_dors_caud.pth"
gen_all_model = "src/models/alt_gen_all.pth"
gen_dors_late_model = "src/models/alt_gen_dors_late.pth"
alt_gen_class_dictionary = "src/models/alt_gen_dict.json"
alt_gen_accuracy_list =  "src/models/alt_gen_accuracies.json"

# Alternate Transformation Files
all_transformation = "src/models/all_transformation.pth"
dors_caud_transformation = "src/models/dors_caud_transformation.pth"
dors_late_transformation = "src/models/dors_late_transformation.pth"

#Genus Model Tracked Files
genus_specific_accuracies = "src/genus_models/genus_specific_accuracies.json"

# YOLO Whole-Image Trainer Model
yolo_model = "src/models/yolov8n_whole_image.pt"

# Cropped Dataset
cropped_dataset = "cropped_dataset"

# Hyperparameters
genus_caud_hypers = "src/hyperparameters/gen_caud_hyperparameters.json"
genus_dors_hypers = "src/hyperparameters/gen_dors_hyperparameters.json"
genus_fron_hypers = "src/hyperparameters/gen_fron_hyperparameters.json"
genus_late_hypers = "src/hyperparameters/gen_late_hyperparameters.json"

species_caud_hypers = "src/hyperparameters/spec_caud_hyperparameters.json"
species_dors_hypers = "src/hyperparameters/spec_dors_hyperparameters.json"
species_fron_hypers = "src/hyperparameters/spec_fron_hyperparameters.json"
species_late_hypers = "src/hyperparameters/spec_late_hyperparameters.json"
