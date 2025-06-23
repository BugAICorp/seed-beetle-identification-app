""" globals.py

When adding new files to the project in the categories of: Models, Transformations, Model Trackers;
Add a variable here with the full path of the file so that the code will only show a simple variable
rather than joining the path in the code.
(Please check spelling to ensure easy access as well)
"""
training_database = "training.db"
class_list = "class_list.txt"
img_height = "height.txt"

# Species Files
spec_caud_model = "spec_caud.pth"
spec_dors_model = "spec_dors.pth"
spec_fron_model = "spec_fron.pth"
spec_late_model = "spec_late.pth"
spec_class_dictionary = "spec_dict.json"
spec_accuracy_list = "spec_accuracies.json"

# Genus Files
gen_caud_model = "gen_caud.pth"
gen_dors_model = "gen_dors.pth"
gen_fron_model = "gen_fron.pth"
gen_late_model = "gen_late.pth"
gen_class_dictionary = "gen_dict.json"
gen_accuracy_list = "gen_accuracies.json"

# Transformation Files
caud_transformation = "caud_transformation.pth"
dors_transformation = "dors_transformation.pth"
fron_transformation = "fron_transformation.pth"
late_transformation = "late_transformation.pth"

# Alternate Species Files
alt_img_height = "src/models/alt_height.txt"
spec_dors_caud_model = "src/models/alt_spec_dors_caud.pth"
spec_all_model = "src/models/alt_spec_all.pth"
spec_dors_late_model = "src/models/alt_spec_dors_late.pth"
alt_spec_class_dictionary = "src/models/alt_spec_dict.json"
alt_spec_accuracy_list = "src/models/alt_spec_accuracies.json"

# Alternate Genus Files
gen_dors_caud_model = "src/models/alt_gen_dors_caud.pth"
gen_all_model = "src/models/alt_gen_all.pth"
gen_dors_late_model = "src/models/alt_gen_dors_late.pth"
alt_gen_class_dictionary = "src/models/alt_gen_dict.json"
alt_gen_accuracy_list = "src/models/alt_gen_accuracies.json"

# Alternate Transformation Files
all_transformations = "src/models/all_transformation.pth"
dors_caud_transformation = "src/models/dors_caud_transformation.pth"
dors_late_transformation = "src/models/dors_late_transformation.pth"

# Genus Model Tracked Files
genus_specific_accuracies = "src/genus_models/genus_specific_accuracies.json"
