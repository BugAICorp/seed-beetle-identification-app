""" one_of_each_simulator.py """
import sys
import os
from PIL import Image
from training_database_reader import DatabaseReader
from model_loader import ModelLoader
from evaluation_method import EvaluationMethod
from genus_evaluation_method import GenusEvaluationMethod
from eval_species_by_genus import EvalSpeciesByGenus
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def evaluate_images(species_eval,
                    genus_eval,
                    late_path,
                    dors_path,
                    fron_path,
                    caud_path) -> tuple:
    """
    Uses the inputted models to classify the species and 
    genus of the inputted bug.
    
    Returns: Tuple of top species list and genus/genus confidence
    """
    # Load the provided images
    LATE_IMG = Image.open(late_path) if late_path else None
    DORS_IMG = Image.open(dors_path) if dors_path else None
    FRON_IMG = Image.open(fron_path) if fron_path else None
    CAUD_IMG = Image.open(caud_path) if caud_path else None

    # Run the species evaluation method
    top_spec = species_eval.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    # Run the genus evaluation method
    top_gen, genus_confidence = genus_eval.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    return (top_spec, top_gen, genus_confidence)

def evaluate_with_genspec(eval,
                    late_path,
                    dors_path,
                    fron_path,
                    caud_path) -> tuple:
    """
    Uses the inputted models to classify the species and
    genus of the inputted bug.

    Returns: Tuple of top species list and genus/genus confidence
    """
    # Load the provided images
    LATE_IMG = Image.open(late_path) if late_path else None
    DORS_IMG = Image.open(dors_path) if dors_path else None
    FRON_IMG = Image.open(fron_path) if fron_path else None
    CAUD_IMG = Image.open(caud_path) if caud_path else None

    top_genus, top_species = eval.classify_images(
        dors=DORS_IMG, caud=CAUD_IMG, late=LATE_IMG, fron=FRON_IMG
    )

    return (top_species, top_genus[0], top_genus[1])

if __name__ == '__main__':
    # Get Species and Genus Class Number
    dbr = DatabaseReader(globals.training_database, class_file_path=globals.class_list)
    SPECIES_OUTPUTS = dbr.get_num_species()
    GENUS_OUTPUTS = dbr.get_num_genus() + 1

    # Get Model Files
    species_model_paths = {
            "caud" : globals.spec_caud_model,
            "dors" : globals.spec_dors_model,
            "fron" : globals.spec_fron_model,
            "late" : globals.spec_late_model
        }

    genus_model_paths = {
            "caud" : globals.gen_caud_model,
            "dors" : globals.gen_dors_model,
            "fron" : globals.gen_fron_model,
            "late" : globals.gen_late_model
        }

    # Load Genus Evaluator
    genus_ml = ModelLoader(genus_model_paths, GENUS_OUTPUTS)
    genus_models = genus_ml.get_models()

    genus_evaluator = GenusEvaluationMethod(globals.img_height, genus_models, 1,
                                            globals.gen_class_dictionary, globals.gen_accuracy_list)

    # Load Species Evaluator
    species_ml = ModelLoader(species_model_paths, SPECIES_OUTPUTS)
    species_models = species_ml.get_models()

    species_evaluator = EvaluationMethod(globals.img_height, species_models, 1,
                                         globals.spec_class_dictionary, globals.spec_accuracy_list)

    genus_spec_evaluator = EvalSpeciesByGenus(genus_models, globals.gen_class_dictionary)

    ###### TO BE CHANGED FOR MULTIPLE TESTS
    #LATE_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT LATE.jpg"
    #DORS_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT DORS.jpg"
    #FRON_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT FRON.jpg"
    #CAUD_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT CAUD.jpg"
    ######
    specimen_inputs = ["GEM_3229201", "GEM_1002207612", "GEM_3229390", "GEM_187667461", "MTEC_18767533",
                       "GEM_187671326", "USDA_187678234B", "GEM_187684634", "GEM_Spn5213N", "GEM_187675132",
                       "GEM_1002217268", "USNM_187686701", "GEM_187678904", "BYU_187676545", "GEM_1002210599",
                       "GEM_187672900", "GEM_187673021", "INBIO_CRI002233334", "BCISP-2764", "BCISP-1525",
                       "GEM_187688519", "GEM_187688578", "GEM_187688727", "GEM_187688617", "USNM_187688162",
                       "BYU_187676786", "GEM_187688395", "GEM_187688357", "GEM_187688292", "USNM_187688212",
                       "USNM_187688221", "USNM_187688229", "MZLU_187688251", "USNM_187685967", "USDA_ALPCA171567715001",
                       "GEM_187688197", "USDA_ASPCA191535428010K", "GEM_187688139", "USNM_3220094", "USNM_3220096",
                       "FSCA_00038073", "USNM_3220092", "INBIOCRI001267906", "INBIOCRI001267768",
                       "WIBF_001604", "USDA_APLTX151550324001A", "WIBF_072458", "TAMU-ENTO_X0187796", "GEM_187688341",
                       "USNM_187676456", "USNM_3220761", "USDA_SFOCA022271051440", "GEM_1002204853",
                       "LPL_11963B", "MTEC_3228103", "WIBF_022950", "GEM_3229153", "GEM_1002210652",
                       "BCISP-3585", "GEM_1002210995", "GEM_1002210916", "USNM_3214019", "GEM_187685294",
                       "USNM_187679768", "USNM_110282", "CNCType15059", "USNM_187687049", "GEM_187675200",
                       "GEM_187673626", "GEM_1002205459", "WIBF_025200", "GEM_1002205468"]
    
    model_to_use = int(input("Input 1 for classic models, 2 for species based on genus: "))
    for imagename in specimen_inputs:
        filtered_images = dbr.dataframe[dbr.dataframe['SpecimenID'] == imagename]
        if not filtered_images.empty:
            file_name_substring = (
                "dataset/" +
                filtered_images.iloc[0]['Genus'] + " " +
                filtered_images.iloc[0]['Species'] + " " +
                filtered_images.iloc[0]['SpecimenID'] + " 5XEXT "
                                )
            LATE_PATH = file_name_substring + "LATE.jpg"
            DORS_PATH = file_name_substring + "DORS.jpg"
            FRON_PATH = file_name_substring + "FRON.jpg"
            CAUD_PATH = file_name_substring + "CAUD.jpg"

            print(f"Results for {file_name_substring}:\n")
            top_species = None
            top_genus = None
            genus_conf_score = None
            if(model_to_use == 1):
                # Genus and Species Evaluation
                top_species, top_genus, genus_conf_score = evaluate_images(
                    species_eval=species_evaluator,
                    genus_eval=genus_evaluator,
                    late_path=LATE_PATH if os.path.exists(LATE_PATH) else None,
                    dors_path=DORS_PATH if os.path.exists(DORS_PATH) else None,
                    fron_path=FRON_PATH if os.path.exists(FRON_PATH) else None,
                    caud_path=CAUD_PATH if os.path.exists(CAUD_PATH) else None)
            
            elif(model_to_use == 2):
                top_species, top_genus, genus_conf_score = evaluate_with_genspec(
                    eval=genus_spec_evaluator,
                    late_path=LATE_PATH if os.path.exists(LATE_PATH) else None,
                    dors_path=DORS_PATH if os.path.exists(DORS_PATH) else None,
                    fron_path=FRON_PATH if os.path.exists(FRON_PATH) else None,
                    caud_path=CAUD_PATH if os.path.exists(CAUD_PATH) else None
                )

            print(f"1. Predicted Species: {top_species[0][0]}, Confidence: {top_species[0][1]:.2f}\n")
            print(f"2. Predicted Species: {top_species[1][0]}, Confidence: {top_species[1][1]:.2f}\n")
            print(f"3. Predicted Species: {top_species[2][0]}, Confidence: {top_species[2][1]:.2f}\n")
            print(f"4. Predicted Species: {top_species[3][0]}, Confidence: {top_species[3][1]:.2f}\n")
            print(f"5. Predicted Species: {top_species[4][0]}, Confidence: {top_species[4][1]:.2f}\n\n")

            print(f"Top Genus: {top_genus}, Confidence: {genus_conf_score:.2f}\n\n")
