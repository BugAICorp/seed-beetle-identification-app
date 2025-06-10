""" genus_specific_simulator.py """
import sys
import os
from PIL import Image
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from genus_specific_model_trainer import GenusSpecificModelTrainer
from model_loader import ModelLoader
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

if __name__ == '__main__':
    #tdc = TrainingDataConverter("dataset")
    #tdc.conversion(globals.training_database)
    dbr = DatabaseReader(globals.training_database, class_file_path=globals.class_list)
    df = dbr.get_dataframe()

    genus_specific_tp = GenusSpecificModelTrainer(df)
    genus_specific_tp.train_genus("Callosobruchus", 5)