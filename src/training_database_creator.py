""" training_database_creator.py """
import sqlite3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class TrainingDataConverter:
    """
    Converter from file directory of images to sqlite database
    """
    def __init__(self, dataset_dir_path):
        """ Initialize converter with file path reference to directory """
        self.dir_path = dataset_dir_path
        self.db = None

    def img_to_binary(self, image_path):
        """
        Translate given file path of image to binary
        Return: binary file in bytes
        """
        with open(image_path, 'rb') as file:
            binary_file = file.read()
        return binary_file

    def build_db(self):
        """
        Builds training database and sets appropriate columns
        Return: None
        """
        table = sqlite3.connect(self.db)
        cursor = table.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS TrainingData (
                Genus TEXT,
                Species TEXT,
                UniqueID TEXT PRIMARY KEY,
                View TEXT,
                SpecimenID TEXT,
                Image BLOB,
                Filename TEXT
            )
        ''')
        table.commit()
        table.close()

    def add_img(self, image_data, image_binary, filename):
        """
        Inserts individual image as record in database.
        Returns: None
        """
        table = sqlite3.connect(self.db)
        cursor = table.cursor()
        try:
            # ensure array contains correct data
            if len(image_data) != 5:
                raise ValueError(
                    "image_data invalid, needs: [genus, species, unique_id, view, specimen_id]"
                    )

            cursor.execute('''
            INSERT INTO TrainingData (Genus, Species, UniqueID, View, SpecimenID, Image, Filename) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', image_data + (image_binary, filename))

            table.commit()
            print(f"Inserted Image UniqueID: {image_data[2]} with Filename: {filename}")
        except sqlite3.IntegrityError:
            print(f"WARNING! Duplicate image dropped: {image_data[2]}")
        finally:
            table.close()

    def parse_name(self, name: str):
        """
        Parses file name into column values
        Returns: Tuple (Genus, Species, UniqueID, View, SpecimenID)
        """
        name_parts = name.split(' ')
        if len(name_parts) != 5:
            return None

        genus = name_parts[0]
        species = f"{genus} {name_parts[1]}"
        specimen_id = name_parts[2]
        image_type = name_parts[3]
        view = name_parts[4].split('.')[0].upper()

        unique_id = f"{specimen_id}_{image_type}_{view}"

        print(species)

        return (genus, species, unique_id, view, specimen_id)

    def conversion(self, db_name):
        """
        Main function handling building database then iterating through images and adding them
        to the database with proper format
        Returns: None
        """
        self.db = db_name
        if not os.path.exists(self.dir_path):
            print(f"Directory, {self.dir_path}, does not exist.")
            return

        self.build_db()

        for filename in os.listdir(self.dir_path):
            # loop through image files
            if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(self.dir_path, filename)
                name_parts = self.parse_name(filename) # placeholder parsing
                if name_parts:
                    image_data = name_parts[:5]
                    image_binary = self.img_to_binary(file_path)
                    self.add_img(image_data, image_binary, filename)
                else:
                    print(f"File, {filename}, has invalid naming format.")

if __name__ == "__main__":
    dir_path = input("Please input the file path of the data set directory: ")
    tdc = TrainingDataConverter(dir_path)
    tdc.conversion("training.db")
    print("Process Completed")
