""" training_database_reader.py """
import os
import sys
import sqlite3
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class DatabaseReader:
    """
    Reads from SQLite database, and creates a pandas dataframe
    """
    def __init__(self, database, connection=None,
                 table="TrainingData", query=None,
                 class_file_path=None, exclude_classes=False,
                 create_other=False):
        """
        Initialize DatabaseReader and loads data into a Pandas DataFrame. 
        
        Arguements:
            database (str): path to the SQLite database file.
            connection (sqlite3.Connection): shared SQLite connection(this is optional).
            table (str): name of the table in the database to query.
            query (str): SQL query to execute(this is optional).
            class_file_path (str): File path of specified class to be in the models
            exclude_classes (bool): This flag lets you control if the class file is inclusion or exclusion
            create_other (bool): If True, includes all data but maps unknown species/genus to 'Other'
        """
        self.database = database
        self.connection = connection
        self.table = table
        self.create_other = create_other

        if class_file_path:
            self.allowed_species = self.load_valid_classes(class_file_path)
            placeholders = ','.join(['?'] * len(self.allowed_species))
            if not create_other:
                if exclude_classes:
                    self.query = f"""
                        SELECT Genus, Species, UniqueID, View, SpecimenID, Image, Filename
                        FROM {self.table}
                        WHERE Species NOT IN ({placeholders})
                    """
                else:
                    self.query = f"""
                        SELECT Genus, Species, UniqueID, View, SpecimenID, Image, Filename
                        FROM {self.table}
                        WHERE Species IN ({placeholders})
                    """
            else:
                # For create_other we fetch all data and later replace non-allowed species/genus
                self.query = f"""
                    SELECT Genus, Species, UniqueID, View, SpecimenID, Image, Filename FROM {self.table}
                """
        else:
            self.allowed_species = None
            default_query = f"""
                SELECT Genus, Species, UniqueID, View, SpecimenID, Image, Filename FROM {self.table}
            """
            self.query = query or default_query

        self.dataframe = self.load_data()

        if self.create_other and self.allowed_species:
            self.dataframe = self.map_unallowed_to_other(self.dataframe, self.allowed_species)

    def load_valid_classes(self, class_file_path):
        """
        Reads in a file containing the valid species entries and initializes the
        allowed classes set.
        Returns:
            Set: the allowed classes to be trained with
        """
        with open(class_file_path, 'r', encoding='utf-8') as file:
            allowed_species = {
                # Only need the species for class limiting
                line.strip().split()[-1] for line in file if line.strip()
            }

        return allowed_species

    def load_data(self):
        """
        Loads data from the database using the provided query. If database is invalid or empty,
        this method returns an empty DataFrame.
        
        Returns: 
            pd.DataFrame: the queried data from the SQLite database as a Pandas DataFrame.
        """
        try:
            parameters = list(self.allowed_species) if self.allowed_species and not self.create_other else []
            if self.connection:
                # use the provided connection
                return pd.read_sql_query(self.query, self.connection, params=parameters)
            # create a new connection if none is provided
            with sqlite3.connect(self.database) as conn:
                return pd.read_sql_query(self.query, conn, params=parameters)
        except (sqlite3.DatabaseError, pd.io.sql.DatabaseError) as e:
            # if error is raised, then print error and return empty DataFrame
            print(f"Error reading database: {e}")
            return pd.DataFrame()

    def map_unallowed_to_other(self, df, allowed_species):
        """
        Replaces species/genus not in allowed list with 'Other'.

        Args:
            df (pd.DataFrame): Data loaded from DB
            allowed_species (Set[str]): Set of species to retain

        Returns:
            pd.DataFrame: Updated DataFrame with other species/genus mapped to 'Other'
        """
        df = df.copy()
        df.loc[~df['Species'].isin(allowed_species), 'Species'] = 'other'
        df.loc[~df['Species'].isin(allowed_species), 'Genus'] = 'Other'
        return df

    def get_dataframe(self):
        """
        Simple getter method that returns the objects DataFrame
        """
        return self.dataframe

    def get_num_species(self):
        """
        Gets number of unique classes within species column
        """
        return self.dataframe['Species'].nunique()

    def get_num_genus(self):
        """
        Gets number of unique classes within genus column
        """
        return self.dataframe['Genus'].nunique()

if __name__ == "__main__":
    db_path = input("Please input the file path of the SQLite database: ")
    class_file = input(
        "Please input path to allowed classes text file (or leave blank to include all): ").strip()
    reader = DatabaseReader(db_path, class_file if class_file else None)
    print("Process Completed")
