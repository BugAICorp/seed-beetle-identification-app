""" test_training_database_reader.py """
import os
import sys
import sqlite3
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from training_database_reader import DatabaseReader

class TestDatabaseReader(unittest.TestCase):
    """
    Unit testing for training database reader
    """
    @classmethod
    def setUpClass(cls):
        """
        Create an in-memory SQLite database for testing DatabaseReader class.
        """
        # use a single shared connection for all tests
        cls.test_db = ":memory:"

        # connect to the database
        cls.connection = sqlite3.connect(cls.test_db)
        cursor = cls.connection.cursor()

        # initialize the test table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS TrainingData (
                Genus TEXT,
                Species TEXT,
                UniqueID TEXT PRIMARY KEY,
                View TEXT,
                Image BLOB,
                SpecimenID TEXT,
                Filename TEXT
            )
        """)

        # insert sample data
        sample_data = [
            ("GenusA", "GenusA SpeciesA", "ID1", "View1", sqlite3.Binary(b'\xff\xd8\xff\xe0...'),
             "ID4", "GenusA SpeciesA GEM_ID4 5XEXT View1.jpg"),
            ("GenusB", "GenusB SpeciesB", "ID2", "View2", sqlite3.Binary(b'\xff\xd8\xff\xe0...'),
             "ID5", "GenusB SpeciesB GEM_ID5 5XEXT View2.jpg"),
            ("GenusC", "GenusC SpeciesC", "ID3", "View3", sqlite3.Binary(b'\xff\xd8\xff\xe0...'),
             "ID6", "GenusC SpeciesC GEM_ID6 5XEXT View3.jpg"),
        ]
        # ensure table is empty before inserting rows
        cursor.execute("DELETE FROM TrainingData")
        cursor.executemany("""
            INSERT OR IGNORE INTO TrainingData (Genus, Species, UniqueID, View, Image, SpecimenID, Filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, sample_data)

        cls.connection.commit()

    @classmethod
    def tearDownClass(cls):
        """
        Close the shared in-memory database connection after tests.
        """
        cls.connection.close()

    def test_load_valid_data(self):
        """
        Test that DatabaseReader loads data from a valid database and table.
        Use correctly formatted mocked database.
        """
        # create DatabaseReader obejct with mocked database
        reader = DatabaseReader(self.test_db, connection=self.connection)
        df = reader.dataframe

        # assert the DataFrame contents(number of rows, columns exist, and data is expected)
        self.assertEqual(len(df), 3)
        self.assertIn("Filename", df.columns)
        self.assertEqual(df.iloc[0]["Genus"], "GenusA")

    def test_custom_query(self):
        """
        Test DatabaseReader using a custom SQL query.
        """
        # create DatabaseReader object with example query
        query = "SELECT Genus, Species FROM TrainingData WHERE Genus = 'GenusA'"
        reader = DatabaseReader(self.test_db, connection=self.connection, query=query)
        df = reader.dataframe

        # assert the DataFrame contents(number of rows and correct data)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Genus"], "GenusA")

    def test_invalid_table(self):
        """
        Test handling of reading from a non-existent table.
        """
        # create DatabaseReader object with an invalid table name
        reader = DatabaseReader(self.test_db, connection=self.connection, table="InvalidTable")
        df = reader.dataframe

        # assert empty dataframe with returned(meaning error was raised while loading dataframe)
        self.assertTrue(df.empty)

    def test_invalid_database(self):
        """
        Test handling of an invalid database path.
        """

        reader = DatabaseReader("fake.db")
        df = reader.dataframe

        # verify an empty DataFrame is returned(meaning error was raised while loading dataframe)
        self.assertTrue(df.empty)

    def test_empty_table(self):
        """
        Test DatabaseReader correctly handles an empty table.
        """
        # create an empty table
        cursor = self.connection.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS EmptyTable (Column1 TEXT)")
        self.connection.commit()

        # create reader object using the empty table
        reader = DatabaseReader(self.test_db, connection=self.connection, table="EmptyTable")
        df = reader.dataframe

        # assert an empty DataFrame is returned(meaning error was raised while loading dataframe)
        self.assertTrue(df.empty)

    def test_class_file_inclusion(self):
        """ Test that DatabaseReader includes only species from class file. """
        # Create a temp file with SpeciesA
        with open("test_classes.txt", "w") as f:
            f.write("GenusA SpeciesA\n")

        reader = DatabaseReader(self.test_db, connection=self.connection, class_file_path="test_classes.txt")
        df = reader.dataframe

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Species"], "GenusA SpeciesA")

        os.remove("test_classes.txt")

    def test_class_file_exclusion(self):
        """ Test that DatabaseReader excludes species from class file. """
        # Create a temp file with SpeciesA
        with open("test_classes.txt", "w") as f:
            f.write("GenusA SpeciesA\n")

        reader = DatabaseReader(self.test_db, connection=self.connection, class_file_path="test_classes.txt", exclude_classes=True)
        df = reader.dataframe

        self.assertEqual(len(df), 2)
        self.assertNotIn("GenusA SpeciesA", df["Species"].values)

        os.remove("test_classes.txt")

    def test_class_file_create_other(self):
        """Test that unlisted species are mapped to 'Other' when create_other is True."""
        # Create a temp file with only SpeciesA
        with open("test_classes.txt", "w") as f:
            f.write("GenusA SpeciesA\n")

        reader = DatabaseReader(
            self.test_db,
            connection=self.connection,
            class_file_path="test_classes.txt",
            create_other=True
        )
        df = reader.get_dataframe()

        # Should include all rows (3 total)
        self.assertEqual(len(df), 3)

        # Check that SpeciesB and SpeciesC are labeled as 'Other'
        self.assertIn("other", df["Species"].values)
        self.assertIn("Other", df["Genus"].values)

        # Check that SpeciesA and GenusA remain unchanged
        row = df[df["Species"] == "SpeciesA"].iloc[0]
        self.assertEqual(row["Genus"], "GenusA")

        os.remove("test_classes.txt")


if __name__ == "__main__":
    unittest.main()
