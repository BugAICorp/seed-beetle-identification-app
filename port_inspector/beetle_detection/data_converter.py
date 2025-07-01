"""data_converter.py"""

# flake8: noqa
import json, os, sys, io
from PIL import Image
from port_inspector_app.models import TrainingDatabase, ValidClasses
from .beetle_cropper import BeetleCropper
from PIL import UnidentifiedImageError

 
class DjangoTrainingDatabaseConverter:

    def __init__(self, dir):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.dir = os.path.join(base_dir, dir)
        self.beetle_cropper = BeetleCropper()

    def img_to_binary(self, image):
        try:
            # Crop the image before saving to the database
            img = Image.open(image).convert("RGB")
            cropped = self.beetle_cropper.crop_beetle(img)

            img_binary = io.BytesIO()
            cropped.save(img_binary, format='JPEG')
            img_binary.seek(0)
            return img_binary.read()

        except UnidentifiedImageError:
                print(f"Cannot identify image file: {image}")
        except OSError as e:
            print(f"OS error processing {image}: {e}")

        return None

    def parse_name(self, name: str):
        name_parts = name.split(' ')
        if len(name_parts) != 5:
            return None
        cur_index = len(name_parts) - 1
        view = name_parts[cur_index][:name_parts[cur_index].find('.')]
        cur_index -= 2
        unique_id = name_parts[cur_index] + view
        specimen_id = name_parts[cur_index]
        cur_index -= 1
        species = name_parts[cur_index]
        cur_index -= 1
        genus = name_parts[cur_index][name_parts[cur_index].find('/')+1:]

        return (genus, species, unique_id, view, specimen_id)

    def conversion(self):
        if not os.path.exists(self.dir):
            print(f"Directory {self.dir} does not exist.")
            return

        for filename in os.listdir(self.dir):
            if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(self.dir, filename)
                name_parts = self.parse_name(filename)
                if name_parts:
                    genus, species, unique_id, view, specimen_id = name_parts
                    image_binary = self.img_to_binary(file_path)

                    if ValidClasses.objects.filter(genus=genus).exists() and ValidClasses.objects.filter(species=species).exists():
                        if not TrainingDatabase.objects.filter(uniqueid=unique_id).exists():
                            TrainingDatabase.objects.create(
                                genus=genus,
                                species=species,
                                uniqueid=unique_id,
                                view=view,
                                specimenid=specimen_id,
                                image=image_binary
                            )
                            print(f"Inserted: {unique_id}")
