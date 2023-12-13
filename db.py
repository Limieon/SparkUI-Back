import os

from dotenv import load_dotenv
from peewee import *

load_dotenv()

SPARKUI_BACK_DB_TYPE = os.getenv("SPARKUI_BACK_DB_TYPE")

db = None

print("Initializing Database...")
print(SPARKUI_BACK_DB_TYPE)
if(SPARKUI_BACK_DB_TYPE == "sqlite"):
    db = SqliteDatabase(database="db.sqlite")
else:
    raise Exception("Currently only SQLite is supported!")

# Database Tables
# -> Models

class BaseModel(Model):
    class Meta:
        database = db

# Represents a SD Checkpoint that contains multiple versions
class Checkpoint(BaseModel):
    handle = CharField(unique = True)                               # The unique handle across all checkpoint groups
    name = CharField()                                              # The display name of the checkpoint
    description = CharField()                                       # The short description of the model
    preview_url = CharField()                                       # A preview image of the checkpoint
    class Meta:
        table_name = "Checkpoint"

# Represents a SD Checkpoint
class CheckpointVariation(BaseModel):
    handle = CharField(unique = True)                               # The unique handle inside the checkpoint group
    checkpoint = ForeignKeyField(Checkpoint, on_delete='CASCADE')   # The reference to the checkpoint group
    file_name = CharField()                                         # The filename thats stored on disk
    inpainting = BooleanField()                                     # If checkpoint was made for used in inpainting
    created_at = TimestampField()                                   # Stores when the checkpoint was uploaded onto the server
    sd_version = CharField()                                        # Stores SD Version such as SD1.5, SD2.1, SDXL or SDXL-Turbo

    # Optional Usage Info
    width = IntegerField(null=True)                                 # The recommended width and height of the image
    height = IntegerField(null=True)
    clip_skip = IntegerField(null=True)                             # The recommended clip skip value
    sampler = CharField(null=True)                                  # The recommended sampling method
    sampling_steps_min = IntegerField(null=True)                    # The recommended min and max sampling steps
    sampling_steps_max = IntegerField(null=True)

    class Meta:
        table_name = "CheckpointVariation"


class Lora(BaseModel):
    handle = CharField(unique=True)                                 # The unique handle across the loras
    name = CharField()                                              # The display name of the lora
    description = CharField()                                       # The description of the lora

    # Optional Usage Info
    weight_min = FloatField(null=True)                              # The recommended min and max weight
    weight_max = FloatField(null=True)

    class Meta:
        table_name = "Lora"

class LoraVersion(BaseModel):
    handle = CharField(unique=True)                                 # The unique handle of the version inside the loras
    file_name = CharField()                                         # The filename of the lora
    trigger_words = CharField()                                     # A list of trigger words recommended for the lora
    created_at = TimestampField()                                   # The timestamp when the lora was upload

    class Meta:
        table_name = "LoraVersion"

class Embedding(BaseModel):
    handle = CharField(unique=True)                                 # The unique handle of the embedding
    name = CharField()                                              # The display name of the embedding
    description = CharField()                                       # A short description of the embedding

    class Meta:
        table_name = "Embedding"

class EmbeddingVersion(BaseModel):
    handle = CharField(unique=True)                                 # The unique handle inside the embeddings
    embedding = ForeignKeyField(Embedding, on_delete='CASCADE')     # Reference to the embedding group
    file_name = CharField()                                         # The file name on the disk
    created_at = TimestampField()                                   # The timestamp when the embedding was uploaded

    class Meta:
        table_name = "EmbeddingVersion"


class VAE(BaseModel):
    handle = CharField()                                            # The unique handle of the VAE
    created_at = TimestampField()                                   # The timestamp when the VAE was uploaded
    file_name = CharField()                                         # The file name stored on disk
    
    class Meta:
        table_name = "VAE"

TABLES = [
    Checkpoint, CheckpointVariation,
    Lora, LoraVersion,
    Embedding, EmbeddingVersion,
    VAE
]

for table in TABLES:
    if not table.table_exists():
        table.create_table()
