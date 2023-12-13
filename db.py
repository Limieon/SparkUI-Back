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


class Checkpoint(BaseModel):
    handle = CharField(unique = True)
    name = CharField()
    description = CharField()
    preview_url = CharField()
    class Meta:
        table_name = "Checkpoint"

class CheckpointVariation(BaseModel):
    handle = CharField(unique = True)
    checkpoint = ForeignKeyField(Checkpoint, on_delete='CASCADE')
    file_name = CharField()
    inpainting = BooleanField()
    created_at = TimestampField()

    # Optional Usage Info
    width = IntegerField(null=True)
    height = IntegerField(null=True)
    clip_skip = FloatField(null=True)
    sampler = CharField(null=True)
    sampling_steps_min = IntegerField(null=True)
    sampling_steps_max = IntegerField(null=True)

    class Meta:
        table_name = "CheckpointVariation"


class Lora(BaseModel):
    handle = CharField(unique=True)
    name = CharField()
    description = CharField()

    # Optional Usage Info
    weight_min = FloatField(null=True)
    weight_max = FloatField(null=True)

    class Meta:
        table_name = "Lora"

class LoraVersion(BaseModel):
    handle = CharField(unique=True)
    file_name = CharField()
    trigger_words = CharField()
    created_at = TimestampField()

    class Meta:
        table_name = "LoraVersion"

class Embedding(BaseModel):
    handle = CharField(unique=True)
    name = CharField()
    description = CharField()

    class Meta:
        table_name = "Embedding"

class EmbeddingVersion(BaseModel):
    handle = CharField(unique=True)
    embedding = ForeignKeyField(Embedding, on_delete='CASCADE')
    file_name = CharField()
    created_at = TimestampField()

    class Meta:
        table_name = "EmbeddingVersion"


class VAE(BaseModel):
    handle = CharField()
    created_at = TimestampField()
    file_name = CharField()
    
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
