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
    displayName = CharField()
    file = CharField()
    civitai_page = CharField()
    class Meta:
        table_name = "Checkpoints"

class CheckpointVariant(BaseModel):
    checkpoint = ForeignKeyField(Checkpoint, on_delete='CASCADE')
    handle = CharField(unique = True)
    name = CharField()
    file_name = CharField()
    inpainting = BooleanField()
    class Meta:
        table_name = "CheckpointVariants"

if(not db.table_exists('checkpoints')):
    Checkpoint.create_table()
if(not db.table_exists('CheckpointVariants')):
    Checkpoint.create_table()
