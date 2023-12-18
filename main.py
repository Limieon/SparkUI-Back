import asyncio
from prisma import Prisma
from prisma.models import Checkpoint, CheckpointVariation

async def main() -> None:
    prisma = Prisma(auto_register=True)
    await prisma.connect()

    # write your queries here
    checkpoint = await Checkpoint.prisma().create(data = {
        'handle': 'dreamshaper',
        'name': 'DreamShaper'
    })
    
    checkpointVariation = await CheckpointVariation.prisma().create(data = {
        'handle': 'v8',
        'name': 'DreamShaper - V8',
        'baseModel': 0,
        'checkpoint_id': checkpoint.id,
        'file': 'dreamshaper_v8.safetensors'
    })

    await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
