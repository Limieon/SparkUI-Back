from prisma import Prisma
from prisma.models import BaseModel, VAE, VAEVersion


class DB:
    async def connect(self):
        self.handle = Prisma(auto_register=True)
        await self.handle.connect()

        if not await VAE.prisma().find_first(where={"id": 1}):
            await VAE.prisma().create(
                data={"id": 1, "handle": "default", "name": "Default VAE"}
            )

            await VAEVersion.prisma().create(
                data={"id": 1, "handle": "default", "name": "Default"}
            )

    async def shutdown(self):
        await self.handle.disconnect()

    handle: Prisma
