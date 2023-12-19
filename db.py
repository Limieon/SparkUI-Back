from prisma import Prisma
from prisma.models import BaseModel

class DB:
    async def connect(self):
        self.handle = Prisma(auto_register=True)
        await self.handle.connect()

    async def shutdown(self):
        await self.handle.disconnect()

    handle: Prisma
