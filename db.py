from prisma import Prisma

class DB:
    def __init__(self):
        self.handle = Prisma()

    async def connect(self):
        await self.handle.connect()

    async def shutdown(self):
        await self.handle.disconnect()

    handle: Prisma
