import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
#import sys
#print(sys.path)

class SentBot(sc2.BotAI):
    async def on_step(self, iteration):
        #what to do every step
        await self.distribute_workers() #in sc2/bot_ai.py

run_game(maps.get("AbyssalReefLE"),[
    Bot(Race.Protoss, SentBot()),
    Computer(Race.Terran, Difficulty.Easy)
    ], realtime=True)


