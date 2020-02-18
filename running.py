import numpy as np
from pprint import pprint
from load import Dataset
import matplotlib.pylab as plt

class RunningSim:
    """
    slot: (time, pos, (events, sub_events), do or be done)
    """
    def __init__(self, match_id,dataset=None, time_inv=None):
        if dataset is None:
            self.dataset = Dataset()
        else:
            self.dataset = dataset
        self.game = self.dataset.fullMatchGroup[match_id].drop(['TeamID'], axis=1)
        # print(self.game.head())

        self.allOpponents = list(set(self.game['OriginPlayerID']).union(set(self.game['DestinationPlayerID'].dropna())))
        self.maxHalfTime = self.getHalfTime()
        # print("Half time is", self.maxHalfTime)
        self.table = self.buildTable()
        self.h_team = list(filter(lambda x: x.startswith('Huskies'), self.allOpponents))
        self.other_team = list(filter(lambda x: not x.startswith('Huskies'), self.allOpponents))
    
    def getNowPos(self, players, t):
        xy = []
        for player in players:
            a = self.getPosAtT(player, t)
            if a is None:
                xy.append((np.inf, np.inf))
            else:
                xy.append(self.getPosAtT(player, t))
        return xy
    
    def getPosAtT(self, player, t):
        start, end = self.startAndend(player, t)
        if start is None:
            return None
        inv = end[0] - start[0]
        pos = start[1] + (end[1] - start[1])*(t-start[0])/inv
        return pos
    
    
    def startAndend(self, player, t1):
        """
        given t1
        find the player time period that include t1
        return:
            time period
            startpos: period start pos
            endpos: period end pos
        """
        eventsSlot = self.table[player]
        # print(eventsSlot[:5])
        start, end = None, None
        for i, event in enumerate(eventsSlot):
            if event[0] >= t1:
                # print(events
                start = eventsSlot[i-1]
                end = eventsSlot[i]
                break
        if start is None:
            # print(player)
            # raise ValueError(f"TIME {t1} is not in the game")
            return None, None
        return start, end

    def getHalfTime(self):
        time = max(self.game[self.game['MatchPeriod'] == '1H']['EventTime'])
        return time

    def buildTable(self):
        """
        return:
            dict(
                'player1': [(time, pos, (events, sub_events), do or be done)]
            )
        """
        table = {}
        numpy_data = self.game.to_numpy()
        for player in self.allOpponents:
            if player.startswith(self.dataset.selfname):
                table[player] = [(0., np.array([0.,0.]), ['open', 'open'], True)]
            else:
                table[player] = [(0., np.array([100.,100.]), ['open', 'open'], True)]
        overhalf = False
        for data in numpy_data:
            # print(data)
            
            if overhalf == True:
                data[3] += self.maxHalfTime
            if overhalf == False and data[2] == '2H':
                data[3] += self.maxHalfTime
                overhalf = True
                intern_time = (data[3] + self.maxHalfTime)/2
                # print("Second Half Start!!!")
                # NOTE should check the logic
                for player in self.allOpponents:
                    if player.startswith(self.dataset.selfname):
                        table[player].append((intern_time, np.array([0.,0.]), ['second_open', 'second_open'], True))
                    else:
                        table[player].append((intern_time, np.array([100., 100.]), ['second_open', 'second_open'], True))
            if data[0].startswith(self.dataset.selfname):
                table[data[0]].append((data[3], data[6:8].astype('float'), data[4:6], True))
            else:
                table[data[0]].append((data[3], 100 - data[6:8].astype('float'), data[4:6], True))
            # 
            try:
                if np.isnan(data[1]):
                    continue
            except TypeError:
                pass
            # 
            if data[1].startswith(self.dataset.selfname):
                table[data[1]].append((data[3], data[8:10].astype('float'), data[4:6], False))
            else:
                table[data[1]].append((data[3], 100 - data[8:10].astype('float'), data[4:6], False))
        for name, slots in table.items():
            for i, step in enumerate(slots):
                if np.isnan(step[1][0]):
                    if name.startswith(self.dataset.selfname):
                        slots[i][1][0], slots[i][1][1] = 0., 0.
                    else:
                        slots[i][1][0], slots[i][1][1] = 100., 100.
        return table
    
    
def plot(running, total_t = 4000):
    running : RunningSim
    plt.ion()
    allplayer = running.allOpponents
    h_team = list(filter(lambda x: x.startswith('Huskies'), allplayer))
    other_team = list(filter(lambda x: not x.startswith('Huskies'), allplayer))
    fig = plt.figure()
    for time in range(total_t):
        h_pos = running.getNowPos(h_team, t=time)
        o_pos = running.getNowPos(other_team, t=time)
        plt.scatter([i[0] for i in h_pos], [i[1] for i in h_pos], c='r', s=5)
        plt.scatter([i[0] for i in o_pos], [i[1] for i in o_pos], c='b', s=5)
        plt.pause(0.01)
        plt.clf()
    
if __name__ == "__main__":
    test = RunningSim(match_id=1)
    pprint(test.table['Huskies_F2'][:10])
    allplayer = test.allOpponents
    pprint(len(allplayer))
    print("1 s")
    pprint(test.getNowPos(['Huskies_F2', 'Huskies_F1'], t=1))
    pprint("1000 s")
    pprint(test.getNowPos(['Huskies_F2', 'Huskies_F1'], t=1000))
    pprint(test.h_team)
    pprint(test.other_team)
    # for name, slots in test.table.items():
    #     print(name)
    #     pprint(slots[:5])
    # plot(test)