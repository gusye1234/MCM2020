from running import RunningSim
from load import Dataset
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from load import PlotNX
import networkx as nx

class TimeSlots:
    def __init__(self, match_id, dataset=None, interval=60, name=None):
        self.interval = interval
        self.dataset : Dataset = dataset if dataset is not None else Dataset()
        self.sim = RunningSim(dataset=self.dataset, match_id=match_id)
        self.gamepass = self.dataset.passingMatchGroup[match_id]        
        self.gamefull = self.dataset.fullMatchGroup[match_id]
        self.getMatchLast()
        self.selfname = self.dataset.selfname if name is None else name
        self.stats = self.build_Slots()

    def getFactor(self, boundary=30):
        passgame = self.gamepass[self.gamepass['TeamID'] == self.selfname]
        distances_passer = []
        distances_receiver = []
        opponents = self.sim.other_team
        for data in passgame.to_numpy():
            starter = data[6:8].astype('float')
            receiver = data[8:].astype('float')
            time   = data[4]
            oppoDis = np.array(self.sim.getNowPos(opponents, time)).astype('float')
            dis_start = TimeSlots.distance(starter, oppoDis)
            # print(dis_start)
            dis_start = dis_start[dis_start <= boundary]
            # print(dis_start)
            dis_recv  = TimeSlots.distance(receiver, oppoDis)
            dis_recv  = dis_recv[dis_recv <= boundary]
            if len(dis_start):
                distances_passer.append(np.mean(dis_start)/len(dis_start))
            if len(dis_recv):
                distances_receiver.append(np.mean(dis_recv)/len(dis_recv))
        return distances_passer, distances_receiver
        
        
    def getPeriodStat(self, t1,t2, touchingGuys=None):
        assert t1 < t2
        passgame = self.gamepass[self.gamepass['TeamID'] == self.selfname]
        assert len(passgame) > 0
        period = passgame[(t1 <= passgame['EventTime']) & (passgame['EventTime'] <= t2)]
        if len(period) == 0:
            return None
        # stat of passing freq and opponents pressure
        PassTimes = len(period)
        starter_dis, receiver_dis = self.getOppoDis(period)
        DG2F      = self.count_DG2F(period)/PassTimes
        if touchingGuys is not None:
            guystouch = []
            for guy in touchingGuys:
                guystouch.append(self.getGuyPass(period, guy)/PassTimes)
        
        # stat of diversity
        action_list = list(period['EventSubType'])
        action_set  = period['EventSubType'].unique()
        action_count = np.array([action_list.count(i) for i in action_set]).astype('float')
        Diversity = TimeSlots.diversity(action_count.astype('float')/np.sum(action_count)) # normalized
        
        # 
        players = len(period['OriginPlayerID'].append(period['DestinationPlayerID']).unique())
        
        # TODO
        
        ans = [PassTimes, starter_dis, receiver_dis, Diversity, players, DG2F]
        if touchingGuys is not None:
            ans.append(guystouch)
        return ans
    
    def count_DG2F(self, period):
        length = 0
        for data in period.to_numpy():
            if (data[1][-2] == "G" or data[1][-2] == "D") and data[2][-2] == 'F':
                length += 1
        return length
    
    def getMatchLast(self):
        h = max(self.gamefull[self.gamefull['MatchPeriod'] == '1H']['EventTime'])
        # print(h)
        b = self.gamepass[self.gamepass['MatchPeriod'] == '2H']['EventTime'].copy()
        self.gamepass['EventTime'][self.gamepass['MatchPeriod'] == '2H'] = b.apply(lambda x: x+h)
        # print(max(self.gamepass['EventTime']))
        b = self.gamefull[self.gamefull['MatchPeriod'] == '2H']['EventTime'].copy()
        self.gamefull['EventTime'][self.gamefull['MatchPeriod'] == '2H'] = b.apply(lambda x: x+h)
        # print(max(self.gamefull['EventTime']))
              
    
    def build_Slots(self):
        t = 0.
        stats = []
        for t in range(self.interval, int(max(self.gamepass['EventTime']))+1, self.interval):
            a = self.getPeriodStat(t-100, t)
            if a is not None:
                stats.append(a)
        stats = np.array(stats).astype('float')
        
        return stats
        
    
    def getOppoDis(self, period, boundary=20):
        """
        NOTE for passingevents.csv 
        return:
            mean #near_opponents/distance!!!
            lager, closer
        """
        opponents = self.sim.other_team
        dis1 = 0.
        dis2 = 0.
        for slot in period.to_numpy():
            starter = slot[6:8].astype('float')
            receiver = slot[8:].astype('float')
            time = slot[4]
            oppoDis = np.array(self.sim.getNowPos(opponents, time)).astype('float')
            dis_start = TimeSlots.distance(starter, oppoDis)
            # print(dis_start)
            dis_start = dis_start[dis_start <= boundary]
            # print(dis_start)
            dis_recv  = TimeSlots.distance(receiver, oppoDis)
            dis_recv  = dis_recv[dis_recv <= boundary]
            if len(dis_start):
                dis1 += np.mean(dis_start)/len(dis_start)
            if len(dis_recv):
                dis2 += np.mean(dis_recv)/len(dis_recv)
        return dis1/len(period) , dis2/len(period)
            
    
    def getGuyPass(self, period, guy):
        if not guy.startswith(self.selfname):
            raise ValueError(f"player not in this model! player{guy}, team{self.selfname}")
        data = period.to_numpy()
        length = 0
        for i in data:
            if data[1] == guy or data[2] == guy:
                length += 1
        return length
    
        
    @staticmethod
    def distance(x1, x2):
        """"""
        return np.sqrt(np.sum(np.square(x1 - x2), axis=1))
        
    @staticmethod
    def diversity(poss : np.ndarray):
        return  -np.sum(poss*np.log2(poss))
    
def plotPassAndDis(stats):
    plt.scatter(stats[:,1]*10, stats[:, 0], s=1, linewidths=2,label="time slot point")
    # plt.axis('scaled')
    plt.ylim([-2,40])
    plt.xlim([0,20])
    plt.title('How passing times related to opponents position (passer)')
    plt.xlabel('opponents position factor (bigger, relaxer)')
    plt.ylabel(f'passing times in a certain time interval({100})')
    plt.legend()
    plt.show()
    
def hAndOther(match_id):
    data_h = TimeSlots(match_id)
    names = data_h.gamepass['TeamID'].unique()
    print(names)
    other_team = names[0] if data_h.selfname == names[1] else names[1]
    data_o = TimeSlots(match_id, name=other_team)
    stats_h = data_h.stats
    stats_o = data_o.stats
    x = np.linspace(0,min(len(stats_h), len(stats_o))-1, 1000)
    f1 = interp1d(range(len(stats_h)), stats_h[:, 0], kind='cubic')
    f2 = interp1d(range(len(stats_o)), stats_o[:, 0], kind='cubic')
    plt.plot(x, f1(x), c='r', linewidth=3,label=data_h.selfname)
    plt.plot(x, f2(x), c='b', linewidth=3,label=data_o.selfname)
    plt.title(f'passing times in blocks in Game {match_id} (receiver)')
    plt.xlabel('blocks')
    plt.ylabel(f'passing times in a certain time interval({60})')
    plt.legend()
    plt.show()
    
def hAndOtherDG2F(match_id):
    data_h = TimeSlots(match_id)
    names = data_h.gamepass['TeamID'].unique()
    print(names)
    other_team = names[0] if data_h.selfname == names[1] else names[1]
    data_o = TimeSlots(match_id, name=other_team)
    stats_h = data_h.stats
    stats_o = data_o.stats
    x = np.linspace(0,min(len(stats_h), len(stats_o))-1, 1000)
    f1 = interp1d(range(len(stats_h)), stats_h[:, 5], kind='quadratic')
    f2 = interp1d(range(len(stats_o)), stats_o[:, 5], kind='quadratic')
    plt.plot(x, f1(x), c='r', linewidth=3,label=data_h.selfname)
    plt.plot(x, f2(x), c='b', linewidth=3,label=data_o.selfname)
    plt.title(f'G, D pass to F in blocks in Game {match_id}')
    plt.ylim([0, 5])
    plt.xlabel('blocks')
    plt.ylabel(f'passing times in a certain time interval({60})')
    plt.legend()
    plt.show()
    
def getImportance(match_id):
    data_h = PlotNX(match_id=match_id)
    names = data_h.gamefull['TeamID'].unique()
    print(names)
    other_team = names[0] if data_h.selfname == names[1] else names[1]
    data_o = PlotNX(match_id=match_id, selfname=other_team)
    score_h = nx.betweenness_centrality(data_h.G)
    score_h = sorted(score_h.items(), key=lambda item:item[1], reverse = True)
    score_o = nx.betweenness_centrality(data_o.G)
    score_o = sorted(score_o.items(), key=lambda item:item[1], reverse = True)
    pprint(score_h)
    pprint(score_o)
    guys_h = score_h[:3]
    guys_o = score_o[:3]
    
def plotBin():
    dis1 = []
    dis2 = []
    for match in range(1, 39):
        print(match)
        data = TimeSlots(match_id=match)
        one, two = data.getFactor()
        dis1.extend(one)
        dis2.extend(two)
    # data = TimeSlots(match_id=1)
    # dis1, dis2 = data.getFactor()
    dis1 = list(filter(lambda x: np.isfinite(x), dis1))
    dis2 = list(filter(lambda x: np.isfinite(x), dis2))
    
    plt.hist(dis1, normed=True,bins=40, facecolor='blue', edgecolor='black')
    plt.xlabel("pressure factor")
    plt.ylabel("freqency")
    plt.title("the passing freqency under the pressure from opponents (passer)")
    plt.show()
    plt.hist(dis2, normed=True,bins=40, facecolor='blue', edgecolor='black')
    plt.xlabel("pressure factor")
    plt.ylabel("freqency")
    plt.title("the passing freqency under the pressure from opponents (receiver)")
    plt.show()
    x, y = getBinandPoint(np.array(dis2))
    plotCurve(x,y)


def getBinandPoint(data : np.ndarray, bins=100):
    assert bins != 0
    max_value = np.max(data)
    min_value = np.min(data)
    interval = (max_value - min_value)/bins
    sorted_data = np.sort(data)
    bin_up   = min_value + interval
    bin_x    = []
    bin_freq = []
    j_start = 0
    
    for i in range(0, bins):
        freq = 0
        while sorted_data[j_start] <= bin_up:
            freq += 1
            j_start += 1
            if j_start >= len(sorted_data):
                break
        bin_freq.append(freq)
        bin_x.append(bin_up - interval/2)
        bin_up += interval
    # plt.plot(bin_x, bin_freq)
    # plt.show()

    return bin_x, bin_freq



def func(x : np.ndarray, a, b, c):
    return a * x**(5/2) * np.exp(-b*x) + c

def function(a, b, c):
    def f(x):
        return a * x**(5/2) * np.exp(-b*x) + c
    return f

# [208.48195307   0.67142628   5.55367265]
# 
def plotCurve(x, y):
    # f1 = interp1d(x, y, kind='cubic')
    popt, pcov = curve_fit(func, x, y)
    # print(popt)
    # print(pcov)
    
    
    x1 = np.linspace(0, 30, 1000)
    # # y1 = f1(x1)
    y1 = func(x1, popt[0], popt[1], popt[2])
    plt.scatter(x, y, c='b')
    plt.plot(x1,y1, c='y', label='fitting curve')
    plt.title("the passing freqency under the pressure from opponents")
    plt.xlabel('pressure factor')
    plt.ylabel('freqency')
    plt.legend()
    plt.show()
    return popt
        
def getCurve(x, y):
    popt, pcov = curve_fit(func, x, y)
    return popt

def getFunct(x,y):
    co = getCurve(x, y)
    return function(*co)
 
# class Weight(RunningSim):
#     def __init__(self):
#         self.dataset : Dataset = Dataset()
#         # self.sim = RunningSim(dataset=self.dataset)
#         self.gamepass = self.dataset.passing.drop(['MatchID'], axis=1)        
#         self.gamefull = self.dataset.fullevents.drop(['MatchID'], axis=1)
#         self.getMatchLast()
#         self.selfname = self.dataset.selfname
#         self.sims = [RunningSim(dataset=self.dataset, match_id=i) for i in range(1,39)]
    
    
    
#     def getWeight(self, player1, player2, match_id):
#         pass
    
    # def getFactor(self, boundary=30):
    #     passgame = self.gamepass[self.gamepass['TeamID'] == self.selfname]
    #     distances_passer = []
    #     distances_receiver = []
    #     opponents = self.sim.other_team
    #     for data in passgame.to_numpy():
    #         starter = data[6:8].astype('float')
    #         receiver = data[8:].astype('float')
    #         time   = data[4]
    #         oppoDis = np.array(self.sim.getNowPos(opponents, time)).astype('float')
    #         dis_start = TimeSlots.distance(starter, oppoDis)
    #         # print(dis_start)
    #         dis_start = dis_start[dis_start <= boundary]
    #         # print(dis_start)
    #         dis_recv  = TimeSlots.distance(receiver, oppoDis)
    #         dis_recv  = dis_recv[dis_recv <= boundary]
    #         if len(dis_start):
    #             distances_passer.append(np.mean(dis_start)/len(dis_start))
    #         if len(dis_recv):
    #             distances_receiver.append(np.mean(dis_recv)/len(dis_recv))
    #     return distances_passer, distances_receiver
        
            

        
if __name__ == "__main__":
    # import os
    # if not os.path.exists('stats_H.npy'):
    #     test = TimeSlots(match_id=1)
    #     # pprint(test.stats)
        
    #     stats = test.stats
    #     for i in range(1,39):
    #         test = TimeSlots(match_id=i)
    #         stats = np.vstack([stats, test.stats])
    #     np.save('stats_H', stats)
    # else:
    #     stats = np.load('stats_H.npy')
    # plotPassAndDis(stats)
    # hAndOther(match_id=9)
    # hAndOtherDG2F(match_id=9)
    # getImportance(match_id=1)
    # plotPassAndDis(stats)
    plotBin()
    # test = Weight()