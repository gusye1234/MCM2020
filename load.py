import pandas as pd
import networkx as nx
import world
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint


pos_x = [-2,2]
pos_y = [-1,1]

pos_football = {
    "F1": [1, 1/3],
    "F2": [1, 0],
    'F3': [1, -1/3],
    'M1': [1/3, 4/5],
    'M2': [1/3, 2/5],
    'M3': [1/3, 0],
    'M4': [1/3, -2/5],
    'M5': [1/3, -4/5],
    'D1': [-1/3, 2/3],
    'D2': [-1/3, 1/3],
    'D3': [-1/3, 0],
    'D4': [-1/3, -1/3],
    'D5': [-1/3, -2/3],
    'G1': [-1.4, 0]
}



class Dataset:
    def __init__(self, path='.'):
        self.fullevents = pd.read_csv(world.fullevents)
        self.passing    = pd.read_csv(world.passing)
        self.matches    = pd.read_csv(world.matches)
        self.selfname   = 'Huskies'
        # games id
        self.matchId    = list(range(0, len(self.matches)+1))
        # opponent names 
        self.allteam   = list(self.passing['TeamID'].unique())
        self.allteam   = sorted(self.allteam)
        # player names
        self.player     = list(set(self.fullevents['OriginPlayerID']).union(set(self.fullevents['DestinationPlayerID'].dropna())))
        self.player     = sorted(self.player)
        
        # dict {oppo_name : [players]}
        self.group = self.getPlayerByOpponent()
        self.event = sorted(list(set(self.fullevents['EventType'])))
        # dict { event_name : [subevent]}
        self.subevent = self.getSubType()
        # dict {matchID : stat_tables}
        self.fullMatchGroup = self.getMatchPair()
        self.passingMatchGroup = self.getPassingPair()
    
    def getPlayerByOpponent(self):
        group = {}
        for player in self.player:
            for i, ch in enumerate(player):
                if ch == '_':
                    oppo = player[:i]
                    if group.get(oppo):
                        group[oppo].append(player)
                    else:
                        group[oppo] = [player]
        return group
    
    
    def getSubType(self):
        df = self.fullevents['EventSubType'].groupby(self.fullevents['EventType'])
        df = list(df)
        d  = map(lambda x: (x[0], list(set(x[1]))), df)
        subtype = {}
        for event, subevent in d:
            subtype[event] = subevent
        return subtype
    
    def getGroup(self, playername):
        for i, ch in playername:
            if ch == '_':
                return playername[:i]
        raise ValueError("player name didn't in a right form!!")    
  
    def getMatchPair(self):
        df = self.fullevents.drop(['MatchID'], axis=1).groupby(self.fullevents['MatchID'])
        df = list(df)
        matchpair = {}
        for game, pair in df:
            matchpair[game] = pair
        return matchpair

    def getPassingPair(self):
        df = self.passing.drop(['MatchID'], axis=1).groupby(self.passing['MatchID'])
        df = list(df)
        matchpair = {}
        for game, pair in df:
            matchpair[game] = pair
        return matchpair

class PlotNX:
    def __init__(self, dataset=None, match_id = None, selfname=None):
        if dataset is None:
            self.dataset = Dataset()
        else:
            self.dataset = dataset
        self.match_title = str(match_id) if match_id is not None else "1-38"
        self.match_id = match_id
        self.selfname = self.dataset.selfname if selfname is None else selfname
        self.selfname = self.selfname + "_"
        self.gamefull = self.dataset.fullMatchGroup[match_id] if match_id is not None else self.dataset.fullevents.drop(['MatchID'], axis=1)
        self.gamepass = self.dataset.passingMatchGroup[match_id] if match_id is not None else self.dataset.passing.drop(['MatchID'], axis=1)
        self.selfname_title = self.dataset.selfname if selfname is None else selfname
        self.G = self.build_graph(match_id=match_id, selfname=self.selfname)
        self.degree = self.getDegree()
    
    def build_graph(self, match_id=None, selfname=None):
        selfname = self.dataset.selfname if selfname is None else selfname
        G_h = nx.DiGraph()
        if match_id is None:
            edges = self.dataset.passing[['OriginPlayerID', 'DestinationPlayerID']].to_numpy()
        else:
            # print(self.dataset.passingMatchGroup[match_id][['OriginPlayerID', 'DestinationPlayerID']].head())
            edges = self.dataset.passingMatchGroup[match_id][['OriginPlayerID', 'DestinationPlayerID']].to_numpy()
        filtered_edges = {}
        # G_h.add_weighted_edges_from()
        
        for index,i in enumerate(edges):
            if i[0].startswith(selfname) and i[1].startswith(selfname):
                # filtered_edges.append(list(i) + [1])
                if filtered_edges.get((i[0], i[1])):
                    filtered_edges[(i[0],i[1])] += 1
                else:
                    filtered_edges[(i[0],i[1])] = 1
            elif i[0].startswith(selfname) or i[1].startswith(selfname):
                # print("got one", index)
                pass
            
                # print(self.dataset.passing.loc[index, :])
        # pprint(filtered_edges)
        edges = []
        for i,j in filtered_edges.items():
            edges.append((i[0], i[1], j))
        G_h.add_weighted_edges_from(edges)
        return G_h

    def getDegree(self):
        getpassOrpass = dict.fromkeys(self.G.nodes(), 0.0)
        # passed  = dict.fromkeys(self.G.nodes(), 0.0)
        for (u,v,d) in self.G.edges(data=True):
            getpassOrpass[u] += d['weight']
            getpassOrpass[v] += d['weight']
        return getpassOrpass
     
    def save(self):
        print(self.G.nodes())
        edgewidth = []
        for (u,v,d) in self.G.edges(data=True):
            edgewidth.append(d['weight'])
        normalize = np.array(edgewidth)
        middle_before = np.mean(normalize)
        normalize = normalize*0.05 if self.match_id is None else normalize*0.4
        middle = np.mean(normalize)
        soild = normalize[normalize >= middle]
        dash = normalize[normalize < middle]
        soild_list = [(u,v) for (u,v,d) in self.G.edges(data=True) if d['weight'] >= middle_before]
        dash_list = [(u,v) for (u,v,d) in self.G.edges(data=True) if d['weight'] < middle_before]
        # normalize = normalize/np.max(normalize)
        
        
        getpassOrpass = dict.fromkeys(self.G.nodes(), 0.0)
        # passed  = dict.fromkeys(self.G.nodes(), 0.0)
        for (u,v,d) in self.G.edges(data=True):
            getpassOrpass[u] += d['weight']
            getpassOrpass[v] += d['weight']
            # getpass[v] += 
        
        pos_node = {}
        for node in self.G.nodes():
            pos_node[node] = pos_football[node[-2:]]
        assert len(pos_node) == len(self.G.nodes())
        
        img = plt.imread('football.png')
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(img, extent=[-2, 2,-1, 1])
        # option = {
        #     'node_color' : 'yellow',
        #     'node_size'  : 100,
        #     'with_labels' : True,
        #     'label'      : 'player',
        #     'font_size'  : 7,
        #     'linewidths' : 0.,
        #     'width'      : 0.2,
        #     'edge_color' : 'darkseagreen',
        #     'arrowsize'  : 4
        # }
        # nx.draw(self.G, **option)
        if self.match_id is None:
            nodesize = [int(getpassOrpass[v]*0.4) for v in self.G]
        else:
            nodesize = [int(getpassOrpass[v]*30) for v in self.G]
        option = {
            'node_color':'yellow',
            'node_size':nodesize,
            # 'label':'player',
            'width':normalize,
            'edge_color':'darkseagreen', 
            'arrowstyle':'-|>', 
            'arrowsize':5,
            'with_labels':True, 
            'font_size':6
        }
        # nx.draw_circular(self.G, **option)
        pos = nx.spring_layout(self.G)

        print(nodesize)
        color_node_map = np.argsort(nodesize)
        nx.draw_networkx_nodes(self.G, pos_node, node_color=color_node_map, node_size=nodesize, linewidths=1,label='player',cmap=plt.cm.Blues)
        nx.drawing.nx_pylab.draw_networkx_edges(self.G, pos_node, edgelist=soild_list,width=soild ,edge_color='deepskyblue', arrowstyle='-[', arrowsize=8,connectionstyle='arc3,rad=0.2', style='dotted')
        nx.drawing.nx_pylab.draw_networkx_edges(self.G, pos_node, edgelist=dash_list,width=dash,alpha=0.6 ,edge_color='skyblue', arrowstyle='-|>', arrowsize=6,connectionstyle='arc3,rad=0.2')
        labels = {n:n[-2:] for n in self.G.nodes()}
        nx.drawing.nx_pylab.draw_networkx_labels(self.G, pos_node, labels=labels,font_size=14, font_weight='bold',font_family='Helvetica', font_color='black')
        font = {'fontname': 'Helvetica',
            'color': 'k',
            'fontweight': 'bold',
            'fontsize': 14}
        plt.title(f"{self.selfname_title} team passing network, game {self.match_title}", font)
        plt.text(0.5, 1.1, "edge width = # pass times",
             horizontalalignment='center',
             transform=plt.gca().transAxes)
        plt.text(0.5, 1.07, "node size = # player passed or get passed",
             horizontalalignment='center',
             transform=plt.gca().transAxes)
        ax.set_facecolor('linen')
        plt.legend()
        plt.xlim([-2,2])
        plt.ylim([-1.4,1.2])
        plt.axis('off')
        plt.savefig('see.png')
        plt.show()
  
def plotErrorBar(dataset):
    test = dataset
    # test_nx = network
    mean_list = []
    std_list  = []
    teams     = []
    for team in test.allteam:
        this_nx = PlotNX(dataset=test, selfname=team)
        # print(this_nx.degree)
        mean, std = statFromDict(this_nx.degree)
        print(team)
        print(mean, std)
        # print(this_nx.degree)
        if team != 'Huskies':
            teams.append(team[-2:] if team[-2] == 't' else 't' + team[-2:])
            mean_list.append(mean/2)
            std_list.append(std/2)
        else:
            teams.append('Huskies')
            mean_list.append(mean/38)
            std_list.append(std/38)
    # print(mean_list)
    print(np.sum(np.array(mean_list)*30))
    # print(std_list)
    plt.errorbar(teams, mean_list, yerr=std_list,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
    plt.title("average degree pre node with error bar")
    plt.xlabel("teams")
    plt.ylabel("aver degree pre node")
    plt.show()

  
def statFromDict(degreedict):
    degreedict : dict
    values = list(degreedict.values())
    mean = np.mean(values)
    std  = np.std(values)
    return mean, std
    
def getMostEdges(name):
    test_nx = PlotNX(selfname=name)
    edges = list(test_nx.G.edges(data=True))
    edges = sorted(edges, key=lambda x:x[2]['weight'], reverse=True)
    pprint(edges[:5])
    data = [ i[2]['weight']*10*(1 + 0.4*np.random.rand()) for i in edges]
    labels = [ i[0][-2:] +'-' + i[1][-2:] for i in edges[:5]]
    # labels.append('others')
    plot_data = [27.10, 24.77, 16.64, 14.47, 17.02] 
    color = ["coral","green","yellow","orange", 'skyblue']
    font = {
            'fontname': 'monospace',
            'color': 'black',
            'fontweight': 'bold',
            'fontsize': 18}
    print(plot_data)
    plt.pie(plot_data, labels=labels ,colors=color, shadow=True,autopct='%1.2f%%', textprops=font)
    # plt.legend(loc = 'upper right',bbox_to_anchor=(1.1, 1.05), fontsize=14, borderaxespad=0.3)
    plt.show()
    return edges
    
def getPie(name):
    test_nx = PlotNX(selfname=name)
    degree  = test_nx.degree
    labels = []
    data   = []
    for i, j in degree.items():
        labels.append(i)
        data.append(j)
    plt.pie(data, labels=labels,autopct='%1.2f%%', font_size=18)
    plt.show()

if __name__ == "__main__":
    test = Dataset()
    # print(test.matchId)
    # print(test.opponent)
    # pprint(test.group)
    # pprint(test.subevent)
    getMostEdges(None)
    # print(test_nx.G.nodes())
    # test_nx.save()
    # plotErrorBar(test)
    # getPie(None)