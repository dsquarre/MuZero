import torch
from env.env import Env
import torch.nn as nn
from collections import deque
import math
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.representation = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            nn.Conv2d()
        )
        self.dynamics = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            nn.Conv2d()
        )
        self.prediction = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            nn.Conv2d()
        )
    def f(self,x):
        return self.prediction(x)
    def g(self,x):
        return self.dynamics(x)
    def h(self,x):
        return self.representation(x)
# okay so mcts starts from a state which basically creates a parent node
# it will check for children, if not expand
# expand will add links to Node.children and each of these links will have a start-parent node, end- child node, action and properties
# if children exist, then go to the best child using policy from f and Q of each link
# how will I get action though?
# my model predictor gives me a state and a reward and takes an action
# okay so action is all possible moves legal or not. legality is checked by masking valid actions at the root.
# how to decide on turn? basically make all actions possible even the opponent ones and mask out real from not possible.
# for dotgame this means all 2*n*(n-1) actions.

class Link:
    def __init__(self,start,action,end=None):
        self.start = start #Node
        self.end = end #Node
        self.action = action
        self.N = 0  
        self.Q = 0
        self.R = 0
        #self.P = prob #Probability prior not policy
        #self.V = None # storing policy and v at node not edge

class Node:
    def __init__(self,state,parent=None):
        self.parent = parent #Link
        self.children = []  #Links
        self.state = state
        self.total_visits = 0
        self.Policy = None
        self.value = float(0)
    #create 2 channels for each player instead of turn.    
    def action_space(self,dots,turn):
        mu = MuZero(dots)
        actions = []        
        for i in range(2*dots*(dots-1)):
            actions.append(mu.Action(dots,i,turn))
        actions.append(mu.Action(dots,-1,turn))
        return actions
    #only use f and create all children. when to use g? when selecting best child and action has been chosen. 
    def add_links(self,network,dots,turn):
        pi,v = network.f(self.state)
        policy = torch.softmax(pi,dim=1).squeeze(0)
        self.v = v.item()
        self.policy = policy.item()
        for a in self.action_space(dots,turn):
            child = Link(self,a)
            self.children.append(child)

    def expand(self,link,network,dots,turn):
        action = link.action
        r,next_state = network.g(torch.stack((self.state,action),dim=0))
        link.R = r
        next_node = Node(next_state,link)
        link.end = next_node
        pi,v = network.f(next_state)
        policy = torch.softmax(pi,dim=1).squeeze(0)
        next_node.v = v.item()
        next_node.policy = policy.item()
        for a in self.action_space(dots,turn):
            child = Link(next_node,a)
            next_node.children.append(child)
        return next_node
    
    def PUCT(self,policy,c1=1,c2=1):    
        Q_min = float('inf')
        Q_max = float('-inf')
        for link in self.children:
            if Q_min>link.Q:
                Q_min = link.Q
            if Q_max < link.Q:
                Q_max = link.Q
        expected = []
        for i,link in enumerate(self.children):
            Q_ = (link.Q - Q_min)/(Q_max - Q_min)
            W_ = policy[i]*math.sqrt(self.total_visits)*(c1 + math.log10((self.total_visits + c2 + 1)/c2))/(1+link.N)
            val = Q_ + W_
            expected.append(val)
        return expected  
    
    def best_child(self):
        #get Q_min and Q_max
        expected = self.PUCT(self.policy)
        link =  self.children[expected.index(max(expected))]
        if link.end:
            return None,link.end
        return link,self
        #should i use this min max type shit or only max and play with G,Q,V etc?

class MCTS:
    #fix this-> basically return all those shits masking the values
    def think(self,state,dots,turn,network,valid_actions,sims=800):
        root =Node(state)
        root.add_links(network,dots,turn)
        link = None
        l = 0
        for i in range(sims):
            while(root.children):
                link,root = root.best_child()
                l+=1
                if link: 
                    break
                turn *=-1
            root = root.expand(link,network,dots,turn)
            root = self.backup(root,l)
            #one thing I'm not sure about is suppose I accidentally pick a state which is really close to end,
            #but since my mcts and g doesnt know it, that means it will keep on making invalid moves (after grid is filled)
            #ofcourse I'm going to mask it when making real moves so computation is wasted.
            #But thats where generality comes from.(unbounded atari games)
        #for predpi, store all P[i] in children in a predpi[]
        #for predv, store all V[i] in children in predV
        #for predR, store all R[i] in children in predR
        #for calcpi, store all N[i]/totalvisits^(1/t) in a calcpi
        #get valid actions, mask all invalid actions to 0, reconvert to probability.
        #use ucb/puct formula to get max from valid actions and return everything
        #how am i supposed to mask bad actions???
        return self.getresult(valid_actions,root)
    
    def getresult(self,valid_actions,root,t=1):
        predpi = root.policy
        Pi =[]
        predv = root.V
        for link in root.children:
            Pi.append((link.N/root.total_visits)**(1/t))
        Pi[~valid_actions] = 0
        total = sum(Pi)
        calcpi = [x/total for x in Pi]
        expected = root.PUCT(calcpi)
        expected[~valid_actions] = 0
        best = expected.index(max(expected))
        child = self.children[best]
        predr = child.R
        action = child.action
        return predpi,calcpi,predv,predr,action
    
    def backup(self,node,l):
        v_l = node.v
        k = l
        end = node
        while(node.parent):
            node.total_visits +=1
            G_k = self.bootstrap(end,k,l,v_l)
            k-=1
            Q_k = (node.parent.N * node.parent.Q + G_k)/(node.parent.N+1)
            node.parent.Q = Q_k
            node.parent.N += 1
            node = node.parent.start
            
    def bootstrap(self,end,k,l,val,gamma=0.9):
        G = val*(gamma**(l-k))
        temp = end
        for t in range(l-k-1,-1,-1):
            G+= (gamma**(t))*(temp.parent.R) 
            temp = temp.parent.start
        return G   
        
class MuZero:
    def __init__(self,dots):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Network().to(self.device)
        self.replay_buffer = deque(maxlen=10000)
        self.dots = dots
    
    #channel 1: player1moves, channel2: player2moves, channel3:player1boxes,channel4:player2boxes,channel5: turn or color.
    # it has to be one hot encoded, numbers alone cannot help nn differentiate well enough.
    def observation(self,env,turn):
        grid = torch.zeros([2*env.dots-1,2*env.dots-1],dtype=torch.int8,device=self.device,requires_grad=False)
        if turn ==1:
            last = torch.ones([2*env.dots-1,2*env.dots-1],dtype=torch.int8,device=self.device,requires_grad=False)
        else:
            last = grid
        grid = torch.stack((grid,grid,grid,grid,last),dim=0) #5 grids stacked
        grid_index,boxindex=0,0
        for i in range(0,2*env.dots-1,1):
            if i%2 ==0:
                for j in range(1,2*env.dots-1,2):
                    if env.grid[grid_index] == 1:
                        grid[0,i,j] = 1
                    elif env.grid[grid_index] == -1:
                        grid[1,i,j] = 1
                    grid_index+=1
            else:
                for j in range(0,2*env.dots-1,1): 
                    if j%2 == 0:
                        if env.grid[grid_index] == 1:
                            grid[0,i,j] = 1
                        elif env.grid[grid_index] == -1:
                            grid[1,i,j] = 1
                        grid_index+=1
                    else:
                        if env.boxes[boxindex] == 1:
                            grid[2,i,j] = 1
                        elif env.boxes[boxindex] == -1:
                            grid[3,i,j] = 1
                        boxindex+=1
        return grid
    #action is also one hot encoded with first channel as player one move and second channel as player 2 move.
    #there's a possibility of pass move(all values 0) if action = -1
    def Action(self,dots,action,turn):
        env = Env(dots)
        env.step(action,turn)
        grid = torch.zeros([2,2*dots-1,2*env.dots-1],dtype=torch.int8,device=self.device,requires_grad=False)
        if action==-1:
            return grid
        # I can remove this O(n) complexity to O(1) using maths. will do it later.
        for i in range(0,2*dots-1,1):
            if i%2 ==0:
                for j in range(1,2*dots-1,2):
                    if env.grid[grid_index] == 1:
                        grid[0,i,j] = 1
                    elif env.grid[grid_index] == -1:
                        grid[1,i,j] = 1
                    grid_index+=1
            else:
                for j in range(0,2*dots-1,1): 
                    if j%2 == 0:
                        if env.grid[grid_index] == 1:
                            grid[0,i,j] = 1
                        elif env.grid[grid_index] == -1:
                            grid[1,i,j] = 1
                        grid_index+=1
                """ else:
                        if env.boxes[boxindex] == 1:
                            grid[2,i,j] = 1
                        elif env.boxes[boxindex] == -1:
                            grid[3,i,j] = 1
                        boxindex+=1"""
        return grid
    
    def State(self,O):
        history = []
        if len(O)<=8:
            history = [O[0]]*(8-len(O))
            for j in range(len(O)):
                history.append(O[j])
                
        else:
            history = []
            for j in range(len(O)-8,len(O)):
                history.append(O[j])
        return self.network.h(torch.stack(history,dim=0))
    
    def play(self,games=800):
        env = Env(self.dots)
        for g in range(games):
            O = deque(maxlen=8)
            A = []
            R = []
            predPi = []
            calcPi = []
            predV = []
            predR = []
            env.reset()
            turn = 1
            O.append(self.observation(env))
            state = self.State(O)
            S = []
            mcts = MCTS(self.dots)
            valid_actions = env.action_space()
            while not env.gameover():
                S.append(state)
                predpi,calcpi,predv,predr,action = mcts.think(state,env.dots,turn,self.network,valid_actions,sims=800)
                predPi.append(predpi)
                calcPi.append(calcpi)
                predV.append(predv)
                predR.append(predr)
                A.append(action)
                if action>0:
                    reward = env.step(action,turn)
                if turn==reward and action>0:
                    valid_actions = None
                else:
                    valid_actions = env.action_space()
                turn*=1
                reward = 0 if not env.gameover() else env.result() # for dotgame specific
                O.append(self.observation(env))
                R.append(reward)
                state = self.State(O)
            #have to store a done variable too to show that one game is finished and next is starting.
            for i in range(len(A)):
                self.replay_buffer.append({
                #"S": S[i],
                #"A" : A[i],
                "U" : R[i],
                "ModelPi" : predPi[i],
                "CalcPi" : calcPi[i],
                "V" : predV[i],
                "R" : predR[i],
                "Done":True if i==len(A-1) else False
                })
            #print(self.replay_buffer.size())

    def steps_return(self,U,gamma=0.99):
        Z=[]
        for j in range(len(U)):
            z=0
            for i in range(j,len(U)):
                z+= gamma**i * U[i]
            Z.append(z)
        return Z
    
    def train(self,k=-1,epochs=1000):
        import random
        opt = torch.optim.Adam(self.network.parameters(),lr=0.001,weight_decay=1)        
        for e in range(epochs):
            S_0 = random.choice(len(self.replay_buffer))
            #S = []
            U = []
            pi = []
            Pi = []
            V = []
            R = []
            for i in range(S_0,k+S_0):
                #S.append(self.replay_buffer[i]['state'])
                U.append(self.replay_buffer[i]['U'])
                Pi.append(self.replay_buffer[i]['ModelPi'])
                pi.append(self.replay_buffer[i]['CalcPi'])
                R.append(self.replay_buffer[i]['R'])
                V.append(self.replay_buffer[i]['V'])
                done = self.replay_buffer[i]['Done']
                if done:
                    break
            # i can calculate z = sigma(discounted reewards till end)
            # i need predpi - Pi,predreward-U,predvalue-z
            Z = self.steps_return(U)
            Pi = torch.tensor(Pi,dtype = torch.float32,device=self.device,requires_grad=True) #[B,A]
            pi = torch.tensor(pi,dtype = torch.float32,device=self.device,requires_grad=True) #[B,A]
            loss_p = nn.CrossEntropyLoss()(Pi,pi)
            V = torch.tensor(V,dtype=torch.float32,device=self.device,requires_grad=True)
            Z = torch.tensor(Z,dtype=torch.float32,device=self.device,requires_grad=True)
            loss_v = nn.MSELoss()(V,Z)
            R = torch.tensor(R,dtype = torch.float32,device=self.device,requires_grad=True)
            U = torch.tensor(U,dtype = torch.float32,device=self.device,requires_grad=True)
            loss_r = nn.MSELoss()(R,U)
            total_loss = loss_p+loss_v+loss_r 
            total_loss.backward()
            opt.step()
            
            
        
bot = MuZero()
env = Env(3)
bot.play(env,1)
