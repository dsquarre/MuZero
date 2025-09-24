import torch
from resnet import ResNet18
import torch.nn as nn
from collections import deque
import math

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
   
    #only use f and create all children. when to use g? when selecting best child and action has been chosen. 
    def add_links(self,network,action_space):
        pi,v = network.f(self.state)
        policy = torch.softmax(pi,dim=1).squeeze(0)
        self.v = v.item()
        self.policy = policy.item()
        for a in action_space:
            child = Link(self,a)
            self.children.append(child)

    def expand(self,link,network,action_space):
        action = link.action
        r,next_state = network.g(torch.stack((self.state,action),dim=0))
        link.R = r
        next_node = Node(next_state,link)
        link.end = next_node
        pi,v = network.f(next_state)
        policy = torch.softmax(pi,dim=1).squeeze(0)
        next_node.v = v.item()
        next_node.policy = policy.item()
        for a in action_space:
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
    def simulate(self,obs,action_space,valid_actions,network,sims=800):
        state = network.h(obs)
        root =Node(state)
        root.add_links(network,action_space)
        link = None
        l = 0
        for i in range(sims):
            while(root.children):
                link,root = root.best_child()
                l+=1
                if link: 
                    break
            root = root.expand(link,network,action_space)
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
        #mask calcpi invalid to 0 and choose actions from predpi or calcpi? 
        # -> choosing prepi to get exp and mask invalid exp to -inf to choose action 
        return self.extract(valid_actions,root)
    
    def extract(self,valid_actions,root,t=1):
        predpi = root.policy
        Pi =[]
        predv = root.V
        for link in root.children:
            Pi.append((link.N/root.total_visits)**(1/t))
        Pi[~valid_actions] = 0
        total = sum(Pi)
        calcpi = [x/total for x in Pi]
        expected = root.PUCT(predpi)
        expected[~valid_actions] = float('-inf')
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
    def __init__(self,Env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Env = Env
        self.replay_buffer = deque(maxlen=10000)
        obs_channels,state_channels,state_action_channels,height,width,actionspace = Env.getparams()
        self.network = ResNet18(obs_channels,state_channels,state_action_channels,height,width,actionspace)
        self.network.to(self.device)
        self.action_space = Env.action_space()
   
    def play(self,games=800):
        for g in range(games):
            #O = []
            A = []
            R = []
            predPi = []
            calcPi = []
            predV = []
            predR = []
            self.Env.reset()
            obs = self.env.Observation()
            mcts = MCTS(self.dots)
            while not self.Env.gameover():
                #O.append(obs)
                valid_actions = self.Env.valid_actions()
                predpi,calcpi,predv,predr,action = mcts.simulate(obs,self.action_space,valid_actions,self.network,sims=800)
                predPi.append(predpi)
                calcPi.append(calcpi)
                predV.append(predv)
                predR.append(predr)
                A.append(action)
                reward = self.Env.step(action)
                obs = self.Env.Observation()
                R.append(reward)
            #have to store a done variable too to show that one game is finished and next is starting.
            for i in range(len(A)):
                self.replay_buffer.append({
                #"O": O[i],
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
            
            
        
