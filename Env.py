from bisect import bisect_left
from State import State
import gym
from gym.utils import seeding
import numpy as np

def get_index_from_time(current_time, time_list):
    index = bisect_left(time_list, current_time)
    if (time_list[index] == current_time):
        return index
    else:
        return index - 1               


DEFAULT_EVENT_COUNT = 100
DEFAULT_TRADE_COUNT = 10
DEFAULT_CANDLE_COUNT = 10

class QRLEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    actions = [(0,0,0),(0,1,0),(1,0,0),(1,1,0),(0,0,1)]
    
    def __init__(self, lob_order, lobs, candles15min_order, candles15min, candles1min_order, candles1min, candles1sec_order, candles1sec, start_random_time, end_random_time,
                 event_count = DEFAULT_EVENT_COUNT,
                 trade_count = DEFAULT_TRADE_COUNT,
                 candles_count = DEFAULT_CANDLE_COUNT,
                 reset_on_close = True,
                 random_ofs_on_reset = True):
        self.lob_order = lob_order
        self.lobs = lobs
        
        self.candles15min_order = candles15min_order
        self.candles15min = candles15min
        
        self.candles1min_order = candles1min_order
        self.candles1min = candles1min
        
        self.candles1sec_order = candles1sec_order
        self.candles1sec = candles1sec
        
        self.random_ofs_on_reset = random_ofs_on_reset
    
        self._state = State(lobs, candles15min, candles1min, candles1sec, event_count, trade_count, candles_count)
        self.action_space = gym.spaces.Discrete(n=len(self.actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        
        self.event_count = event_count
        self.candles_count = candles_count
        
        self.start_random_time = start_random_time
        self.end_random_time = end_random_time
               
        self.seed()        
        
    
    def reset(self):        
        timepoint = 0
        if self.random_ofs_on_reset:    
            # select random timepoint 
            offset = self.np_random.choice(self.end_random_time - self.start_random_time)  +  self.start_random_time           
        else:
            # select large enough start time            
            offset = self.start_random_time         
                
        timepoint = self.lob_order[offset]
         
         # get pointers from timepoint
        lob_p = get_index_from_time(timepoint,self.lob_order)
        qmin_p = get_index_from_time(timepoint, self.candles15min_order)
        smin_p = get_index_from_time(timepoint, self.candles1min_order)
        ssec_p = get_index_from_time(timepoint, self.candles1sec_order)
        
        self._state.reset(lob_p, qmin_p, smin_p, ssec_p, offset)                             
            
        return self._state.encode()    
    
    def step(self, action_idx):
        action = self.actions[action_idx]
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": "btc"}
        return obs, reward, done, info 
    
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
    
    def close(self):
        pass
    
    def render(self, mode='human', close=False):
        pass
    
    
    
        
        
        



