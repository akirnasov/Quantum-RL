from State_MM import State_MM
from State_AP import State_AP
import numpy as np


class State:
    def __init__(self, lobs, candles15min, candles1min, candles1sec, event_count, trade_count, candles_count):
        self.max_shape = (event_count, lobs[0].num_levels*4)
        self.shape = (8, event_count, lobs[0].num_levels*4)
        self.state_mm = State_MM(lobs, candles15min, candles1min, candles1sec, 0, 0, 0, 0, event_count, trade_count, candles_count)
        self.state_ap = State_AP(self.max_shape)   

    # supply action
    def step(self, a):
        lob = self.state_mm.GetLOB()
        event_p = self.state_mm.GetEventPointer()
        event = lob.event_queue[event_p]

        done = self.state_mm.step()
        reward, done_ap = self.state_ap.step(a, lob, event)

        done |= done_ap
        
        return reward, done
    
    def encode(self):
        obs_mm = self.state_mm.encode()
        obs_ap = self.state_ap.encode()
        obs = obs_mm + obs_ap
        obs = np.array(obs)
        return(obs)
        
    def reset(self, lob_p, qmin_p, smin_p, ssec_p, offset):
        self.state_mm.reset(lob_p, qmin_p, smin_p, ssec_p, offset)
        self.state_ap.reset()
        



        
                






            








