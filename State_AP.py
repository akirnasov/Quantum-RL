import numpy as np

class State_AP:
    def __init__(self, encoding_shape):
        self.sell_order = False
        self.sell_size = 0
        self.buy_order = False
        self.buy_size = 0
        self.position = 0
        self.max_position= 10000.0
        self.max_shape = encoding_shape
        
        # number of fields, encoding position
        self.num_position_fields = 5


    def reset(self):
        self.sell_order = False
        self.sell_size = 0
        self.buy_order = False
        self.buy_size = 0
        self.position = 0
    
    def compute_new_level_size(self, event, lob, is_sell):
        
        reward = 0.0
        remaining_size = 0
        if is_sell == 1:
            remaining_size = self.sell_size
        else:
            remaining_size = self.buy_size

        if event.type == 3:
            # this is a trade
            # check trade direction:
            is_trade_buy = 0
            if event.size > 0:
                is_trade_buy = 1

            # check that trade is matching currently checked limit order 
            if is_trade_buy == 0 and is_sell == 1:
                return reward

            if is_trade_buy == 1 and is_sell == 0:
                return reward

            if  abs(event.size) >= remaining_size:
                # we compute reward in 1000's of Satoshi (10^-5 BTC)                
                rebate = (1500.0/event.price)     
                # assume our order will be also taken
                if is_sell == 1:
                    self.sell_order = 0
                    self.sell_size = 0
                    self.position -= 100
                    reward = (-10000000.0/event.price) + rebate
                else:
                    self.buy_order = 0
                    self.buy_size = 0
                    self.position += 100
                    reward = (10000000.0/event.price) + rebate
        else:
            # regular event type
            # we are only interested in events on best bid and ask levels matching limit order direction
            if event.level != 1 and event.level != -1:
                return reward
            if event.level == 1 and is_sell == 0:
                return reward
            if event.level == -1 and is_sell == 1:
                return reward

            if event.type == 2:
                
                level_size = 0
                event_size = abs(event.size)
                if event.level == 1:
                    level_size = lob.ask_sizes[0]
                else:
                    level_size = lob.bid_sizes[0]

                if event_size >= level_size:
                    # update is increasing size
                    return reward 
                else:
                    # update is decreasing size
                    # need to compute new estimated remaining size
                    ratio = (float)(remaining_size) / (float)(level_size)
                    delta = (int)(ratio * (float)(level_size - event_size))
                    remaining_size -= delta
                    if is_sell == 1:
                        self.sell_size = remaining_size
                    else:
                        self.buy_size = remaining_size
            # insert event
            elif event.type == 1:
                if is_sell == 1:
                    self.sell_size = event.size
                else:
                    self.buy_size = event.size
            # delete level event
            else:
                if is_sell == 1:
                    self.sell_size = lob.ask_sizes[1]
                else:
                    self.buy_size = lob.bid_sizes[1]
           

        return reward 


    # we supply action, event and lob to find next agent position
    def step(self, a, lob, event):
        
        reward = 0
        
        done = False

        if a[2] == 1:
            done = True
            price = 0.0                        
            # request to close position
            if self.position != 0:
                if self.position > 0:
                    # closing long position with best bid price
                    price = lob.bid_prices[0]
                elif self.position < 0:
                    # closing short position with best ask price
                    price = lob.ask_prices[0]

                commission = (7500.0/price)
                reward = ((-1.0) * ((float)(self.position) / price)) - commission
            else:
                reward = 0
            return reward, done                  
        
        # first check what limit orders remain and what orders would be canceled
        if a[0] == 0:
            # cancell sell order
            self.sell_order = 0
            self.sell_size = 0
        else:
            # check if that is a new sell order
            if self.sell_order == 0:
                # set size based on lob 
                self.sell_order = 1
                self.sell_size = lob.ask_sizes[0]
            else:
                # if order existed before - check how size is possibly altered
                reward = self.compute_new_level_size(event, lob, 1)
                
        if a[1] == 0:
            # cancel buy order 
            self.buy_order = 0
            self.buy_size = 0
        else:
            # check if that is a new buy order
            if self.buy_order == 0:
                # set size based on lob
                self.buy_order = 1
                self.buy_size = lob.bid_sizes[0]
            else:
                reward = self.compute_new_level_size(event, lob, 0)

        return reward, done 
    
    def encode(self):
        res = []        
        ap_res = np.zeros(self.max_shape, dtype=np.float32)
       
        ap_res[0][0] = (float)(self.sell_order) 
        ap_res[0][1] = (float)(self.sell_size) / (float)(self.max_position)        
        ap_res[0][2] = (float)(self.buy_order) 
        ap_res[0][3] = (float)(self.buy_size) / (float)(self.max_position)        
        ap_res[0][4]= (float)(self.position) / (float)(self.max_position)
        
        res.append(ap_res)
        
        return res
        





            








