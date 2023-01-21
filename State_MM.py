import numpy as np

class State_MM:
    def __init__(self, lobs, qmins, smins, ssecs, lob_p, qmin_p, smin_p, ssec_p, events_length, trades_length, candles_length):
        
        # prev 100 events and 100 trades
        self.prev_events = []
        self.prev_trades = []

        # prev 10 candles
        self.prev_15min = []
        self.prev_1min = []
        self.prev_1sec = []
        
        # pointers in big arrays
        self.offset = 0
        self.event_p = 0
        self.lob_p = lob_p
        self.qmin_p = qmin_p
        self.smin_p = smin_p
        self.ssec_p = ssec_p
        
        # big arrays
        self.lobs = lobs
        self.qmins = qmins
        self.smins = smins
        self.ssecs = ssecs

        # current lob
        self.lob = self.lobs[lob_p]

        # current timestamp
        self.timestamp = self.lob.open_ms
        self.event_timestamp = 0
        self.trades_timestamp = 0

        # length of events / trades queues
        self.events_length = events_length
        self.trades_length = trades_length
        self.candles_length = candles_length
        
        # assumed max btc price for price normalization
        self.max_price = 50000.0
        # assumed max volume on lob's level
        self.max_size = 500000.0

        self.max_delta = 10.0

        # max update number
        self.max_update_type = 3.0
        # max ms diff type 
        self.max_ms_delta = 100.0
        
        # length of a vector, representing an update
        self.update_encoding_length = 7
        
        # length of a vector, representing a trade
        self.trade_encoding_length = 5
        
        # length of a vector, representing a candle
        self.candle_encoding_length = 4

    def reset(self, lob_p, qmin_p, smin_p, ssec_p, _offset):
        # running structures
        self.lob = self.lobs[lob_p]
        self.lob_p = lob_p
        self.qmin_p = qmin_p
        self.smin_p = smin_p
        self.ssec_p = ssec_p
        self.offset = _offset
        self.event_p = 0
        
        self.prev_15min.clear()
        self.prev_1min.clear()
        self.prev_1sec.clear()
        
        # current timestamp
        self.event_timestamp = self.lob.event_timestamp
        self.trades_timestamp = self.lob.trades_timestamp
        
        self.prev_events.clear()
        self.prev_trades.clear()

        # fill candles queue
        for ind in range(self.candles_length):
            rev_ind = self.candles_length - 1 - ind
            candle_15min = self.qmins[qmin_p - rev_ind]
            self.prev_15min.append(candle_15min)

            candle_1min = self.smins[smin_p - rev_ind]
            self.prev_1min.append(candle_1min)

            candle_1sec = self.ssecs[ssec_p - rev_ind]
            self.prev_1sec.append(candle_1sec)
            
        # step until we have enough events and trades in a queue
        while (len(self.prev_events) < self.events_length) or (len(self.prev_trades) < self.trades_length):
            self.step()
                                        
    def GetLOB(self):
        return self.lob

    def GetEventPointer(self):
        return self.event_p
    
    def step(self):
        # process lob 
        if self.event_p < len(self.lob.event_queue):
            next_event = self.lob.event_queue[self.event_p]
            if next_event.type != 3:
                # apply current event to LOB
                self.lob.ApplyUpdate(next_event)
                self.event_timestamp += next_event.ms_delta
                self.timestamp = self.event_timestamp
                self.prev_events.append(next_event)
                if len(self.prev_events) > self.events_length:
                    self.prev_events.pop(0)
            else:
                # apply current trade 
                self.trades_timestamp += next_event.ms_delta
                self.timestamp = self.trades_timestamp
                # determine trade sign                
                if next_event.size > 0:
                    next_event.is_trade_buy = 1
                else:
                    next_event.is_trade_buy = 0
                self.prev_trades.append(next_event)
                if len(self.prev_trades) > self.events_length:
                    self.prev_trades.pop(0)
        
        # bump pointer in event queue
        if (self.event_p + 1) < len(self.lob.event_queue):
            self.event_p += 1
        else:            
            # shift pointers to LOB and events_queue
            self.lob_p += 1
            if self.lob_p == len(self.lobs):
                # can't shift further
                return True
            else:
                # next 100 ms frame containg new LOB and new event_queue
                self.lob = self.lobs[self.lob_p]
                self.event_timestamp = self.lob.event_timestamp
                self.trades_timestamp = self.lob.trades_timestamp
                self.event_p = 0

        # process candles
        # process 1 sec candles
        if self.timestamp > (self.prev_1sec[self.candles_length - 1].open_ms + 1000):
            self.prev_1sec.pop(0)
            self.ssec_p += 1
            if self.ssec_p == len(self.ssecs):
                return True
            else:
                next_1sec = self.ssecs[self.ssec_p]
                self.prev_1sec.append(next_1sec)

        # process 1 min candles
        if self.timestamp > (self.prev_1min[self.candles_length - 1].open_ms + 60*1000):
            self.prev_1min.pop(0)
            self.smin_p += 1
            if self.smin_p == len(self.smins):
                return True
            else:
                next_1min = self.smins[self.smin_p]
                self.prev_1min.append(next_1min)

        # process 15 min candles
        if self.timestamp > (self.prev_15min[self.candles_length - 1].open_ms + 15*60*1000):
            self.prev_15min.pop(0)
            self.qmin_p += 1
            if self.qmin_p == len(self.qmins):
                return True
            else:
                next_15min = self.qmins[self.qmin_p]
                self.prev_15min.append(next_15min)
            
        return False    
    
    def encode(self):
        # LOB is encoded as vector of bid_prices, bid_sizes, ask_prices, ask_sizes        
        max_price = self.max_price
        max_size = self.max_size
        max_delta = self.max_delta
        max_ms_delta = self.max_ms_delta
        lob = self.lob
        num_levels = lob.num_levels
        max_height = self.events_length
        max_width = 4 * num_levels
        max_shape = (max_height, max_width) 
        res = []
        lob_res = np.zeros(max_shape, dtype=np.float32)
        for ind in range(num_levels):
            lob_res[0][2*ind] = (lob.ask_prices[ind] / max_price)
            lob_res[0][2*ind + 1] = (lob.ask_sizes[ind] / max_size)
        for ind in range(num_levels):
            lob_res[0][2*num_levels + 2*ind] = (lob.bid_prices[ind] / max_price)
            lob_res[0][2*num_levels + 2*ind + 1] = (lob.bid_sizes[ind] / max_size)
        
        res.append(lob_res)
        
        # updates queue is encoded as np.array of shape (100, num_update_fields) - 100 previous updates
        updates_res = np.zeros(max_shape, dtype=np.float32)
        
        event_queue = self.prev_events
        for rev_ind in range(self.events_length):            
            ind = self.events_length - rev_ind - 1
            event = event_queue[ind]
            event_type = event.type
            if event_type == 0:
                updates_res[ind][0] = 0.0
                updates_res[ind][1] = 0.0
            elif event_type == 1:
                updates_res[ind][0] = 0.0
                updates_res[ind][1] = 1.0    
            elif event_type == 2:
                updates_res[ind][0] = 1.0
                updates_res[ind][1] = 0.0
                                
            updates_res[ind][2] = event.price / max_price
            updates_res[ind][3] = event.size / max_size
            updates_res[ind][4] = event.size_delta / max_delta
            updates_res[ind][5] = (float)(event.level) / (float)(num_levels)            
            updates_res[ind][6] = (float)(event.ms_delta) / (float)(max_ms_delta)
            
        res.append(updates_res)
        
        # trades queue is encoded as np.array of shape (100, num_trade_fields) - 100 previous trades       
        trades_res = np.zeros(max_shape, dtype=np.float32)
        
        trade_queue = self.prev_trades
        for rev_ind in range(self.trades_length):
            ind = self.trades_length - rev_ind - 1
            trade = trade_queue[ind]            
                                
            trades_res[ind][0] = trade.price / max_price
            trades_res[ind][1] = trade.size / max_size
            trades_res[ind][2] = (float)(trade.level) / (float)(num_levels)
            trades_res[ind][3] = (float)(trade.ms_delta) / (float)(max_ms_delta)
            trades_res[ind][4] = (float)(trade.is_trade_buy)
            
        res.append(trades_res)
        
        # se support/ resistance levels on 15 min, 1 min and 1 sec frequency
        sup_res = np.zeros(max_shape, dtype=np.float32)
        
        # encode candles - candle queue is encoded as np.array of shape (10, num_candle_fields) - 10 previous candles
        # 15 min candles     
        qmin_res = np.zeros(max_shape, dtype=np.float32)        
        qmin_queue = self.prev_15min
        
        for rev_ind in range(self.candles_length):            
            ind = self.candles_length - rev_ind - 1
            candle = qmin_queue[ind]            
                                
            qmin_res[ind][0] = candle.open / max_price
            qmin_res[ind][1] = candle.close / max_price
            qmin_res[ind][2] = candle.max / max_price
            qmin_res[ind][3] = candle.min / max_price
            if ind == 0:
                sup_res[0][0] = candle.running_max / max_price
                sup_res[0][1] = candle.running_min / max_price
            
        res.append(qmin_res)
        
        # 1 min candles        
        min_res = np.zeros(max_shape, dtype=np.float32)        
        min_queue = self.prev_1min
        
        for rev_ind in range(self.candles_length):            
            ind = self.candles_length - rev_ind - 1
            candle = min_queue[ind]            
                                
            min_res[ind][0] = candle.open / max_price
            min_res[ind][1] = candle.close / max_price
            min_res[ind][2] = candle.max / max_price
            min_res[ind][3] = candle.min / max_price
            if ind == 0:
                sup_res[0][2] = candle.running_max / max_price
                sup_res[0][3] = candle.running_min / max_price
            
        res.append(min_res)
        
        # 1 sec candles      
        sec_res = np.zeros(max_shape, dtype=np.float32)        
        sec_queue = self.prev_1sec
        
        for rev_ind in range(self.candles_length):            
            ind = self.candles_length - rev_ind - 1
            candle = sec_queue[ind]            
                                
            sec_res[ind][0] = candle.open / max_price
            sec_res[ind][1] = candle.close / max_price
            sec_res[ind][2] = candle.max / max_price
            sec_res[ind][3] = candle.min / max_price
            if ind == 0:
                sup_res[0][4] = candle.running_max / max_price
                sup_res[0][5] = candle.running_min / max_price
            
        res.append(sec_res)
        
        res.append(sup_res)
        
        return res
        

        
        
        
            
            
            
            
            
                        
            
            
            
            
            
            
            
            
            
            
            
        
        
        
            
        
        









