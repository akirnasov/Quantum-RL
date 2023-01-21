import torch
import torch.nn as nn

# number of LSTM layers
LSTM_LAYER_NUM=1

# number of features in LSTM HL for EVENTS
EVENT_LSTM_HL=128

# number of features in LSTM HL for TRADES
TRADES_LSTM_HL=128

# number of neurons for LOB LINEAR layer
LOB_LINEAR_NN=64

# number of features in LSTM HL for CANDLES
CANDLES_LSTM_HL= 64

# number of neurons for Sup/Res vector
SUPRES_LINEAR_NN=64

# number or neurons for Agent Position Linear layer
AP_LINEAR_NN=64


class DQN(nn.Module):
    def __init__(self, lob_level_num, event_len, event_width, candle_len, candle_width, trade_len, trade_width, ap_width, supres_width, actions_n):
        super(DQN, self).__init__()

        self.lob_level_num = lob_level_num
        self.event_len = event_len
        self.event_width = event_width
        self.candle_len = candle_len
        self.candle_width = candle_width
        self.trade_len = trade_len
        self.trade_width = trade_width
        self.ap_width = ap_width
        self.supres_width = supres_width
        self.actions_n = actions_n
        
        # LSTM for events
        self.events_lstm = nn.Sequential(
                    nn.LSTM(event_width, EVENT_LSTM_HL, LSTM_LAYER_NUM)
                )
        
        out_size_events = event_len * EVENT_LSTM_HL
        
         # LSTM for trades
        self.trades_lstm = nn.Sequential(
                    nn.LSTM(trade_width, TRADES_LSTM_HL, LSTM_LAYER_NUM)
                )
        
        out_size_trades = trade_len * TRADES_LSTM_HL
        
        # LSTM for 15 min candles
        self.min15_lstm = nn.Sequential(
                    nn.LSTM(candle_width, CANDLES_LSTM_HL, LSTM_LAYER_NUM)
                )
        
        out_size_min15 = candle_len * CANDLES_LSTM_HL
        
        # LSTM for 1 min candles
        self.min1_lstm = nn.Sequential(
                    nn.LSTM(candle_width, CANDLES_LSTM_HL, LSTM_LAYER_NUM)
                )
        
        out_size_min1 = candle_len * CANDLES_LSTM_HL
        
        # LSTM for 1 sec candles
        self.sec1_lstm = nn.Sequential(
                    nn.LSTM(candle_width, CANDLES_LSTM_HL, LSTM_LAYER_NUM)
                )
        
        out_size_sec1 = candle_len * CANDLES_LSTM_HL

        self.lob_linear = nn.Sequential(
                nn.Linear(4*self.lob_level_num, 64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,10)
                )
        out_size_lob = 10
        
        self.supres_linear = nn.Sequential(
                nn.Linear(supres_width, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
                )
        
        out_size_supres = 10
        
        self.ap_linear = nn.Sequential(
                nn.Linear(ap_width, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
                )      
        
        out_size_ap = 10
        
        out_size = out_size_lob + out_size_events + out_size_trades + out_size_min15 + out_size_min1 + out_size_sec1 + out_size_supres + out_size_ap
        
        
        self.fc_val = nn.Sequential(
            #nn.LSTM(400, 128),
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
           # nn.LSTM(400, 128),
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
                
        lob = x[:, 0, 0, :]
        events = x[:, 1, :, 0:self.event_width]
        trades = x[:, 2, 0:self.trade_len, 0:self.trade_width]
        min15 = x[:, 3, 0:self.candle_len, 0:self.candle_width]
        min1 = x[:, 4, 0:self.candle_len, 0:self.candle_width]
        sec1 = x[:, 5, 0:self.candle_len, 0:self.candle_width]
        supres = x[:, 6, 0, 0:self.supres_width]
        ap = x[:, 7, 0, 0:self.ap_width]
        
        lob_out = self.lob_linear(lob)
        lob_out = lob_out.view(x.size()[0],-1)
        
        events_out, states = self.events_lstm(events)
        events_out = events_out.view(x.size()[0], -1)
        
        trades_out, states = self.trades_lstm(trades)
        trades_out = trades_out.view(x.size()[0], -1)
        
        min15_out, states = self.min15_lstm(min15)
        min15_out = min15_out.view(x.size()[0], -1)
        
        min1_out, states = self.min1_lstm(min1)
        min1_out = min1_out.view(x.size()[0], -1)
        
        sec1_out, states = self.sec1_lstm(sec1)
        sec1_out = sec1_out.view(x.size()[0], -1)
        
        supres_out = self.supres_linear(supres)
        supres_out = supres_out.view(x.size()[0], -1)
        
        ap_out = self.ap_linear(ap)
        ap_out = ap_out.view(x.size()[0], -1)
        
        # concatanate outputs
        total_out = torch.cat((lob_out, events_out, trades_out, min15_out, min1_out, sec1_out, supres_out, ap_out), 1)
        
        # one more linear layer for the output                
        val = self.fc_val(total_out)
        adv = self.fc_adv(total_out)        
        
        return val + adv - adv.mean(dim=1, keepdim=True)
        
        
        
        
        
        
        
            
            
        
        
        









