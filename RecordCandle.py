import csv

class RecordCandle:
    def __init__(self, open_ms, open, max, min, close, running_max, running_min):
        self.open_ms = open_ms
        self.open = open
        self.max = max
        self.min = min
        self.close = close
        self.running_max = running_max
        self.running_min = running_min

    def print(self):
        print(self.open_ms, self.open, self.max, self.min, self.close)
        print(self.running_max,self.running_min)


def ReadCandles(candle_file):
    with open(candle_file,'rt',encoding='utf-8') as fd:
        reader = csv.reader(fd)
        candles = []
        candle_order = []        
        for row in reader:
            open_ms = (int)(row[0])
            open_p = (float)(row[1])
            max = (float)(row[2])
            min = (float)(row[3])
            close = (float)(row[4])
            run_max = (float)(row[5])
            run_min = (float)(row[6])
            cur_candle = RecordCandle(open_ms,open_p,max,min,close,run_max,run_min)
            candles.append(cur_candle)
            candle_order.append(open_ms)            


    return candle_order,candles









