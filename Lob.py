import csv

from collections import deque
from Update import Update

class Lob:
    def __init__(self, open_ms, ask_prices, ask_sizes,  bid_prices, bid_sizes, event_queue, num_levels):
        self.open_ms = open_ms
        self.ask_prices = ask_prices
        self.ask_sizes = ask_sizes
        self.bid_prices = bid_prices
        self.bid_sizes = bid_sizes        
        self.event_queue = event_queue
        self.num_levels = num_levels
        self.event_timestamp = 0
        self.trades_timestamp = 0

    def print(self):
        print(self.open_ms)
        for ind in range(self.num_levels):
            rev_ind = self.num_levels - ind - 1
            print(self.ask_prices[rev_ind], self.ask_sizes[rev_ind])
        print('---------')
        for ind in range(self.num_levels):    
            print(self.bid_prices[ind],self.bid_sizes[ind])    

    def Cleanup(self, is_bid):
        if is_bid:
            # find 0 position or last position
            found = False
            for ind in range(len(self.bid_prices)):
                if self.bid_sizes[ind] == 0:
                    self.bid_prices.pop(ind)
                    self.bid_sizes.pop(ind)
                    found = True
                    break
                if found:
                    ind = self.num_levels
                    self.bid_prices.pop(ind)
                    self.bid_sizes.pop(ind)
        else:
            # find 0 position or last position
            found = False
            for ind in range(len(self.ask_prices)):
                if self.ask_sizes[ind] == 0:
                    self.ask_prices.pop(ind)
                    self.ask_sizes.pop(ind)
                    found = True
                    break
                if found:
                    ind = self.num_levels
                    self.ask_prices.pop(ind)
                    self.ask_sizes.pop(ind)

    def ApplyUpdate(self, update):

        # skip updates on high levels
        if abs(update.level) > len(self.ask_prices):
            return

        if (update.type == 2 or update.type == 1):
            # find update pos
            if (update.level < 0):
                for ind in range(len(self.bid_prices)):
                    if self.bid_prices[ind] == update.price:
                        if update.type == 2:
                            self.bid_sizes[ind] = update.size
                        else:
                            self.bid_sizes[ind] = 0
                        break
            else:
                for ind in range(len(self.ask_prices)):
                    if self.ask_prices[ind] == update.price:
                        if update.type == 2:
                            self.ask_sizes[ind] = abs(update.size)
                        else:
                            self.ask_sizes[ind] = 0
                        break
        else:
            if (update.level < 0):
                # find insert position
                found = False
                for ind in range(len(self.bid_prices)):
                    if self.bid_prices[ind] < update.price:
                        self.bid_prices.insert(ind, update.price)
                        self.bid_sizes.insert(ind, update.size)
                        found = True
                        break
                if found:
                    self.Cleanup(True)
            else:
                # find insert position
                found = False
                for ind in range(len(self.ask_prices)):
                    if self.ask_prices[ind] > update.price:
                        self.ask_prices.insert(ind, update.price)
                        self.ask_sizes.insert(ind, abs(update.size))
                        found = True
                        break
                if found:
                    self.Cleanup(False)

    def AddUpdates(self, update_queue, global_event_ms, global_trade_ms, global_time_ms):
        if self.event_timestamp == 0:
            self.event_timestamp = global_event_ms
        if self.trades_timestamp == 0:
            self.trades_timestamp = global_trade_ms
        for update in update_queue:
            type_elem = update.type
            global_time_ms += update.ms_delta
            if type_elem == 3:
                # trade type
                if global_trade_ms == 0:
                    global_trade_ms = global_time_ms
                    continue
                else:
                    trade_delta = global_time_ms - global_trade_ms
                    global_trade_ms = global_time_ms
                    update.ms_delta = trade_delta
            else:
                # event type
                if global_event_ms == 0:
                    global_event_ms = global_time_ms
                    continue
                else:
                    update_delta = global_time_ms - global_event_ms
                    global_event_ms = global_time_ms
                    update.ms_delta = update_delta

            self.event_queue.append(update)

        return global_event_ms, global_trade_ms, global_time_ms


def CreateEventQueueFromRow(row):

    event_queue = deque()

    total_elem = len(row) - 1
    num_elem = total_elem // 6

    for ind in range(num_elem):
        type_ind = 6*ind + 1
        type_elem = (int)(row[type_ind])
        price_ind = 6*ind + 2
        price = (float)(row[price_ind])
        level_ind = 6*ind + 3
        level = (int)(row[level_ind])
        size_ind = 6*ind + 4
        size = (float)(row[size_ind])
        size_delta_ind = 6*ind + 5
        size_delta = (float)(row[size_delta_ind])
        ms_delta_ind = 6*ind+6
        ms_delta = (int)(row[ms_delta_ind])

        update = Update(type_elem, price, level, size, size_delta, ms_delta)
        event_queue.appendleft(update)

    return event_queue


def IsLobRow(row):
    size = (float)(row[2])

    if (size < 0):
        return True
    else:
        return False


def CreateLOBFromRow(row, num_levels):

    ask_prices = []
    ask_sizes = []
    bid_prices = []
    bid_sizes = []
    event_queue = []

    open_ms = (int)(row[0])

    row_len = len(row)
    num_lob_elem = (row_len - 1) // 2

    num_ask = 0
    num_bid = 0
    for ind in range(num_lob_elem):
        price = (float)(row[2 * (ind) + 1])
        size = (int)(row[2 * (ind) + 2])

        if size < 0:
            num_ask += 1
            ask_prices.append(price)
            ask_sizes.append(-size)
        else:
            num_bid += 1
            bid_prices.append(price)
            bid_sizes.append(size)

    # fix rows having less than num_levels (price,size) pairs
    if num_ask < num_levels:
        last_ask_price = ask_prices[num_ask - 1]
        last_ask_size = ask_sizes[num_ask - 1]

        while num_ask < num_levels:
            next_ask_price = last_ask_price + 0.5
            ask_prices.append(next_ask_price)
            ask_sizes.append(last_ask_size)
            num_ask += 1

    if num_bid < num_levels:
        last_bid_price = bid_prices[num_bid - 1]
        last_bid_size = bid_sizes[num_bid - 1]

        while num_bid < num_levels:
            next_bid_price = last_bid_price - 0.5
            bid_prices.append(next_bid_price)
            bid_sizes.append(last_bid_size)
            num_bid += 1

    lob = Lob(open_ms, ask_prices, ask_sizes, bid_prices, bid_sizes, event_queue, num_levels)

    return lob


def ReadLobs(lobs_file, num_levels=10):
    with open(lobs_file, 'rt', encoding='utf-8') as fd:
        reader = csv.reader(fd)
        lobs = []
        lob_order = []

        prev_lob = None
        next_lob = None

        ind = 0
        global_trade_ms = 0
        global_event_ms = 0
        global_time_ms = 0

        for row in reader:
            print(ind)
            ind += 1
            # TODO remove this test condition
           # if ind > 300000:
              #  break
            if IsLobRow(row):
                # lob row
                new_lob = CreateLOBFromRow(row, num_levels)
                if next_lob is None:
                    next_lob = new_lob
                else:
                    if prev_lob is None:
                        prev_lob = next_lob
                        next_lob = new_lob
                    else:
                        lobs.append(prev_lob)
                        lob_order.append(prev_lob.open_ms)
                        prev_lob = next_lob
                        next_lob = new_lob
            else:
                # update row
                if prev_lob is None:
                    global_time_ms = (int)(row[0])
                    continue
                else:
                    update_queue = CreateEventQueueFromRow(row)
                    global_event_ms, global_trade_ms, global_time_ms = prev_lob.AddUpdates(update_queue, global_event_ms, global_trade_ms, global_time_ms)

    return lob_order, lobs








