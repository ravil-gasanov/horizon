from queue import PriorityQueue

class PQ(PriorityQueue):
    '''
    Priority Queue: 
    - descending order
    - element must be a list
    '''
    def push(self, item):
        item[0] = -item[0]
        self.put_nowait(item)

    def pop(self):
        item = self.get_nowait()
        item[0] = -item[0]
        return item