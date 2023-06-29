from structs import PQ

class TestPQ:
    def test_order_is_descending(self):
        pq = PQ()

        pq.push([1])
        pq.push([2])

        el_1 = pq.pop()
        el_2 = pq.pop()

        print(el_1, el_2)
        assert el_1 > el_2

        
