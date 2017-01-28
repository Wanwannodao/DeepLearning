import multiprocessing as mp

def work(queue, data):
    total = 0
    for x in data:
        total += x

    queue.put(total)

if __name__ == "__main__":
    data = [[1,2,3], [4,5,6]]
    queue = mp.Queue()

    workers = []
    for d in data:
        worker = mp.Process(target=work, args=(queue, d,))
        workers.append(worker)
        worker.start()

    [w.join() for w in workers]
    
    print queue.get()
    print queue.get()
