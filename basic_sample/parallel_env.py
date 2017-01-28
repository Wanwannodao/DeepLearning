import multiprocessing as mp
import gym

# this sample launchs N OpenAI gym environment 

ENV_NAME='CartPole-v0'

def work(render):
    env = gym.make(ENV_NAME)
    obserbation = env.reset()
    if render:
        env.render()

if __name__ == "__main__":
    workers = []
    for _ in xrange(mp.cpu_count()):
        worker = mp.Process(target=work, args=(True,))
        workers.append(worker)
        worker.start()

    
