

from typing import Any
import random
import time
import numpy as np


def batch_index_generator(length, batch_size):
    if length < batch_size:
        return None, None
    for i in range(0, length, batch_size):
        if i + batch_size < length:
            yield i, i + batch_size
        else:
            yield i, length

def batch_interator(samples, batch_size):
    length = len(samples)
    if length < batch_size:
        return None

    for i in range(0, length, batch_size):
        if i + batch_size < length:
            res = samples[i:i + batch_size]
        else:
            res = samples[i:]
        yield res

def test_batch_interator():
    samples = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for batch in batch_interator(samples, 6):
        print(batch)

class SampleBuffer:
    def __init__(self):
        self.mem = []

    def store_transition(self, state, action, reward, state_, done):
        self.mem.append((state, action, reward, state_, done))

    def reset(self):
        self.mem = []

    def last_transition(self):
        return self.mem[-1]

    def size(self):
        return len(self.mem)

    def get_samples(self, start, end):
        states, actions, rewards, states_, dones = zip(*self.mem[start:end])
        return np.array(states), actions, rewards, np.array(states_), dones

    def get_all_samples(self):
        states, actions, rewards, states_, dones = zip(*self.mem)
        return np.array(states), actions, rewards, np.array(states_), dones
         
class NStepBuffer:
    def __init__(self, nstep, gamma):
        self.mem = []
        self.nstep = nstep
        self.gamma = gamma

    def store_transition(self, state, action, reward, state_, done):
        if self.nstep == len(self.mem):
            pass
        else:
            self.mem.append((state, action, reward, state_, done))

    def fold_buffer(self):
        state, action, _, _, _= self.mem[0]
        _, _, _, state_, done = self.mem[-1]
        
        n_reward = 0

        n_reward = sum(self.gamma**i * trans[2] for i, trans in enumerate(self.mem))
        
        # min_len = min(len(self.mem), self.nstep)
        # for i in reversed(range(min_len)):
        #     _, _, r, s_, d = self.mem[i]
        #     n_reward = r + self.gamma * (1 - d) * n_reward
        #     if d:
        #         state_ = s_
        #         done = d
        return [(state, action, n_reward, state_, done)]
        
        # samples = []
        # while len(self.mem) > 0:
        #     r = sum([self.mem[i][2]*(self.gamma**i) for i in range(len(self.mem))])
        #     state, action, _, state_, done = self.mem.pop(0)
        #     samples.append((state, action, r, state_, done))
        # return samples 

        
    def reset(self):
        self.mem = []

class replayBuffer:
    def __init__(self, max_size, nstep=1, gamma=1.0, min_replay_size=0):
        self.max_mem_size = max_size
        self.mem_loc = 0
        self.mem = []

        self.replay_size = min_replay_size
        if self.replay_size > self.max_mem_size or self.replay_size == 0:
            self.replay_size = self.max_mem_size

        if nstep > 1:
            self.nstep_buffer = NStepBuffer(nstep, gamma)
        else:
            self.nstep_buffer = None

    def store_transition(self, state, action, reward, state_, done):
        if self.nstep_buffer:
            self.nstep_buffer.store_transition(state, action, reward, state_, done)
            if len(self.nstep_buffer.mem) == self.nstep_buffer.nstep:
                samples = self.nstep_buffer.fold_buffer()
                for sample in samples:
                    self._store_transition_in_buffter(*sample)
                self.nstep_buffer.reset()
        else:
            self._store_transition_in_buffter(state, action, reward, state_, done)
            
    def _store_transition_in_buffter(self, state, action, reward, state_, done):
        if self.max_mem_size > len(self.mem):
            self.mem.append((state, action, reward, state_, done))
        else:
            index = self.mem_loc
            self.mem[index] = (state, action, reward, state_, done)
            self.mem_loc = (self.mem_loc + 1) % self.max_mem_size

    def sample_buffer(self, batch_size):
        if len(self.mem) < batch_size or len(self.mem) < self.replay_size:
            return None, None, None, None, None
        else:
            batch = random.sample(self.mem, batch_size)
            states, actions, rewards, states_, terminal = zip(*batch)
            return np.array(states), np.array(actions), rewards, np.array(states_), terminal
    
    def size(self):
        return len(self.mem)

    @classmethod
    def test(cls):
        print("ReplayBuffer is working")
        rb = cls(3)
        rb.store_transition(1, 2, 3, 4, 5)
        rb.store_transition(6, 7, 8, 9, 10)
        rb.store_transition(11, 12, 13, 14, 15)
        rb.store_transition(16, 17, 18, 19, 20)
        rb.store_transition(21, 22, 23, 24, 25)

        print(rb.batch_buffer(2))

# SampleBuffer.test() 




