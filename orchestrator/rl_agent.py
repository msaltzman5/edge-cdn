import random
from collections import defaultdict

class PrefetchQLearning:
    def __init__(self, actions=None, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions or ["prefetch", "no_prefetch"]

    def _key(self, state, action):
        return (state, action)

    def get_q(self, state, action):
        return self.q[self._key(state, action)]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        # greedy
        values = {a: self.get_q(state, a) for a in self.actions}
        return max(values, key=values.get)

    def update(self, state, action, reward, next_state):
        old = self.get_q(state, action)
        next_max = max(self.get_q(next_state, a) for a in self.actions)
        new = old + self.alpha * (reward + self.gamma * next_max - old)
        self.q[self._key(state, action)] = new
