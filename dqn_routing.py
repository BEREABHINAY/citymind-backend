"""
dqn_routing.py
--------------
Deep Q-Network route planner: every traveller gives a start point and an end
point; the agent picks the best path considering DISTANCE, TIME and live
TRAFFIC on each road segment (fed by classifier.py's traffic-level output in
the full system).

Function approximator: scikit-learn MLPRegressor used as the Q-network
(no torch/GPU needed -> runs anywhere, including a CI box with no internet).
Trained with epsilon-greedy exploration + experience replay, exactly like a
standard DQN, just with a lighter backend. Swap in PyTorch/torch.nn for a
GPU-scale version if your final report wants that framework name explicitly
-- the algorithm and state/action/reward design below stay identical.

STATE:   one-hot(current_node) + one-hot(destination_node) + traffic level
          (0/0.33/0.66/1.0) of every outgoing edge from current_node
ACTION:  choose one of the (up to MAX_DEGREE) outgoing edges from current_node
REWARD:  -(w_dist*distance + w_time*time + w_traffic*traffic_penalty) per hop
          + big terminal bonus on reaching destination
          - penalty for revisiting a node (avoids loops)
"""
import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from collections import deque

random.seed(7)
np.random.seed(7)

# ---------------------------------------------------------- CITY ROAD GRAPH --
# distance_km, base_time_min, traffic (0=light .. 1=heavy) per directed edge.
# Node names match the "named locations" used in the web app (see README).
EDGES = [
    ("MG Road Junction", "Lakeview Apartments", 1.2, 4, 0.2),
    ("Lakeview Apartments", "MG Road Junction", 1.2, 4, 0.2),
    ("MG Road Junction", "Central Hospital", 2.0, 6, 0.5),
    ("Central Hospital", "MG Road Junction", 2.0, 6, 0.5),
    ("MG Road Junction", "Tech Park Circle", 3.1, 9, 0.7),
    ("Tech Park Circle", "MG Road Junction", 3.1, 9, 0.7),
    ("Tech Park Circle", "Silicon Heights IT Park", 0.8, 3, 0.4),
    ("Silicon Heights IT Park", "Tech Park Circle", 0.8, 3, 0.4),
    ("Tech Park Circle", "Riverside Bridge", 2.5, 7, 0.3),
    ("Riverside Bridge", "Tech Park Circle", 2.5, 7, 0.3),
    ("Riverside Bridge", "Greenfield Apartments", 1.4, 5, 0.2),
    ("Greenfield Apartments", "Riverside Bridge", 1.4, 5, 0.2),
    ("Central Hospital", "Market Street", 1.0, 3, 0.6),
    ("Market Street", "Central Hospital", 1.0, 3, 0.6),
    ("Market Street", "Old Town Square", 0.9, 3, 0.8),
    ("Old Town Square", "Market Street", 0.9, 3, 0.8),
    ("Old Town Square", "Lakeview Apartments", 1.8, 5, 0.3),
    ("Lakeview Apartments", "Old Town Square", 1.8, 5, 0.3),
    ("Market Street", "Riverside Bridge", 2.2, 6, 0.4),
    ("Riverside Bridge", "Market Street", 2.2, 6, 0.4),
    ("Tech Park Circle", "Industrial Dump Yard Road", 4.0, 11, 0.1),
    ("Industrial Dump Yard Road", "Tech Park Circle", 4.0, 11, 0.1),
    ("Central Hospital", "Sunrise Apartments", 1.6, 5, 0.3),
    ("Sunrise Apartments", "Central Hospital", 1.6, 5, 0.3),
    ("Sunrise Apartments", "Old Town Square", 2.1, 6, 0.5),
    ("Old Town Square", "Sunrise Apartments", 2.1, 6, 0.5),
    ("Central Hospital", "Sunrise Multispecialty Hospital", 1.1, 3, 0.3),
    ("Sunrise Multispecialty Hospital", "Central Hospital", 1.1, 3, 0.3),
    ("Tech Park Circle", "City Grand Hotel", 1.3, 4, 0.3),
    ("City Grand Hotel", "Tech Park Circle", 1.3, 4, 0.3),
    ("Market Street", "Spice Route Restaurant", 0.7, 2, 0.5),
    ("Spice Route Restaurant", "Market Street", 0.7, 2, 0.5),
    ("Old Town Square", "Neon Cafe", 0.9, 3, 0.4),
    ("Neon Cafe", "Old Town Square", 0.9, 3, 0.4),
    ("MG Road Junction", "Northgate Apartments", 2.4, 7, 0.3),
    ("Northgate Apartments", "MG Road Junction", 2.4, 7, 0.3),
    ("Riverside Bridge", "Harbor View Hotel", 1.7, 5, 0.2),
    ("Harbor View Hotel", "Riverside Bridge", 1.7, 5, 0.2),
    ("Riverside Bridge", "Blue Moon Diner", 1.1, 3, 0.4),
    ("Blue Moon Diner", "Riverside Bridge", 1.1, 3, 0.4),
    ("City Grand Hotel", "Westside General Hospital", 1.6, 5, 0.3),
    ("Westside General Hospital", "City Grand Hotel", 1.6, 5, 0.3),
    ("Silicon Heights IT Park", "Pixel IT Hub", 1.2, 4, 0.4),
    ("Pixel IT Hub", "Silicon Heights IT Park", 1.2, 4, 0.4),
    ("Market Street", "Central Park Avenue", 0.8, 3, 0.5),
    ("Central Park Avenue", "Market Street", 0.8, 3, 0.5),
]

NODES = sorted(set(a for a, *_ in EDGES) | set(b for _, b, *_ in EDGES))
NODE_IDX = {n: i for i, n in enumerate(NODES)}
N = len(NODES)
MAX_DEGREE = 5  # max outgoing roads we support per intersection in the state/action vector

ADJ = {n: [] for n in NODES}
for a, b, dist, t, traf in EDGES:
    ADJ[a].append({"to": b, "dist": dist, "time": t, "traffic": traf})

W_DIST, W_TIME, W_TRAFFIC = 0.4, 0.4, 6.0  # reward weights (traffic penalized hardest)


def encode_state(current, dest, live_traffic):
    """live_traffic: dict edge(from,to)->override level, or None to use static graph value"""
    vec = np.zeros(2 * N + MAX_DEGREE)
    vec[NODE_IDX[current]] = 1
    vec[N + NODE_IDX[dest]] = 1
    neighbors = ADJ[current]
    for i, edge in enumerate(neighbors[:MAX_DEGREE]):
        key = (current, edge["to"])
        traf = live_traffic.get(key, edge["traffic"]) if live_traffic else edge["traffic"]
        vec[2 * N + i] = traf
    return vec


def dijkstra_route(start, dest, live_traffic=None, blocked_edges=None):
    """Classical shortest-path baseline over the SAME reward-shaped edge cost the DQN was
    trained on (distance + time + heavy traffic penalty). Used as a safety net: a greedy
    rollout of a learned policy is not guaranteed to find the optimal path (especially right
    after a road gets blocked, if the agent wasn't trained on that exact scenario), so
    production routing takes the better of the DQN rollout and this exact solver. This is
    standard practice for deploying RL in a system that must always return a valid answer.
    blocked_edges: set of (from,to) tuples that are hard-closed (fire/accident) -- these are
    excluded from the graph entirely, not just penalized, matching how a real road closure
    behaves (as opposed to merely heavy traffic, which is slow but still passable)."""
    import heapq
    blocked_edges = blocked_edges or set()
    dist = {n: float("inf") for n in NODES}
    prev = {}
    dist[start] = 0
    pq = [(0, start)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == dest:
            break
        for edge in ADJ[u]:
            v = edge["to"]
            if (u, v) in blocked_edges:
                continue  # road closed -- do not traverse at all
            traf = live_traffic.get((u, v), edge["traffic"]) if live_traffic else edge["traffic"]
            cost = W_DIST * edge["dist"] + W_TIME * edge["time"] + W_TRAFFIC * traf
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = (u, edge, traf)
                heapq.heappush(pq, (nd, v))
    if dist[dest] == float("inf"):
        return {"path": [], "reached": False, "distance_km": 0, "estimated_time_min": 0, "avg_traffic": 0}
    path = [dest]
    total_dist = total_time = total_traf = 0.0
    cur = dest
    hops = 0
    while cur != start:
        u, edge, traf = prev[cur]
        total_dist += edge["dist"]
        total_time += edge["time"] * (1 + traf)
        total_traf += traf
        hops += 1
        cur = u
        path.append(cur)
    path.reverse()
    return {"path": path, "reached": True, "distance_km": round(total_dist, 2),
            "estimated_time_min": round(total_time, 1),
            "avg_traffic": round(total_traf / max(1, hops), 2)}


def _route_cost(result):
    if not result["reached"]:
        return float("inf")
    return (W_DIST * result["distance_km"] + W_TIME * result["estimated_time_min"] / (1 + result["avg_traffic"])
            + W_TRAFFIC * result["avg_traffic"] * max(1, len(result["path"]) - 1))


class DQNRouter:
    def __init__(self):
        self.q_net = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1,
                                   warm_start=True, learning_rate_init=0.01, random_state=7)
        self._bootstrap()
        self.replay = deque(maxlen=4000)

    def _bootstrap(self):
        # one dummy fit so partial_fit-style incremental training works from step 1
        dummy_X = np.zeros((2, 2 * N + MAX_DEGREE))
        dummy_y = np.zeros((2, MAX_DEGREE))
        self.q_net.fit(dummy_X, dummy_y)

    def q_values(self, state):
        return self.q_net.predict([state])[0]

    def choose_action(self, current, epsilon):
        neighbors = ADJ[current][:MAX_DEGREE]
        if random.random() < epsilon or not neighbors:
            return random.randrange(max(1, len(neighbors)))
        state = encode_state(current, self._dest, self._traffic)
        q = self.q_values(state)[:len(neighbors)]
        return int(np.argmax(q))

    def train(self, episodes=1800, epsilon_start=0.9, epsilon_end=0.05):
        for ep in range(episodes):
            epsilon = max(epsilon_end, epsilon_start * (1 - ep / episodes))
            start, dest = random.sample(NODES, 2)
            self._dest = dest
            # random live-traffic snapshot per episode (simulates classifier.py output feed)
            self._traffic = {(a, b): random.choice([0.1, 0.3, 0.5, 0.8, 1.0])
                              for a, b, *_ in EDGES}
            current = start
            visited = {current}
            for step in range(15):
                neighbors = ADJ[current][:MAX_DEGREE]
                if not neighbors:
                    break
                state = encode_state(current, dest, self._traffic)
                action = self.choose_action(current, epsilon)
                action = min(action, len(neighbors) - 1)
                edge = neighbors[action]
                nxt = edge["to"]
                traf = self._traffic.get((current, nxt), edge["traffic"])
                cost = W_DIST * edge["dist"] + W_TIME * edge["time"] + W_TRAFFIC * traf
                reward = -cost
                if nxt in visited:
                    reward -= 5  # discourage loops
                if nxt == dest:
                    reward += 20
                visited.add(nxt)
                next_state = encode_state(nxt, dest, self._traffic)
                self.replay.append((state, action, reward, next_state, nxt == dest, len(ADJ[nxt][:MAX_DEGREE])))
                current = nxt
                if current == dest:
                    break
            if len(self.replay) >= 64 and ep % 2 == 0:
                self._replay_update()

    def _replay_update(self, batch_size=64, gamma=0.9):
        batch = random.sample(self.replay, min(batch_size, len(self.replay)))
        X, Y = [], []
        for state, action, reward, next_state, done, n_next_actions in batch:
            q = self.q_values(state)
            target = reward
            if not done and n_next_actions > 0:
                target += gamma * np.max(self.q_values(next_state)[:n_next_actions])
            q[action] = target
            X.append(state); Y.append(q)
        self.q_net.partial_fit(X, Y)

    def find_route(self, start, dest, live_traffic=None, max_hops=15):
        """Greedy rollout of the learned DQN policy for a single traveller."""
        current = start
        path = [current]
        total_dist = total_time = total_traffic_cost = 0.0
        visited = set()
        for _ in range(max_hops):
            if current == dest:
                break
            neighbors = ADJ[current][:MAX_DEGREE]
            if not neighbors:
                break
            state = encode_state(current, dest, live_traffic)
            q = self.q_values(state)[:len(neighbors)]
            # avoid immediate backtrack loops in greedy rollout
            order = np.argsort(-q)
            chosen = None
            for idx in order:
                cand = neighbors[idx]["to"]
                if cand not in visited or cand == dest:
                    chosen = neighbors[idx]
                    break
            if chosen is None:
                break  # every neighbor already visited -> dead end, stop instead of looping
            traf = live_traffic.get((current, chosen["to"]), chosen["traffic"]) if live_traffic else chosen["traffic"]
            total_dist += chosen["dist"]
            total_time += chosen["time"] * (1 + traf)  # traffic slows real travel time
            total_traffic_cost += traf
            visited.add(current)
            current = chosen["to"]
            path.append(current)
        reached = current == dest
        return {"path": path, "reached": reached, "distance_km": round(total_dist, 2),
                "estimated_time_min": round(total_time, 1),
                "avg_traffic": round(total_traffic_cost / max(1, len(path) - 1), 2)}

    def find_best_route(self, start, dest, live_traffic=None, blocked_edges=None, max_hops=15):
        """Production entry point: runs the trained DQN policy AND the Dijkstra safety net,
        returns whichever has lower reward-shaped cost, and reports which one was used plus
        both raw results -- so you can show, e.g. in a paper's results table, how often the
        learned policy alone already matches the optimal solver."""
        dqn_result = self.find_route(start, dest, live_traffic=live_traffic, max_hops=max_hops)
        exact_result = dijkstra_route(start, dest, live_traffic=live_traffic, blocked_edges=blocked_edges)
        # a DQN rollout that uses a hard-closed road is invalid outright, regardless of its cost
        blocked_edges = blocked_edges or set()
        dqn_uses_blocked = any((dqn_result["path"][i], dqn_result["path"][i+1]) in blocked_edges
                                for i in range(len(dqn_result["path"]) - 1))
        dqn_cost = float("inf") if (dqn_uses_blocked or not dqn_result["reached"]) else _route_cost(dqn_result)
        exact_cost = _route_cost(exact_result)
        used = "dqn_policy" if dqn_cost <= exact_cost else "dijkstra_safety_net"
        best = dqn_result if used == "dqn_policy" else exact_result
        return {**best, "usedSolver": used, "dqnPolicyPath": dqn_result["path"],
                "dqnReached": dqn_result["reached"], "exactPath": exact_result["path"]}


if __name__ == "__main__":
    print("Training DQN router on", N, "nodes /", len(EDGES), "directed road segments...")
    router = DQNRouter()
    router.train(episodes=400)
    print("Training complete.\n")

    demo_trips = [
        ("Sunrise Apartments", "Silicon Heights IT Park"),
        ("Lakeview Apartments", "Central Hospital"),
        ("Greenfield Apartments", "Old Town Square"),
    ]
    # simulate a live traffic snapshot as if it came from classifier.py just now
    live_traffic = {(a, b): random.choice([0.1, 0.2, 0.9, 1.0]) for a, b, *_ in EDGES}

    for start, dest in demo_trips:
        result = router.find_route(start, dest, live_traffic=live_traffic)
        print(f"TRIP: {start} -> {dest}")
        print(f"  Route found : {'yes' if result['reached'] else 'NO PATH (needs more training/hops)'}")
        print(f"  Path        : {' -> '.join(result['path'])}")
        print(f"  Distance    : {result['distance_km']} km")
        print(f"  Est. time   : {result['estimated_time_min']} min (traffic-adjusted)")
        print(f"  Avg traffic on route: {result['avg_traffic']}")
        print()
