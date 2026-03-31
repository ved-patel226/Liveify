import itertools
import numpy as np

drivers = [
    ("Russell", 29.5, 164),
    ("Antonelli", 28.7, 173),
    ("Leclerc", 26.3, 162),
    ("Norris", 24.7, 118),
    ("Verstappen", 24.6, 133),
    ("Piastri", 24.1, 91),
    ("Hamilton", 23.4, 158),
    ("Gasly", 21.7, 143),
    ("Lawson", 18.0, 140),
    ("Hadjar", 17.7, 117),
    ("Alonso", 16.2, 79),
    ("Sainz", 14.9, 115),
    ("Hulkenberg", 13.3, 95),
    ("Ocon", 12.7, 130),
    ("Bortoleto", 11.5, 98),
    ("Bearman", 11.2, 142),
    ("Stroll", 11.0, 69),
    ("Albon", 9.9, 70),
    ("Lindblad", 9.7, 131),
    ("Colapinto", 9.0, 117),
    ("Perez", 7.5, 97),
]
constructors = [
    ("Merc", 29.3, 175),
    ("Ferrari", 24.6, 161),
    ("Racing Bulls", 17.3, 120),
    ("Alpine", 17.5, 120),
    ("Red Bull", 21.8, 116),
    ("Haas", 11.2, 114),
    ("McLaren", 23.8, 106),
    ("Audi", 10.8, 92),
    ("Cadillac", 8.4, 80),
    ("Williams", 12.7, 80),
    ("Aston Martin", 12.8, 69),
]
BUDGET = 107.2

d_names = [d[0] for d in drivers]
d_costs = np.array([d[1] for d in drivers])
d_pts = np.array([d[2] for d in drivers])
d_cheap = d_costs < 18  # eligible for doubling

c_names = [c[0] for c in constructors]
c_costs = np.array([c[1] for c in constructors])
c_pts = np.array([c[2] for c in constructors])

combos = np.array(list(itertools.combinations(range(len(drivers)), 5)))

combo_costs = d_costs[combos].sum(axis=1)  # shape (20349,)
combo_pts = d_pts[combos].sum(axis=1)

eligible_mask = d_cheap[combos]  # (20349, 5) bool
eligible_pts = np.where(eligible_mask, d_pts[combos], 0)
best_double = eligible_pts.max(axis=1)  # (20349,)

min_c_cost = c_costs.min()
valid_combos = combo_costs <= BUDGET - min_c_cost  # prune early

best_score = -1
best_result = None

for ci, (cn, cc, cp) in enumerate(zip(c_names, c_costs, c_pts)):
    mask = valid_combos & (combo_costs + cc <= BUDGET)
    if not mask.any():
        continue
    total_pts = combo_pts[mask] + cp + best_double[mask]
    idx = total_pts.argmax()
    score = total_pts[idx]
    if score > best_score:
        best_score = score
        combo_idx = np.where(mask)[0][idx]
        doubled_idx = (
            eligible_pts[combo_idx].argmax() if best_double[combo_idx] > 0 else None
        )
        best_result = {
            "drivers": [d_names[i] for i in combos[combo_idx]],
            "constructor": cn,
            "cost": combo_costs[combo_idx] + cc,
            "points": score,
            "doubled": (
                d_names[combos[combo_idx][doubled_idx]]
                if doubled_idx is not None
                else None
            ),
        }

print("Best Team:", best_result)
