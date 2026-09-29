import json, collections
r = json.load(open('/root/art/combat-report.json'))
tr = r['training']['trace']
fire_ids = {'fire','forward_fire','back_fire','strafe_left_fire','strafe_right_fire','turn_left_fire','turn_right_fire'}
# damage observed at step t (behavior action outcome) vs trigger at t, t-1, t-2
def dmg(x): return (x.get('outcome') or {}).get('damageDealt', 0) > 0
n = len(tr)
for lag in (0, 1, 2, 3):
    fired = [i for i in range(lag, n) if tr[i-lag]['behaviorAction'] in fire_ids]
    not_fired = [i for i in range(lag, n) if tr[i-lag]['behaviorAction'] not in fire_ids]
    pf = sum(dmg(tr[i]) for i in fired) / max(1, len(fired))
    pn = sum(dmg(tr[i]) for i in not_fired) / max(1, len(not_fired))
    print(f'lag {lag}: P(damage | fire at t-{lag}) = {pf:.3f}  (n={len(fired)})   P(damage | no fire) = {pn:.3f}')
# informative steps: was a fire action issued in the previous 1-2 steps?
inf = [i for i in range(n) if tr[i]['probeSpread'] > 1e-9]
prev_fire = sum(1 for i in inf if i > 0 and (tr[i-1]['behaviorAction'] in fire_ids or (i > 1 and tr[i-2]['behaviorAction'] in fire_ids)))
print('informative steps preceded by fire within 2 steps:', prev_fire, '/', len(inf))
print('behavior fire rate overall:', sum(1 for x in tr if x['behaviorAction'] in fire_ids) / n)
big = [i for i in inf if tr[i]['probeSpread'] >= 0.1]
print('spread>=0.1 informative:', len(big), 'targets', collections.Counter(tr[i]['targetTop'] for i in big).most_common())
