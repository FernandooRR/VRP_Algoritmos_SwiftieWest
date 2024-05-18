import sys
import random
import math
from google.colab import files
import csv

uploaded = files.upload()





def read_csv(filepath):
    vrp = {'nodes': []}
    try:
        with open(filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['type'] == 'param' and row['label'] == 'capacity':
                    vrp['capacity'] = float(row['demand'])
                    if vrp['capacity'] <= 0:
                        raise ValueError('Capacity must be greater than zero.')
                elif row['type'] == 'node':
                    node = {
                        'label': row['label'],
                        'demand': float(row['demand']),
                        'posX': float(row['posX']),
                        'posY': float(row['posY'])
                    }
                    if node['demand'] <= 0:
                        raise ValueError(f"Demand of node {node['label']} must be greater than zero.")
                    if node['demand'] > vrp['capacity']:
                        raise ValueError(f"Demand of node {node['label']} exceeds vehicle capacity.")
                    vrp['nodes'].append(node)
        if 'capacity' not in vrp:
            raise ValueError('Missing capacity parameter.')
        if len(vrp['nodes']) == 0:
            raise ValueError('No nodes found.')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    return vrp


vrp = read_csv('in.csv')
if vrp is None:
    print("Failed to read VRP data.")
    exit(1)




def distance(n1, n2):
    dx = n2['posX'] - n1['posX']
    dy = n2['posY'] - n1['posY']
    return math.sqrt(dx * dx + dy * dy)

def fitness(p):
    s = distance(vrp['nodes'][0], vrp['nodes'][p[0]])
    for i in range(len(p) - 1):
        prev = vrp['nodes'][p[i]]
        next = vrp['nodes'][p[i + 1]]
        s += distance(prev, next)
    s += distance(vrp['nodes'][p[len(p) - 1]], vrp['nodes'][0])
    return s

def adjust(p):
    repeated = True
    while repeated:
        repeated = False
        for i1 in range(len(p)):
            for i2 in range(i1):
                if p[i1] == p[i2]:
                    haveAll = True
                    for nodeId in range(len(vrp['nodes'])):
                        if nodeId not in p:
                            p[i1] = nodeId
                            haveAll = False
                            break
                    if haveAll:
                        del p[i1]
                    repeated = True
                if repeated: break
            if repeated: break
    i = 0
    s = 0.0
    cap = vrp['capacity']
    while i < len(p):
        s += vrp['nodes'][p[i]]['demand']
        if s > cap:
            p.insert(i, 0)
            s = 0.0
        i += 1
    i = len(p) - 2
    while i >= 0:
        if p[i] == 0 and p[i + 1] == 0:
            del p[i]
        i -= 1

popsize = 50
iterations = 100

pop = []

for i in range(popsize):
    p = list(range(1, len(vrp['nodes'])))
    random.shuffle(p)
    pop.append(p)
for p in pop:
    adjust(p)

for i in range(iterations):
    nextPop = []
    for j in range(int(len(pop) / 2)):
        parentIds = set()
        while len(parentIds) < 4:
            parentIds |= {random.randint(0, len(pop) - 1)}
        parentIds = list(parentIds)
        parent1 = pop[parentIds[0]] if fitness(pop[parentIds[0]]) < fitness(pop[parentIds[1]]) else pop[parentIds[1]]
        parent2 = pop[parentIds[2]] if fitness(pop[parentIds[2]]) < fitness(pop[parentIds[3]]) else pop[parentIds[3]]
        cutIdx1, cutIdx2 = random.randint(1, min(len(parent1), len(parent2)) - 1), random.randint(1, min(len(parent1), len(parent2)) - 1)
        cutIdx1, cutIdx2 = min(cutIdx1, cutIdx2), max(cutIdx1, cutIdx2)
        child1 = parent1[:cutIdx1] + parent2[cutIdx1:cutIdx2] + parent1[cutIdx2:]
        child2 = parent2[:cutIdx1] + parent1[cutIdx1:cutIdx2] + parent2[cutIdx2:]
        nextPop += [child1, child2]
    if random.randint(1, 15) == 1:
        ptomutate = nextPop[random.randint(0, len(nextPop) - 1)]
        i1 = random.randint(0, len(ptomutate) - 1)
        i2 = random.randint(0, len(ptomutate) - 1)
        ptomutate[i1], ptomutate[i2] = ptomutate[i2], ptomutate[i1]
    for p in nextPop:
        adjust(p)
    pop = nextPop

better = None
bf = float('inf')
for p in pop:
    f = fitness(p)
    if f < bf:
        bf = f
        better = p

print(' route:')
print('depot')
for nodeIdx in better:
    print(vrp['nodes'][nodeIdx]['label'])
print('depot')
print(' cost:')
print('%f' % bf)
