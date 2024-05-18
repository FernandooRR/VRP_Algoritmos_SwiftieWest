import itertools

def calculate_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i]][route[i+1]]
    total_distance += distance_matrix[route[-1]][route[0]]  
    return total_distance

def vrp_brute_force(num_vehicles, distance_matrix):
    num_locations = len(distance_matrix)
    locations = list(range(1, num_locations))
    best_route = None
    min_distance = float('inf')

    for routes in itertools.permutations(locations):
        distance = calculate_distance([0] + list(routes) + [0], distance_matrix)
        if distance < min_distance:
            min_distance = distance
            best_route = routes

    return best_route, min_distance


distance_matrix = [
    [0, 29, 20, 21],
    [29, 0, 15, 17],
    [20, 15, 0, 28],
    [21, 17, 28, 0]
]
num_vehicles = 1

best_route, min_distance = vrp_brute_force(num_vehicles, distance_matrix)
print("Best route:", best_route)
print("Minimum distance:", min_distance)
