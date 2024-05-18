def vrp_greedy(num_vehicles, distance_matrix):
    n = len(distance_matrix)
    unvisited = set(range(1, n))
    current_location = 0
    route = [current_location]
    total_distance = 0

    while unvisited:
        next_location = min(unvisited, key=lambda x: distance_matrix[current_location][x])
        total_distance += distance_matrix[current_location][next_location]
        route.append(next_location)
        unvisited.remove(next_location)
        current_location = next_location

    total_distance += distance_matrix[current_location][0]
    route.append(0)
    
    return route, total_distance


best_route, min_distance = vrp_greedy(num_vehicles, distance_matrix)
print("Best route (Greedy):", best_route)
print("Minimum distance:", min_distance)
