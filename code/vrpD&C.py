
def divide_and_conquer_vrp(locations, distance_matrix):
    if len(locations) <= 2:
        return locations, calculate_distance(locations, distance_matrix)

    mid = len(locations) // 2
    left_locations = locations[:mid]
    right_locations = locations[mid:]

    left_route, left_distance = divide_and_conquer_vrp(left_locations, distance_matrix)
    right_route, right_distance = divide_and_conquer_vrp(right_locations, distance_matrix)

    combined_route = left_route + right_route
    combined_distance = left_distance + right_distance + distance_matrix[left_route[-1]][right_route[0]]

    return combined_route, combined_distance

locations = [0, 1, 2, 3]
best_route, min_distance = divide_and_conquer_vrp(locations, distance_matrix)
print("Best route:", best_route)
print("Minimum distance:", min_distance)
