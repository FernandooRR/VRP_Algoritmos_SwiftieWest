def tsp_dynamic_programming(distance_matrix):
    n = len(distance_matrix)
    all_visited = (1 << n) - 1
    memo = [[None] * n for _ in range(1 << n)]

    def visit(mask, pos):
        if mask == all_visited:
            return distance_matrix[pos][0]
        if memo[mask][pos] is not None:
            return memo[mask][pos]

        ans = float('inf')
        for city in range(n):
            if mask & (1 << city) == 0:
                new_mask = mask | (1 << city)
                ans = min(ans, distance_matrix[pos][city] + visit(new_mask, city))

        memo[mask][pos] = ans
        return ans

    return visit(1, 0)

min_distance = tsp_dynamic_programming(distance_matrix)
print("Minimum distance (TSP):", min_distance)
