import tqdm

# import tqdm.notebook as tqdm
import hashlib
import numpy as np

from . import rectangular_kruskal_maze


def make_maze(n: int) -> np.ndarray:
    algo = rectangular_kruskal_maze.KruskalRectangular(n)
    spanning_tree, edges = algo.kruskal_spanning_tree()

    size = 2 * n + 1
    result = np.zeros((size, size), dtype=np.int32)

    for i in range(n):
        for j in range(n):
            y, x, node = 2 * i + 1, 2 * j + 1, i * n + j

            result[y, x] = 1
            result[y, x + 1] = j + 1 < n and spanning_tree[node][algo.RIGHT] or result[y, x + 1]
            result[y + 1, x] = i + 1 < n and spanning_tree[node][algo.BOTTOM] or result[y + 1, x]

    result[1, 0] = result[-2, -1] = 1

    return result, edges


def generate_mazes(n: int, maze_size: int) -> np.ndarray:
    result = []
    hashes = {"PLACEHOLDER"}

    for _ in tqdm.trange(n):
        key = "PLACEHOLDER"
        while key in hashes:
            maze, edges = make_maze(maze_size)
            key = hashlib.sha256(str(edges).encode()).hexdigest()
        else:
            hashes.add(key)

        result.append(maze)

    return np.stack(result)


mazes = generate_mazes(10000, 10)
