import matplotlib
matplotlib.use('TkAgg')  # Ensure interactive backend

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from collections import deque

maze = [
    ['S', 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    [1,   0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [0,   0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    [0,   1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
    [0,   1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1,   1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0,   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0,   1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [0,   1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0,   1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [0,   0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1,   1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0,   1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0,   1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0,   0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1,   1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [0,   0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0,   1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0,   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1,   1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 'G'],
]


ROWS, COLS = len(maze), len(maze[0])

def to_numeric_grid(maze):
    grid = np.zeros((ROWS, COLS))
    for r in range(ROWS):
        for c in range(COLS):
            val = maze[r][c]
            if val == 1:
                grid[r][c] = 1  # Wall
            elif val == 'S':
                grid[r][c] = 2  # Start
            elif val == 'G':
                grid[r][c] = 3  # Goal
            else:
                grid[r][c] = 0  # Free
    return grid

def find_pos(value):
    for r in range(ROWS):
        for c in range(COLS):
            if maze[r][c] == value:
                return (r, c)

def get_neighbors(r, c):
    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and maze[nr][nc] != 1:
            yield (nr, nc)

def bfs_or_dfs(algorithm='bfs'):
    start = find_pos('S')
    goal = find_pos('G')
    grid = to_numeric_grid(maze)
    visited = set([start])
    frontier = deque([(start, [start])]) if algorithm == 'bfs' else [(start, [start])]

    cmap = matplotlib.colors.ListedColormap([
        'white',   # 0: Free
        'black',   # 1: Wall
        'orange',  # 2: Start
        'red',     # 3: Goal
        'green',   # 4: Visited
        'blue'     # 5: Final path
    ])

    plt.ion()
    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=5)
    ax.set_title(f"{algorithm.upper()} Maze Search")
    plt.show()

    while frontier:
        (r, c), path = frontier.popleft() if algorithm == 'bfs' else frontier.pop()

        if (r, c) == goal:
            for pr, pc in path:
                if grid[pr][pc] in [0, 4]:
                    grid[pr][pc] = 5  # Final path
                    im.set_data(grid)
                    plt.draw()
                    plt.pause(0.01)
            break

        if grid[r][c] == 0:
            grid[r][c] = 4  # Visited
            im.set_data(grid)
            plt.draw()
            plt.pause(0.01)

        for neighbor in get_neighbors(r, c):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, path + [neighbor]))

    plt.ioff()
    plt.show(block=True)

# --- Ask user
algo = input("Enter search algorithm (bfs or dfs): ").strip().lower()
if algo in ['bfs', 'dfs']:
    bfs_or_dfs(algo)
else:
    print("Invalid input. Please enter 'bfs' or 'dfs'.")
