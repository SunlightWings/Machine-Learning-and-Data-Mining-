A* (A-star) is a best-first search algorithm that finds the shortest path from a start node to a goal node by considering both:

    * Cost so far (g) – distance from the start to the current node.

    * Estimated cost to goal (h) – a heuristic (usually Manhattan or Euclidean distance).

It evaluates nodes using the formula:
   * f(n)=g(n)+h(n)

Where:

    - g(n) = actual cost from start to current node

    - h(n) = estimated cost from current node to goal (heuristic)

    - f(n) = total estimated cost of the path through this node
