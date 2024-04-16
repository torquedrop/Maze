from PIL import Image, ImageDraw
from collections import deque

# Constants for visual representation
SPACE = 100
THICKNESS = SPACE // 10
RADIUS = THICKNESS
dx = [0, 1, 0, -1]
dy = [-1, 0, 1, 0]


# Class for creating and drawing panels
class Panel:
    def __init__(self, n, m) -> None:
        # Initialize board as a white canvas
        self.board = Image.new("RGB", (m * SPACE, n * SPACE), color="white")
        # Create a drawing object
        self.pan = ImageDraw.Draw(self.board)
        pass

    # Draw walls based on flags
    def draw(self, f, i, j):
        x = j * SPACE
        y = i * SPACE
        color = "blue"

        if f[0]:
            self.pan.line([(x, y), (x - SPACE, y)], fill=color, width=THICKNESS)
        if f[1]:
            self.pan.line([(x, y), (x, y + SPACE)], fill=color, width=THICKNESS)
        if f[2]:
            self.pan.line([(x, y), (x + SPACE, y)], fill=color, width=THICKNESS)
        if f[3]:
            self.pan.line([(x, y), (x, y - SPACE)], fill=color, width=THICKNESS)

    # Draw walls with rotation flags
    def draw_rt(self, f, is_sac: int, i, j):
        x = j * SPACE
        y = i * SPACE
        color = "blue"
        sac_color = "red"
        # Draw sacs if exist
        if is_sac:
            offset = SPACE // 5
            self.pan.line(
                [(x + offset, y + offset), (x + SPACE - offset, y + SPACE - offset)],
                fill=sac_color,
                width=THICKNESS,
            )
            self.pan.line(
                [(x + SPACE - offset, y + offset), (x + offset, y + SPACE - offset)],
                fill=sac_color,
                width=THICKNESS,
            )
        # Draw walls based on flags
        if f[0]:
            self.pan.line([(x, y), (x, y + SPACE)], fill=color, width=THICKNESS)
        if f[1]:
            self.pan.line(
                [(x, y + SPACE), (x + SPACE, y + SPACE)], fill=color, width=THICKNESS
            )
        if f[2]:
            self.pan.line(
                [(x + SPACE, y + SPACE), (x + SPACE, y)], fill=color, width=THICKNESS
            )
        if f[3]:
            self.pan.line([(x, y), (x + SPACE, y)], fill=color, width=THICKNESS)

    # Draw pillars at specified positions
    def draw_pillar(self, i, j):
        x, y = j * SPACE, i * SPACE
        pillar_color = "green"
        self.pan.ellipse(
            [(x - RADIUS, y - RADIUS), (x + RADIUS, y + RADIUS)],
            fill=pillar_color,
            outline=None,
        )

    # Draw a path on the panel
    def draw_path(self, path):
        for i in range(1, len(path)):
            st_x, st_y = (
                SPACE * path[i - 1][1] + SPACE // 2,
                SPACE * path[i - 1][0] + SPACE // 2,
            )
            ed_x, ed_y = (
                SPACE * path[i][1] + SPACE // 2,
                SPACE * path[i][0] + SPACE // 2,
            )
            self.pan.line([(st_x, st_y), (ed_x, ed_y)], fill="yellow", width=THICKNESS)

    # Display the panel
    def display(self):
        self.board.show()


# Class for handling point-based mazes
class PointStandard:
    def __init__(self, inputs) -> None:
        # Initialize maze dimensions
        n = len(inputs) + 1
        m = len(inputs[0]) + 1

        # Initialize point flags for walls
        point_flags = [[[0 for k in range(0, 4)] for j in range(m)] for i in range(n)]

        # Determine wall flags based on input values
        for i in range(1, n):
            for j in range(1, m):
                mz = int(inputs[i - 1][j - 1])
                if mz > 1:
                    point_flags[i][j][1] = point_flags[i + 1][j][3] = 1
                if mz % 2 == 1:
                    point_flags[i][j][2] = point_flags[i][j + 1][0] = 1
        # Create panel object
        panel = Panel(n, m)

        # Assign attributes
        self.n = n
        self.m = m
        self.point_flags = point_flags
        self.panel = panel

    # Count the number of walls and pillar positions
    def get_wall_pillar_count(self):
        """
        - iterate all points and run bfs for each point if it's not visited yet.
        count of calling bfs is the number of walls.
        
        - if left, right, top, bottom line is empty, it's a pillar
        """
        visited = set()
        wall_cnt = 0
        pillar_pos_list = []

        for _x in range(1, self.n):
            for _y in range(1, self.m):
                if (_x, _y) not in visited:
                    is_pillar = (
                        sum([self.point_flags[_x][_y][i] for i in range(0, 4)]) == 0
                    )
                    if is_pillar:
                        pillar_pos_list.append((_x, _y))
                        continue
                    wall_cnt += 1
                    visited.add((_x, _y))
                    q = deque([(_x, _y)])
                    while q:
                        node = q.popleft()

                        for i in range(0, 4):
                            if self.point_flags[node[0]][node[1]][i] == 1:
                                x = node[0] + dx[i]
                                y = node[1] + dy[i]
                                if x == 0 or y == 0 or x == self.n or y == self.m:
                                    continue
                                if (x, y) not in visited:
                                    q.append((x, y))
                                    visited.add((x, y))
        return wall_cnt, pillar_pos_list

    # Draw the maze on the panel
    def draw(self):
        for i in range(0, self.n):
            for j in range(0, self.m):
                self.panel.draw(self.point_flags[i][j], i, j)

    # Display the panel
    def show(self):
        self.panel.display()


# Class for handling rectangle-based mazes
class RectStandard:
    def __init__(self, inputs) -> None:
        # Initialize maze dimensions
        n = len(inputs) + 1
        m = len(inputs[0]) + 1
        # Initialize rectangle flags for walls
        rect_flags = [[[0 for k in range(0, 4)] for j in range(m)] for i in range(n)]

        # Determine rectangle flags based on input values
        for i in range(1, n):
            for j in range(1, m):
                mz = int(inputs[i - 1][j - 1])
                if mz > 1:
                    rect_flags[i][j][0] = rect_flags[i][j - 1][2] = 1
                if mz % 2 == 1:
                    rect_flags[i][j][3] = rect_flags[i - 1][j][1] = 1
        # Create panel object
        panel = Panel(n, m)

        # Assign attributes
        self.n = n
        self.m = m
        self.rect_flags = rect_flags
        self.panel = panel

        # Identify sacs, paths
        self.get_not_blue()

    # Find cutting edges and paths
    def find_cutting_edges(self, gates, rect_flags, n, m):
        # Initialize variables
        found_cnt = 0
        cutting_edges = []
        path_list = []
        found = [[0 for j in range(m)] for i in range(n)]

        # Define depth-first search function
        def dfs_iterative(start):
            visited = set()
            parent = {}
            low = {}
            disc = {}
            time = 0
            nonlocal found_cnt
            stack = [(start, None)]
            while stack:
                u, parent_node = stack[-1]
                if u not in visited:
                    visited.add(u)
                    disc[u] = time
                    low[u] = time
                    time += 1
                found_unvisited = False

                ux, uy = u

                if not found[ux][uy]:
                    found[ux][uy] = 1
                    found_cnt += 1

                v_list = []
                if u in gates and u != start:
                    v_list.append(start)
                else:
                    for i in range(0, 4):
                        if not rect_flags[ux][uy][i]:
                            vx, vy = ux + dx[i], uy + dy[i]
                            if vx < 0 or vy < 0 or vx >= n or vy >= m:
                                continue
                            v_list.append((vx, vy))

                for v in v_list:
                    vx, vy = v
                    if v == parent_node:
                        continue
                    if v not in visited:
                        parent[v] = u
                        stack.append((v, u))
                        found_unvisited = True
                        break
                    else:
                        low[ux, uy] = min(low[ux, uy], disc[vx, vy])
                if not found_unvisited:
                    stack.pop()
                    if parent_node is not None:
                        px, py = parent_node
                        low[px, py] = min(low[px, py], low[ux, uy])
                        if low[ux, uy] > disc[px, py]:
                            cutting_edges.append((parent_node, u))

        # Define function to find accessible cul-de-sacs
        def find_sacs(start, obstacle):
            stack = [start]
            visited = set()
            while stack:
                u = stack[-1]
                if u not in visited:
                    visited.add(u)
                stack.pop()

                ux, uy = u
                v_list = []
                for i in range(0, 4):
                    if not rect_flags[ux][uy][i]:
                        v = (ux + dx[i], uy + dy[i])
                        vx, vy = v
                        if vx < 0 or vy < 0 or vx >= n or vy >= m:
                            continue
                        if v == obstacle or v in visited:
                            continue
                        stack.append(v)
            return visited

        # Define function to find paths
        def find_path(start):
            current = (start, None)
            path = []
            flag = 1
            while flag:
                u, parent = current
                path.append(u)
                ux, uy = u
                if u != start and u in gates:
                    path_list.append(path)
                v_list = []
                for i in range(0, 4):
                    if not rect_flags[ux][uy][i]:
                        v = (ux + dx[i], uy + dy[i])
                        vx, vy = v
                        if (
                            vx < 0
                            or vy < 0
                            or vx >= n
                            or vy >= m
                            or v == parent
                            or self.is_sac[vx][vy]
                        ):
                            continue
                        v_list.append(v)
                if len(v_list) == 1:
                    current = (v_list[0], u)
                else:
                    break

        # Loop through gates to find cutting edges and paths
        for node in gates:
            # if node not in visited:
            nx, ny = node
            if not found[nx][ny]:
                found_cnt = 0
                cutting_edges = []
                dfs_iterative(node)
                for edge in cutting_edges:
                    u, v = edge
                    vx, vy = v
                    if not self.is_sac[vx][vy]:
                        sac_list = find_sacs(v, u)
                        for sac in sac_list:
                            self.is_sac[sac[0]][sac[1]] = 1

                if found_cnt > 1:
                    self.access_cnt += 1

                find_path(node)

        # Count inaccessible areas
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                if found[i][j] == 0:
                    self.inaccess_cnt += 1

        return path_list

    # Identify non-blue areas in the maze
    def get_not_blue(self):
        n = self.n
        m = self.m
        #insert gates to the variable
        gates = set()
        for i in range(1, n - 1):
            self.rect_flags[i][0][1] = self.rect_flags[i][0][3] = 1
            self.rect_flags[i][m - 1][1] = self.rect_flags[i][m - 1][3] = 1
            gates.add((i, 0))
            gates.add((i, m - 1))

        for j in range(1, m - 1):
            self.rect_flags[0][j][0] = self.rect_flags[0][j][2] = 1
            self.rect_flags[n - 1][j][0] = self.rect_flags[n - 1][j][2] = 1
            gates.add((0, j))
            gates.add((n - 1, j))

        self.is_sac = [[0 for j in range(m)] for i in range(n)]
        self.inaccess_cnt = 0
        self.access_cnt = 0
        self.path_list = self.find_cutting_edges(gates, self.rect_flags, n, m)

    # Draw the maze on the panel
    def draw(self, pillar_pos_list):
        for i in range(1, self.n - 1):
            for j in range(1, self.m - 1):
                self.panel.draw_rt(self.rect_flags[i][j], self.is_sac[i][j], i, j)
        for pos in pillar_pos_list:
            self.panel.draw_pillar(pos[0], pos[1])
        for path in self.path_list:
            self.panel.draw_path(path)

    # Get the count of gates in the maze
    def get_gate_count(self):
        """
        for 4 side of maze, get the count of gate
        """
        cnt = 0
        for i in range(1, self.n - 1):
            if self.rect_flags[i][1][0] == 0:
                cnt += 1
            if self.rect_flags[i][self.m - 1][0] == 0:
                cnt += 1
        for j in range(1, self.m - 1):
            if self.rect_flags[0][j][1] == 0:
                cnt += 1
            if self.rect_flags[self.n - 1][j][3] == 0:
                cnt += 1
        return cnt

    # Get the count of inaccessible inner points in the maze
    def get_inaccess_count(self):
        return self.inaccess_cnt

    # Get the count of accessible areas in the maze
    def get_access_count(self):
        return self.access_cnt

    # Get the count of sets of accessible cul-de-sacs that are all connected
    def get_sacs_group_count(self):
        """
        run bfs to get count of cul-de-sacs.
        number of calling bfs is the number of cul-de-sacs group
        """
        n = self.n
        m = self.m
        visited = [[0 for j in range(m)] for i in range(n)]
        sacs_set_cnt = 0
        for _x in range(1, n - 1):
            for _y in range(1, m - 1):
                if not visited[_x][_y] and self.is_sac[_x][_y]:
                    sacs_set_cnt += 1
                    visited[_x][_y] = 1
                    q = deque([(_x, _y)])
                    while q:
                        x, y = q.popleft()

                        for i in range(0, 4):
                            if self.rect_flags[x][y][i] == 0:
                                nx = x + dx[i]
                                ny = y + dy[i]
                                if nx == 0 or ny == 0 or nx == n - 1 or ny == m - 1:
                                    continue
                                if not visited[nx][ny] and self.is_sac[nx][ny]:
                                    q.append((nx, ny))
                                    visited[nx][ny] = 1
        return sacs_set_cnt

    # Get the count of unique entry-exit paths with no intersections not to cul-de-sacs
    def get_unique_path_count(self):
        return len(self.path_list)

    # Show the maze on the panel
    def show(self):
        self.panel.display()


class MazeError(Exception):
    def __init__(self, message="Incorrect input."):
        super().__init__(message)


class Maze:
    def __init__(self, file_name) -> None:

        try:
            with open(file_name, "r") as file:
                lines = file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{file_name}' does not exist.")

        try:
            inputs = []
            n = len(lines)
            m = -1
            for line in lines:
                nline = line.strip()
                if m != -1 and len(nline) != m:
                    raise MazeError()
                m = len(nline)
                if m == 0:
                    raise MazeError()
                for c in nline:
                    if not (c >= "0" and c <= "3"):
                        raise MazeError()

                inputs.append(nline)

            # if not (n >= 2 and n <= 31 and m >=

            """
            consider the problem into two parts.
            - each cell on RectStandard indicates whether there's a line on 4 directions.
                here, the rectangle is the main cell
            - each cell on PointStandard indicates whether there's a line on 4 directions.
                here, the point is the cell
            
            """
            self.pt_st = PointStandard(inputs)
            self.rt_st = RectStandard(inputs)
        except Exception as e:
            raise MazeError("Incorrect input.")

    def display_information_by_count(self, cnt, patterns):
        match cnt:
            case 0:
                print(patterns[0])
            case 1:
                print(patterns[1])
            case _:
                print(f"The maze has {cnt}{patterns[2]}")

    def display_gate_count(self, gate_cnt):
        self.display_information_by_count(
            gate_cnt,
            (
                "The maze has no gate.",
                "The maze has a single gate.",
                " gates.",
            ),
        )

    def display_wall_count(self, wall_cnt):
        self.display_information_by_count(
            wall_cnt,
            (
                "The maze has no wall.",
                "The maze has walls that are all connected.",
                " sets of walls that are all connected.",
            ),
        )

    def display_inaccess_count(self, inaccess_cnt):
        self.display_information_by_count(
            inaccess_cnt,
            (
                "The maze has no inaccessible inner point.",
                "The maze has a unique inaccessible inner point.",
                " inaccessible inner points.",
            ),
        )

    def display_access_count(self, access_cnt):
        self.display_information_by_count(
            access_cnt,
            (
                "The maze has no accessible area.",
                "The maze has a unique accessible area.",
                " accessible areas.",
            ),
        )

    def display_sacs_set_count(self, sacs_set_cnt):
        self.display_information_by_count(
            sacs_set_cnt,
            (
                "The maze has no accessible cul-de-sac.",
                "The maze has accessible cul-de-sacs that are all connected.",
                " sets of accessible cul-de-sacs that are all connected.",
            ),
        )

    def display_path_count(self, path_cnt):
        self.display_information_by_count(
            path_cnt,
            (
                "The maze has no entry-exit path with no intersection not to cul-de-sacs.",
                "The maze has a unique entry-exit path with no intersection not to cul-de-sacs.",
                " entry-exit paths with no intersections not to cul-de-sacs.",
            ),
        )

    # Analyze the maze
    def analyze(self):
        # Get the count of gates in the maze
        gate_cnt = self.rt_st.get_gate_count()
        # Get the count of walls and pillar positions in the maze
        wall_cnt, pillar_pos_list = self.pt_st.get_wall_pillar_count()
        # Get the count of inaccessible inner points in the maze
        inaccess_cnt = self.rt_st.get_inaccess_count()
        # Get the count of accessible areas in the maze
        access_cnt = self.rt_st.get_access_count()
        # Get the count of sets of accessible cul-de-sacs that are all connected
        sacs_set_cnt = self.rt_st.get_sacs_group_count()
        # Get the count of unique entry-exit paths with no intersections not to cul-de-sacs
        path_cnt = self.rt_st.get_unique_path_count()

        # Display analysis results
        # Display the number of gates in the maze
        self.display_gate_count(gate_cnt)
        # Display the number of walls in the maze
        self.display_wall_count(wall_cnt)
        # Display the number of inaccessible inner points in the maze
        self.display_inaccess_count(inaccess_cnt)
        # Display the number of accessible areas in the maze
        self.display_access_count(access_cnt)
        # Display the number of sets of accessible cul-de-sacs that are all connected
        self.display_sacs_set_count(sacs_set_cnt)
        # Display the number of unique entry-exit paths with no intersections not to cul-de-sacs
        self.display_path_count(path_cnt)
        # Draw the maze
        self.rt_st.draw(pillar_pos_list)

    def display(self):
        self.rt_st.show()
