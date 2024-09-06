##imoprt dependencies
from collections import deque

##read input and create field
n,m,s,t,q = map(int,input().split())
field = [['*'] * m for _ in range(n)]
for i in range(q):
    coord1,coord2 = map(int, input().split())
    field[coord1-1][coord2-1] = '#'
visited = [[False] * m for _ in range(n)]

##implement breadth-first-search
def bfs(fld, vsd,start_x,start_y,n,m):
    queue = deque([(start_x-1,start_y-1,0)])
    vsd[start_x-1][start_y-1] = True
    #create grid of moves
    dx = [-2, -2, -1, 1, 2, 2, 1, -1]
    dy = [-1, 1, 2, 2, 1, -1, -2, -2]
    sum_of_paths = 0

    while queue:
        #unpack the first queue element
        x,y,depth = queue.popleft()
        vsd[x][y]=True

        #make all the potential moves
        for (loc1,loc2) in zip(dx,dy):
            nx,ny = x+loc1, y+loc2
            # check if the points is within the margins
            if (0<= nx < n) and (0<=ny< m) and not vsd[nx][ny]:
                vsd[nx][ny] = True
                if fld[nx][ny]=='#':
                    sum_of_paths += depth + 1
                    fld[nx][ny] = '*'
                    queue.append((nx, ny, depth + 1))
                else:
                    queue.append((nx,ny, depth+1))
    fld[start_x-1][start_y-1] = '*'
    for rows in fld:
        if '#' in rows:
            return -1
    return sum_of_paths

print(bfs(field,visited,s,t,n,m))
