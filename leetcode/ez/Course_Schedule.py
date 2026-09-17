from collections import deque

def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    indegree = [0] * numCourses

    for course, prerequisite in prerequisites:
        graph[prerequisite].append(course)
        indegree[course] += 1

    queue = deque(
        course for course in range(numCourses)
        if indegree[course] == 0
    )

    finished = 0

    while queue:
        course = queue.popleft()
        finished += 1

        for next_course in graph[course]:
            indegree[next_course] -= 1

            if indegree[next_course] == 0:
                queue.append(next_course)

    return finished == numCourses

numCourses = 2
prerequisites = [[1, 0]]

print(canFinish(numCourses, prerequisites))
