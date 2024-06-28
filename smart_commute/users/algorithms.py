import os

import numpy as np
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import folium
import pandas as pd
from sys import maxsize
from itertools import permutations
import random

# Shift timings for EC are 8am to 5pm
pick_time_hr = 8
pick_time_min = 0

drop_time_hr = 17
drop_time_min = 0


def clarke_wright_savings_complete(coordinates, shift_id, type_transit, max_capacity, num_routes):
    second_level_algo = 'TSP'
    # max_capacity = 8
    # num_routes = distance_matrix.shape[0] + 1
    # df_lat_lon = pd.read_csv('/content/lat_lon_ec.csv')
    # coordinates = df_lat_lon[['Latitude', 'Longitude']].values.tolist()
    # shift_id = "EC"
    distance_matrix = np.zeros((81, 81))
    if shift_id == "EC":
        if type_transit == 'pick':
            df = pd.read_csv('smart_commute/DM_pickup0800arrival.csv', header=None)
            distance_matrix = df.to_numpy()
            df_t = pd.read_csv('smart_commute/TM_pickup0800arrival.csv', header=None)
            time_matrix = df_t.to_numpy()
        elif type_transit == 'drop':
            df = pd.read_csv('smart_commute/DM_dropoff1700departure.csv', header=None)
            distance_matrix = df.to_numpy()
            distance_matrix = distance_matrix.transpose()
            df_t = pd.read_csv('smart_commute/TM_dropoff1700departure.csv', header=None)
            time_matrix = df_t.to_numpy()
            time_matrix = time_matrix.transpose()
        else:
            print('invalid transit type')
    else:
        print("calculate distance matrix")
    num_routes = distance_matrix.shape[0]
    routes = clarke_wright_savings(distance_matrix, num_routes=num_routes, max_route_length=max_capacity)

    # Output the routes
    num_points_optimized = 0
    routes_dict = {}
    for idx, route in enumerate(routes):
        if idx != 0:
            routes_dict[idx - 1] = route
            num_points_optimized += len(route)
    clusters = routes_dict
    print("routes as per clarke wright savings")
    print(routes_dict)
    print(f"total points {num_points_optimized}")

    coordinates_list = []
    for i in routes_dict.keys():
        selected_vertices = get_coord_list(routes_dict, i, coordinates)
        coordinates_list += [selected_vertices]
    save_folium_map(coordinates_list, 'clarke_wright_savings')

    final_routed_dict = {}

    # running TSP if required
    if second_level_algo == 'TSP':
        for i in clusters:
            print('optimizing route # ', i)
            # Example lists
            list1 = clusters[i]
            '''if len(list1) > 8:
              list1= list1[0:7]'''
            list2 = coordinates

            # Finding indices
            indices = list1

            # print(indices)
            indices = [0] + indices  # including origin
            # print(indices)
            graph = distance_matrix[np.ix_(indices, indices)]
            graph_time = time_matrix[np.ix_(indices, indices)]
            s = 0
            V = len(indices)
            if type_transit == 'drop':
                distance, min_time, indv_distances, indv_durations, path = travellingSalesmanProblem(graph.transpose(),
                                                                                                     graph_time.transpose(),
                                                                                                     s, type_transit, V)
            else:
                distance, min_time, indv_distances, indv_durations, path = travellingSalesmanProblem(graph, graph_time,
                                                                                                     s, type_transit, V)
            routed_indices = [indices[value] for value in path]
            # final_routed_dict[i] = {'route_vertex_index': routed_indices,'distance':distance}

            list_coord = []
            for v in routed_indices:
                # print(coord[v])
                list_coord += [coordinates[v]]

            origin = list_coord[0]
            destination = list_coord[-1]
            optimized_waypoints = list_coord[1:-1]

            optimized_waypoints_str = "|".join([f"{lat},{lon}" for lat, lon in optimized_waypoints])

            google_maps_url = (
                f"https://www.google.com/maps/dir/?api=1&origin={origin[0]},{origin[1]}"
                f"&destination={destination[0]},{destination[1]}"
                f"&waypoints={optimized_waypoints_str}"
            )

            final_routed_dict[i] = {'route_vertex_index': routed_indices, 'distance': distance / 1000,
                                    'duration': min_time / 60, 'URL': google_maps_url,
                                    'individual_distances': indv_distances, 'individual_durations': indv_durations}

        print(final_routed_dict)
        return final_routed_dict


def clarke_wright_savings(d_matrix, num_routes, max_route_length):
    n = len(d_matrix)
    routes = [[i] for i in range(n)]  # Initially, each coordinate is its own route

    # Calculate savings
    savings = []
    for i in range(n):
        for j in range(i + 1, n):
            if i != j:
                savings.append((d_matrix[i][0] + d_matrix[0][j] - d_matrix[i][j], i, j))
    # print(savings)
    savings.sort(reverse=True, key=lambda x: x[0])

    # print("sorted_savings")
    # print(savings)
    def find_route(point, routes):
        for route in routes:
            if point in route:
                return route
        return None

    for saving, i, j in savings:
        route_i = find_route(i, routes)
        route_j = find_route(j, routes)
        if route_i is not route_j and len(route_i) + len(route_j) <= max_route_length:
            routes.remove(route_i)
            routes.remove(route_j)
            routes.append(route_i + route_j)

    # Ensure we have exactly num_routes routes
    # ix = 0
    print(f"len routes: {len(routes)}")
    # print(f"num routes: {num_routes}")
    while len(routes) > num_routes:
        # Merge smallest routes
        routes.sort(key=len)
        route1 = routes.pop(0)
        route2 = routes.pop(0)
        if len(route1) + len(route2) <= max_route_length:
            routes.append(route1 + route2)
        else:
            routes.extend([route1, route2])
        print('r')

    return routes


def k_means_algorithm(coordinates):
    # df_lat_lon = pd.read_csv('lat_lon_ec.csv')
    # coordinates = df_lat_lon[['Latitude', 'Longitude']].values.tolist()
    print(os.getcwd())
    df = pd.read_csv('smart_commute/Distance_Matrix.csv', header=None)

    distance_matrix = df.to_numpy()

    # Convert list of lists to list of tuples
    vertices = [tuple(inner_list) for inner_list in coordinates[1:]]

    # Kmeans Clustering
    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(vertices)
    clusters = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(vertices[idx])
    # print(clusters)

    final_routed_dict = {}
    for i in clusters:
        print('optimizing route # ', i)
        # Example lists
        list1 = clusters[i]
        '''if len(list1) > 8:
          list1= list1[0:7]'''
        list2 = coordinates

        # Finding indices
        indices = [list2.index(list(value)) for value in list1]

        # print(indices)
        indices = [0] + indices  # including origin
        # print(indices)
        graph = distance_matrix[np.ix_(indices, indices)]
        s = 0
        V = len(indices)
        distance, path = travellingSalesmanProblem(graph, s, V)
        routed_indices = [indices[value] for value in path]
        final_routed_dict[i] = {'route_vertex_index': routed_indices, 'distance': distance}
    # print(final_routed_dict)
    return final_routed_dict


# Routing Through Traveling Salesman Problem
# implementation of traveling Salesman Problem

def travellingSalesmanProblem(graph, graph_time, s, type_transit='drop', V=4):
    optim_route = []
    # store all vertex apart from source vertex
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)
    # print(vertex)
    # store minimum weight Hamiltonian Cycle
    if len(vertex) > 8:
        min_path, optim_route = 99999, [s] + list(vertex)
        # print(min_path, optim_route)
        return min_path, optim_route
    min_path = maxsize
    # print(min_path)
    next_permutation = permutations(vertex)
    optim_path_trail = []
    optim_time_trail = []
    if type_transit == 'drop':
        for i in next_permutation:
            # print(i)

            # store current Path weight(cost)
            current_pathweight = 0
            current_pathtime = 0
            path_trail = []
            time_trail = []

            # compute current path weight
            k = s
            for j in i:
                current_pathweight += graph[k][j]
                path_trail += [graph[k][j] / 1000]
                current_pathtime += graph_time[k][j]
                time_trail += [graph_time[k][j] / 60]
                k = j
            # current_pathweight += graph[k][s]

            # update minimum
            min_path = min(min_path, current_pathweight)

            # print(current_pathweight,min_path)

            if min_path >= current_pathweight:
                optim_route = [s] + list(i)
                # print("at optim")
                min_time = current_pathtime
                optim_path_trail = path_trail
                optim_time_trail = time_trail
                # print(min_path)
                # print(min_path, optim_route)
    else:
        # in case of pickup, we will be having factory in the end
        for i in next_permutation:
            # print(i)

            # store current Path weight(cost)
            current_pathweight = 0
            current_pathtime = 0
            path_trail = []
            time_trail = []

            # compute current path weight
            k = i[0]
            for j in i[1:]:
                current_pathweight += graph[k][j]
                path_trail += [graph[k][j] / 1000]
                current_pathtime += graph_time[k][j]
                time_trail += [graph_time[k][j] / 60]
                k = j
            current_pathweight += graph[k][s]
            current_pathtime += graph_time[k][s]
            path_trail += [graph[k][s] / 1000]
            time_trail += [graph_time[k][s] / 60]

            # update minimum
            min_path = min(min_path, current_pathweight)

            # print(current_pathweight,min_path)

            if min_path >= current_pathweight:
                optim_route = list(i) + [s]
                # print("at optim")
                min_time = current_pathtime
                optim_path_trail = path_trail
                optim_time_trail = time_trail
                # print(min_path)
                # print(min_path, optim_route)
    # print(optim_path_trail)
    individual_distances = accumulated_sum(optim_path_trail, type_transit)
    # print(optim_time_trail)
    individual_times = accumulated_sum(optim_time_trail, type_transit)
    return min_path, min_time, individual_distances, individual_times, optim_route


def get_coord_list(routes, route_idx, coord):
    list_coord = []
    for v in routes[route_idx]:
        # print(coord[v])
        list_coord += [coord[v]]
    return list_coord


def accumulated_sum(input_list, type_transit):
    if type_transit == 'drop':
        accumulated_list = []
        running_total = 0

        for value in input_list:
            running_total += value
            accumulated_list.append(running_total)
    else:
        accumulated_list = []
        running_total = 0

        # Iterate through the list in reverse
        for value in reversed(input_list):
            running_total += value
            accumulated_list.append(running_total)

        # Reverse the trailing list to match the original order
        accumulated_list.reverse()

    return accumulated_list


def save_folium_map(coordinates_list, algorithm_type):
    # List of colors for markers
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
              'lightred', 'beige', 'darkblue', 'darkgreen']

    # Create a Folium map centered on the first coordinate
    m = folium.Map(location=coordinates_list[0][0], zoom_start=11)

    # Add markers to the map
    for idx, coords in enumerate(coordinates_list):
        for coord in coords:
            folium.Marker(
                location=coord,
                icon=folium.Icon(color=colors[idx % len(colors)])
            ).add_to(m)

    # Save the map to an HTML file
    m.save(f'map_{algorithm_type}.html')
