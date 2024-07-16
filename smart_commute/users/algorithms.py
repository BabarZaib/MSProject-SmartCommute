import os

import numpy as np
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import folium
import pandas as pd
from sys import maxsize
from itertools import permutations
import random
import googlemaps
import polyline
from datetime import datetime, timedelta
import pytz
import requests

api_key = 'AIzaSyDKuB9ZdvWA6BvD65w2X-P88Ejzj79_s8I'




def call_final_algo(coordinates, shift_id, type_transit, max_capacity, num_routes, first_level_algo, second_level_algo):
    # second_level_algo = 'TSP'
    # max_capacity = 8
    # num_routes = distance_matrix.shape[0] + 1
    # df_lat_lon = pd.read_csv('/content/lat_lon_ec.csv')
    # coordinates = df_lat_lon[['Latitude', 'Longitude']].values.tolist()
    # shift_id = "EC"

    distance_matrix = np.zeros((81, 81))
    if shift_id == "EC":
        # Shift timings for EC are 8am to 5pm
        pick_time_hr = 8
        pick_time_min = 0

        drop_time_hr = 17
        drop_time_min = 0
        if type_transit == 'pick':
            df = pd.read_csv('smart_commute/Algorithm/input_data/EC/DM_pickup0800arrival.csv', header=None)
            distance_matrix = df.to_numpy()
            df_t = pd.read_csv('smart_commute/Algorithm/input_data/EC/TM_pickup0800arrival.csv', header=None)
            time_matrix = df_t.to_numpy()
            shift_time_hr = pick_time_hr
            shift_time_min = pick_time_min

        elif type_transit == 'drop':
            df = pd.read_csv('smart_commute/Algorithm/input_data/EC/DM_dropoff1700departure.csv', header=None)
            distance_matrix = df.to_numpy()
            distance_matrix = distance_matrix.transpose()
            df_t = pd.read_csv('smart_commute/Algorithm/input_data/EC/TM_dropoff1700departure.csv', header=None)
            time_matrix = df_t.to_numpy()
            time_matrix = time_matrix.transpose()
            shift_time_hr = drop_time_hr
            shift_time_min = drop_time_min
        else:
            print('invalid transit type')

    elif shift_id == 'GS':
        # Shift timings for General Shift are 7am to 5pm
        pick_time_hr = 7
        pick_time_min = 0

        drop_time_hr = 17
        drop_time_min = 0
        if type_transit == 'pick':
            df = pd.read_csv('smart_commute/Algorithm/input_data/GS/DM_pickup_gen0700arrival.csv', header=None)
            distance_matrix = df.to_numpy()
            df_t = pd.read_csv('smart_commute/Algorithm/input_data/GS/TM_pickup_gen0700arrival.csv', header=None)
            time_matrix = df_t.to_numpy()
            shift_time_hr = pick_time_hr
            shift_time_min = pick_time_min

        elif type_transit == 'drop':
            df = pd.read_csv('smart_commute/Algorithm/input_data/GS/DM_dropoff_gen1700departure.csv', header=None)
            distance_matrix = df.to_numpy()
            distance_matrix = distance_matrix.transpose()
            df_t = pd.read_csv('smart_commute/Algorithm/input_data/GS/TM_dropoff_gen1700departure.csv', header=None)
            time_matrix = df_t.to_numpy()
            time_matrix = time_matrix.transpose()
            shift_time_hr = drop_time_hr
            shift_time_min = drop_time_min
        else:
            print('invalid transit type')

    else:
        print("calculate distance matrix")

    clusters = []
    if first_level_algo == 'CWS':
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

    elif first_level_algo == 'min_dist':
        clusters = minimum_distance(distance_matrix, coordinates, max_capacity, num_routes)
    elif first_level_algo == 'manual':
        clusters = manual_process()

    elif first_level_algo == 'GA':
        # EC Routing through GA
        clusters = genetic_algorithm(num_routes, max_capacity)
        routes_dict = clusters

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
                                                                                                     s,
                                                                                                     first_level_algo,
                                                                                                     type_transit, V
                                                                                                     )
            else:
                distance, min_time, indv_distances, indv_durations, path = travellingSalesmanProblem(graph, graph_time,
                                                                                                     s,
                                                                                                     first_level_algo,
                                                                                                     type_transit, V
                                                                                                     )
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
            encoded_polyline = get_encoded_polyline(api_key, origin, destination, optimized_waypoints)

            google_maps_url = (
                f"https://www.google.com/maps/dir/?api=1&origin={origin[0]},{origin[1]}"
                f"&destination={destination[0]},{destination[1]}"
                f"&waypoints={optimized_waypoints_str}"
            )

            final_routed_dict[i] = {'route_vertex_index': routed_indices, 'distance': distance / 1000,
                                    'duration': min_time / 60, 'URL': google_maps_url,
                                    'individual_distances': indv_distances, 'individual_durations': indv_durations,
                                    'polyline': encoded_polyline}

        print(final_routed_dict)
        return final_routed_dict

    elif second_level_algo == 'GMO':
        final_routed_dict_google_optim = {}
        API_KEY = 'AIzaSyDKuB9ZdvWA6BvD65w2X-P88Ejzj79_s8I'
        # type_transit = 'drop'
        for i in clusters.keys():
            # print(clusters[i])
            indices, distance, duration, google_maps_url, polyline = get_google_optimized_route(clusters[i], i,
                                                                                                coordinates, API_KEY,
                                                                                                type_transit,
                                                                                                distance_matrix,
                                                                                                shift_time_min,
                                                                                                shift_time_hr)
            final_routed_dict_google_optim[i] = {'route_vertex_index': indices, 'distance': distance,
                                                 'duration': duration, 'URL': google_maps_url,
                                                 'polyline': polyline}

            for i in final_routed_dict_google_optim:
                indv_distances, indv_durations = get_indv_distance_durations(
                    final_routed_dict_google_optim[i]['route_vertex_index'], type_transit, distance_matrix, time_matrix)
                # print(final_routed_dict_google_optim[i]['route_vertex_index'])
                final_routed_dict_google_optim[i]['individual_distances'] = indv_distances
                final_routed_dict_google_optim[i]['individual_durations'] = indv_durations

        return final_routed_dict_google_optim


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


def manual_process():
    clusters = {0: [1, 2, 3, 4],
                1: [5, 6, 7, 8, 9, 10, 11, 12],
                2: [13, 14, 15, 16, 17, 18],
                3: [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                4: [29, 30, 31, 32, 33, 34],
                5: [35, 36, 37, 38, 39],
                6: [40, 41, 42, 43, 44, 45, 46],
                7: [47, 48, 49, 50, 51, 52, 53],
                8: [54, 54, 56, 57, 58, 59, 60, 61, 62, 63],
                9: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74],
                10: [75, 76, 77, 78, 79, 80]}
    return clusters


def genetic_algorithm(num_routes, max_capacity):
    if num_routes == 10 and max_capacity == 8:
        clusters = {0: [52, 40, 49, 51, 50, 48, 7, 5],
                    1: [44, 45, 46, 43, 42, 41, 6, 10],
                    2: [72, 70, 66, 67, 64, 47, 23, 27],
                    3: [60, 62, 68, 61, 54, 55, 56, 58],
                    4: [14, 53, 24, 25, 39, 35, 38, 37],
                    5: [79, 78, 76, 63, 59, 36, 1, 4],
                    6: [30, 34, 28, 19, 22, 26, 18, 13],
                    7: [73, 71, 33, 15, 20, 17, 3, 2],
                    8: [57, 69, 31, 16, 11, 12, 8, 9],
                    9: [80, 77, 75, 65, 74, 32, 29, 21]}
    elif num_routes == 10 and max_capacity == 9:
        clusters = {0: [22, 33, 32, 29, 21, 14, 16],
                    1: [47, 52, 40, 45, 6, 10, 7, 5, 3],
                    2: [78, 79, 68, 61, 54, 55, 63, 59],
                    3: [56, 62, 57, 31, 28, 19, 18, 13, 2],
                    4: [44, 46, 9, 39, 35, 38, 15, 20, 17],
                    5: [72, 73, 60, 30, 34, 37, 36],
                    6: [77, 75, 76, 65, 70, 74, 66, 67, 58],
                    7: [12, 8, 11, 1, 4],
                    8: [23, 27, 80, 71, 69, 64, 24, 26, 25],
                    9: [48, 50, 51, 53, 49, 43, 42, 41]}
    return clusters

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
        distance, path = travellingSalesmanProblem(graph, s, 'k-means', V)
        routed_indices = [indices[value] for value in path]
        final_routed_dict[i] = {'route_vertex_index': routed_indices, 'distance': distance}
    # print(final_routed_dict)
    return final_routed_dict


def minimum_distance(distance_matrix, coordinates, veh_capacity, num_of_routes):
    # As the data contains factory location, time and distance data at index 0 of each file, we exclude that for the
    # calculations among the points and take it separately
    distance_matrix_excl = distance_matrix[1:, 1:]
    coordinates_excl = coordinates[1:]
    vertices = coordinates_excl

    reference_vertex = coordinates[0]  # Lucky Motor Corp

    # Calculate distance matrix and select top N the farthest vertices as per the number_routes required
    initial_vertex = find_initial_vertex(vertices, reference_vertex)
    selected_vertices_indices = select_farthest_vertices(vertices, distance_matrix_excl, initial_vertex,
                                                         k=num_of_routes)

    selected_vertices = [vertices[i] for i in selected_vertices_indices]

    # Find the unique nearest neighbors for the selected vertices
    nearest_neighbors_indices = find_unique_nearest_neighbors(selected_vertices_indices, vertices, distance_matrix_excl)
    print(nearest_neighbors_indices)

    nearest_neighbors = {}
    nearest_neighbors = {tuple(vertices[i]): vertices[nearest_neighbors_indices[i]] for i in nearest_neighbors_indices}

    # Create a folium map centered around the average location of selected vertices
    average_lat = np.mean([vertex[0] for vertex in selected_vertices])
    average_lon = np.mean([vertex[1] for vertex in selected_vertices])
    map_center = (average_lat, average_lon)
    mymap = folium.Map(location=map_center, zoom_start=11)

    # Add markers for each selected vertex
    for vertex in selected_vertices:
        folium.Marker(location=vertex, popup=f'Selected Vertex: {vertex}', icon=folium.Icon(color='blue')).add_to(mymap)

    # Add markers and lines for the nearest neighbors
    for sv, nv in nearest_neighbors.items():
        folium.Marker(location=nv, popup=f'Nearest Vertex: {nv}', icon=folium.Icon(color='green')).add_to(mymap)
        folium.PolyLine([sv, nv], color="red", weight=2.5, opacity=1).add_to(mymap)

    # Save the map to an HTML file
    mymap.save('selected_vertices_with_nearest_neighbors_map.html')

    index_for_df = list(range(distance_matrix_excl.shape[1]))
    distance_matrix_df = pd.DataFrame(distance_matrix_excl, index=index_for_df, columns=index_for_df)

    routes_dict = {}
    c = 0
    for i in nearest_neighbors_indices:
        routes_dict[c] = [i, nearest_neighbors_indices[i]]
        c += 1
    print(routes_dict)

    last_stops = {}
    all_selected_stops = []
    for x, y in routes_dict.items():
        print(x, y)
        last_stops[x] = y[-1]
        all_selected_stops += y

    all_stops = list(range(0, len(coordinates_excl)))
    available_stops = [x for x in all_stops if x not in all_selected_stops]
    len(available_stops)

    for r in range(len(coordinates)):
        if len(available_stops) == 0:
            break
        available_last_stops = {}
        available_routes_dict = {}
        for key, value in routes_dict.items():
            if len(value) < veh_capacity:
                available_routes_dict[key] = value
                available_last_stops[key] = value[-1]

        df_min_dist = distance_matrix_df.loc[list(available_last_stops.values()), available_stops]

        min_value = df_min_dist.min().min()

        # Step 2: Locate the position of this minimum value
        result = df_min_dist.stack().idxmin()

        print("Row label (from):", result[0])
        print("Column label (to):", result[1])
        print("minimum distance: ", min_value)

        def get_key_from_value(d, value):
            return next((key for key, val in d.items() if val == value), None)

        key = get_key_from_value(last_stops, result[0])
        routes_dict[key] = routes_dict[key] + [result[1]]
        last_stops[key] = result[1]
        all_selected_stops += [result[1]]
        available_stops = [x for x in all_stops if x not in all_selected_stops]
        # print(len(available_stops))

        print(key)
        print(routes_dict)
        # print(last_stops)
        # print(all_selected_stops)
        # print(available_stops)
    # Adjusting data to cater for Factory location at 0
    increased_data = {key: [value + 1 for value in values] for key, values in routes_dict.items()}
    routes_dict = increased_data
    clusters = routes_dict
    return clusters


# Routing Through Traveling Salesman Problem
# implementation of traveling Salesman Problem

def travellingSalesmanProblem(graph, graph_time, s, first_level_algo, type_transit='drop', V=4):
    optim_route = []
    # store all vertex apart from source vertex
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)
    # print(vertex)
    # store minimum weight Hamiltonian Cycle
    if first_level_algo == 'CWS':
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


def get_encoded_polyline(api_key, origin, destination, waypoints):
    gmaps = googlemaps.Client(key=api_key)
    directions_result = gmaps.directions(origin, destination, waypoints=waypoints, mode="driving")

    encoded_polyline = directions_result[0]['overview_polyline']['points']

    return encoded_polyline


def get_google_optimized_route(cluster, i, coordinates, API_KEY, type_transit, distance_matrix, shift_time_min,
                               shift_time_hr):
    # Function to get the optimized route using Google Maps Directions API
    def get_optimized_route(origin, waypoints, destination):
        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        waypoints_str = "|".join([f"{lat},{lon}" for lat, lon in waypoints])

        dt = int(get_elapsed_seconds(shift_time_hr, shift_time_min, 0))

        if type_transit == 'drop':
            # dt = int(get_elapsed_seconds(shift_time_hr,shift_time_min,0))
            params = {
                "origin": f"{origin[0]},{origin[1]}",
                "destination": f"{destination[0]},{destination[1]}",  ## End at the farthest waypoint for simplicity
                "waypoints": f"optimize:true|{waypoints_str}",
                "key": API_KEY,
                "departure_time": dt
            }
        else:
            # pick
            # dt = int(get_elapsed_seconds(17,0,0))
            params = {
                "origin": f"{origin[0]},{origin[1]}",
                "destination": f"{destination[0]},{destination[1]}",  ## End at the farthest waypoint for simplicity
                "waypoints": f"optimize:true|{waypoints_str}",
                "key": API_KEY,
                "arrival_time": dt
            }

        # dt = 1719194400 + 3*86400

        '''params = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",  ## End at the farthest waypoint for simplicity
            "waypoints": f"optimize:true|{waypoints_str}",
            "key": API_KEY,
            "departure_time": dt
        }'''

        response = requests.get(base_url, params=params)
        route = response.json()
        # print(route)

        if route['status'] == 'OK':
            legs = route['routes'][0]['legs']
            total_distance = sum(leg['distance']['value'] for leg in legs) / 1000  # in km
            total_duration = sum(leg['duration']['value'] for leg in legs) / 60  # in minutes

            optimized_waypoints_order = route['routes'][0]['waypoint_order']
            optimized_waypoints = [waypoints[i] for i in optimized_waypoints_order]

            return total_distance, total_duration, optimized_waypoints, route['routes'][0]['overview_polyline'][
                'points']
        else:
            print(f"Error: {route['status']}")
            if 'error_message' in route:
                print(f"Error Message: {route['error_message']}")
            return None

    # indices = [coordinates.index(list(value)) for value in cluster] #it returns indices of coordinates against list of waypoints
    indices = [0] + cluster
    cluster = [coordinates[idx] for idx in indices]
    print(cluster)
    pop_index = distance_matrix[indices[1:], indices[0]].argmax()  # selecting farthest point
    # Get the origin and waypoints
    origin_original = cluster[0]
    waypoints = cluster[1:]
    # print(waypoints)
    destination_original = waypoints.pop(pop_index)
    # print(waypoints)

    if type_transit == 'drop':
        origin = origin_original
        destination = destination_original
    else:
        destination = origin_original
        origin = destination_original

    # Get the optimized route details
    result = get_optimized_route(origin, waypoints, destination)

    if result:
        total_distance, total_duration, optimized_waypoints, polyline = result

        # Generate Google Maps URL
        optimized_waypoints_str = "|".join([f"{lat},{lon}" for lat, lon in optimized_waypoints])
        google_maps_url = (
            f"https://www.google.com/maps/dir/?api=1&origin={origin[0]},{origin[1]}"
            f"&destination={destination[0]},{destination[1]}"
            f"&waypoints={optimized_waypoints_str}"
        )

        print(f"Total Distance: {total_distance} km")
        print(f"Total Duration: {total_duration / 60} hours")
        print(f"Google Maps URL: {google_maps_url}")
    else:
        print("Failed to fetch route details.")
    complete_route_coordinates = [origin] + optimized_waypoints + [destination]
    indices = [coordinates.index(list(value)) for value in complete_route_coordinates]
    # print(complete_route_coordinates)
    # print(indices)
    # print(polyline)
    return indices, total_distance, total_duration, google_maps_url, polyline


def get_next_weekday(target_hour, target_minute, target_weekday):
    # Define the PST timezone
    pst = pytz.timezone('Asia/Karachi')

    # Get the current date and time in PST
    now = datetime.now(pst)

    # Calculate the next occurrence of the specified weekday at the given time
    days_ahead = target_weekday - now.weekday()
    if days_ahead <= 0:  # If the day is today or has passed this week
        days_ahead += 7

    next_weekday = now + timedelta(days=days_ahead)
    target_time_pst = next_weekday.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

    return target_time_pst


def get_elapsed_seconds(th, tm, wd):
    # Get the next occurrence of the specified weekday at 17:00 PST (let's assume Monday)
    target_time_pst = get_next_weekday(target_hour=th, target_minute=tm, target_weekday=wd)  # 0 is Monday

    # Convert the target time to UTC
    target_time_utc = target_time_pst.astimezone(pytz.utc)

    # Define the Unix epoch start time in UTC
    epoch_start = datetime(1970, 1, 1, 0, 0, 0, tzinfo=pytz.utc)

    # Calculate the difference in seconds
    elapsed_time = (target_time_utc - epoch_start).total_seconds()

    # Subtract 20 minutes (1200 seconds) from the elapsed time
    adjusted_elapsed_time = elapsed_time - 20 * 60

    return elapsed_time


def get_indv_distance_durations(routes_list, type_transit, distance_matrix, time_matrix):
    distance_trail = []
    time_trail = []
    # if statement check for drop
    # routes_list = final_routed_dict_google_optim[0]['route_vertex_index']
    for i in range(len(routes_list) - 1):
        # print(routes_list[i],routes_list[i+1])
        from_vertex_index = routes_list[i]
        to_vertex_index = routes_list[i + 1]
        if type_transit == 'drop':
            distance_trail += [distance_matrix[to_vertex_index, from_vertex_index] / 1000]
            time_trail += [time_matrix[to_vertex_index, from_vertex_index] / 60]
        else:
            distance_trail += [distance_matrix[from_vertex_index, to_vertex_index] / 1000]
            time_trail += [time_matrix[from_vertex_index, to_vertex_index] / 60]

    indv_distances = accumulated_sum(distance_trail, type_transit)
    indv_durations = accumulated_sum(time_trail, type_transit)
    return indv_distances, indv_durations


# Function to find initial vertex farthest from reference vertex
def find_initial_vertex(vertices, reference_vertex):
    distances = [geodesic(vertex, reference_vertex).km for vertex in vertices]
    return np.argmax(distances)


# Function to select farthest vertices
def select_farthest_vertices(vertices, distance_matrix, initial_vertex, k=10):
    selected_vertices = [initial_vertex]
    for _ in range(1, k):
        max_min_distance = -1
        next_vertex = -1
        for i in range(len(vertices)):
            if i not in selected_vertices:
                min_distance = min([distance_matrix[i][j] for j in selected_vertices])
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    next_vertex = i
        selected_vertices.append(next_vertex)
    return selected_vertices


# Function to find the nearest vertex that is not closer to any other selected vertex
def find_unique_nearest_neighbors(selected_vertices, vertices, distance_matrix):
    nearest_neighbors = {}
    for s in selected_vertices:
        distances = [distance_matrix[s][v] for v in range(len(vertices))]
        sorted_distances = sorted([(dist, v) for v, dist in enumerate(distances) if v not in selected_vertices])
        for dist, v in sorted_distances:
            if v not in nearest_neighbors.values():
                nearest_neighbors[s] = v
                break
    return nearest_neighbors
