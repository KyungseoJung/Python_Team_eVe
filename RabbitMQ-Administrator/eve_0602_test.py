import cplex
from cplex.exceptions import CplexSolverError
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import openrouteservice
from openrouteservice import client, places
ors = client.Client(key='5b3ce3597851110001cf6248caed29210bfa4c9ebb65a1cfee84a1f4')

import math
from itertools import permutations
import networkx as nx
import random
import copy
import time
import sys
import csv
import json
import os
import psutil
import gc
import ray


@ray.remote
class KShortestPathActor:
    def __init__(self, graph_data):
        self.G = nx.node_link_graph(graph_data)
    
    def calculate_k_shortest_paths(self, source, target, k):
        try:
            k_shortest_paths = list(nx.shortest_simple_paths(self.G, source=source, target=target, weight='weight'))
            k_shortest_paths = k_shortest_paths[:k]
            return (source, target, k_shortest_paths)
        except nx.NetworkXNoPath:
            return (source, target, [])
        
class two_phase_heuristic:
    def __init__(self, datafile, battery_json_path, truck_csv_path, drawroute_json_path,  M=100000, lamb=50000, TN=4, CT=2, num_cpus=4):
        self.datafile = datafile
        self.battery_json_path = battery_json_path
        self.truck_csv_path = truck_csv_path
        self.drawroute_json_path = drawroute_json_path

        self.M = M
        self.lamb = lamb
        self.TN = TN
        self.CT = CT
        self.T = list(range(1, TN + 1))
        self.B = [str(x) for x in range(1, len(self.T) * 2 + 1)]
        self.CB_l = [75] * len(self.B)
        self.C_i=[]

        self.G = None   # 그래프 초기화
        # self.all_k_shortest_paths_result = None
        self.all_k_shortest_paths_result = {}
        self.loaded_k_sp_result = []

        self.elecvery_visited_nodes_set = set()
        self.elecvery_battery_visited_nodes = {}
        self.elecvery_battery_used_capacity = set()

        self.cluster_visited_nodes_set = set()
        self.cluster_battery_visited_nodes = {}
        self.cluster_battery_used_capacity = set()

        self.visited_nodes_set = set()
        self.battery_visited_nodes = {} 
        self.initial_visited_nodes = set()

        # self.manager = None
        self.data = None

        if ray.is_initialized():
            ray.shutdown()

        ray.init(num_cpus=num_cpus, log_to_driver=True)  


    def read_datafile(self):
        with open(self.datafile, 'r') as file:
            file_content = file.read()

        lines = file_content.strip().split('\n')[10:]    # depot 제외
        N = []  
        x_coord = []
        y_coord = []
        R_i = []
        E_i = []
        L_i = []

        for line in lines:
            data = line.split()
            cust_no = data[0]
            cust_no_with_quotes = '' + cust_no + ''
            N.append(cust_no_with_quotes)     # N: string
            x_coord.append(float(data[1]))
            y_coord.append(float(data[2]))
            R_i.append(int(data[3]))
            E_i.append(int(data[4]))
            L_i.append(int(data[5]))

        N_hat = ['h' + item for item in N]
        line_d = file_content.strip().split('\n')[9]
        E_0 = int(line_d.split()[4])
        L_n = int(line_d.split()[5])
        xd_coord = float(line_d.split()[1])
        yd_coord = float(line_d.split()[2])
        
        is_cloned = {original: cloned for original, cloned in zip(N, N_hat)}
        V = N + N_hat
        V_0 = V.copy()
        V_0.append(str(0))
        V_1 = V.copy()
        V_1.append(str(len(N) + 1))
        V_2 = V.copy()
        V_2.append(str(0))
        V_2.append(str(len(N) + 1))
        A_1 = {(i, j) for i in V for j in V if i != j and (j not in is_cloned or is_cloned[j] != j)}
        A_2 = {(original, cloned) for original, cloned in is_cloned.items()}
        A_3 = {(str(0), j) for j in V}
        A_4 = {(i, str(len(N) + 1)) for i in V}
        A_5 = {(cloned, original) for original, cloned in is_cloned.items()}
        A = A_1.union(A_2, A_3, A_4) - A_5
        A_p = A - A_2
        S_i = [5] * len(V)
        C_i = R_i
        coordinates = [(float(x), float(y)) for x, y in zip(x_coord, y_coord)]
        xd_coord, yd_coord = float(xd_coord), float(yd_coord)


        # OpenRouteService API를 사용하여 duration과 distance 계산
        locations = [[xd_coord, yd_coord]] + coordinates   # 디포를 시작에 추가

        extended_locations = [locations[i] for i in range(1, len(coordinates) + 1)] + [locations[i] for i in range(1, len(coordinates) + 1)] + [locations[0]] + [locations[0]]

        print("Extended locations:", extended_locations)
    
        request = {
            'locations': extended_locations,
            'profile': 'driving-car',
            'metrics': ['duration', 'distance']
        }
        
        matrix = ors.distance_matrix(**request)
        durations = np.array(matrix['durations'])
        distances = np.array(matrix['distances'])

        # print(f"Locations: {len(locations)}, Durations: {len(durations)}, Distances: {len(distances)}")

        T_ij = {}   # time - durations
        D_ij = {}   # distance

        # T_ij 및 D_ij 생성
        for i, origin in enumerate(V_0):    # i: index, origin: node
            # print(f"Index: {i}, Origin: {origin}")
            for j, destination in enumerate(V_1):
                # 제약 조건 처리
                if i == j or (origin.startswith('h') and origin[1:] == destination):
                    continue

                origin_idx = i
                destination_idx = j
                
                if (origin == '0' and destination == str(len(coordinates) + 1)) or (origin == str(len(coordinates) + 1) and destination == '0'):
                    T_ij[(origin, destination)] = 0
                    D_ij[(origin, destination)] = 0
                else:
                    if origin == destination:
                        T_ij[(origin, destination)] = 0
                        D_ij[(origin, destination)] = 0
                    elif destination.startswith('h') and destination[1:] == origin:
                        T_ij[(origin, destination)] = R_i[int(origin) - 1]
                        D_ij[(origin, destination)] = 0
                    else:
                        T_ij[(origin, destination)] = durations[origin_idx][destination_idx] / 60  # 초를 분으로 변환
                        D_ij[(origin, destination)] = distances[origin_idx][destination_idx]

        # 소수점 둘째 자리까지 반올림
        T_ij = {key: round(value, 2) for key, value in T_ij.items()}

        T_ij_ot = T_ij.copy()

        for key in list(T_ij.keys()):
            if key[1] == '11':
                new_key = (key[0], '0')
                T_ij_ot[new_key] = T_ij[key]

        # print("T_ij: ", T_ij)
        # print("D_ij: ", D_ij)
        
        self.depot_operation_time = L_n - E_0
        self.N = N
        self.N_hat = N_hat
        self.E_0 = E_0
        self.L_n = L_n
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.xd_coord = xd_coord
        self.yd_coord = yd_coord
        self.R_i = R_i
        self.E_i = E_i
        self.L_i = L_i

        self.V=V
        self.V_0=V_0
        self.V_1=V_1
        self.V_2=V_2
        self.A_1=A_1
        self.A_2=A_2
        self.A_3=A_3
        self.A_4=A_4
        self.A_5=A_5
        self.A = A
        self.A_p=A_p
        self.S_i=S_i
        self.C_i = C_i
        self.T_ij=T_ij
        self.T_ij_ot = T_ij_ot
        self.D_ij = D_ij

        self.node_to_index = {node: index for index, node in enumerate(self.N)}

    def construct_graph(self):
        self.G = nx.DiGraph()
        for node in self.N:
            self.G.add_node(node) 
        for i in range(len(self.N)):
            for j in range(len(self.N)):
                if i != j:
                    node1 = self.N[i]
                    node2 = self.N[j]
                    # distance = self.calculate_distance(node1, node2)  
                    distance = self.D_ij[(node1, node2)]
                    time = self.T_ij[(node1, node2)]
                    if time <= self.depot_operation_time and self.check_time_window(node1, node2):
                        self.G.add_edge(node1, node2, weight=distance)  

    # def get_coordinates(self, node):
    #     index = self.N.index(node)
    #     x = self.x_coord[index]
    #     y = self.y_coord[index]
    #     return x, y
    
    # def calculate_distance(self, node1, node2):
    #     x1, y1 = self.get_coordinates(node1)
    #     x2, y2 = self.get_coordinates(node2)
    #     distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    #     return distance

    def check_time_window(self, node1, node2):
        # travel_time = self.calculate_distance(node1, node2)   # 차량 속도 고려 X
        travel_time = self.T_ij[(node1, node2)]     # 차량 속도 고려 
        arrive_time = self.E_i[self.N.index(node1)] + travel_time
        end_window = self.L_i[self.N.index(node2)]
        return arrive_time <= end_window   


    # 1
    @staticmethod
    @ray.remote
    def calculate_k_shortest_paths(graph, source, target, k):
        try:
            k_shortest_paths = list(nx.shortest_simple_paths(graph, source=source, target=target, weight='weight'))
            k_shortest_paths = k_shortest_paths[:k]
            return (source, target, k_shortest_paths)
        except nx.NetworkXNoPath:
            return (source, target, [])
        

    def all_k_shortest_paths(self, k):
        nodes = list(self.G.nodes())
        futures = []
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if nx.has_path(self.G, node1, node2):
                    futures.append(two_phase_heuristic.calculate_k_shortest_paths.remote(self.G, node1, node2, k))
        results = ray.get(futures)
        for result in results:
            source, target, paths = result
            self.all_k_shortest_paths_result[(source, target)] = paths
            if nx.has_path(self.G, target, source):
                self.all_k_shortest_paths_result[(target, source)] = [list(reversed(path)) for path in paths]
        return self.all_k_shortest_paths_result
    

    def elecvery(self):
        valid_nodes = []
        invalid_nodes = []
        for node in self.N:
            index = int(node) - 1
            if self.E_i[index] + self.R_i[index] + 10  <= self.L_i[index] and self.L_i[index] - 10 - self.R_i[index] >= self.E_0 + self.T_ij['0',node] :
                valid_nodes.append(node)
            else:
                invalid_nodes.append(node)
        
        T_ij_new = {}

        for (i, j), distance in self.T_ij.items():
            if i in self.N or i == '0' or i == str(len(self.N) + 1):
                if j in self.N or j == '0' or j == str(len(self.N) + 1):
                    T_ij_new[(i, j)] = distance

        W = self.N
        W_0 = {(i,j) for i in W for j in W if i != j}
        W_1 = {(str(0), j) for j in W}
        W_2 = {(i, str(len(self.N) + 1)) for i in W}
        AW = W_0.union(W_1, W_2)

        fitness_function = {}
        for (i,j) in AW:
            if i != '0' and j != str(len(self.N)+1):
                fitness_function[(i, j)] = (self.E_i[int(j)-1]) - T_ij_new[(i,j)] - self.L_i[int(i)-1]
 
            elif i == '0':
                fitness_function[(i, j)] = 1/((self.E_i[int(j)-1]) - self.E_0 - T_ij_new[(i,j)])
            elif j == str(len(self.N)+1):
                fitness_function[(i, j)] = 0

        fitness_function_1 = {}

        for (i,j) in AW:
            if fitness_function[(i,j)] > 0:
                fitness_function_1[(i,j)] = fitness_function[(i,j)]
            # elif fitness_function[(i,j)] <= 0:
            #     fitness_function_1[(i,j)] = 0

        clusters = []
        perform_nodes = valid_nodes[:]
        total_capacity = 0
        max_capacity = 75

        while perform_nodes:
            max_j = None
            max_value = float('-inf')
            for j in perform_nodes:
                if ('0', j) in fitness_function_1 and fitness_function_1[('0', j)] > max_value:
                    max_value = fitness_function_1[('0', j)]
                    max_j = j
            if max_j:
                cluster = [max_j]
                total_capacity += self.R_i[int(max_j) - 1]
                perform_nodes.remove(max_j)
                
                current_node = max_j
                while True:
                    min_j = None
                    min_value = float('-inf')
                    for j in perform_nodes:
                        if (current_node, j) in fitness_function_1 and fitness_function_1[(current_node, j)] > min_value:
                            if total_capacity + self.R_i[int(j) - 1] <= max_capacity:
                                min_value = fitness_function_1[(current_node, j)]
                                min_j = j
                    if min_j:
                        cluster.append(min_j)
                        total_capacity += self.R_i[int(min_j) - 1]
                        perform_nodes.remove(min_j)
                        current_node = min_j
                    else:
                        break
                
                clusters.append(cluster)
                total_capacity = 0
            else:
                break

        clusters = clusters[:self.TN]
        
        cluster_value ={}
        for i, j in enumerate(clusters, start=1):
            for k in j:
                cluster_value[k] = str(i)

        # cluster_visited_nodes_set = set(map(int, valid_nodes))
        elecvery_visited_nodes_set = {int(key) for key in cluster_value.keys()}

        battery_node = {}
        for key, value in cluster_value.items():
            battery_node.setdefault(value, []).append(key)

        battery_usage = {}
        for key, value in battery_node.items():
            battery_usage[key] = sum(self.R_i[int(node)-1] for node in value)

        unused_values = set(self.B) - set(map(str, battery_node.keys()))
        for new_node in unused_values:
            battery_node[int(new_node)] = []
            battery_usage[int(new_node)] = 0

        elecvery_battery_visited_node = battery_node.copy()
        elecvery_battery_used_capacity = {int(key): int(value) for key, value in battery_usage.items()}
        elecvery_battery_visited_nodes = {int(key): list(map(int, value)) for key, value in elecvery_battery_visited_node.items()}

        # self.elecvery_visited_nodes_set = elecvery_visited_nodes_set
        # self.elecvery_battery_visited_nodes = elecvery_battery_visited_nodes
        # self.elecvery_battery_used_capacity = elecvery_battery_used_capacity

        return elecvery_visited_nodes_set , elecvery_battery_visited_nodes, elecvery_battery_used_capacity


    def Greed_Cluster(self):
        valid_nodes = []
        invalid_nodes = []
        for node in self.N:
            index = int(node) - 1
            if self.E_i[index] + self.R_i[index] + 10  <= self.L_i[index] and self.L_i[index] - 10 - self.R_i[index] >= self.E_0 + self.T_ij['0',node] :
                valid_nodes.append(node)
            else:
                invalid_nodes.append(node)
        
        T_ij_new = {}

        for (i, j), distance in self.T_ij.items():
            if i in self.N or i == '0' or i == str(len(self.N) + 1):
                if j in self.N or j == '0' or j == str(len(self.N) + 1):
                    T_ij_new[(i, j)] = distance

        W = self.N
        W_0 = {(i,j) for i in W for j in W if i != j}
        W_1 = {(str(0), j) for j in W}
        W_2 = {(i, str(len(self.N) + 1)) for i in W}
        AW = W_0.union(W_1, W_2)

        fitness_function = {}
        for (i,j) in AW:
            if i != '0' and j != str(len(self.N)+1):
                fitness_function[(i, j)] = (self.L_i[int(j)-1] - 10 - self.R_i[int(j)-1]) - T_ij_new[(i,j)] - (self.L_i[int(i)-1])

            elif i == '0':
                fitness_function[(i, j)] = 1/((self.L_i[int(j)-1] - 10 - self.R_i[int(j)-1]) - self.E_0 - T_ij_new[(i,j)])
            elif j == str(len(self.N)+1):
                fitness_function[(i, j)] = 0

        fitness_function_1 = {}

        for (i,j) in AW:
            if fitness_function[(i,j)] >= 0:
                fitness_function_1[(i,j)] = fitness_function[(i,j)]

        clusters = []
        perform_nodes = valid_nodes[:]
        max_capacity = 75
        P_of_best_choice = 0.6
        Q_of_best_choice = 1.0 - P_of_best_choice

        while perform_nodes:
            max_j_candidates = []  # List to hold candidate j values
            max_value = float('-inf')
            
            for j in perform_nodes:
                if ('0', j) in fitness_function_1 and fitness_function_1[('0', j)] > max_value:
                    max_value = fitness_function_1[('0', j)]
                    max_j_candidates.append((j, fitness_function_1[('0', j)]))
            
            # Sort candidates based on fitness value in descending order and keep top 3
            max_j_candidates.sort(key=lambda x: x[1], reverse=True)
            max_j_candidates = max_j_candidates[:3]
            
            if max_j_candidates:
                # Calculate probabilities dynamically
                num_candidates = len(max_j_candidates)
                probabilities = [P_of_best_choice * (Q_of_best_choice ** i) for i in range(num_candidates)]
                probabilities = [p / sum(probabilities) for p in probabilities]  # Normalize probabilities
                
                # Choose max_j based on probabilities
                choices = [candidate[0] for candidate in max_j_candidates]
                chosen_max_j = random.choices(choices, weights=probabilities, k=1)[0]
                total_capacity = 0
                # Check capacity before adding to cluster
                if total_capacity + self.R_i[int(chosen_max_j) - 1] <= max_capacity:
                    cluster = [chosen_max_j]
                    total_capacity += self.R_i[int(chosen_max_j) - 1]
                    perform_nodes.remove(chosen_max_j)
                    
                    current_node = chosen_max_j
                    while True:
                        min_j_candidates = []
                        min_value = float('inf')
                        for j in perform_nodes:
                            if (current_node, j) in fitness_function_1 and fitness_function_1[(current_node, j)] < min_value:
                                if total_capacity + self.R_i[int(j) - 1] <= max_capacity:
                                    min_j_candidates.append((j, fitness_function_1[(current_node, j)]))
                        
                        # Sort candidates based on fitness value in ascending order and keep top 3
                        min_j_candidates.sort(key=lambda x: x[1])
                        min_j_candidates = min_j_candidates[:3]
                        
                        if min_j_candidates:
                            # Calculate probabilities dynamically
                            num_candidates = len(min_j_candidates)
                            probabilities = [P_of_best_choice * (Q_of_best_choice ** i) for i in range(num_candidates)]
                            probabilities = [p / sum(probabilities) for p in probabilities]  # Normalize probabilities
                            
                            # Choose min_j based on probabilities
                            choices = [candidate[0] for candidate in min_j_candidates]
                            chosen_min_j = random.choices(choices, weights=probabilities, k=1)[0]
                            
                            # Check capacity before adding to cluster
                            if total_capacity + self.R_i[int(chosen_min_j) - 1] <= max_capacity:
                                cluster.append(chosen_min_j)
                                total_capacity += self.R_i[int(chosen_min_j) - 1]
                                perform_nodes.remove(chosen_min_j)
                                current_node = chosen_min_j
                            else:
                                break
                        else:
                            break
                    
                    clusters.append(cluster)
                    total_capacity = 0
            else:
                break


        clusters = clusters[:len(self.B)]

        cluster_value ={}
        for i, j in enumerate(clusters, start=1):
            for k in j:
                cluster_value[k] = str(i)

        # cluster_visited_nodes_set = set(map(int, valid_nodes))
        cluster_visited_nodes_set = {int(key) for key in cluster_value.keys()}

        battery_node = {}
        for key, value in cluster_value.items():
            battery_node.setdefault(value, []).append(key)

        battery_usage = {}
        for key, value in battery_node.items():
            battery_usage[key] = sum(self.R_i[int(node)-1] for node in value)

        unused_values = set(self.B) - set(map(str, battery_node.keys()))
        for new_node in unused_values:
            battery_node[int(new_node)] = []
            battery_usage[int(new_node)] = 0

        cluster_battery_visited_node = battery_node.copy()
        cluster_battery_used_capacity = {int(key): int(value) for key, value in battery_usage.items()}
        cluster_battery_visited_nodes = {int(key): list(map(int, value)) for key, value in cluster_battery_visited_node.items()}

        # self.clusters = clusters
        # self.cluster_value = cluster_value
        # self.valid_nodes = valid_nodes
        # self.cluster_visited_nodes_set = cluster_visited_nodes_set
        # self.cluster_battery_visited_nodes = cluster_battery_visited_nodes
        # self.cluster_battery_used_capacity = cluster_battery_used_capacity
        
        return cluster_visited_nodes_set , cluster_battery_visited_nodes, cluster_battery_used_capacity

    
    # ortools - battery route with cvrptw 
    def create_data_model(self):
        data = {}
        depot_window = (self.E_0, self.L_n)
        data['locations'] = [(self.xd_coord, self.yd_coord)] + \
                            [(x, y) for x, y in zip(self.x_coord, self.y_coord)]
        data['time_windows'] = [depot_window] + [(E_i, L_i) for E_i, L_i in zip(self.E_i, self.L_i)]
        data['charge_times'] = [0] + self.R_i  # 충전 요구량당 1분 걸린다고 가정
        data['num_trucks'] = self.TN
        data['truck_capacities'] = self.CT
        data['num_batteries'] = data['num_trucks'] * data['truck_capacities']
        data['battery_capacities'] = [75] * data['num_batteries'] 
        data['depot'] = 0
        data['demands'] = [0] + self.R_i
        data['speed'] = 10
        
        return data
    
    # def compute_euclidean_distance_matrix(self, locations):
    #     """Creates callback to return distance between points."""
    #     distances = {}
    #     for from_counter, from_node in enumerate(locations):
    #         distances[from_counter] = {}
    #         for to_counter, to_node in enumerate(locations):
    #             if from_counter == to_counter:
    #                 distances[from_counter][to_counter] = 0
    #             else:
    #                 # Euclidean distance
    #                 # distances[from_counter][to_counter] = (int(
    #                 #     math.hypot((from_node[0] - to_node[0]),
    #                 #                (from_node[1] - to_node[1]))))
    #                 distance = math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))
    #                 distances[from_counter][to_counter] = round(distance, 2)  # 거리를 반올림하여 저장
    #     return distances
        
    def ORTools_cvrptw(self):
        visited_nodes_set = set()
        battery_visited_nodes = {} 
        battery_used_capacity = {}

        if self.data is None:
            self.data = self.create_data_model()

        manager = pywrapcp.RoutingIndexManager(
            len(self.data['locations']),
            self.data['num_batteries'], 
            self.data['depot']
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # 각 노드에서의 충전하는 시간 추가
            charge_time = self.data['charge_times'][from_node]

            try:
                return self.T_ij_ot[(str(from_node), str(to_node))] + charge_time
            except KeyError:
                # print(f"KeyError with from_node: {from_node}, to_node: {to_node}")
                return charge_time 
            # return int(distance_matrix[from_node][to_node] / self.data['speed']) + charge_time

        def demand_callback(from_index):  
            """Returns the demand of the node."""
            from_node = manager.IndexToNode(from_index)
            return self.data['demands'][from_node]

        # distance_matrix = self.compute_euclidean_distance_matrix(self.data['locations'])

        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        depot_earlest_time = self.data['time_windows'][0][0]
        depot_latest_time = self.data['time_windows'][0][1]

        # time dimension 추가
        routing.AddDimension(
                transit_callback_index,
                depot_latest_time,   # allow waiting time, 위치에서의 대기 시간
                depot_latest_time,   # maximum time per vehicle in a route, 한 차가 route를 도는데 쓸 수 있는 최대 시간
                False,  # Don't force start cumul to zero.
                'Time')
       
        # time window 제약 추가
        time_dimension = routing.GetDimensionOrDie('Time')

        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(self.data['time_windows']):
            if location_idx == 0:
                continue
            index = manager.NodeToIndex(location_idx)
            
            time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])            
            routing.AddDisjunction([index], 1000)  # 모든 node 방문 안해도 ok

        # Add time window constraints for each vehicle start node.  
        for vehicle_id in range(self.data['num_batteries']):
            index = routing.Start(vehicle_id)
            time_dimension.CumulVar(index).SetRange(depot_earlest_time, depot_latest_time)

        # 최적화 대상으로 설정
        # 각 차량의 시작 지점과 끝 지점의 누적 변수를 최소화하도록 
        # for i in range(data['num_batteries']):
        #     routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
        #     routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        # demand dimension
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.data['battery_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        # search_parameters.local_search_metaheuristic = (
        #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        # search_parameters.time_limit.FromSeconds(1)

        assignment = routing.SolveWithParameters(search_parameters)
      
        # Print solution on console.
        if assignment:                        
            for vehicle_id in range(self.data['num_batteries']):
                index = routing.Start(vehicle_id)
                plan_output_demand = f'Amount of Battery {vehicle_id + 1} used: '
                plan_output_customer = f'Number of customers visited by Battery {vehicle_id + 1}: '
                route_demand = 0
                customer_count = 0
                visited_nodes = []

                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    route_demand += self.data['demands'][node]
                    
                    if node != self.data['depot']:
                        customer_count += 1
                        visited_nodes_set.add(node)
                        visited_nodes.append(node) 

                    plan_output_demand += f'{route_demand} -> '
                    index = assignment.Value(routing.NextVar(index))

                node = manager.IndexToNode(index)
                plan_output_demand += f'{route_demand + self.data["demands"][node]}'
                plan_output_customer += f'{customer_count}'

                battery_visited_nodes[vehicle_id + 1] = visited_nodes
                battery_used_capacity[vehicle_id + 1] = route_demand

            #     print(plan_output_demand)
            #     print(plan_output_customer)

            # self.visited_nodes_set = visited_nodes_set
            # self.battery_visited_nodes = battery_visited_nodes
            # self.battery_used_capacity = battery_used_capacity
            
            return visited_nodes_set, battery_visited_nodes, battery_used_capacity


    # def node_removal_1(self, route, battery_route, battery_used_capacity):

    #     route_copy = copy.deepcopy(route)
    #     new_battery_route = copy.deepcopy(battery_route)
    #     new_battery_used_capacity = copy.deepcopy(battery_used_capacity)

    #     node = random.choice(list(route_copy))    # 삭제할 노드 랜덤으로 선택
    #     new_route = route_copy - {node}
    #     print("삭제 node: ", node)

    #     for battery_id, nodes in new_battery_route.items():
    #         if node in nodes:
    #             new_battery_route[battery_id].remove(node)

    #     # 배터리 용량 업데이트
    #     node_index = self.node_to_index[str(node)]
    #     node_demand = self.R_i[node_index]
    #     new_battery_used_capacity[battery_id] -= node_demand
        
    #     return new_route, new_battery_route, new_battery_used_capacity
    
    def node_removal(self, route, battery_route, battery_used_capacity):
        new_route = copy.deepcopy(route)
        new_battery_route = copy.deepcopy(battery_route)
        new_battery_used_capacity = copy.deepcopy(battery_used_capacity)

        # 방문한 노드 수가 많은 배터리들 중에서 배터리 선택
        max_visited_battery_ids = [battery_id for battery_id, nodes in new_battery_route.items() if nodes]
        if not max_visited_battery_ids:
            print("노드를 방문한 배터리가 한 개도 없습니다.")
            return new_route, new_battery_route, new_battery_used_capacity

        battery_id = random.choice(max_visited_battery_ids)
        visited_nodes = new_battery_route[battery_id]

        # 방문한 노드들 중에서 랜덤으로 삭제할 노드 선택
        node = random.choice(visited_nodes)
        new_route.remove(node)

        # 배터리 경로에서 노드 삭제
        new_battery_route[battery_id].remove(node)

        # 배터리 용량 업데이트
        node_index = self.node_to_index[str(node)]
        node_demand = self.R_i[node_index]
        new_battery_used_capacity[battery_id] -= node_demand

        return new_route, new_battery_route, new_battery_used_capacity

    def node_exchange(self, route, battery_route, battery_used_capacity):
        new_route = copy.deepcopy(route)
        new_battery_route = copy.deepcopy(battery_route)
        new_battery_used_capacity = copy.deepcopy(battery_used_capacity)

        battery_ids = list(new_battery_route.keys())
        non_empty_battery_ids = [bid for bid in battery_ids if new_battery_route[bid]]  
        
        if len(non_empty_battery_ids) < 2:
            return None, None, None

        battery_id1, battery_id2 = random.sample(non_empty_battery_ids, 2)

        # # 2개 이상의 노드를 방문하는 배터리 ID와 1개 또는 0개의 노드를 방문하는 배터리 ID 구분
        # multi_node_battery_ids = []
        # single_node_battery_ids = []

        # for bid, visited_nodes in new_battery_route.items():
        #     if len(visited_nodes) >= 2:
        #         multi_node_battery_ids.append(bid)
        #     else:
        #         single_node_battery_ids.append(bid)
        

        # if multi_node_battery_ids:
        #     if len(multi_node_battery_ids) == 1:
        #         # 1개의 multi-node 배터리만 있는 경우
        #         battery_id1 = multi_node_battery_ids[0]
        #         # 나머지 배터리 ID 중에서 또 다른 배터리 ID 선택
        #         remaining_battery_ids = [bid for bid in single_node_battery_ids if bid != battery_id1]
        #         battery_id2 = random.choice(remaining_battery_ids)
        #     else:
        #         # 2개 이상의 multi-node 배터리가 있는 경우
        #         battery_id1, battery_id2 = random.sample(multi_node_battery_ids, 2)
        # elif single_node_battery_ids:
        #     # 1개의 노드를 방문하는 배터리 ID가 있으면 그 중 하나 선택
        #     battery_id1 = random.choice(single_node_battery_ids)
        #     # 나머지 배터리 ID 중에서 또 다른 배터리 ID 선택
        #     remaining_battery_ids = [bid for bid in single_node_battery_ids if bid != battery_id1]
        #     battery_id2 = random.choice(remaining_battery_ids)
        # else:
        #     print("방문한 노드가 2개 이상인 배터리 ID 또는 1개 이상의 노드를 방문하지 않는 배터리 ID가 부족합니다.")
        #     return None

        visited_nodes1 = new_battery_route.get(battery_id1, [])
        visited_nodes2 = new_battery_route.get(battery_id2, [])

        # 각 배터리에서 마지막 노드 선택
        node1 = visited_nodes1[-1]
        node2 = visited_nodes2[-1]

        # 노드 교환
        visited_nodes1[-1] = node2
        visited_nodes2[-1] = node1

        # 배터리 1의 사용 용량 업데이트
        battery_id1_demand = sum(self.R_i[self.node_to_index[str(node)]] for node in visited_nodes1)
        new_battery_route[battery_id1] = visited_nodes1
        new_battery_used_capacity[battery_id1] = battery_id1_demand

        # 배터리 2의 사용 용량 업데이트
        battery_id2_demand = sum(self.R_i[self.node_to_index[str(node)]] for node in visited_nodes2)
        new_battery_route[battery_id2] = visited_nodes2
        new_battery_used_capacity[battery_id2] = battery_id2_demand
        
        return new_route, new_battery_route, new_battery_used_capacity


    def node_insertion(self, route, battery_route, battery_used_capacity):
        new_route = copy.deepcopy(route)
        new_battery_route = copy.deepcopy(battery_route)
        new_battery_used_capacity = copy.deepcopy(battery_used_capacity)

        # 모든 노드가 이미 추가된 경우를 확인
        all_nodes = set(self.N)
        if all_nodes.issubset(new_route):
            print("모든 노드가 이미 추가되었습니다. 더 이상 추가할 노드가 없습니다.")
            return route, battery_route, battery_used_capacity

        # battery_route를 방문한 노드의 수를 기준으로 정렬
        sorted_battery_route = sorted(new_battery_route.items(), key=lambda x: len(x[1]))

        # 방문한 노드가 적은 순서대로 배터리 선택
        battery_ids = [battery_id for battery_id, visited_nodes in sorted_battery_route if visited_nodes]

        if not battery_ids:
            return None   # 수정 필요
        
        # 방문한 노드가 가장 적은 배터리 선택
        battery_id = battery_ids[0]        
        visited_nodes = new_battery_route[battery_id]

        last_node_index = visited_nodes[-1] - 1
        last_node = self.N[last_node_index]

        neighbors = [] 
        found_neighbors = False

        while not found_neighbors:
            for successor_node in self.G.successors(last_node):
                if int(successor_node) not in new_route:
                    neighbors.append(int(successor_node))
                    found_neighbors = True

            if not found_neighbors:
                print("마지막 노드의 이웃이 없습니다. 새로운 마지막 노드를 선택합니다.")
                last_node_index -= 1
                if last_node_index < 0:
                    print("모든 노드가 이미 추가되었습니다. 더 이상 추가할 노드가 없습니다.")
                    return route, battery_route, battery_used_capacity
                last_node = self.N[last_node_index]
                print('last_node: ', last_node)

        if not neighbors:
            print("마지막 노드의 이웃이 없습니다.")
            return None

        new_node = random.choice(neighbors)
        # 추가될 노드의 요구량 계산
        node_index = self.node_to_index[str(new_node)]
        node_demand = self.R_i[node_index]

        # 배터리 용량을 고려하여 노드를 추가할 배터리 결정
        battery_capacity = self.CB_l[battery_id-1]

        battery_used_capacity = new_battery_used_capacity[battery_id]

        if battery_capacity >= battery_used_capacity + node_demand:  # 배터리가 요구량을 수용할 수 있는 경우
            new_route.add(new_node) 
            new_battery_route[battery_id].append(new_node)  # 노드를 해당 배터리에 추가
            new_battery_used_capacity[battery_id] += node_demand
        
        else:
            print(f"배터리 {battery_id}는 노드의 요구량을 수용할 용량이 부족합니다.")
            empty_battery_ids = [bid for bid in battery_route.keys() if not battery_route[bid]]

            if empty_battery_ids:
                selected_battery_id = random.choice(empty_battery_ids)
                new_battery_route[selected_battery_id].append(new_node)
                new_battery_used_capacity[selected_battery_id] += node_demand
                print(f"노드 {new_node}를 배터리 {selected_battery_id}에 추가했습니다.")
                
            else:
                print("모든 배터리가 요구량을 수용할 용량이 부족합니다.")

        return new_route, new_battery_route, new_battery_used_capacity


    def calculate_objective(self, route, battery_route):
        print("*****문제 풀기")

        self.process_route(route, battery_route)

        try:
            self.MIP_solver()
            objective_value = self.cpx.solution.get_objective_value()
            variable_names = self.cpx.variables.get_names()
            variable_values = self.cpx.solution.get_values()

            print("문제를 푼 방문노드: ", route)
            print("목적함수 값: ", objective_value)
            print("Variable Values:")
         
            # for name, value in zip(variable_names, variable_values):
            #     if value >= 0.0001:
            #         print(name, "=", value) 

            print("Solution status:", self.cpx.solution.get_status())
            
            return objective_value, variable_names, variable_values
        
        except CplexSolverError as e:
            print("CPLEX Error:", e)

            return None, None, None
        
    def calculate_objective_CT1(self, route, battery_route):
        print("*****문제 풀기")

        self.process_route(route, battery_route)

        try:
            self.MIP_solver_CT1()
            objective_value = self.cpx.solution.get_objective_value()
            variable_names = self.cpx.variables.get_names()
            variable_values = self.cpx.solution.get_values()

            print("문제를 푼 방문노드: ", route)
            print("목적함수 값: ", objective_value)
            print("Variable Values:")
         
            # for name, value in zip(variable_names, variable_values):
            #     if value >= 0.0001:
            #         print(name, "=", value) 

            print("Solution status:", self.cpx.solution.get_status())
            
            return objective_value, variable_names, variable_values
        
        except CplexSolverError as e:
            print("CPLEX Error:", e)

            return None, None, None
        

    @ray.remote
    def greed_cluster_initial_solution(self):
        initial_solutions = []
        greed_cluster_start_time = time.time()
        while time.time() - greed_cluster_start_time < 15:
            cluster_route, cluster_battery_route, cluster_battery_used_capacity = self.Greed_Cluster()
            cluster_energy, cluster_variable_names, cluster_variable_values = self.calculate_objective(cluster_route, cluster_battery_route)
            if cluster_energy is not None:
                print("*****cluster_energy:", cluster_energy)
                initial_solutions.append(("Greed_Cluster", cluster_energy, cluster_route, cluster_battery_route, cluster_battery_used_capacity, cluster_variable_names, cluster_variable_values))
                # break
            else:
                print("Invalid Greed_Cluster solution, retrying...")
        return initial_solutions

    @ray.remote
    def ortools_initial_solution(self):
        initial_solutions = []
        ortools_start_time = time.time()
        while time.time() - ortools_start_time < 15:
            ortools_route, ortools_battery_route, ortools_battery_used_capacity = self.ORTools_cvrptw()
            current_route = ortools_route
            current_battery_route = ortools_battery_route
            current_battery_used_capacity = ortools_battery_used_capacity
            current_energy, current_variable_names, current_variable_values = self.calculate_objective(current_route, current_battery_route)
            print("*****current_energy:", current_energy)
            
            while current_energy is None and time.time() - ortools_start_time < 15:
                new_route, new_battery_route, new_battery_used_capacity = self.node_removal(current_route, current_battery_route, current_battery_used_capacity)
                new_energy, new_variable_names, new_variable_values = self.calculate_objective(new_route, new_battery_route)
                
                if new_energy is not None:  
                    current_route = copy.deepcopy(new_route)
                    current_battery_route = copy.deepcopy(new_battery_route)
                    current_battery_used_capacity = copy.deepcopy(new_battery_used_capacity)
                    current_energy = new_energy
                    current_variable_names = copy.deepcopy(new_variable_names)
                    current_variable_values = copy.deepcopy(new_variable_values)
                else:
                    print("Invalid ortools solution, retrying after node removal...")
                    current_route = copy.deepcopy(new_route)
                    current_battery_route = copy.deepcopy(new_battery_route)
                    current_battery_used_capacity = copy.deepcopy(new_battery_used_capacity)

            if current_energy is not None:
                initial_solutions.append(("ORTools", current_energy, current_route, current_battery_route, current_battery_used_capacity, current_variable_names, current_variable_values))
                # break  # 유효한 솔루션을 찾으면 루프 종료
        return initial_solutions

    @ray.remote
    def elecvery_initial_solution(self):
        initial_solutions = []
        route, battery_route, battery_used_capacity = self.elecvery()
        energy, variable_names, variable_values = self.calculate_objective_CT1(route, battery_route)
        if energy is not None:
            print("***** elecvery_energy:", energy)
            initial_solutions.append(("elecvery", energy, route, battery_route, battery_used_capacity, variable_names, variable_values))
        return initial_solutions

    def simulated_annealing(self, initial_temperature=100, cooling_rate=0.95, max_iterations=1000, time_limit=40):
        start_time = time.time()
        initial_solutions = []

        # 병렬로 원격 함수 시작
        greed_cluster_id = self.greed_cluster_initial_solution.remote(self)
        ortools_id = self.ortools_initial_solution.remote(self)
        elecvery_id = self.elecvery_initial_solution.remote(self)

        # 결과를 기다림
        greed_cluster_result = ray.get(greed_cluster_id)
        ortools_result = ray.get(ortools_id)
        elecvery_result = ray.get(elecvery_id)

        # 결과를 병합
        initial_solutions = greed_cluster_result + ortools_result + elecvery_result

        if not initial_solutions:
            raise ValueError("No valid initial solution found")
        
        # 모든 initial_solutions 값을 출력
        for solution in initial_solutions:
            source, energy, route, battery_route, _, _, _ = solution
            print(f"Source: {source}")
            print(f"Energy: {energy}")
            print(f"Route: {route}")
            print(f"Battery Route: {battery_route}")
            print("=====================================")
        
        # 가장 좋은 초기해 선택
        best_initial_solution = max(initial_solutions, key=lambda x: x[1])
        _, best_initial_energy, best_initial_route, best_initial_battery_route, best_initial_battery_used_capacity, best_initial_variable_names, best_initial_variable_values = best_initial_solution

        # 가장 좋은 초기해로 초기화
        current_route = best_initial_route
        current_battery_route = best_initial_battery_route
        current_battery_used_capacity = best_initial_battery_used_capacity
        current_energy = best_initial_energy
        current_variable_names = best_initial_variable_names
        current_variable_values = best_initial_variable_values
        
        best_route = copy.deepcopy(current_route)
        best_battery_route = copy.deepcopy(current_battery_route)
        best_battery_used_capacity = copy.deepcopy(current_battery_used_capacity)
        best_energy = current_energy
        best_variable_names = copy.deepcopy(current_variable_names)
        best_variable_values = copy.deepcopy(current_variable_values)

        print(f"***** initial route: {current_route}")
        print(f"***** initial battery route: {current_battery_route}")
        print(f"***** initial battery used capacity: {current_battery_used_capacity}")
        print(f"***** initial energy: {current_energy}")
            
        temperature = initial_temperature
        iterations = 0

        # 연산자 사용 횟수와 개선 횟수를 추적하기 위한 변수
        operator_usage_count = {
            'node_exchange': 0,
            'node_removal': 0,
            'node_insertion': 0
        }
        operator_improvement_count = {
            'node_exchange': 0,
            'node_removal': 0,
            'node_insertion': 0
        }

        probabilities = [0.8, 0.1, 0.1]  

        while temperature > 0.1 and iterations < max_iterations:
            print("***** starting SA")
            if time.time() - start_time >= time_limit:
                break
            
            # probabilities = [0.8, 0.1, 0.1]  
            selected_operator = random.choices([self.node_exchange, self.node_removal, self.node_insertion], weights=probabilities)[0]
            
            operator_name = selected_operator.__name__
            operator_usage_count[operator_name] += 1
            
            if len(current_route) > 1:
                new_route, new_battery_route, new_battery_used_capacity = selected_operator(current_route, current_battery_route, current_battery_used_capacity)
                print(f"***** Operator: {selected_operator.__name__}")
                # print(f"문제 input 방문 노드: {new_route}")
                # print(f"문제 input 배터리별 방문 노드: {new_battery_route}")

                new_energy, new_variable_names, new_variable_values = self.calculate_objective(new_route, new_battery_route)
                print(f"새 목적함수 값: {new_energy}")   

                if new_energy is not None:
                    if new_energy > best_energy:  # new_energy가 더 크거나 같으면 갱신
                        best_route = copy.deepcopy(new_route)
                        best_battery_route = copy.deepcopy(new_battery_route) 
                        best_battery_used_capacity = copy.deepcopy(new_battery_used_capacity)
                        best_energy = new_energy
                        best_variable_names = copy.deepcopy(new_variable_names)
                        best_variable_values = copy.deepcopy(new_variable_values)

                        current_route = copy.deepcopy(new_route)
                        current_battery_route = copy.deepcopy(new_battery_route)
                        current_battery_used_capacity = copy.deepcopy(new_battery_used_capacity)
                        
                        operator_improvement_count[operator_name] += 1

                        # 확률 업데이트
                        if operator_name == 'node_exchange':
                            probabilities[0] += 0.005
                            probabilities[1] -= 0.0025
                            probabilities[2] -= 0.0025
                        elif operator_name == 'node_removal':
                            probabilities[0] -= 0.0025
                            probabilities[1] += 0.005
                            probabilities[2] -= 0.0025
                        elif operator_name == 'node_insertion':
                            probabilities[0] -= 0.0025
                            probabilities[1] -= 0.0025
                            probabilities[2] += 0.005

                        # 확률 정규화
                        probabilities = [max(0, p) for p in probabilities]
                        total = sum(probabilities)
                        probabilities = [p / total for p in probabilities]

                    else:
                        # new_energy가 더 작은 경우, 수락 확률을 계산
                        accept_probability = math.exp((new_energy - best_energy) / temperature)
                        if random.random() < accept_probability:
                            current_route = copy.deepcopy(new_route)
                            current_battery_route = copy.deepcopy(new_battery_route)
                            current_battery_used_capacity = copy.deepcopy(new_battery_used_capacity)
                            current_variable_names = copy.deepcopy(new_variable_names)
                            current_variable_values = copy.deepcopy(new_variable_values)
                        else:
                            print("***** no improvement")
                            current_route = copy.deepcopy(best_route)
                            current_battery_route = copy.deepcopy(best_battery_route)
                            current_battery_used_capacity = copy.deepcopy(best_battery_used_capacity)
                            current_variable_names = copy.deepcopy(best_variable_names)
                            current_variable_values = copy.deepcopy(best_variable_values)

                temperature *= cooling_rate
                iterations += 1
                print(f"***** Iteration {iterations}, Temperature: {temperature}, Current Energy: {current_energy}, Best Energy: {best_energy}")

        print("Best solution:", best_route)
        print("Best battery route:", best_battery_route)
        print("Best battery used capacity:", best_battery_used_capacity)
        print("Best objective value:", best_energy)
        
        # 연산자 사용 횟수 및 개선 횟수 출력
        print("Operator usage count:", operator_usage_count)
        print("Operator improvement count:", operator_improvement_count)
        
        # 최종 probabilities 값 출력
        print("Final probabilities:", probabilities)

        self.save_solution(best_energy, best_variable_names, best_variable_values)

        return best_route, best_battery_route, best_battery_used_capacity, best_energy, best_variable_names, best_variable_values
    

    def expand_node_pairs(self, paths):
        self.expanded_pairs = set()

        for path in paths:
            path_list = list(path) 

            ## (0, 1), (0, h1)
            self.expanded_pairs.add(('0', str(path_list[0])))
            self.expanded_pairs.add(('0', 'h' + str(path_list[0])))

            # (1, n+1), (h1, n+1)
            # self.expanded_pairs.add((str(path_list[0]), str(len(self.N) + 1)))
            # self.expanded_pairs.add(('h' + str(path_list[0]), str(len(self.N) + 1)))

            for i in range(len(path_list) - 1):
                self.expanded_pairs.add((str(path_list[i]), str(path_list[i + 1])))
                self.expanded_pairs.add((str(path_list[i]), 'h' + str(path_list[i])))
                self.expanded_pairs.add(('h' + str(path_list[i]), str(path_list[i + 1])))
                self.expanded_pairs.add(('h' + str(path_list[i]), 'h' + str(path_list[i + 1])))

            # (0, 3), (0, h3)
            # self.expanded_pairs.add(('0', str(path_list[-1])))
            # self.expanded_pairs.add(('0', 'h' + str(path_list[-1])))

            ## (3, n+1), (h3, n+1)
            self.expanded_pairs.add((str(path_list[-1]), str(len(self.N) + 1)))
            self.expanded_pairs.add(('h' + str(path_list[-1]), str(len(self.N) + 1)))

        # Remove duplicates and (1, 1), (2, 2), ...
        self.expanded_pairs = {(a, b) for a, b in self.expanded_pairs if a != b}

        return self.expanded_pairs
    
    
    def process_route(self, route, battery_route):
        self.a = set()           # Battery Visited Nodes (i, hi, l)
        self.a_c = set()         # Battery Non-Visited Nodes (i, hi, l)
        self.a_c_nodes = set()   # Battery Non-Visited Nodes

        self.b = set()           # Battery Visited Nodes (i, j) 조합, k-sp에서 가져올 조합
        
        self.k_shortest_paths = set()
        self.expanded_pairs = set()

        self.c = set()

        # a 
        for vehicle_id, visited_nodes in battery_route.items():
            battery_id = str(vehicle_id)
            # print(f'Visited nodes by Battery {battery_id}: {visited_nodes}')

            if not visited_nodes:
                continue

            for i in range(0, len(visited_nodes)):
                self.a.add((str(visited_nodes[i]), 'h' + str(visited_nodes[i]), battery_id))
                # print(f'Added to self.a: {(str(visited_nodes[i]), "h" + str(visited_nodes[i]), battery_id)}')
        
        # print("$$$$$ a, Battery Visited Nodes (i,i\hat,l): ", self.a)

        # a_c
        route_str = set(map(str, route))
        self.a_c_nodes = set(self.N) - route_str
        # print("$$$$$ a_c_nodes: ", self.a_c_nodes)

        for node in self.a_c_nodes:
            for l in self.B:
                self.a_c.add((node, 'h' + node, l))
                # print(f'Added to self.a_c: {(node, "h" + node, l)}')

        # intersection = self.a.intersection(self.a_c)
        # if intersection:
        #     print("겹치는 요소가 있습니다: ", intersection)
        # else:
        #     print("겹치는 요소가 없습니다.")

        # b
        for i, j in permutations(route, 2):
            if self.G.has_edge(str(i), str(j)):  # 길이 있다면 조합 생성
                self.b.add((str(i), str(j)))

        # b에 해당하는 k-sp 가져오기
        for pair in self.b:
            i, j = pair
            k_sp = self.all_k_shortest_paths_result.get((i, j), [])
            for path in k_sp:
                self.k_shortest_paths.add(tuple(path))

        # 불러온 k-sp 확장
        self.expanded_pairs = self.expand_node_pairs(self.k_shortest_paths)

        # 디포에서 나오고 들어가는 경로 추가
        for node in route:
            self.expanded_pairs.add(('0', str(node)))
            self.expanded_pairs.add(('0', 'h' + str(node)))
            self.expanded_pairs.add((str(node), str(len(self.N) + 1)))
            self.expanded_pairs.add(('h' + str(node), str(len(self.N) + 1)))

        # 방문하지 않는 노드(a_c_nodes)로 들어가거나 나가는 arc들을 제거
        self.expanded_pairs = {
            (a, b) for a, b in self.expanded_pairs
            if a not in self.a_c_nodes and b not in self.a_c_nodes and
            (a[0] != 'h' or a[1:] not in self.a_c_nodes) and
            (b[0] != 'h' or b[1:] not in self.a_c_nodes)
        }

        expanded_pairs_set = set(self.expanded_pairs)

        pairs_to_expand = [(node1, node2) for node1 in self.a_c_nodes for node2 in self.N if node1 != node2]
        pairs_to_expand += [(node2, node1) for node1 in self.a_c_nodes for node2 in self.N if node1 != node2]

        expanded_pairs_from_ac_nodes = self.expand_node_pairs(pairs_to_expand)

        for pair in expanded_pairs_from_ac_nodes:
            if pair not in expanded_pairs_set:
                self.c.add(pair)

        # Find and exclude intersections
        intersection_to_exclude = expanded_pairs_set.intersection(self.c)

        self.expanded_pairs = expanded_pairs_set - intersection_to_exclude


    def MIP_solver(self):
        self.cpx = cplex.Cplex()
            
        self.cpx.objective.set_sense(self.cpx.objective.sense.maximize)

        x_i_j = lambda i, j: 'x_%s_%s' % (i, j)
        y_i_j_l = lambda i, j, l: 'y_%s_%s_%s' % (i, j, l)
        z_i_l = lambda i, l: 'z_%s_%s' % (i, l)
        o_i = lambda i: 'o_%s' % (i)
        u_i = lambda i: 'u_%s' % (i)
        v_i = lambda i: 'v_%s' % (i)

        self.cpx.variables.add(
            obj=[-self.D_ij[(i,j)] for (i, j) in self.A],
            ub=[1 for (i, j) in self.A],
            lb=[0 for (i, j) in self.A],
            types=['B' for (i, j) in self.A],
            names=[x_i_j(str(i), str(j)) for (i, j) in self.A]
        )

        self.cpx.variables.add(
            obj=[0 for (i, j) in self.A for l in self.B],
            ub=[1 for (i, j) in self.A for l in self.B],
            lb=[0 for (i, j) in self.A for l in self.B],
            types=['B' for (i, j) in self.A for l in self.B],
            names=[y_i_j_l(str(i), str(j), str(l)) for (i, j) in self.A for l in self.B]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.N for l in self.B],
            ub=[1 for i in self.N for l in self.B],
            lb=[0 for i in self.N for l in self.B],
            types=['B' for i in self.N for l in self.B],
            names=[z_i_l(str(i), str(l)) for i in self.N for l in self.B]
        )

        self.cpx.variables.add(
            obj=[self.lamb for i in self.N],
            ub=[1 for i in self.N],
            lb=[0 for i in self.N],
            types=['B' for i in self.N],
            names=[o_i(str(i)) for i in self.N]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.V_1],
            lb=[0 for i in self.V_1],
            types=['C' for i in self.V_1],
            names=[u_i(str(i)) for i in self.V_1]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.V_0],
            lb=[0 for i in self.V_0],
            types=['C' for i in self.V_0],
            names=[v_i(str(i)) for i in self.V_0]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.N],
            ub=[self.L_i[self.N.index(i)] for i in self.N],
            lb=[self.E_i[self.N.index(i)] for i in self.N],
            types=['C' for i in self.N],
            names=[u_i(str(i)) for i in self.N]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.N],
            ub=[self.L_i[self.N.index(i)] for i in self.N],
            lb=[self.E_i[self.N.index(i)] for i in self.N],
            types=['C' for i in self.N],
            names=[v_i(str(i)) for i in self.N]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.N_hat],
            ub=[self.L_i[self.N_hat.index(i)] for i in self.N_hat],
            lb=[self.E_i[self.N_hat.index(i)] for i in self.N_hat],
            types=['C' for i in self.N_hat],
            names=[u_i(str(i)) for i in self.N_hat]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.N_hat],
            ub=[self.L_i[self.N_hat.index(i)] for i in self.N_hat],
            lb=[self.E_i[self.N_hat.index(i)] for i in self.N_hat],
            types=['C' for i in self.N_hat],
            names=[v_i(str(i)) for i in self.N_hat]
        ) 

        for (i, j, l) in self.a:
            self.cpx.variables.set_lower_bounds(y_i_j_l(i, j, l), 1)

        for (i, j, l) in self.a_c:
            self.cpx.variables.set_upper_bounds(y_i_j_l(i, j, l), 0)

        # for (i, j) in A:
        #     for l in self.B:
        #         if (i, j, l) not in self.a:
        #             self.cpx.variables.set_upper_bounds(y_i_j_l(i, j, l), 0)
        
        
        for (i, j) in self.A:
            if i != str(0) and j != str(len(self.N)):
                if (i, j) not in self.expanded_pairs:
                    # print(f"Processing pair: ({i}, {j})")
                    self.cpx.variables.set_upper_bounds(x_i_j(i, j), 0)

            
        #2 모든 트럭은 안나가도 괜찮다
        T_len = len(self.T)
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(0, i) for i in self.V],
                    val=[1 for i in self.V]
                )
            ],
            senses=['L'],
            rhs=[T_len],
            names=['const2']
        )

        #3 
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(a, b) for (a,b) in self.A if b == j ],
                    val=[1 for (a,b) in self.A if b == j  ]
                ) for j in self.V
            ],
            senses=['L' for j in self.V],
            rhs=[1 for j in self.V],
            names=['const3_%s' % j for j in self.V]
        )

        #4 x flow 제약식
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(a, b) for (a, b) in self.A if a == i] + [x_i_j(c, d) for (c,d) in self.A if d == i],
                    val=[1 for (a, b) in self.A if a == i] + [-1 for (c, d) in self.A if d == i]
                ) for i in self.V
            ],
            
            senses=['E' for i in self.V] ,
            rhs=[0 for i in self.V],
            names=['const4_%s' % (i) for i in self.V]
        )

        #5
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(a, b) for (a, b) in self.A if b == i ] + [x_i_j(c, d) for (c, d) in self.A if d == j],
                    val=[1 for (a, b) in self.A if b == i] + [-1 for (c, d) in self.A if d == j]
                ) for (i,j) in self.A_2
            ], 
            senses=['E' for (i,j) in self.A_2],
            rhs=[0 for (i,j) in self.A_2],
            names=['const5_%s_%s' % (i, j) for (i,j) in self.A_2]
        ) 

        #6
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[y_i_j_l(0, i, l) for i in self.V],
                    val=[1 for i in self.V]
                ) for l in self.B
            ],
            senses=['L' for l in self.B],
            rhs=[1 for l in self.B],
            names=['const6_%s' % (l) for l in self.B]
        )

        #7
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[y_i_j_l(a, b, l) for (a, b) in self.A if a == i] + [y_i_j_l(c, d, l) for (c, d) in self.A if d == i],
                    val=[1] * len([y_i_j_l(a, b, l) for (a, b) in self.A if a == i]) + [-1] * len([y_i_j_l(c, d, l) for (c, d) in self.A if d == i])
                ) for i in self.V for l in self.B
            ],
            senses=['E' for i in self.V for l in self.B] ,
            rhs=[0 for i in self.V for l in self.B] ,
            names=['const7_%s_%s' % (i, l) for i in self.V for l in self.B]
        )

        #8
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(i, j)] + [y_i_j_l(i, j, l) for l in self.B],
                    val=[self.CT] + [-1 for l in self.B]
                ) for (i,j) in self.A_p
            ],
            senses=['G' for (i,j) in self.A_p],
            rhs=[0 for (i,j) in self.A_p],
            names=['const8_%s_%s' % (i,j) for (i,j) in self.A_p]
        )

        #9
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[z_i_l(i, l) for i in self.N],
                    val=[self.R_i[self.N.index(i)] for i in self.N]
                ) for l in self.B
            ],
            senses=['L' for l in self.B],
            rhs=[self.CB_l[self.B.index(l)] for l in self.B],
            names=['const9_%s' % (l) for l in self.B]
        )
        #10
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[z_i_l(i,l) for l in self.B],
                    val=[1 for l in self.B]
                ) for i in self.N
            ],
            senses=['L' for i in self.N],
            rhs=[1 for i in self.N],
            names=['const10_%s' % (i) for i in self.N]
        )
        #11
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[o_i(i)] + [z_i_l(i, l) for l in self.B],
                    val=[1] + [-1 for l in self.B]
                ) for i in self.N
            ],
            senses=['E' for i in self.N],
            rhs=[0 for i in self.N],
            names=['const11_%s' % (i) for i in self.N]
        )
        #12
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[y_i_j_l(i, j, l)] + [z_i_l(i, l)],
                    val=[1] + [-1]
                ) for (i, j) in self.A_2 for l in self.B
            ],
            senses=['E' for (i, j) in self.A_2 for l in self.B],
            rhs=[0 for (i, j) in self.A_2 for l in self.B],
            names=['const12_%s_%s_%s' % (i, j, l) for (i, j) in self.A_2 for l in self.B]
        )
        #13
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(i)] + [u_i(i)],
                    val=[1] + [-1]
                ) for i in self.V
            ],
            senses=['G' for i in self.V],
            rhs=[self.S_i[i] for i in range(0,len(self.V))],
            names=['const13_%s' % (i) for i in self.V]
        )
        #14
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(i)] + [u_i(j)] + [x_i_j(i,j)],
                    val=[1] + [-1] + [self.M]
                ) for (i,j) in self.A
            ],
            senses=['L' for (i,j) in self.A],
            rhs=[ - self.T_ij[(i,j)] + self.M for (i,j) in self.A],
            names=['const14_%s_%s' % (i,j) for (i,j) in self.A]
        )

        #15
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[u_i(j)] + [v_i(i)],
                    val=[1] + [-1]
                ) for (i, j) in self.A_2
            ],
            senses=['G' for (i, j) in self.A_2],
            rhs=[self.C_i[self.N.index(i)] for (i, j) in self.A_2],
            names=['const15_%s_%s' % (i,j) for (i, j) in self.A_2]
        )
        #16 떠나는 v_i_hat lower bound 
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(j)],
                    val=[1]
                ) for (i, j) in self.A_2
            ],
            senses=['G' for (i, j) in self.A_2],
            rhs=[self.E_i[self.N.index(i)] + self.S_i[self.N.index(i)] + self.C_i[self.N.index(i)] + self.S_i[self.N_hat.index(j)] for (i, j) in self.A_2],
            names=['const16_%s_%s' % (i,j) for (i, j) in self.A_2]
        )
        #17 들어오는 u_i upper bound 최소 이전에는 들어와야함
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[u_i(i)],
                    val=[1]
                ) for (i, j) in self.A_2
            ],
            senses=['L' for (i, j) in self.A_2],
            rhs=[ self.L_i[self.N.index(i)] - self.S_i[self.N.index(i)] - self.C_i[self.N.index(i)] - self.S_i[self.N_hat.index(j)] for (i, j) in self.A_2],
            names=['const17_%s_%s' % (i,j) for (i, j) in self.A_2]
        )
        #18 떠나는 v_i lower bound 
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(i)],
                    val=[1]
                ) for i in self.N
            ],
            senses=['G' for i in self.N],
            rhs=[self.E_i[self.N.index(i)] + self.S_i[self.N.index(i)] for i in self.N],
            names=['const18_%s' % (i) for i in self.N]
        )
        #19
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(str(0))],
                    val=[1]
                )
            ],
            senses=['G'],
            rhs=[self.E_0],
            names=['const19']
        )
        #20
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[u_i(str(len(self.N)+1))],
                    val=[1]
                )
            ],
            senses=['L'],
            rhs=[self.L_n],
            names=['const20']
        )

        self.cpx.solve()
       
        self.x_i_j=x_i_j
        self.y_i_j_l=y_i_j_l
        self.z_i_l=z_i_l
        self.o_i=o_i
        self.u_i=u_i
        self.v_i=v_i

        variable_names = self.cpx.variables.get_names()
        variable_values = self.cpx.solution.get_values()
        self.variable_names = variable_names
        self.variable_values = variable_values
    






    def MIP_solver_CT1(self):
        self.cpx = cplex.Cplex()
            
        self.cpx.objective.set_sense(self.cpx.objective.sense.maximize)

        x_i_j = lambda i, j: 'x_%s_%s' % (i, j)
        y_i_j_l = lambda i, j, l: 'y_%s_%s_%s' % (i, j, l)
        z_i_l = lambda i, l: 'z_%s_%s' % (i, l)
        o_i = lambda i: 'o_%s' % (i)
        u_i = lambda i: 'u_%s' % (i)
        v_i = lambda i: 'v_%s' % (i)

        self.cpx.variables.add(
            obj=[-self.D_ij[(i,j)] for (i, j) in self.A],
            ub=[1 for (i, j) in self.A],
            lb=[0 for (i, j) in self.A],
            types=['B' for (i, j) in self.A],
            names=[x_i_j(str(i), str(j)) for (i, j) in self.A]
        )

        self.cpx.variables.add(
            obj=[0 for (i, j) in self.A for l in self.B],
            ub=[1 for (i, j) in self.A for l in self.B],
            lb=[0 for (i, j) in self.A for l in self.B],
            types=['B' for (i, j) in self.A for l in self.B],
            names=[y_i_j_l(str(i), str(j), str(l)) for (i, j) in self.A for l in self.B]
        )
        
        self.cpx.variables.add(
            obj=[0 for i in self.N for l in self.B],
            ub=[1 for i in self.N for l in self.B],
            lb=[0 for i in self.N for l in self.B],
            types=['B' for i in self.N for l in self.B],
            names=[z_i_l(str(i), str(l)) for i in self.N for l in self.B]
        )

        self.cpx.variables.add(
            obj=[self.lamb for i in self.N],
            ub=[1 for i in self.N],
            lb=[0 for i in self.N],
            types=['B' for i in self.N],
            names=[o_i(str(i)) for i in self.N]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.V_1],
            lb=[0 for i in self.V_1],
            types=['C' for i in self.V_1],
            names=[u_i(str(i)) for i in self.V_1]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.V_0],
            lb=[0 for i in self.V_0],
            types=['C' for i in self.V_0],
            names=[v_i(str(i)) for i in self.V_0]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.N],
            ub=[self.L_i[self.N.index(i)] for i in self.N],
            lb=[self.E_i[self.N.index(i)] for i in self.N],
            types=['C' for i in self.N],
            names=[u_i(str(i)) for i in self.N]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.N],
            ub=[self.L_i[self.N.index(i)] for i in self.N],
            lb=[self.E_i[self.N.index(i)] for i in self.N],
            types=['C' for i in self.N],
            names=[v_i(str(i)) for i in self.N]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.N_hat],
            ub=[self.L_i[self.N_hat.index(i)] for i in self.N_hat],
            lb=[self.E_i[self.N_hat.index(i)] for i in self.N_hat],
            types=['C' for i in self.N_hat],
            names=[u_i(str(i)) for i in self.N_hat]
        )

        self.cpx.variables.add(
            obj=[0 for i in self.N_hat],
            ub=[self.L_i[self.N_hat.index(i)] for i in self.N_hat],
            lb=[self.E_i[self.N_hat.index(i)] for i in self.N_hat],
            types=['C' for i in self.N_hat],
            names=[v_i(str(i)) for i in self.N_hat]
        ) 

        for (i, j, l) in self.a:
            self.cpx.variables.set_lower_bounds(y_i_j_l(i, j, l), 1)

        for (i, j, l) in self.a_c:
            self.cpx.variables.set_upper_bounds(y_i_j_l(i, j, l), 0)

        # for (i, j) in A:
        #     for l in self.B:
        #         if (i, j, l) not in self.a:
        #             self.cpx.variables.set_upper_bounds(y_i_j_l(i, j, l), 0)
        
        
        # for (i, j) in self.A:
        #     if i != str(0) and j != str(len(self.N)):
        #         if (i, j) not in self.expanded_pairs:
        #             # print(f"Processing pair: ({i}, {j})")
        #             self.cpx.variables.set_upper_bounds(x_i_j(i, j), 0)

            
        #2 모든 트럭은 안나가도 괜찮다
        T_len = len(self.T)
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(0, i) for i in self.V],
                    val=[1 for i in self.V]
                )
            ],
            senses=['L'],
            rhs=[T_len],
            names=['const2']
        )

        #3 
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(a, b) for (a,b) in self.A if b == j ],
                    val=[1 for (a,b) in self.A if b == j  ]
                ) for j in self.V
            ],
            senses=['L' for j in self.V],
            rhs=[1 for j in self.V],
            names=['const3_%s' % j for j in self.V]
        )

        #4 x flow 제약식
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(a, b) for (a, b) in self.A if a == i] + [x_i_j(c, d) for (c,d) in self.A if d == i],
                    val=[1 for (a, b) in self.A if a == i] + [-1 for (c, d) in self.A if d == i]
                ) for i in self.V
            ],
            
            senses=['E' for i in self.V] ,
            rhs=[0 for i in self.V],
            names=['const4_%s' % (i) for i in self.V]
        )

        #5
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(a, b) for (a, b) in self.A if b == i ] + [x_i_j(c, d) for (c, d) in self.A if d == j],
                    val=[1 for (a, b) in self.A if b == i] + [-1 for (c, d) in self.A if d == j]
                ) for (i,j) in self.A_2
            ], 
            senses=['E' for (i,j) in self.A_2],
            rhs=[0 for (i,j) in self.A_2],
            names=['const5_%s_%s' % (i, j) for (i,j) in self.A_2]
        ) 

        #6
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[y_i_j_l(0, i, l) for i in self.V],
                    val=[1 for i in self.V]
                ) for l in self.B
            ],
            senses=['L' for l in self.B],
            rhs=[1 for l in self.B],
            names=['const6_%s' % (l) for l in self.B]
        )

        #7
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[y_i_j_l(a, b, l) for (a, b) in self.A if a == i] + [y_i_j_l(c, d, l) for (c, d) in self.A if d == i],
                    val=[1] * len([y_i_j_l(a, b, l) for (a, b) in self.A if a == i]) + [-1] * len([y_i_j_l(c, d, l) for (c, d) in self.A if d == i])
                ) for i in self.V for l in self.B
            ],
            senses=['E' for i in self.V for l in self.B] ,
            rhs=[0 for i in self.V for l in self.B] ,
            names=['const7_%s_%s' % (i, l) for i in self.V for l in self.B]
        )

        #8
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(i, j)] + [y_i_j_l(i, j, l) for l in self.B],
                    val=[1] + [-1 for l in self.B]
                ) for (i,j) in self.A
            ],
            senses=['G' for (i,j) in self.A],
            rhs=[0 for (i,j) in self.A],
            names=['const8_%s_%s' % (i,j) for (i,j) in self.A]
        )

        #9
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[z_i_l(i, l) for i in self.N],
                    val=[self.R_i[self.N.index(i)] for i in self.N]
                ) for l in self.B
            ],
            senses=['L' for l in self.B],
            rhs=[self.CB_l[self.B.index(l)] for l in self.B],
            names=['const9_%s' % (l) for l in self.B]
        )
        #10
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[z_i_l(i,l) for l in self.B],
                    val=[1 for l in self.B]
                ) for i in self.N
            ],
            senses=['L' for i in self.N],
            rhs=[1 for i in self.N],
            names=['const10_%s' % (i) for i in self.N]
        )
        #11
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[o_i(i)] + [z_i_l(i, l) for l in self.B],
                    val=[1] + [-1 for l in self.B]
                ) for i in self.N
            ],
            senses=['E' for i in self.N],
            rhs=[0 for i in self.N],
            names=['const11_%s' % (i) for i in self.N]
        )
        #12
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[y_i_j_l(i, j, l)] + [z_i_l(i, l)],
                    val=[1] + [-1]
                ) for (i, j) in self.A_2 for l in self.B
            ],
            senses=['E' for (i, j) in self.A_2 for l in self.B],
            rhs=[0 for (i, j) in self.A_2 for l in self.B],
            names=['const12_%s_%s_%s' % (i, j, l) for (i, j) in self.A_2 for l in self.B]
        )
        #13
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(i)] + [u_i(i)],
                    val=[1] + [-1]
                ) for i in self.V
            ],
            senses=['G' for i in self.V],
            rhs=[self.S_i[i] for i in range(0,len(self.V))],
            names=['const13_%s' % (i) for i in self.V]
        )
        #14
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(i)] + [u_i(j)] + [x_i_j(i,j)],
                    val=[1] + [-1] + [self.M]
                ) for (i,j) in self.A
            ],
            senses=['L' for (i,j) in self.A],
            rhs=[ - self.T_ij[(i,j)] + self.M for (i,j) in self.A],
            names=['const14_%s_%s' % (i,j) for (i,j) in self.A]
        )

        #15
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[u_i(j)] + [v_i(i)],
                    val=[1] + [-1]
                ) for (i, j) in self.A_2
            ],
            senses=['G' for (i, j) in self.A_2],
            rhs=[self.C_i[self.N.index(i)] for (i, j) in self.A_2],
            names=['const15_%s_%s' % (i,j) for (i, j) in self.A_2]
        )
        #16 떠나는 v_i_hat lower bound 
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(j)],
                    val=[1]
                ) for (i, j) in self.A_2
            ],
            senses=['G' for (i, j) in self.A_2],
            rhs=[self.E_i[self.N.index(i)] + self.S_i[self.N.index(i)] + self.C_i[self.N.index(i)] + self.S_i[self.N_hat.index(j)] for (i, j) in self.A_2],
            names=['const16_%s_%s' % (i,j) for (i, j) in self.A_2]
        )
        #17 들어오는 u_i upper bound 최소 이전에는 들어와야함
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[u_i(i)],
                    val=[1]
                ) for (i, j) in self.A_2
            ],
            senses=['L' for (i, j) in self.A_2],
            rhs=[ self.L_i[self.N.index(i)] - self.S_i[self.N.index(i)] - self.C_i[self.N.index(i)] - self.S_i[self.N_hat.index(j)] for (i, j) in self.A_2],
            names=['const17_%s_%s' % (i,j) for (i, j) in self.A_2]
        )
        #18 떠나는 v_i lower bound 
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(i)],
                    val=[1]
                ) for i in self.N
            ],
            senses=['G' for i in self.N],
            rhs=[self.E_i[self.N.index(i)] + self.S_i[self.N.index(i)] for i in self.N],
            names=['const18_%s' % (i) for i in self.N]
        )
        #19
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(str(0))],
                    val=[1]
                )
            ],
            senses=['G'],
            rhs=[self.E_0],
            names=['const19']
        )
        #20
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[u_i(str(len(self.N)+1))],
                    val=[1]
                )
            ],
            senses=['L'],
            rhs=[self.L_n],
            names=['const20']
        )

        self.cpx.solve()
        
        
        self.x_i_j=x_i_j
        self.y_i_j_l=y_i_j_l
        self.z_i_l=z_i_l
        self.o_i=o_i
        self.u_i=u_i
        self.v_i=v_i

        variable_names = self.cpx.variables.get_names()
        variable_values = self.cpx.solution.get_values()
        self.variable_names = variable_names
        self.variable_values = variable_values



    def save_solution(self, objective_value, variable_names, variable_values):
        try:
            file_name = os.path.basename(self.datafile)
            base_name = os.path.splitext(file_name)[0]

            x_values, y_values, z_values, o_values, u_values, v_values = {}, {}, {}, {}, {}, {}

            for var, val in zip(variable_names, variable_values):
                if val > 0.0001:
                    if var.startswith('x'):
                        x_values[var] = val
                    elif var.startswith('y'):
                        y_values[var] = val
                    elif var.startswith('z'):
                        z_values[var] = val
                    elif var.startswith('o'):
                        o_values[var] = val
                    elif var.startswith('u'):
                        u_values[var] = val
                    elif var.startswith('v'):
                        v_values[var] = val

            def group_by_i(i_values):
                grouped_i_values = {}
                for key, value in i_values.items():
                    i_value = key.split('_')[1]
                    if i_value not in grouped_i_values:
                        grouped_i_values[i_value] = {}
                    grouped_i_values[i_value][key] = value
                return grouped_i_values

            def group_by_j(j_values):
                grouped_j_values = {}
                for key, value in j_values.items():
                    j_value = key.split('_')[2]
                    if j_value not in grouped_j_values:
                        grouped_j_values[j_value] = {}
                    grouped_j_values[j_value][key] = value
                return grouped_j_values

            def group_by_l(l_values):
                grouped_l_values = {}
                for key, value in l_values.items():
                    l_value = key.split('_')[-1]
                    if l_value not in grouped_l_values:
                        grouped_l_values[l_value] = {}
                    grouped_l_values[l_value][key] = value
                return grouped_l_values

            Truck_routes = {}
            start_nodes = [node for node in x_values if node.startswith('x_0')]
            truck_num = 1

            while start_nodes:
                start_node = start_nodes.pop(0)
                current_node = start_node.split('_')[2]
                route = ['0', current_node]
                visited = set(route)
                while True:
                    next_nodes = [node for node in x_values if node.startswith(f'x_{current_node}_') and node not in visited]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0].split('_')[2]
                    if next_node in visited or next_node == route[0]:
                        break
                    route.append(next_node)
                    visited.add(next_node)
                    current_node = next_node
                Truck_routes[f'Truck{truck_num}'] = route
                truck_num += 1

            grouped_y_values = group_by_l(y_values)

            Battery_routes = {}

            for i in grouped_y_values.keys():
                start_nodes = [node for node in grouped_y_values[i] if node.startswith('y_0')]

                while start_nodes:
                    start_node = start_nodes.pop(0)
                    current_node = start_node.split('_')[2]
                    route = ['0', current_node]

                    while True:
                        next_nodes = [node for node in grouped_y_values[i] if node.startswith(f'y_{current_node}') and node != start_node]
                        if not next_nodes:
                            break
                        next_node = next_nodes[0].split('_')[2]
                        if next_node == route[0]:
                            break
                        route.append(next_node)
                        current_node = next_node

                    Battery_routes[f'Battery{i}'] = route

            node_time = {}
            grouped_u_values = group_by_i(u_values)
            grouped_v_values = group_by_i(v_values)

            for key in grouped_u_values.keys():
                if key in grouped_v_values:
                    u_value = grouped_u_values[key]['u_' + key]
                    v_value = grouped_v_values[key]['v_' + key]
                    node_time[key] = [u_value, v_value]

            node_battery = {}

            for key, value in group_by_i(z_values).items():
                j_value = int(list(value.keys())[0].split('_')[-1])
                node_battery[key] = j_value

            battery_node = {}

            for key, value in node_battery.items():
                if value not in battery_node:
                    battery_node[value] = [key]
                else:
                    battery_node[value].append(key)

            node_serviced = []
            node_serviced_1 = {} 

            for key, value in o_values.items():
                if value == 1.0:
                    i_value = int(key.split('_')[-1]) 
                    node_serviced.append(i_value)
                    node_serviced_1[i_value] = 'serviced' 
            u_values['u_0'] = 0

            Battery_state = {}
            for battery in Battery_routes.keys():
                Battery_state[battery] = [75] + [75] * (len(Battery_routes[battery]) - 1)
                
            for battery, route in Battery_routes.items():
                battery_number = int(battery.replace('Battery', ''))
                for i, point in enumerate(route[1:], start=1):  
                    if Battery_state[battery][i] == 75:  
                        Battery_state[battery][i] = Battery_state[battery][i - 1]  
                    if point in battery_node[battery_number]:  
                        if i+1 < len(Battery_state[battery]): 
                            Battery_state[battery][i+1] = Battery_state[battery][i] - self.R_i[int(point) - 1]  

            u_value_1 = {key: round(value, 0) for key, value in u_values.items()}

            Battery_info = {}

            for battery, routes in Battery_routes.items():
                battery_info = []
                battery_number = int(battery.replace('Battery', ''))
                battery_state_key = f'Battery{battery_number}'
                if battery_state_key in Battery_state:
                    for i, route in enumerate(routes):
                        u_key = f'u_{route}'
                        
                        if u_key in u_value_1:
                            
                            battery_info.append([route, u_value_1[u_key], Battery_state[battery_state_key][i]])
                Battery_info[battery_number] = battery_info

            for i in Battery_info:
                if Battery_info[i][0][0] == '0':
                    Battery_info[i][0][1] = Battery_info[i][1][1] - self.T_ij['0', Battery_info[i][1][0]]
            for i in Battery_info:
                if Battery_info[i][-1][0] == str(len(self.N)+1):
                    Battery_info[i][-1][1] = 0
                    Battery_info[i][-1][1] = Battery_info[i][-2][1] + 5 + self.T_ij[Battery_info[i][-2][0], Battery_info[i][-1][0]]
            for i in Battery_info:
                for sublist in Battery_info[i]:
                    sublist[1] = round(sublist[1])


            # battery json
            # battery_json_filename = os.path.join(self.battery_json_path, f'battery_output_{base_name}.json')

            # with open(battery_json_filename, 'w', encoding='cp949') as jsonfile:
            # #with open(battery_json_filename, 'w', encoding='utf-8') as jsonfile:
            #     json.dump(Battery_info, jsonfile, ensure_ascii=False, indent=4)
            # print(f"데이터가 {battery_json_filename} 파일에 저장되었습니다.")


            # battery csv
            battery_json_filename = os.path.join(self.battery_json_path, f'battery_output_{base_name}.csv')

            # with open(battery_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            with open(battery_json_filename, 'w', newline='', encoding='cp949') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Battery_Num', 'Node', 'Time', 'State'])
                for key, value in Battery_info.items():
                    for sublist in value:
                        writer.writerow([key] + sublist)

            # truck csv
            headers = ['driver_id', 'client_id', 'delivery_type', 'latitude', 'longitude', 'time' ,'battery_id']

            def determine_delivery_type(stop):
                return '수거' if 'h' in stop else '배달'

            def minutes_to_time(minutes):
                hours = int(minutes) // 60
                minutes = int(minutes) % 60
                return f"{hours:02d}:{minutes:02d}"
            def get_j_from_time_key(values, tkey):
                # time_key에 해당하는 key 찾기
                key = f'z_{tkey}_'
                for k in values.keys():
                    if k.startswith(key):
                        # key에서 j 값을 추출
                        j_value = k.split('_')[-1]
                        return j_value
            
            truck_csv_filename = os.path.join(self.truck_csv_path, f'Truck_routes_{base_name}.csv')
            
            with open(truck_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            #with open(truck_csv_filename, 'w', newline='', encoding='cp949') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)  # 헤더 작성
                
                for truck, stops in Truck_routes.items():
                    driver_id = truck.replace('Truck', '')  # 트럭 이름에서 번호 추출
                    for stop in stops:
                        if stop not in ['0', str(len(self.N)+1)]:
                            delivery_type = determine_delivery_type(stop)
                            client_id = stop.replace('h', '')  # 'h' 제거하여 클라이언트 ID 얻기
                            latitude = self.y_coord[int(client_id)-1]  # client_id에 해당하는 x_coord 가져오기
                            longitude = self.x_coord[int(client_id)-1]  # client_id에 해당하는 y_coord 가져오기
                            if stop.startswith('h'):
                                time_key = stop[1:]
                            else:
                                time_key= (stop)
                            time_minutes = u_value_1['u_'+stop]  # 시간 정보 가져오기
                            time = minutes_to_time(time_minutes)  # 시간을 시:분 형식의 문자열로 변환
                            battery_id = get_j_from_time_key(z_values, time_key)
                            writer.writerow([driver_id, client_id, delivery_type, latitude, longitude, time , battery_id])
            print(f"데이터가 {truck_csv_filename} 파일에 저장되었습니다.")

            # draw route json
            drawroute_json_filename = os.path.join(self.drawroute_json_path, f'drawroute_{base_name}.json')
            data = {}

            for truck, stops in Truck_routes.items():
                driver_id = truck.replace('Truck', '')  # 트럭 이름에서 번호 추출
                if driver_id not in data:
                    data[driver_id] = []
                for stop in stops:
                    if stop not in ['0', str(len(self.N)+1)]:
                        client_id = stop.replace('h', '')  # 'h' 제거하여 클라이언트 ID 얻기
                        latitude = self.y_coord[int(client_id)-1]  # client_id에 해당하는 y_coord 가져오기
                        longitude = self.x_coord[int(client_id)-1]  # client_id에 해당하는 x_coord 가져오기
                        time_key = stop if 'h' not in stop else stop[2:]  # 'h'가 있으면 'h'를 제외하고 가져오기
                        time_minutes = u_value_1['u_'+stop]  # 시간 정보 가져오기
                        time = minutes_to_time(time_minutes)  # 시간을 시:분 형식의 문자열로 변환
                        data[driver_id].append({
                            "latitude": latitude,
                            "longitude": longitude,
                            "time": time
                        })
                    elif stop == '0':
                        latitude = self.yd_coord  # yd_coord 가져오기
                        longitude = self.xd_coord  # xd_coord 가져오기
                        time = minutes_to_time(self.E_0)  # 시간을 시:분 형식의 문자열로 변환
                        data[driver_id].append({
                            "latitude": latitude,
                            "longitude": longitude,
                            "time": time
                        })
                    elif stop == str(len(self.N)+1):
                        latitude = self.yd_coord  # yd_coord 가져오기
                        longitude = self.xd_coord  # xd_coord 가져오기
                        time = minutes_to_time(self.L_n)  # 시간을 시:분 형식의 문자열로 변환
                        data[driver_id].append({
                            "latitude": latitude,
                            "longitude": longitude,
                            "time": time
                        })

            with open(drawroute_json_filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(data, jsonfile, ensure_ascii=False, indent=4)

        except Exception as e:
            print("An error occurred:", e)


def solve(datafile, battery_json_path, truck_json_path , drawroute_json_path):
    try:
        solver = two_phase_heuristic(datafile, battery_json_path, truck_json_path ,drawroute_json_path)
        solver.read_datafile()
        solver.construct_graph()
        # self.print_memory_usage()  # 메모리 사용량 출력
        
        print("Calculating all k shortest paths...")
        solver.all_k_shortest_paths(k=3)
        
        print("Solving battery route...")
        solver.simulated_annealing()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise  

# def print_memory_usage(self):
#     process = psutil.Process()
#     memory_info = process.memory_info()
#     print(f"RSS: {memory_info.rss / (1024 * 1024):.2f} MB, VMS: {memory_info.vms / (1024 * 1024):.2f} MB")
