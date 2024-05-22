import cplex
from cplex.exceptions import CplexSolverError
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# %matplotlib inline
# import matplotlib.patches as patches
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import math
from itertools import permutations
import networkx as nx
import pickle
import random
import copy
import time
import sys
import csv
import os


class two_phase_heuristic:
    def __init__(self, datafile, pickle_path, truck_csv_path, battery_csv_path, M=100000, lamb=1000, TN=5, CT=2):
        # self.cpx = cplex.Cplex()
        self.datafile = datafile
        self.M = M
        self.lamb = lamb
        self.TN = TN
        self.CT = CT
        self.T = list(range(1, TN + 1))
        self.B = [str(x) for x in range(1, len(self.T) * 2 + 1)]
        self.CB_l = [75] * len(self.B)
        self.C_i=[]

        self.G = None    # 그래프 초기화
        self.all_k_shortest_paths_result = None

        # self.file_path = None
        # self.battery_csv_filename = None
        # self.truck_csv_filename = None

        self.file_path = pickle_path
        self.truck_csv_path = truck_csv_path
        self.battery_csv_path = battery_csv_path

        self.visited_nodes_set = set()
        self.battery_visited_nodes = {} 

        self.initial_visited_nodes = set()

        self.memoization_cache = {}

        self.manager = None
        self.data = None

        self.loaded_k_sp_result = []



    def read_solomon(self):
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
            x_coord.append(int(data[1]))
            y_coord.append(int(data[2]))
            R_i.append(int(data[3]))
            E_i.append(int(data[4]))
            L_i.append(int(data[5]))

        N_hat = ['h' + item for item in N]

        line_d = file_content.strip().split('\n')[9]
        E_0 = int(line_d.split()[4])
        L_n = int(line_d.split()[5])
        xd_coord = int(line_d.split()[1])
        yd_coord = int(line_d.split()[2])

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
                    distance = self.calculate_distance(node1, node2)  
                    if distance <= self.depot_operation_time and self.check_time_window(node1, node2):
                        self.G.add_edge(node1, node2, weight=distance)  
        # print("Graph edges: ", self.G.edges())
 
    def get_coordinates(self, node):
        index = self.N.index(node)
        x = self.x_coord[index]
        y = self.y_coord[index]
        return x, y
    
    def calculate_distance(self, node1, node2):
        x1, y1 = self.get_coordinates(node1)
        x2, y2 = self.get_coordinates(node2)
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return distance

    def check_time_window(self, node1, node2):
        travel_time = self.calculate_distance(node1, node2)   # 현재 차량 속도 고려 X
        arrive_time = self.E_i[self.N.index(node1)] + travel_time
        end_window = self.L_i[self.N.index(node2)]
        return arrive_time <= end_window   
    

    def calculate_k_shortest_paths(self, source, target, k):
        try:
            k_shortest_paths = list(nx.shortest_simple_paths(self.G, source=source, target=target, weight='weight'))
            k_shortest_paths = k_shortest_paths[:k]
        except nx.NetworkXNoPath:
            k_shortest_paths = []
        
        return k_shortest_paths


    def all_k_shortest_paths(self, k):
        if self.all_k_shortest_paths_result is None:
            self.all_k_shortest_paths_result = {}
            nodes = list(self.G.nodes())
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    if nx.has_path(self.G, node1, node2):
                        k_shortest_paths = self.calculate_k_shortest_paths(node1, node2, k)
                        self.all_k_shortest_paths_result[(node1, node2)] = k_shortest_paths
                        if nx.has_path(self.G, node2, node1):
                            self.all_k_shortest_paths_result[(node2, node1)] = [list(reversed(path)) for path in k_shortest_paths]
            self.save_k_shortest_paths()
        
        # print('*****all_k_shortest_paths_result:', self.all_k_shortest_paths_result)
        return self.all_k_shortest_paths_result


    def save_k_shortest_paths(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.all_k_shortest_paths_result, f)

    # def set_file_path(self, file_path):
    #     self.file_path = file_path

    # def set_csv_paths(self, battery_csv_path, truck_csv_path):
    #     self.battery_csv_filename = battery_csv_path
    #     self.truck_csv_filename = truck_csv_path

    def load_k_shortest_paths(self):
        with open(self.file_path, 'rb') as f:
            self.all_k_shortest_paths_result = pickle.load(f)
        return self.all_k_shortest_paths_result
    
    
    # ortools - battery route with cvrptw 
    def create_data_model(self):
        data = {}
        depot_window = (self.E_0, self.L_n)
        data['locations'] = [(self.xd_coord, self.yd_coord)] + \
                            [(x, y) for x, y in zip(self.x_coord, self.y_coord)]
        data['time_windows'] = [depot_window] + [(E_i, L_i) for E_i, L_i in zip(self.E_i, self.L_i)]
        data['charge_times'] = [0] + self.R_i  # 충전 요구량당 1분 걸린다고 가정
        data['num_trucks'] = 5
        data['truck_capacities'] = 2
        data['num_batteries'] = data['num_trucks'] * data['truck_capacities']
        data['battery_capacities'] = [75] * data['num_batteries'] 
        data['depot'] = 0
        data['demands'] = [0] + self.R_i
        data['speed'] = 10
        
        return data
    
    def compute_euclidean_distance_matrix(self, locations):
        """Creates callback to return distance between points."""
        distances = {}
        for from_counter, from_node in enumerate(locations):
            distances[from_counter] = {}
            for to_counter, to_node in enumerate(locations):
                if from_counter == to_counter:
                    distances[from_counter][to_counter] = 0
                else:
                    # Euclidean distance
                    # distances[from_counter][to_counter] = (int(
                    #     math.hypot((from_node[0] - to_node[0]),
                    #                (from_node[1] - to_node[1]))))
                    distance = math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))
                    distances[from_counter][to_counter] = round(distance, 2)  # 거리를 반올림하여 저장
        return distances

    # def print_solution(self, data, manager=None, routing=None, assignment=None):
    #     if manager is not None and routing is not None and assignment is not None:
    #         time_dimension = routing.GetDimensionOrDie('Time')
            
    #         for vehicle_id in range(1, self.data['num_batteries'] + 1):
    #             node_solution = []
    #             arc_solution = []
    #             index = routing.Start(vehicle_id)
                
    #             print(f'Battery {vehicle_id}: ', end = '')
                
    #             while not routing.IsEnd(index):
    #                 time_var = time_dimension.CumulVar(index)
    #                 i = manager.IndexToNode(index)
                    
    #                 print(f'{i}[{assignment.Min(time_var)}, {assignment.Max(time_var)}]', end='')
                    
    #                 node_solution.append(i)
                    
    #                 previous_index = index
    #                 index = assignment.Value(routing.NextVar(index))
    #                 j = manager.IndexToNode(index)
    #                 arc_solution.append((i,j))
                    
    #                 print(f' >--{self.time_callback(manager.NodeToIndex(i), manager.NodeToIndex(j))}-> ', end='')
                
    #             time_var = time_dimension.CumulVar(index)
    #             i = manager.IndexToNode(index)
    #             print(f'{i}[{assignment.Min(time_var)}, {assignment.Max(time_var)}]')

    
    def solve_battery_route(self):
        self.visited_nodes_set = set()
        self.battery_visited_nodes = {} 
        self.battery_used_capacity = {}

        if self.data is None:
            self.data = self.create_data_model()

        self.loaded_k_sp_result = self.load_k_shortest_paths()

        self.manager = pywrapcp.RoutingIndexManager(
            len(self.data['locations']),
            self.data['num_batteries'], 
            self.data['depot']
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(self.manager)

        def time_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)

            # 각 노드에서의 충전하는 시간 추가
            charge_time = self.data['charge_times'][from_node]
            return int(distance_matrix[from_node][to_node] / self.data['speed']) + charge_time
        
        self.time_callback = time_callback

        def demand_callback(from_index):  
            """Returns the demand of the node."""
            from_node = self.manager.IndexToNode(from_index)
            return self.data['demands'][from_node]

        distance_matrix = self.compute_euclidean_distance_matrix(self.data['locations'])

        transit_callback_index = routing.RegisterTransitCallback(time_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # time dimension 추가
        routing.AddDimension(
                transit_callback_index,
                2000,   # allow waiting time, 위치에서의 대기 시간
                2000,   # maximum time per vehicle in a route, 한 차가 route를 도는데 쓸 수 있는 최대 시간
                False,  # Don't force start cumul to zero.
                'Time')
        # time window 제약 추가
        time_dimension = routing.GetDimensionOrDie('Time')

        # Add time window constraints for each location except depot.
        for location_idx, time_window in enumerate(self.data['time_windows']):
            if location_idx == 0:
                continue
            index = self.manager.NodeToIndex(location_idx)
                        
            # 충전 시간을 고려하여 유효한 time window 생성
            adjusted_earliest_time = max(time_window[0], self.data['charge_times'][location_idx])
            adjusted_latest_time = time_window[1]
            time_dimension.CumulVar(index).SetRange(adjusted_earliest_time, adjusted_latest_time)
            
            routing.AddDisjunction([index], 1000)  # 모든 node 방문 안해도 ok

        # Add time window constraints for each vehicle start node.
        depot_earlest_time = self.data['time_windows'][0][0]
        depot_latest_time = self.data['time_windows'][0][1]
  
        for vehicle_id in range(1, self.data['num_batteries'] + 1):
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
            for vehicle_id in range(1, self.data['num_batteries'] + 1):
                index = routing.Start(vehicle_id)
                plan_output_demand = f'Amount of Battery {vehicle_id} used: '
                plan_output_customer = f'Number of customers visited by Battery {vehicle_id}: '
                route_demand = 0
                customer_count = 0
                visited_nodes = []

                while not routing.IsEnd(index):
                    node = self.manager.IndexToNode(index)
                    route_demand += self.data['demands'][node]
                    
                    if node != self.data['depot']:
                        customer_count += 1
                        self.visited_nodes_set.add(node)
                        visited_nodes.append(node) 

                    plan_output_demand += f'{route_demand} -> '
                    index = assignment.Value(routing.NextVar(index))

                node = self.manager.IndexToNode(index)
                plan_output_demand += f'{route_demand + self.data["demands"][node]}'
                plan_output_customer += f'{customer_count}'

                self.battery_visited_nodes[vehicle_id] = visited_nodes
                # self.battery_used_capacity[vehicle_id].append(route_demand)
                self.battery_used_capacity[vehicle_id] = route_demand

                print(plan_output_demand)
                print(plan_output_customer)

            print("ortools_visited nodes set: ", self.visited_nodes_set)
            print("ortools_battery visited nodes: ", self.battery_visited_nodes)
            print("ortools_Battery별 사용량: ", self.battery_used_capacity)
            
            result = self.simulated_annealing(self.visited_nodes_set, self.battery_visited_nodes, self.battery_used_capacity)  
            print('++++++++++++++++++++++result: ',result)
            
            _, _, _, best_energy, best_variable_names, best_variable_values = result

            print("+++++best_energy:", best_energy)
            print("+++++best_variable_names:", best_variable_names)
            print("+++++best_variable_values:", best_variable_values)

            self.process_csv(best_energy, best_variable_names, best_variable_values)

            return result


    def node_removal(self, route, battery_route, battery_used_capacity):
        print("***** node removal 함수 받아온 route: ", route)
        print("***** node removal 함수 받아온 battery_route: ", battery_route)
        print("***** node removal 함수 받아온 battery_used_capacity: ",battery_used_capacity)

        route_copy = copy.deepcopy(route)
        new_battery_route = copy.deepcopy(battery_route)
        new_battery_used_capacity = copy.deepcopy(battery_used_capacity)

        node = random.choice(list(route_copy))     # 삭제할 노드 랜덤으로 선택
        new_route = route_copy - {node}
        print("삭제 node: ", node)

        for battery_id, nodes in new_battery_route.items():
            if node in nodes:
                new_battery_route[battery_id].remove(node)

        node_index = self.manager.NodeToIndex(node)
        # node_index = self.node_to_index[str(node)]
        print('node_index: ', node_index)

        node_demand = self.R_i[node_index-1]
        print('노드 demand: ', node_demand)

        # 배터리 용량 및 사용 용량 업데이트
        for battery_id, used_capacity in new_battery_used_capacity.items():
            print("$$$$ battery_route[battery_id]:", battery_route[battery_id])
            
            if isinstance(used_capacity, int):
                if node in battery_route.get(battery_id, []): 
                    new_battery_used_capacity[battery_id] -= node_demand
                else:
                    print("노드가 배터리 경로 안에 없음:", node, battery_id)
            else:
                print("int 값 아님")
                
        print("node removal 후 route 결과: ", new_route)
        print("node removal 후 battery route 결과: ", new_battery_route)
        print("node removal 후 battery used capacity 결과: ", new_battery_used_capacity)
        
        return new_route, new_battery_route, new_battery_used_capacity
    
    def node_removal2(self, route, battery_route, battery_used_capacity):
        print("***** node removal 함수 받아온 route: ", route)
        print("***** node removal 함수 받아온 battery_route: ", battery_route)
        print("***** node removal 함수 받아온 battery_used_capacity: ", battery_used_capacity)

        new_route = copy.deepcopy(route)
        new_battery_route = copy.deepcopy(battery_route)
        new_battery_used_capacity = copy.deepcopy(battery_used_capacity)

        # 방문한 노드 수가 많은 배터리들 중에서 배터리 선택
        max_visited_battery_ids = [battery_id for battery_id, nodes in new_battery_route.items() if nodes]
        if not max_visited_battery_ids:
            print("노드를 방문한 배터리가 없습니다.")
            return new_route, new_battery_route, new_battery_used_capacity

        battery_id = random.choice(max_visited_battery_ids)
        visited_nodes = new_battery_route[battery_id]

        # 방문한 노드들 중에서 랜덤으로 노드 선택
        node = random.choice(visited_nodes)
        new_route.remove(node)
        print("삭제 node: ", node)

        # 배터리 경로에서 노드 삭제
        new_battery_route[battery_id].remove(node)

        node_index = self.manager.NodeToIndex(node)
        print('node_index: ', node_index)

        node_demand = self.R_i[node_index - 1]
        print('노드 demand: ', node_demand)

        # 배터리 용량 업데이트
        new_battery_used_capacity[battery_id] -= node_demand

        print("node removal 후 route 결과: ", new_route)
        print("node removal 후 battery route 결과: ", new_battery_route)
        print("node removal 후 battery used capacity 결과: ", new_battery_used_capacity)

        return new_route, new_battery_route, new_battery_used_capacity



    def node_exchange(self, route, battery_route, battery_used_capacity):
        print("***** node exchange 함수 받아온 route: ", route)
        print("***** node exchange 함수 받아온 battery_route: ", battery_route)
        print("***** node exchange 함수 받아온 battery_used_capacity: ", battery_used_capacity)

        new_route = copy.deepcopy(route)
        new_battery_route = copy.deepcopy(battery_route)
        new_battery_used_capacity = copy.deepcopy(battery_used_capacity)

        # 무작위로 선택된 두 배터리 ID 선택
        battery_ids = list(new_battery_route.keys())
        non_empty_battery_ids = [bid for bid in battery_ids if new_battery_route[bid]]  
        
        if len(non_empty_battery_ids) < 2:
            return None

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
        battery_id1_demand = sum(int(self.R_i[self.manager.NodeToIndex(node) - 1]) for node in visited_nodes1)
    
        new_battery_route[battery_id1] = visited_nodes1
        new_battery_used_capacity[battery_id1] = battery_id1_demand

        # 배터리 2의 사용 용량 업데이트
        battery_id2_demand = sum(int(self.R_i[self.manager.NodeToIndex(node) - 1]) for node in visited_nodes2)

        new_battery_route[battery_id2] = visited_nodes2
        new_battery_used_capacity[battery_id2] = battery_id2_demand

        # print("node exchange 후 route 결과: ", route)
        print("node exchange 후 battery route 결과: ", new_battery_route)
        print("node exchange 후 battery used capacity: ", new_battery_used_capacity)
        
        return new_route, new_battery_route, new_battery_used_capacity


    def node_insertion(self, route, battery_route, battery_used_capacity):
        print("***** node insertion 함수 받아온 route: ", route)
        print("***** node insertion 함수 받아온 battery_route: ", battery_route)
        print("***** node insertion 함수 받아온 battery_used_capacity: ", battery_used_capacity)

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
        print('$$$$$battery_ids: ', battery_ids)

        if not battery_ids:
            return None   # 수정 필요
        
        # 방문한 노드가 가장 적은 배터리 선택
        battery_id = battery_ids[0]

        # battery_id = random.choice(list(new_battery_route.keys()))
        
        visited_nodes = new_battery_route[battery_id]

        last_node_index = visited_nodes[-1] - 1
        last_node = self.N[last_node_index]

        print('last_node: ', last_node)
        
        # neighbors = [] 

        # for successor_node in self.G.successors(last_node):
        #     if int(successor_node) not in new_route:
        #         neighbors.append(int(successor_node))


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


        print('random으로 선택된 battery_id: ', battery_id)
        print('해당 battery_id의 visited_nodes: ', visited_nodes)
        print('$$$$$ last_node neighbors: ', neighbors)

        if not neighbors:
            print("마지막 노드의 이웃이 없습니다.")
            return None

        new_node = random.choice(neighbors)
        print('new_node: ', new_node)
        
        # new_route.add(new_node)
        
        # 추가될 노드의 요구량 계산
        node_index = self.manager.NodeToIndex(int(new_node))
        # node_index = self.node_to_index[new_node]
        print('node_index: ', node_index)

        node_demand = self.R_i[node_index-1]
        print('노드 요구 demand: ', node_demand)

        # 배터리 용량을 고려하여 노드를 추가할 배터리 결정
        battery_capacity = self.data['battery_capacities'][battery_id]

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

        print("node insertion 후 route 결과: ", new_route)
        print("node insertion 후 battery route 결과: ", new_battery_route)
        print("node insertion 후 battery used capacity 결과: ", new_battery_used_capacity)
        
        return new_route, new_battery_route, new_battery_used_capacity
    


    def calculate_objective(self, route, battery_route):
        print("*****문제 풀기")
        # self.memoization_cache.clear() 

        # cache_key = (tuple(route), tuple(battery_route))
        # if cache_key in self.memoization_cache:
        #     return self.memoization_cache[cache_key]

        self.process_route(route, battery_route)

        try:
            self.MIP_solver()
            objective_value = self.cpx.solution.get_objective_value()
            # self.memoization_cache[cache_key] = objective_value

            print("현재 문제를 푼 방문노드: ", route)
            print("목적함수 값: ", objective_value)
            
            # Variable Values 출력
            print("Variable Values:")
            variable_names = self.cpx.variables.get_names()
            variable_values = self.cpx.solution.get_values()

            for name, value in zip(variable_names, variable_values):
                if value >= 0.0001:
                    print(name, "=", value) 

            print("Solution status:", self.cpx.solution.get_status())
            
            return objective_value, variable_names, variable_values
        
        except CplexSolverError as e:
            print("CPLEX Error:", e)

            return None, None, None
        


    def simulated_annealing(self, initial_route, initial_battery_route, initial_battery_used_capacity, initial_temperature=100, cooling_rate=0.95, max_iterations=10, time_limit=30):
        start_time = time.time()
        initial_solutions = []

        while time.time() - start_time < time_limit:
            current_route = initial_route
            current_battery_route = initial_battery_route 
            current_battery_used_capacity = initial_battery_used_capacity
            current_energy, current_variable_names, current_variable_values = self.calculate_objective(current_route, current_battery_route)
            print("*****current_energy:", current_energy)
            
            while current_energy is None :
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
                    current_route = copy.deepcopy(new_route)
                    current_battery_route = copy.deepcopy(new_battery_route)
                    current_battery_used_capacity = copy.deepcopy(new_battery_used_capacity)

            # 현재 해를 저장
            initial_solutions.append((current_energy, current_route, current_battery_route, current_battery_used_capacity, current_variable_names, current_variable_values))

        # 60초가 지난 후에도 초기해가 없는 경우 계속해서 초기해를 구할 때까지 반복
        while not initial_solutions:
            current_route = initial_route
            current_battery_route = initial_battery_route 
            current_battery_used_capacity = initial_battery_used_capacity
            current_energy, current_variable_names, current_variable_values = self.calculate_objective(current_route, current_battery_route)
            print("*****current_energy:", current_energy)

            while current_energy is None:
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
                    current_route = copy.deepcopy(new_route)
                    current_battery_route = copy.deepcopy(new_battery_route)
                    current_battery_used_capacity = copy.deepcopy(new_battery_used_capacity)

            # 현재 해를 저장
            initial_solutions.append((current_route, current_battery_route, current_battery_used_capacity, current_energy, current_variable_names, current_variable_values))

        # # 가장 좋은 초기해 선택
        # best_initial_energy, best_initial_route, best_initial_battery_route, best_initial_battery_used_capacity = max(initial_solutions, key=lambda x: x[0])

        # 가장 좋은 초기해 선택
        best_initial_solution = max(initial_solutions, key=lambda x: x[0])
        best_initial_energy, best_initial_route, best_initial_battery_route, best_initial_battery_used_capacity, best_initial_variable_names, best_initial_variable_values = best_initial_solution

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

        print(f"***** 초기해 current route: {current_route}")
        print(f"***** 초기해 current battery route: {current_battery_route}")
        print(f"***** 초기해 current battery used capacity: {current_battery_used_capacity}")
        print(f"***** 초기해 current energy: {current_energy}")
            
        temperature = initial_temperature
        iterations = 0

        while temperature > 0.1 and iterations < max_iterations:
            print("***** SA 시작")
            
            probabilities = [0.3, 0.05, 0.65]  
            selected_operator = random.choices([self.node_exchange, self.node_removal2, self.node_insertion], weights=probabilities)[0]

            if len(current_route) > 1:
                new_route, new_battery_route, new_battery_used_capacity = selected_operator(current_route, current_battery_route, current_battery_used_capacity)
                print(f"***** Operator: {selected_operator.__name__}")
                print(f"문제 input 방문 노드: {new_route}")
                print(f"문제 input 배터리별 방문 노드: {new_battery_route}")

                new_energy, new_variable_names, new_variable_values = self.calculate_objective(new_route, new_battery_route)
                print(f"새 목적함수 값: {new_energy}")   

                if new_energy is not None:
                    if new_energy >= best_energy:  # new_energy가 더 크거나 같으면 갱신
                        best_route = copy.deepcopy(new_route)
                        best_battery_route = copy.deepcopy(new_battery_route) 
                        best_battery_used_capacity = copy.deepcopy(new_battery_used_capacity)
                        best_energy = new_energy
                        best_variable_names = copy.deepcopy(new_variable_names)
                        best_variable_values = copy.deepcopy(new_variable_values)

                        current_route = copy.deepcopy(new_route)
                        current_battery_route = copy.deepcopy(new_battery_route)
                        current_battery_used_capacity = copy.deepcopy(new_battery_used_capacity)
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
                            print("***** 개선이 없었음")
                            current_route = copy.deepcopy(best_route)
                            current_battery_route = copy.deepcopy(best_battery_route)
                            current_battery_used_capacity = copy.deepcopy(best_battery_used_capacity)
                            current_variable_names = copy.deepcopy(best_variable_names)
                            current_variable_values = copy.deepcopy(best_variable_values)

                # temperature *= 1 - cooling_rate
                temperature *= cooling_rate
                iterations += 1
                print(f"***** Iteration {iterations}, Temperature: {temperature}, Current Energy: {current_energy}, Best Energy: {best_energy}")

        print("Best solution:", best_route)
        print("Best battery route:", best_battery_route)
        print("Best battery used capacity:", best_battery_used_capacity)
        print("Best objective value:", best_energy)

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
        
        # loaded_k_sp_result = []

        self.k_shortest_paths = set()
        self.expanded_pairs = set()

        self.c = set()

        # a 
        for vehicle_id, visited_nodes in battery_route.items():
            battery_id = str(vehicle_id)
            print(f'Visited nodes by Battery {battery_id}: {visited_nodes}')

            if not visited_nodes:
                continue

            # self.a.add((str(visited_nodes[-1]), 'h' + str(visited_nodes[-1]), battery_id))

            # for i in range(len(visited_nodes) - 1):
            for i in range(0, len(visited_nodes)):
                self.a.add((str(visited_nodes[i]), 'h' + str(visited_nodes[i]), battery_id))
                print(f'Added to self.a: {(str(visited_nodes[i]), "h" + str(visited_nodes[i]), battery_id)}')
        
        print("$$$$$ a, Battery Visited Nodes (i,i\hat,l): ", self.a)
        
        # route_name_list = [self.N[node-1] for node in route]
        # print('$$$$$ route_name_list: ', route_name_list)

        # a_c
        print("$$$$$ self.N: ", self.N)
        print("$$$$$ route: ", route)

        route_str = set(map(str, route))
        
        self.a_c_nodes = set(self.N) - route_str
        print("$$$$$ a_c_nodes: ", self.a_c_nodes)

        for node in self.a_c_nodes:
            for l in self.B:
                self.a_c.add((node, 'h' + node, l))
                print(f'Added to self.a_c: {(node, "h" + node, l)}')
        
        print("$$$$$ a_c: ", self.a_c)

        # Check for intersection
        intersection = self.a.intersection(self.a_c)
        if intersection:
            print("겹치는 요소가 있습니다: ", intersection)
        else:
            print("겹치는 요소가 없습니다.")


        # b
        for i, j in permutations(route, 2):
            if self.G.has_edge(str(i), str(j)):     # 만약 G에 edge가 있다면
                self.b.add((str(i), str(j)))

        print("$$$$$ Battery Visited Nodes (i,j): ", self.b)
        print("len b: ", len(self.b))

        # b에 해당하는 k-sp 가져오기
        for pair in self.b:
            i, j = pair
            k_sp = self.loaded_k_sp_result.get((i, j), [])
            for path in k_sp:
                self.k_shortest_paths.add(tuple(path))

        print('k-shortest paths:', self.k_shortest_paths)

        # k-sp 확장
        self.expanded_pairs = self.expand_node_pairs(self.k_shortest_paths)

        for node in route:
            self.expanded_pairs.add(('0', str(node)))
            self.expanded_pairs.add(('0', 'h' + str(node)))
            self.expanded_pairs.add((str(node), str(len(self.N) + 1)))
            self.expanded_pairs.add(('h' + str(node), str(len(self.N) + 1)))

        print('expanded_pairs: ', self.expanded_pairs)

        #####
        expanded_pairs_set = set(self.expanded_pairs)
        print('&&&&& expanded_pairs_set: ', expanded_pairs_set)


        pairs_to_expand = [(node1, node2) for node1 in self.a_c_nodes for node2 in self.N if node1 != node2]
        pairs_to_expand += [(node2, node1) for node1 in self.a_c_nodes for node2 in self.N if node1 != node2]

        expanded_pairs_from_ac_nodes = self.expand_node_pairs(pairs_to_expand)

        for pair in expanded_pairs_from_ac_nodes:
            if pair not in expanded_pairs_set:
                self.c.add(pair)

        print('&&&&& non_expanded_pairs: ', self.c)

        # Find and exclude intersections
        intersection_to_exclude = expanded_pairs_set.intersection(self.c)
        print('&&&&& intersection_to_exclude: ', intersection_to_exclude)

        self.expanded_pairs = expanded_pairs_set - intersection_to_exclude

        print('&&&&& filtered expanded_pairs: ', self.expanded_pairs)


    def MIP_solver(self):
        self.cpx = cplex.Cplex()
            
        is_cloned = {original: cloned for original, cloned in zip(self.N, self.N_hat)}
        V = self.N + self.N_hat
        V_0 = V.copy()
        V_0.append(str(0))
        V_1 = V.copy()
        V_1.append(str(len(self.N) + 1))
        V_2 = V.copy()
        V_2.append(str(0))
        V_2.append(str(len(self.N) + 1))

        A_1 = {(i, j) for i in V for j in V if i != j and (j not in is_cloned or is_cloned[j] != j)}
        A_2 = {(original, cloned) for original, cloned in is_cloned.items()}
        A_3 = {(str(0), j) for j in V}
        A_4 = {(i, str(len(self.N) + 1)) for i in V}
        A_5 = {(cloned, original) for original, cloned in is_cloned.items()}
        A = A_1.union(A_2, A_3, A_4) - A_5
        A_p = A - A_2
        S_i = [5] * len(V)
        C_i = self.R_i
        coordinates = [(float(x), float(y)) for x, y in zip(self.x_coord, self.y_coord)]
        xd_coord, yd_coord = float(self.xd_coord), float(self.yd_coord)

        def calculate_distance(coord1, coord2):
            return round(math.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2), 2)

        T_ij = {}
        for i, j in A:
            if (i, j) in A_2:
                T_ij[(i, j)] = C_i[self.N.index(i)]
            else:
                if i == '0':
                    coord1 = (xd_coord, yd_coord)
                elif i in self.N:
                    coord1 = coordinates[self.N.index(i)]
                else:
                    coord1 = coordinates[self.N_hat.index(i)]

                if j == str(len(self.N) + 1):
                    coord2 = (xd_coord, yd_coord)
                elif j in self.N:
                    coord2 = coordinates[self.N.index(j)]
                else:
                    coord2 = coordinates[self.N_hat.index(j)]

                T_ij[(i, j)] = calculate_distance(coord1, coord2)

        
        self.cpx.objective.set_sense(self.cpx.objective.sense.maximize)

        x_i_j = lambda i, j: 'x_%s_%s' % (i, j)
        y_i_j_l = lambda i, j, l: 'y_%s_%s_%s' % (i, j, l)
        z_i_l = lambda i, l: 'z_%s_%s' % (i, l)
        o_i = lambda i: 'o_%s' % (i)
        u_i = lambda i: 'u_%s' % (i)
        v_i = lambda i: 'v_%s' % (i)


        self.cpx.variables.add(
            obj=[-T_ij[(i,j)] for (i, j) in A],
            ub=[1 for (i, j) in A],
            lb=[0 for (i, j) in A],
            types=['B' for (i, j) in A],
            names=[x_i_j(str(i), str(j)) for (i, j) in A]
        )

        self.cpx.variables.add(
            obj=[0 for (i, j) in A for l in self.B],
            ub=[1 for (i, j) in A for l in self.B],
            lb=[0 for (i, j) in A for l in self.B],
            types=['B' for (i, j) in A for l in self.B],
            names=[y_i_j_l(str(i), str(j), str(l)) for (i, j) in A for l in self.B]
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
            obj=[0 for i in V_1],
            lb=[0 for i in V_1],
            types=['C' for i in V_1],
            names=[u_i(str(i)) for i in V_1]
        )

        self.cpx.variables.add(
            obj=[0 for i in V_0],
            lb=[0 for i in V_0],
            types=['C' for i in V_0],
            names=[v_i(str(i)) for i in V_0]
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


        # total = [(i, j, l) for (i, j) in A for l in self.B]
        # print('total (i,j,l) 갯수: ', len(total))


        # for (i, j) in A:
        #     for l in self.B:
        #         if (i, j, l) not in self.a:
        #             self.cpx.variables.set_upper_bounds(y_i_j_l(i, j, l), 0)
        
        
        for (i, j) in A:
            if i != str(0) and j != str(len(self.N)):
                if (i, j) not in self.expanded_pairs:
                    # print(f"Processing pair: ({i}, {j})")
                    self.cpx.variables.set_upper_bounds(x_i_j(i, j), 0)

            
        #2 모든 트럭은 안나가도 괜찮다
        T_len = len(self.T)
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(0, i) for i in V],
                    val=[1 for i in V]
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
                    ind=[x_i_j(a, b) for (a,b) in A if b == j ],
                    val=[1 for (a,b) in A if b == j  ]
                ) for j in V
            ],
            senses=['L' for j in V],
            rhs=[1 for j in V],
            names=['const3_%s' % j for j in V]
        )

        #4 x flow 제약식
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(a, b) for (a, b) in A if a == i] + [x_i_j(c, d) for (c,d) in A if d == i],
                    val=[1 for (a, b) in A if a == i] + [-1 for (c, d) in A if d == i]
                ) for i in V
            ],
            
            senses=['E' for i in V] ,
            rhs=[0 for i in V],
            names=['const4_%s' % (i) for i in V]
        )

        #5
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(a, b) for (a, b) in A if b == i ] + [x_i_j(c, d) for (c, d) in A if d == j],
                    val=[1 for (a, b) in A if b == i] + [-1 for (c, d) in A if d == j]
                ) for (i,j) in A_2
            ], 
            senses=['E' for (i,j) in A_2],
            rhs=[0 for (i,j) in A_2],
            names=['const5_%s_%s' % (i, j) for (i,j) in A_2]
        ) 

        #6
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[y_i_j_l(0, i, l) for i in V],
                    val=[1 for i in V]
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
                    ind=[y_i_j_l(a, b, l) for (a, b) in A if a == i] + [y_i_j_l(c, d, l) for (c, d) in A if d == i],
                    val=[1] * len([y_i_j_l(a, b, l) for (a, b) in A if a == i]) + [-1] * len([y_i_j_l(c, d, l) for (c, d) in A if d == i])
                ) for i in V for l in self.B
            ],
            senses=['E' for i in V for l in self.B] ,
            rhs=[0 for i in V for l in self.B] ,
            names=['const7_%s_%s' % (i, l) for i in V for l in self.B]
        )

        #8
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[x_i_j(i, j)] + [y_i_j_l(i, j, l) for l in self.B],
                    val=[self.CT] + [-1 for l in self.B]
                ) for (i,j) in A_p
            ],
            senses=['G' for (i,j) in A_p],
            rhs=[0 for (i,j) in A_p],
            names=['const8_%s_%s' % (i,j) for (i,j) in A_p]
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
                ) for (i, j) in A_2 for l in self.B
            ],
            senses=['E' for (i, j) in A_2 for l in self.B],
            rhs=[0 for (i, j) in A_2 for l in self.B],
            names=['const12_%s_%s_%s' % (i, j, l) for (i, j) in A_2 for l in self.B]
        )
        #13
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(i)] + [u_i(i)],
                    val=[1] + [-1]
                ) for i in V
            ],
            senses=['G' for i in V],
            rhs=[S_i[i] for i in range(0,len(V))],
            names=['const13_%s' % (i) for i in V]
        )
        #14
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(i)] + [u_i(j)] + [x_i_j(i,j)],
                    val=[1] + [-1] + [self.M]
                ) for (i,j) in A
            ],
            senses=['L' for (i,j) in A],
            rhs=[ - T_ij[(i,j)] + self.M for (i,j) in A],
            names=['const14_%s_%s' % (i,j) for (i,j) in A]
        )

        #15
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[u_i(j)] + [v_i(i)],
                    val=[1] + [-1]
                ) for (i, j) in A_2
            ],
            senses=['G' for (i, j) in A_2],
            rhs=[C_i[self.N.index(i)] for (i, j) in A_2],
            names=['const15_%s_%s' % (i,j) for (i, j) in A_2]
        )
        #16 떠나는 v_i_hat lower bound 
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[v_i(j)],
                    val=[1]
                ) for (i, j) in A_2
            ],
            senses=['G' for (i, j) in A_2],
            rhs=[self.E_i[self.N.index(i)] + S_i[self.N.index(i)] + C_i[self.N.index(i)] + S_i[self.N_hat.index(j)] for (i, j) in A_2],
            names=['const16_%s_%s' % (i,j) for (i, j) in A_2]
        )
        #17 들어오는 u_i upper bound 최소 이전에는 들어와야함
        self.cpx.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(
                    ind=[u_i(i)],
                    val=[1]
                ) for (i, j) in A_2
            ],
            senses=['L' for (i, j) in A_2],
            rhs=[ self.L_i[self.N.index(i)] - S_i[self.N.index(i)] - C_i[self.N.index(i)] - S_i[self.N_hat.index(j)] for (i, j) in A_2],
            names=['const17_%s_%s' % (i,j) for (i, j) in A_2]
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
            rhs=[self.E_i[self.N.index(i)] + S_i[self.N.index(i)] for i in self.N],
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


    def process_csv(self, objective_value, variable_names, variable_values):
        print("+++++objective_value: ", objective_value)
        print("+++++variable_names: ", variable_names)
        print("+++++variable_values: ", variable_values)

        try:
            file_name = os.path.basename(self.datafile)
            base_name = os.path.splitext(file_name)[0]

            # objective_value = solver.cpx.solution.get_objective_value()
            # variable_values = solver.variable_values
            
            # if objective_value is not None:
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

            print("+++++++++++++++x_values:", x_values)
            # else:
            #     return None, None

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

                while True:
                    next_nodes = [node for node in x_values if node.startswith(f'x_{current_node}') and node != start_node]
                    if not next_nodes:
                        break
                    next_node = next_nodes[0].split('_')[2]
                    if next_node == route[0]:
                        break
                    route.append(next_node)
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
                for i, point in enumerate(route[1:], start=1):  # Skip the first point as it's always '0'
                    if Battery_state[battery][i] == 75:  # Check if the point is None (not updated yet)
                        Battery_state[battery][i] = Battery_state[battery][i - 1]  # Set current state as previous state
                    if point in battery_node[int(battery[-1])]:  # Check if the point is in the battery's nodes
                        if i+1 < len(Battery_state[battery]):  # Check if next state index is within the range
                            Battery_state[battery][i+1] = Battery_state[battery][i] - self.R_i[int(point) - 1]  # Update next state considering energy consumption

            Battery_info = {}

            for battery, routes in Battery_routes.items():
                battery_info = []
                battery_number = int(battery.replace('Battery', ''))
                battery_state_key = f'Battery{battery_number}'
                if battery_state_key in Battery_state:
                    for i, route in enumerate(routes):
                        u_key = f'u_{route}'
                        
                        if u_key in u_values:
                            
                            battery_info.append([route, u_values[u_key], Battery_state[battery_state_key][i]])
                Battery_info[battery_number] = battery_info

            # battery_csv_filename = f'battery_output_0516_{base_name}.csv'
            battery_csv_filename = os.path.join(self.battery_csv_path, f'battery_output_{base_name}.csv')
            
            with open(battery_csv_filename, 'w', newline='', encoding='cp949') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Battery_Num', 'Node', 'Time', 'State'])
                
                for key, value in Battery_info.items():
                    for sublist in value:
                        writer.writerow([key] + sublist)  

            # truck_csv_filename = f'Truck_routes_{base_name}.csv'
            truck_csv_filename = os.path.join(self.truck_csv_path, f'Truck_routes_{base_name}.csv')

            u_value_1 = {key: round(value, 0) for key, value in u_values.items()}

            headers = ['driver_id', 'client_id', 'delivery_type', 'latitude', 'longitude', 'time']

            def determine_delivery_type(stop):
                return '수거' if 'h' in stop else '배달'

            def minutes_to_time(minutes):
                hours = int(minutes) // 60
                minutes = int(minutes) % 60
                return f"{hours:02d}:{minutes:02d}"

            # try:
            with open(truck_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            # with open(truck_csv_filename, 'w', newline='', encoding='cp949') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)  # 헤더 작성
                
                for truck, stops in Truck_routes.items():
                    driver_id = truck.replace('Truck', '')  # 트럭 이름에서 번호 추출
                    for stop in stops:
                        if stop not in ['0', '11']:
                            delivery_type = determine_delivery_type(stop)
                            client_id = stop.replace('h', '')  # 'h' 제거하여 클라이언트 ID 얻기
                            latitude = self.x_coord[int(client_id)-1]  # client_id에 해당하는 x_coord 가져오기
                            longitude = self.y_coord[int(client_id)-1]  # client_id에 해당하는 y_coord 가져오기
                            time_key = stop if 'h' not in stop else stop[2:]  # 'h'가 있으면 'h'를 제외하고 가져오기
                            time_minutes = u_value_1['u_'+stop]  # 시간 정보 가져오기
                            time = minutes_to_time(time_minutes)  # 시간을 시:분 형식의 문자열로 변환
                            writer.writerow([driver_id, client_id, delivery_type, latitude, longitude, time])
            print(f"데이터가 {truck_csv_filename} 파일에 저장되었습니다.")
            # except Exception as e:
            #    print(f"파일을 저장하는 중 오류가 발생했습니다: {e}")

        except Exception as e:
            print("An error occurred:", e)


def solve(datafile, pickle_path, battery_csv_path, truck_csv_path):
    solver_1 = two_phase_heuristic(datafile, pickle_path, battery_csv_path, truck_csv_path)
    solver_1.read_solomon()
    solver_1.construct_graph()
    solver_1.all_k_shortest_paths(k=3)
    # solver_1.load_k_shortest_paths()
    solver_1.solve_battery_route()

    # return result

# if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     if sys.argv[1] == "solve":
    #         solve()
    #     elif sys.argv[1] == "calculate_k_shortest_paths":
    #         result = calculate_k_shortest_paths()
    #         for key, value in result.items():
    #             print(f"Source: {key[0]}, Target: {key[1]} - K-shortest paths: {value}")
    #     else:
    #         print("Invalid argument. Use 'solve' or 'calculate_k_shortest_paths'.")
    # else:
    #     print("Please provide an argument: 'solve' or 'calculate_k_shortest_paths'.")


    # best_route, best_battery_route, best_battery_used_capacity, best_energy = result
    # print("Final Best solution:", best_route)
    # print("Final Best battery route:", best_battery_route)
    # print("Final Best objective value:", best_energy)
    
    # # 결과를 파일에 저장
    # with open("output_log.txt", "w") as f:
    #     f.write(f"Final Best solution: {best_route}\n")
    #     f.write(f"Final Best battery route: {best_battery_route}\n")
    #     f.write(f"Final Best objective value: {best_energy}\n")