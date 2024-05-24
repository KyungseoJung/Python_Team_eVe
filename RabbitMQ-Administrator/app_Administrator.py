# // T-Map appkey = 5RB8KXlDuB6uLZGhugCqS9OJMViZ73P93dRPbphu 

# //#26 관리자(Administrator) 웹화면 디자인 구성 
# //#27 관리자 웹페이지 디자인 
# //#28 수리모형 코드 통합 - [경로 계산하기] 버튼 누르면, 수리모형 코드 실행되도록 - csv 파일 지정한 파일 위치에 저장되도록
# //#29 배터리 강화학습 코드 통합

from flask import Flask, render_template
from flask import jsonify, request # //#28 수리모형 코드 통합을 위한 import

import pandas as pd
import eve_0523_test1 # //#28 수리모형 코드 통합 (Import 수리모형 함수를 포함한 Python file )

app = Flask(__name__)

#===================================================================================================
# //#29 배터리 강화학습 코드 통합 - 여기부터


import pandas as pd
import numpy as np

import math
import glob
import os
from datetime import datetime

import pybamm

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from flask import Flask, render_template, jsonify


#=========================================================================================================
#class BM 은 강화학습환경 > batt3.zip 으로 저장되어있는 모델을 불러왔어 class BM에 적용되어 강화학습으로 충전을 진행
#=========================================================================================================
class BM(gym.Env):
    def __init__(self,start_soc, goal_time,goal_soc,render_mode = None):
        options = {"thermal": "lumped"}
        self.model = pybamm.lithium_ion.SPMe(options)

        self.params = pybamm.ParameterValues("Chen2020").copy()
        init_input = {
            'Number of cells connected in series to make a battery': 4164,
            'Upper voltage cut-off [V]': 5,
        }
        self.params.update(init_input)

        self.solutions = []

        # 기본 세팅
        self.r_max_temp = 273 + 35
        self.r_max_volt = 4.2
        self.SoC_desired = goal_soc

        self.volt = 0
        self.temp = 0
        
        self.time_goal = 0
        self.ep_num = 0
        self.time_step = 0
        self.MAX_time_step = 3600*2

        # test 모델의 축
        self.time_goal  = goal_time
        self.init_soc = start_soc

        self.observation_space = spaces.Box(low=0, high=400, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(30)



    def step(self,action):
        
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        try:

            if self.SoC >= self.SoC_desired or self.SoC >= 1:
                terminated = True
                self.SoC = 1

            else:
                terminated = False

            experiment = pybamm.Experiment([f"Charge at {action/10}C for 30 sec"])
            sim = pybamm.Simulation(model, experiment=experiment, parameter_values=self.params)
            step_solution = sim.solve(starting_solution=self.solutions[-1].last_state)
            self.solutions += [step_solution.last_state]
            

            # Calculate reward based on various factors
            if self.time_step >= self.MAX_time_step:
                terminated = True
                reward = -10000


            self.temp = step_solution["X-averaged cell temperature [K]"].entries[-1]
            self.volt = step_solution["Terminal voltage [V]"].entries[-1]
            Q = self.params["Nominal cell capacity [A.h]"]
            DC = step_solution["Discharge capacity [A.h]"].entries[-1]
            self.SoC = self.SoC-DC/Q
            if self.time_step <= self.time_goal:
                crie = self.SoC > (0.6/self.time_goal)*self.time_step+0.2
                if crie:
                    r_soc = 0
                else:
                    r_soc = 10*(self.SoC - ((0.6/self.time_goal)*self.time_step+0.2))
            else :
                r_soc = (self.time_goal - self.time_step)   
            r_temp = -5 * abs(self.temp - (273+35)) if self.temp> (273+35) else 0
            r_volt = -200 * abs(self.volt - self.r_max_volt) if self.volt > self.r_max_volt else 0

            reward = r_temp +r_volt + r_soc
            self.time_step +=1
        except:
            reward = -1000
            terminated = True


        observation = self._get_obs()
        info = self._get_info()


        # print(self.time_step , observation,reward,"|",float(action),"|")
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        #print(self.init_soc)
        super().reset(seed=seed)
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        self.SoC = self.init_soc
        experiment = pybamm.Experiment(["Rest for 30 min"])
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=self.params)
        step_solution = sim.solve(initial_soc=self.init_soc)
        self.solutions += [step_solution.last_state]

        observation = self._get_obs()
        info = self._get_info()
        self.ep_num +=1
        self.time_step = 0
        return observation, info

    def generate_random_number(self,time_goal):
        self.time_goal = time_goal * 2 -5
        #in test 지정
    
    def update_init_soc(self, init_soc):
        self.init_soc = init_soc
    
    def get_last_soc(self):
        print("Last_soc:",self.SoC)
        return self.SoC
    
    def get_last_solution(self):
        return self.solutions
        
    
    def _get_obs(self):
        return np.array([self.SoC,self.volt,self.temp,self.time_goal-self.time_step], dtype=np.float32)

    def _get_info(self):
        return {"distance": self.SoC_desired - self.SoC}
#=====================================================================================================================
# SumulateBattery 보연이형이 준 파일을 바탕으로 강화학습과 배터리 시뮬레이터를 통해 배터리 데이터를 분단위로 기록
#=====================================================================================================================
class SumulateBattery:
    def RoadRoutData(self,BattID):
        # 파일 경로 설정
        directory_path = './data_input' # 보연이형이 만든 파일 배터리 경로 

        # glob을 사용하여 패턴에 맞는 파일 목록 얻기
        csv_pattern = os.path.join(directory_path, '*.csv')
        csv_files = glob.glob(csv_pattern) # 파일안에 들어 있는 모든 csv 파일 리스트를 탐색

        # 빈 리스트 생성 > 보연이형이 생성한 배터리 데이터를 한 파일로 읽어오기
        dataframes = []

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dataframes.append(df)

        # 리스트의 모든 DataFrame을 하나로 연결
        cdv = pd.concat(dataframes, ignore_index=True)

        # 배터리 충전 전략을 위한 데이터 형식으로 바꾸기
        cdv_1 =cdv[['Battery_Num','Node','Time','State']]
        battery = cdv_1[cdv_1['Battery_Num'] == BattID]
        BatterySituation = []
        
        for i in range(len(battery) - 1):
            time_vari = battery.iloc[i + 1]["Time"] - battery.iloc[i]["Time"]
            state_vari = battery.iloc[i + 1]["State"] - battery.iloc[i]["State"]
            BatterySituation.append((time_vari, state_vari))

        return BatterySituation

    # 배터리 별로 충전 시뮬레이션을 진행
    def Simulation(self,BattID):
        BatterySituation = self.RoadRoutData(BattID) # RoadRoutData 함수를 통해 데이터를 불러오기
        SoC = 1.0 # 초기 SoC는 100% (배터리 충전 상태)
        solutions = [] # 배터리 시뮬레이션의 해를 저장
        SoC_imin_list = [] # 1분 단위 SoC를 저장

        # 배터리 시뮬레이터 불러오기 PyBamm 사용
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        params = pybamm.ParameterValues("Chen2020").copy()
        # 초기 배터리 상태 구현 93KwH 
        # 세부 파라미터 조정 전
        init_input = {
            'Number of cells connected in series to make a battery': 4164,
            'Upper voltage cut-off [V]': 5,
        }
        params.update(init_input)

        experiment = pybamm.Experiment(["Rest for 1 min"])
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)

        step_solution = sim.solve(initial_soc=1)
        solutions += [step_solution.last_state]
        # 배터리 데이터를 순차적으로 불러와 충전과 방전을 반복
        for i in range(len(BatterySituation)):
            
            time = BatterySituation[i][0]
            SoC_state = BatterySituation[i][1]
            CargeRate = (SoC_state / time) * (60 / 75)
            # 충전 상태에 따라 별도의 전략을 실행
            if SoC_state == 0: # Rest > 대기
                options = {"thermal": "lumped"}
                model = pybamm.lithium_ion.SPMe(options)
                params = pybamm.ParameterValues("Chen2020").copy()
                init_input = {
                    'Number of cells connected in series to make a battery': 4164,
                    'Upper voltage cut-off [V]': 5,
                }
                params.update(init_input)
                ChargeMethod = "Rest"
                experiment = pybamm.Experiment([f"{ChargeMethod} at {CargeRate:.2f}C for {int(time)} min"])
                sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
                step_solution = sim.solve(starting_solution=solutions[-1].last_state)

                Q = params["Nominal cell capacity [A.h]"]
                DC = step_solution["Discharge capacity [A.h]"].entries[-1]

                SoC_list = list(SoC - (step_solution["Discharge capacity [A.h]"].entries) / Q)
                SoC_ = SoC - (step_solution["Discharge capacity [A.h]"].entries[-1]) / Q
                SoC = SoC_
                solutions += [step_solution.last_state]
                SoC_imin_list.append(SoC_list)
                 
            elif SoC_state < 0: # 방전 (고객에세 배터리 충전)
                options = {"thermal": "lumped"}
                model = pybamm.lithium_ion.SPMe(options)
                params = pybamm.ParameterValues("Chen2020").copy()
                init_input = {
                    'Number of cells connected in series to make a battery': 4164,
                    'Upper voltage cut-off [V]': 5,
                }
                params.update(init_input)
                ChargeMethod = "Discharge"
                CargeRate = -CargeRate
                experiment = pybamm.Experiment([f"{ChargeMethod} at {CargeRate:.2f}C for {int(time)} min"])
                sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
                step_solution = sim.solve(starting_solution=solutions[-1].last_state)

                Q = params["Nominal cell capacity [A.h]"]
                DC = step_solution["Discharge capacity [A.h]"].entries[-1]
                # print(sum(step_solution["Discharge capacity [A.h]"].entries) / Q)
                SoC_ = SoC - (step_solution["Discharge capacity [A.h]"].entries[-1]) / Q
                SoC_list = list(SoC - step_solution["Discharge capacity [A.h]"].entries / Q)
                SoC = SoC_
                SoC_imin_list.append(SoC_list)
                solutions += [step_solution.last_state]
                
            elif SoC_state > 0: # 충전용 배터리를 강화학습으로 진행
                env = BM(SoC,time,SoC + SoC_state)
                vec_env = make_vec_env(lambda: env, n_envs=1)
                model = PPO.load("batt_3")
                obs = vec_env.reset()
                method_SoC = []
                while True:
                    action, _states = model.predict(obs)
                    observation, reward, terminated,  info  = vec_env.step(action)
                    method_SoC.append(observation[0][0])
                    # Handle the end of an episode
                    if terminated == True:  # Check if any environment in the vectorized env is done
                        break   
                SoC = max(method_SoC)
                SoC = min(SoC,1)
                method_solution_list = env.get_last_solution()
                method_SoC_imin = method_SoC[0::2]
                SoC_imin_list.append(method_SoC_imin)
                step_solution = method_solution_list[-2]

                solutions += [step_solution.last_state]
        #=====================================================================================================================

        SoC_imin_list_1 = sum(SoC_imin_list, [])
        del BatterySituation
        return SoC_imin_list_1
    
    # 원형 그래프를 통해서 표시할 데이터를 생성
    def make_battery_data(self):
        for i in range(1,6): # 배터리 개수 별로 csv를 생성
            BattID = i
            SoC_imin_list_1 = self.Simulation(BattID)
            # 현재 날짜와 시간 얻기
            now = datetime.now()

            # 년, 월, 일 추출
            current_year = now.year
            current_month = now.month
            current_day = now.day

            start_time = pd.Timestamp(f'{current_year}-{current_month}-{current_day} 00:00')
            i  = 1440
            date_range = [start_time + pd.Timedelta(minutes=i) for i in range(1440)]
            date_baterry_state = pd.DataFrame()
            date_baterry_state['Timestamp'] = date_range
            date_baterry_state['State'] = None
            # 리스트의 길이를 데이터프레임의 길이에 맞추기 위해 마지막 값을 반복 사용
            if len(SoC_imin_list_1) < len(date_baterry_state):
                last_value = SoC_imin_list_1[-1]
                SoC_imin_list_1_extended = SoC_imin_list_1 + [last_value] * (len(date_baterry_state) - len(SoC_imin_list_1))
            else:
                SoC_imin_list_1_extended = SoC_imin_list_1[:len(date_baterry_state)]

            date_baterry_state['state'] = SoC_imin_list_1_extended
            date_baterry_state.to_csv(f'./data/battery_{BattID}.csv', index=False)


# #29 주석
# app = Flask(__name__)

# 5개의 CSV 파일 로드
battery_dfs = [pd.read_csv(f'./data/battery_{i}.csv', parse_dates=['Timestamp']) for i in range(1, 6)]
battery_indices = [0] * len(battery_dfs)  # 각 배터리의 현재 인덱스를 저장하는 리스트

def get_battery_state(battery_index):
    df = battery_dfs[battery_index]
    index = battery_indices[battery_index]
    if index >= len(df):
        index = 0  # 인덱스가 데이터프레임의 길이를 초과하면 다시 처음부터 시작
    state = df.iloc[index]['state']
    battery_indices[battery_index] = index + 1  # 다음 호출 시 사용할 인덱스를 업데이트
    return state

@app.route('/battery/<int:battery_index>')
def battery_state(battery_index):
    if battery_index < 0 or battery_index >= len(battery_dfs):
        return jsonify({"error": "Invalid battery index"}), 404
    state = get_battery_state(battery_index)
    return jsonify([state])

# //#29 배터리 강화학습 코드 통합 - 여기까지
#===================================================================================================


@app.route('/calculate_path', methods=['POST'])
def calculate_path():
        # import eve_0521_test
    try:
        # //#28 내가 불러올 데이터
        datafile = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/orderData/S_03.txt"

        # //#28 내가 지정하는 경로 (파일 저장)
        # //#28 fix: pickle_path 코드 수정 - 주석 처리
        # pickle_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/all_k_shortest_paths.pickle_S_02"
        battery_csv_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"
        truck_csv_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"

        # eve_0522_test3.solve(datafile, pickle_path, battery_csv_path, truck_csv_path) # //#28 fix: pickle_path 코드 수정 - 주석 처리 
        eve_0523_test1.solve(datafile, battery_csv_path, truck_csv_path)

        # Assuming there's a function in eve_0522_test.py to calculate the path and save a CSV
        # result = eve_0522_test.calculate_and_save()
        return jsonify({'success': True, 'message': '경로 계산 및 CSV 저장 완료!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

    
@app.route('/')
def administrator_page():
    return render_template('screen_Administrator.html')


if __name__ == '__main__':
    app.run(port=5004,debug=True)