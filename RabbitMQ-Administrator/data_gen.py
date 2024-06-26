import os
import glob
import pandas as pd
import numpy as np
import pybamm
from datetime import datetime
import ray
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

class BM(gym.Env):
    def __init__(self, start_soc, goal_time, goal_soc, render_mode=None):
        options = {"thermal": "lumped"}
        self.model = pybamm.lithium_ion.SPMe(options)
        self.params = pybamm.ParameterValues("Chen2020").copy()
        init_input = {
            'Number of cells connected in series to make a battery': 5210,
            'Upper voltage cut-off [V]': 5,
        }
        self.params.update(init_input)
        
        self.solutions = []
        self.Current_list = []
        self.SoC_list = []
        self.volt_list = []
        self.temp_list = []

        self.r_max_temp = 273 + 35
        self.r_max_volt = 4.2
        self.SoC_desired = goal_soc

        self.volt = 0
        self.temp = 0
        self.curr = 0
        
        

        self.ep_num = 0
        self.time_step = 0
        self.MAX_time_step = 3600*2

        self.time_goal = goal_time
        self.init_soc = start_soc

        self.observation_space = spaces.Box(low=0, high=400, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(30)

    def step(self, action):
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
            self.solutions.append(step_solution.last_state)

            if self.time_step >= self.MAX_time_step:
                terminated = True   
                reward = -10000

            self.temp = step_solution["X-averaged cell temperature [K]"].entries[-1]
            self.volt = step_solution["Terminal voltage [V]"].entries[-1]
            self.curr = step_solution['Current [A]'].entries[-1]
            Q = self.params["Nominal cell capacity [A.h]"]
            DC = step_solution["Discharge capacity [A.h]"].entries[-1]
            self.SoC = self.SoC-DC/Q

            if self.time_step <= self.time_goal:
                crie = self.SoC > (0.6/self.time_goal)*self.time_step + 0.2
                if crie:
                    r_soc = 0
                else:
                    r_soc = 10 * (self.SoC - ((0.6/self.time_goal)*self.time_step + 0.2))
            else:
                r_soc = (self.time_goal - self.time_step)
            
            r_temp = -5 * abs(self.temp - (273+35)) if self.temp > (273+35) else 0
            r_volt = -200 * abs(self.volt - self.r_max_volt) if self.volt > self.r_max_volt else 0

            reward = r_temp + r_volt + r_soc
            self.time_step += 1
        except:
            reward = -1000
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        self.SoC = self.init_soc
        experiment = pybamm.Experiment(["Rest for 30 min"])
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=self.params)
        step_solution = sim.solve(initial_soc=self.init_soc)
        self.solutions.append(step_solution.last_state)

        observation = self._get_obs()
        info = self._get_info()
        self.ep_num += 1
        self.time_step = 0
        return observation, info

    def get_last_soc(self):
        print("Last_soc:", self.SoC)
        return np.array(self.SoC)
    
    def get_curr(self):
        return self.curr
    
    def get_last_solution(self):
        return self.solutions
    
    def _get_obs(self):
        return np.array([self.SoC, self.volt, self.temp, self.time_goal - self.time_step], dtype=np.float32)

    def _get_info(self):
        return {"distance": self.SoC_desired - self.SoC}

class SimulateBattery_RL:
    def __init__(self):
        ray.init()

    def create_custom_list(self, c_rate, twos_count, zeros_count):
        twos_count = int(twos_count)
        zeros_count = int(zeros_count)
        return [float(c_rate)] * twos_count + [0.0] * zeros_count


    def RoadRoutData(self, BattID):
        directory_path = './data_input'
        csv_pattern = os.path.join(directory_path, '*.csv')
        csv_files = glob.glob(csv_pattern)
        
        csv_files_list = [pd.read_csv(csv_file) for csv_file in csv_files]
        cdv = pd.concat(csv_files_list, ignore_index=True)
        cdv = cdv.sort_values(by='Time')

        battery = cdv[cdv['Battery_Num'] == BattID]

        initial_data = pd.DataFrame({
            'Battery_Num': [BattID],
            'Node': [0],
            'Time': [0],
            'State': [75],
        })

        if len(battery) <= 1:
            additional_data = pd.DataFrame({
                'Battery_Num': [BattID],
                'Node': [0],
                'Time': [2],
                'State': [75],
            })
            battery = pd.concat([battery, additional_data], ignore_index=True)

        battery = pd.concat([battery, initial_data], ignore_index=True)
        battery = battery.sort_values(by='Time')

        BatterySituation = [
            (battery.iloc[i + 1]["Time"] - battery.iloc[i]["Time"], 
             battery.iloc[i + 1]["State"] - battery.iloc[i]["State"])
            for i in range(len(battery) - 1)
        ]

        return BatterySituation

    @ray.remote
    def Simulation(self, BattID):
        BatterySituation = self.RoadRoutData(BattID)

        SoC = 0.8
        solutions = []
        SoC_imin_list = []
        Temp_imin_list = []
        C_imin_list = []
        Volt_imin_list = []

        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        params = pybamm.ParameterValues("Chen2020").copy()
        init_input = {
            'Number of cells connected in series to make a battery': 5210,
        }
        params.update(init_input)
        Q = params["Nominal cell capacity [A.h]"]

        experiment = pybamm.Experiment(["Rest for 1 min"])
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
        step_solution = sim.solve(initial_soc=SoC)
        solutions.append(step_solution)

        for time, SoC_state in BatterySituation:
            charge_time = (30 / 100) * SoC_state

            if SoC_state == 0 or SoC_state < 0:
                if SoC_state == 0:
                    if time == 0:
                        continue
                    else:
                        ChargeMethod = "Rest"
                        experiment = pybamm.Experiment([f"{ChargeMethod} at 2C for {time} min"])
                        C_list = self.create_custom_list(0, 0, time)
                elif SoC_state < 0:
                    ChargeMethod = "Discharge"
                    charge_time = -charge_time
                    experiment = pybamm.Experiment([f"{ChargeMethod} at 2C for {charge_time} min", f"Rest for {time - charge_time} min"])
                    C_list = self.create_custom_list(2, charge_time, time - charge_time)

                sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
                step_solution = sim.solve(starting_solution=solutions[-1].last_state)
                solutions.append(step_solution)

                SoC_list = list(SoC - step_solution["Discharge capacity [A.h]"].entries / Q)
                Temp_list = step_solution["X-averaged cell temperature [K]"].entries
                Volt_list = step_solution['Voltage [V]'].entries

                SoC_imin_list.extend(SoC_list[:-1])
                Temp_imin_list.extend(Temp_list[:-1])
                C_imin_list.extend(C_list[:])
                Volt_imin_list.extend(Volt_list[:-1])

                DC = step_solution["Discharge capacity [A.h]"].entries[-1]
                SoC -= DC / Q
            else:
                env = BM(SoC, time, SoC + SoC_state/100)
                vec_env = make_vec_env(lambda: env, n_envs=1)
                RL_model = PPO.load("batt_0526_long")
                obs = vec_env.reset()
                method_SoC = []
                method_volt = []
                method_temp = []
                method_crate = []
                while True:
                    action, _states = RL_model.predict(obs)
                    observation, reward, terminated,  info = vec_env.step(action)
                    method_SoC.append(observation[0][0])
                    method_volt.append(observation[0][1])
                    method_temp.append(observation[0][2])
                    method_crate.append(-int(action[0])/10)

                    if terminated:
                        break

                SoC = max(method_SoC)
                SoC = min(SoC, 1)

                method_solution_list = env.get_last_solution()
                method_SoC_imin = method_SoC[0::2]
                method_C_imin = method_crate[0::2]
                method_temp_imin = method_temp[0::2]
                method_volt_imin = method_volt[0::2]

                SoC_imin_list.extend(method_SoC_imin[:-1])
                C_imin_list.extend(method_C_imin[:])
                Temp_imin_list.extend(method_temp_imin[:-1])
                Volt_imin_list.extend(method_volt_imin[:-1])
                
                step_solution = method_solution_list[-2]
                solutions.append(step_solution.last_state)

                remain_time = time - len(method_SoC_imin)
                if remain_time > 0:
                    experiment = pybamm.Experiment([f"Rest for {remain_time} min"])
                    C_list = self.create_custom_list(0, 0, remain_time)
                    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
                    step_solution = sim.solve(starting_solution=solutions[-1].last_state)
                    solutions.append(step_solution)

                    SoC_list = list(SoC - step_solution["Discharge capacity [A.h]"].entries / Q)
                    Temp_list = step_solution["X-averaged cell temperature [K]"].entries
                    Volt_list = step_solution['Voltage [V]'].entries

                    SoC_imin_list.extend(SoC_list[:-1])
                    Temp_imin_list.extend(Temp_list[:-1])
                    C_imin_list.extend(C_list[:])
                    Volt_imin_list.extend(Volt_list[:-1])

                    DC = step_solution["Discharge capacity [A.h]"].entries[-1]
                    SoC -= DC / Q

        return SoC_imin_list, C_imin_list, Volt_imin_list, Temp_imin_list

    def extend_list(self, lst, target_len):
        return lst + [lst[-1]] * (target_len - len(lst))

    def make_battery_data(self):
        results = []
        for BattID in range(1, 9):
            results.append(self.Simulation.remote(self, BattID))

        outputs = ray.get(results)

        for BattID, (SoC_imin_list_1, C_imin_list, Volt_imin_list_1, Temp_imin_list_1) in enumerate(outputs, start=1):
            now = datetime.now()
            start_time = pd.Timestamp(f'{now.year}-{now.month}-{now.day} 00:00')
            date_range = [start_time + pd.Timedelta(minutes=i) for i in range(2200)]
            date_battery_state = pd.DataFrame({'Timestamp': date_range})

            date_battery_state['state'] = self.extend_list(SoC_imin_list_1, len(date_battery_state))
            date_battery_state['c_rate'] = self.extend_list(C_imin_list, len(date_battery_state))
            date_battery_state['volt'] = self.extend_list(Volt_imin_list_1, len(date_battery_state))
            date_battery_state['temp'] = self.extend_list(Temp_imin_list_1, len(date_battery_state))
            
            date_battery_state.to_csv(f'./RL_data/batt{BattID}.csv', index=False)

class SimulateBattery_CC:
    def __init__(self):
        ray.init()

    def create_custom_list(self, c_rate, twos_count, zeros_count):
        twos_count = int(twos_count)
        zeros_count = int(zeros_count)
        return [float(c_rate)] * twos_count + [0.0] * zeros_count


    def RoadRoutData(self, BattID):
        directory_path = './data_input'
        csv_pattern = os.path.join(directory_path, '*.csv')
        csv_files = glob.glob(csv_pattern)
        
        csv_files_list = [pd.read_csv(csv_file) for csv_file in csv_files]
        cdv = pd.concat(csv_files_list, ignore_index=True)
        cdv = cdv.sort_values(by='Time')

        battery = cdv[cdv['Battery_Num'] == BattID]

        initial_data = pd.DataFrame({
            'Battery_Num': [BattID],
            'Node': [0],
            'Time': [0],
            'State': [75],
        })

        if len(battery) <= 1:
            additional_data = pd.DataFrame({
                'Battery_Num': [BattID],
                'Node': [0],
                'Time': [2],
                'State': [75],
            })
            battery = pd.concat([battery, additional_data], ignore_index=True)

        battery = pd.concat([battery, initial_data], ignore_index=True)
        battery = battery.sort_values(by='Time')

        BatterySituation = [
            (battery.iloc[i + 1]["Time"] - battery.iloc[i]["Time"], 
             battery.iloc[i + 1]["State"] - battery.iloc[i]["State"])
            for i in range(len(battery) - 1)
        ]

        return BatterySituation
    
    @ray.remote
    def Simulation(self, BattID):
        BatterySituation = self.RoadRoutData(BattID)

        SoC = 0.8
        solutions = []
        SoC_imin_list = []
        Temp_imin_list = []
        C_imin_list = []
        Volt_imin_list = []

        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options)
        params = pybamm.ParameterValues("Chen2020").copy()
        init_input = {
            'Number of cells connected in series to make a battery': 5210,
        }
        params.update(init_input)
        Q = params["Nominal cell capacity [A.h]"]

        experiment = pybamm.Experiment(["Rest for 1 min"])
        sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
        step_solution = sim.solve(initial_soc=SoC)
        solutions.append(step_solution)

        for time, SoC_state in BatterySituation:
            charge_time = (30 / 100) * SoC_state

            if SoC_state == 0:
                if time == 0:
                    continue
                else:
                    ChargeMethod = "Rest"
                    experiment = pybamm.Experiment([f"{ChargeMethod} at 2C for {time} min"])
                    C_list = self.create_custom_list(0, 0, time)
            elif SoC_state < 0:
                ChargeMethod = "Discharge"
                charge_time = -charge_time
                if time - charge_time > 0:
                    experiment = pybamm.Experiment([f"{ChargeMethod} at 2C for {charge_time} min", f"Rest for {time - charge_time} min"])
                    C_list = self.create_custom_list(2, charge_time, time - charge_time)
                else:
                    experiment = pybamm.Experiment([f"{ChargeMethod} at 2C for {charge_time} min"])
                    C_list = self.create_custom_list(2, charge_time)
            else:
                ChargeMethod = "Charge"
                if time - charge_time > 0:
                    experiment = pybamm.Experiment([f"{ChargeMethod} at 2C for {charge_time} min", f"Rest for {time - charge_time} min"])
                    C_list = self.create_custom_list(-2, charge_time, time - charge_time)
                else:
                    experiment = pybamm.Experiment([f"{ChargeMethod} at 2C for {charge_time} min"])
                    C_list = self.create_custom_list(-2, charge_time)

            sim = pybamm.Simulation(model, experiment=experiment, parameter_values=params)
            step_solution = sim.solve(starting_solution=solutions[-1].last_state)
            solutions.append(step_solution)

            SoC_list = list(SoC - step_solution["Discharge capacity [A.h]"].entries / Q)
            Temp_list = step_solution["X-averaged cell temperature [K]"].entries
            Volt_list = step_solution['Voltage [V]'].entries

            SoC_imin_list.extend(SoC_list[:-1])
            Temp_imin_list.extend(Temp_list[:-1])
            C_imin_list.extend(C_list[:])
            Volt_imin_list.extend(Volt_list[:-1])

            DC = step_solution["Discharge capacity [A.h]"].entries[-1]
            SoC -= DC / Q

        return SoC_imin_list, C_imin_list, Volt_imin_list, Temp_imin_list

    def extend_list(self, lst, target_len):
        return lst + [lst[-1]] * (target_len - len(lst))

    def make_battery_data(self):
        results = []
        for BattID in range(1, 9):
            results.append(self.Simulation.remote(self, BattID))

        outputs = ray.get(results)

        for BattID, (SoC_imin_list_1, C_imin_list, Volt_imin_list_1, Temp_imin_list_1) in enumerate(outputs, start=1):
            now = datetime.now()
            start_time = pd.Timestamp(f'{now.year}-{now.month}-{now.day} 00:00')
            date_range = [start_time + pd.Timedelta(minutes=i) for i in range(1440)]
            date_battery_state = pd.DataFrame({'Timestamp': date_range})

            date_battery_state['state'] = self.extend_list(SoC_imin_list_1, len(date_battery_state))
            date_battery_state['c_rate'] = self.extend_list(C_imin_list, len(date_battery_state))
            date_battery_state['volt'] = self.extend_list(Volt_imin_list_1, len(date_battery_state))
            date_battery_state['temp'] = self.extend_list(Temp_imin_list_1, len(date_battery_state))
            
            date_battery_state.to_csv(f'./data/batt{BattID}.csv', index=False)

if __name__ == "__main__":
    try:
        print("start_cc")
        simulate_battery_cc = SimulateBattery_CC()
        simulate_battery_cc.make_battery_data()
        ray.shutdown()
    except:
        ray.shutdown()
        simulate_battery_cc = SimulateBattery_CC()
        simulate_battery_cc.make_battery_data()
        ray.shutdown()

    try:
        print("start")
        simulate_battery_rl = SimulateBattery_RL()
        simulate_battery_rl.make_battery_data()
        ray.shutdown()
    except:
        ray.shutdown()
        simulate_battery_rl = SimulateBattery_RL()
        simulate_battery_rl.make_battery_data()
        ray.shutdown()