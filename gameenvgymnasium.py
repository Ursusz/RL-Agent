# from gym import Env
import gymnasium as gym
# from gym.spaces import Discrete, Box, Dict as GymDict
from gymnasium.spaces import Discrete, Box, Dict as GymDict
import numpy as np
import random
import requests
from typing import Optional, Any, Dict
import time

class GameEnv(gym.Env):
    def __init__(self, action_space_size: int):
        super(GameEnv, self).__init__()

        # HTTP methods to communicate with the game
        self.server = "http://127.0.0.1:5002"
        self.get_state_url = self.server + "/get_state"
        self.send_action_url = self.server + "/send_action"
        self.reset_game_url = self.server + "/reset_game"

        # Actions we can take, up, down, left, right, reset
        self.action_space = Discrete(action_space_size)

        # The amount of steps of each episode, for reward
        self.step_count = 0

        # Computers coordinates, static
        self.computer_locations = [
            (360, 168),
            (1704, 168),
            (888, 552),
            (360, 936),
            (1704, 936)
        ]

        # Traps locations, static
        self.traps_locations = [
            (1512, 216),
            (504, 792)
        ]
        
        # The observation space with dynamic data, player coords, zombies coords, the distance to zombies and the completion of each computer
        self.observation_space = Box(
            low=np.array([0.0]*16, dtype=np.float32),
            high=np.array([1920.0, 1080.0, 100.0] + [1920.0, 1080.0]*4 + [12.0]*5, dtype=np.float32),
            dtype=np.float32
        )

        self.previous_state: Optional[Dict[str, Any]] = None

    def step(self, action):
        action_data = {"action": int(action)}
        # print("\n\n", action_data)
        try:
            # Send HTTP method with the action that the agent takes
            requests.post(self.send_action_url, json=action_data)
        except requests.exceptions.ConnectionError:
            print("Error connecting to the server")
            # Observation space all zeros, -100 reward, done=True, info={}
            return np.zeros(self.observation_space.shape, dtype=np.float32), -100.0, True, False, {}

        
        try:
            # Get the current game state
            state_response = requests.get(self.get_state_url)
            current_state = state_response.json()

            if current_state['player']['hp'] <= 0:
                return np.zeros(self.observation_space.shape, dtype=np.float32), -50, True, False, {}
        except requests.exceptions.ConnectionError:
            print("Error connecting to the server")
            # Observation space all zeros, -100 reward, done=True, info={}
            return np.zeros(self.observation_space.shape, dtype=np.float32), -100.0, True, False, {}

        reward = self._calculate_reward(self.previous_state, current_state)
        new_observation = self._process_state_data(current_state)
        self.previous_state = current_state

        done = current_state["player"]["hp"] <= 0
        # print(f"\n[STEP] Player HP: {current_state['player']['hp']} -> Done: {done}\n")
        info = {}
        terminated = done
        truncated = False

        return new_observation, reward, terminated, truncated, info

    def render(self):
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        info = {}
        max_retries = 10
        for attempt in range(max_retries):
            try:
                requests.post(self.reset_game_url)
                time.sleep(0.1)  # dă timp serverului să proceseze reset-ul
                state_response = requests.get(self.get_state_url)
                initial_state = state_response.json()

                if initial_state['player']['hp'] > 0:
                    self.previous_state = initial_state
                    self.step_count = 0
                    return self._process_state_data(initial_state), info
                # else:
                    # print(f"[RESET] Attempt {attempt+1}: Invalid HP ({initial_state['player']['hp']}). Retrying...")

            except requests.exceptions.ConnectionError:
                print("Error connecting to the server")
                return np.zeros(self.observation_space.shape, dtype=np.float32), info

        print("Failed to reset with valid HP after multiple attempts.")
        return np.zeros(self.observation_space.shape, dtype=np.float32), info
    
    def _calculate_reward(self, previous_state: Optional[Dict[str, Any]], current_state: Optional[Dict[str, Any]]) -> float:
        if previous_state is None:
            return 0.0
        
        reward = 0.0

        if self.step_count > 1000 and current_state['player']['hp'] > 0:
            reward += 10.0
        
        player_pos = (current_state['player']['x'], current_state['player']['y'])
        player_prev_pos = (previous_state['player']['x'], previous_state['player']['y'])

        if np.linalg.norm(np.array(player_pos) - np.array(player_prev_pos)) < 0.1:
            reward -= 0.1

        for comp_loc in self.computer_locations:
            curr_distance_to_comp = np.linalg.norm(np.array(player_pos) - np.array(comp_loc))
            prev_distance_to_comp = np.linalg.norm(np.array(player_prev_pos) - np.array(comp_loc))
            if curr_distance_to_comp < 150:
                reward += 0.1
            # if curr_distance_to_comp < prev_distance_to_comp:
            #     reward += 0.3
        
        for i in range(len(current_state['computers'])):
            current_completion = current_state['computers'][i]['completion']
            previous_completion = 0.0

            if i < len(previous_state['computers']):
                previous_completion = previous_state['computers'][i]['completion']

            if current_completion > previous_completion:
                reward += 4.0
                # print("\n Stau langa PC")
            if current_completion == 12 and previous_completion < 12:
                reward += 15.0

        all_completed = True
        for computer in current_state['computers']:
            if computer['completion'] < 12:
                all_completed = False
                break
        if all_completed:
            reward += 100.0

        zombie_penalties = 0
        for i in range(len(current_state['zombies'])):
            if i < len(previous_state['zombies']):
                zombie_current_pos = np.array([current_state['zombies'][i]['x'], current_state['zombies'][i]['y']])
                zombie_prev_pos = np.array([previous_state['zombies'][i]['x'], previous_state['zombies'][i]['y']])

                current_dist_to_zombie = np.linalg.norm(player_pos - zombie_current_pos)
                prev_dist_to_zombie = np.linalg.norm(player_prev_pos - zombie_prev_pos)

                if current_dist_to_zombie < prev_dist_to_zombie and current_dist_to_zombie < 500:
                    zombie_penalties -= 0.1
                elif current_dist_to_zombie > prev_dist_to_zombie and current_dist_to_zombie > 500:
                    zombie_penalties += 0.1
                # if current_dist_to_zombie < 100:
                #     zombie_penalties -= 10.0

        reward += zombie_penalties

        hp_loss = current_state['player']['hp'] - previous_state['player']['hp']
        # print(current_state['player']['hp'] , " ----- " , previous_state['player']['hp'])
        if hp_loss > 0:
            reward -= hp_loss * 0.5
            # print("\n PIERD HP")
        if current_state['player']['hp'] <= 0:
            reward -= 20.0

        return reward


    def _process_state_data(self, current_state: Optional[Dict[str, Any]]) -> np.ndarray:
        if current_state is None:
            # done = True
            # reward = -100.0
            # new_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            # info = {}
            # return new_observation, reward, done, info
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        player_obs = np.array([
            current_state['player']['x'],
            current_state['player']['y'],
            current_state['player']['hp']
        ], dtype=np.float32)
        
        zombie_obs = []
        zombies = current_state.get('zombies', [])

        for i in range(4):
            if i < len(zombies):
                zombie_obs.append(zombies[i]['x'])
                zombie_obs.append(zombies[i]['y'])
            else:
                zombie_obs.extend([0.0, 0.0])  # padding

        zombie_obs = np.array(zombie_obs, dtype=np.float32)

        # completion_obs = np.array(current_state['computer_completion'], dtype=np.float32)

        completion = current_state.get('computer_completion', [])
        completion_obs = np.array(completion + [0.0]*(5 - len(completion)), dtype=np.float32)
        # return {
        #     "player": player_obs,
        #     "zombies": zombie_obs,
        #     "computer_completion": completion_obs
        # }
        return np.concatenate([player_obs, zombie_obs, completion_obs]).astype(np.float32)