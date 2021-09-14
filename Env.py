# Import routines
from itertools import permutations, product
import numpy as np
import random


class CabDriver:
    """Environment Class for Cab agent"""

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # initialising hyperparameters
        self.m = 5  # number of cities, ranges from 1 ..... m
        self.t = 24  # number of hours, ranges from 0 .... t-1
        self.d = 7  # number of days, ranges from 0 ... d-1
        self.C = 5  # Per hour fuel and other costs
        self.R = 9  # per hour revenue from a passenger

        # noinspection PyTypeChecker
        self.action_space = [(0, 0)] + list(permutations([loc for loc in range(self.m)], 2))
        self.state_space = self.__initialize_state_space()
        self.state_init = random.choice(self.state_space)

        # start the first round
        self.reset()

    def __initialize_state_space(self) -> list:
        """
        Initializing the state space for current episode.
        Current state space is represented as (current_location, current_time, current_day)
        """
        return list(
            product([loc for loc in range(self.m)], [tm for tm in range(self.t)], [day for day in range(self.d)])
        )

    # encoding state (or state-action) for NN input
    def state_encode_arch1(self, state) -> list:
        """
        convert the state into a vector so that it can be fed to the NN.
        This method converts a given state into a vector format.
        returns: The vector of size m + t + d.
        """
        state_encoder = [0 for _ in range((self.m + self.t + self.d))]
        # state in form = (location, time, day)
        state_encoder[state[0]] = 1
        state_encoder[self.m + state[1]] = 1
        state_encoder[self.m + self.t + state[2]] = 1
        return state_encoder

    # use this function if using architecture-2: states and actions as input
    # def state_encode_arch2(self, state, action):
    #     """ We are not using this architecture.
    #     convert the (state-action) into a vector so that it can be fed to the NN.
    #     This method converts a given state-action pair into a vector format.
    #     returns: The vector of size m + t + d + m + m. (last 2 m signifies action space length)
    #     """
    #     state_encoder = [0 for _ in range((self.m + self.t + self.d + self.m + self.m))]
    #     # state in form = (location, time, day)
    #     state_encoder[state[0]] = 1
    #     state_encoder[state[1]] = 1
    #     state_encoder[state[2]] = 1
    #     # action in form = (pickup, drop)
    #     if action[0] != 0:
    #         state_encoder[self.m + self.t + self.d + action[0]] = 1
    #
    #     if action[1] != 0:
    #         state_encoder[self.m + self.t + self.d + self.m + action[1]] = 1
    #
    #     return state_encoder

    # get number of requests per location i.e., at current location
    def get_requests(self, current_state) -> tuple:
        """
        Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations.
        """
        current_location = current_state[0]
        lambda_distribution = [2, 12, 4, 7, 8]

        # get number of requests based on poisson distribution for specific lambda value
        num_of_requests = np.random.poisson(lambda_distribution[current_location])

        # setting limit for maximum number of possible requests to 15
        if num_of_requests > 15:
            num_of_requests = 15

        # (0,0) is not considered as customer request. The cab drive is free to reject all customer requests.
        possible_actions_index = random.sample(range(1, ((self.m - 1) * self.m) + 1), num_of_requests) + [0]
        allowed_actions = [self.action_space[i] for i in possible_actions_index]
        # allowed_actions.append([0, 0])   # hence adding the index of action (0,0)
        return possible_actions_index, allowed_actions

    def get_next_state(self, state: tuple, action: tuple, time_matrix) -> tuple:
        """Takes state and action as input and returns next state"""
        current_location, current_time, current_day = state
        pickup_location, drop_location = action
        total_time, transit_time = 0, 0
        wait_time, ride_time = 0, 0

        # three possible scenarios:
        # 1) Reject all customer requests
        # 2) Driver already present at pickup location
        # 3) Driver not present at pickup location
        if pickup_location == 0 and drop_location == 0:
            # refuse all customer requests, wait time is 1hr and next location is current location
            wait_time = 1
            next_location = current_location
        elif current_location == pickup_location:
            # driver already present at pickup location. Hence wait and transit time are 0.
            ride_time = time_matrix[current_location][drop_location][current_time][current_day]
            next_location = drop_location  # next location is drop location
        else:
            # Driver not at pickup location and he needs to travel to pickup location.
            # transit time taken to reach the pickup location
            transit_time = time_matrix[current_location][pickup_location][current_time][current_day]
            new_time, new_day = self.__revised_time_day(current_time, current_day, transit_time)

            # after reaching the pickup location, time taken to drop the customer
            ride_time = time_matrix[pickup_location][drop_location][new_time][new_day]
            next_location = drop_location  # next location is drop location

        # calculating total ride time including the wait time and transit time to pickup and drop location
        total_time = wait_time + ride_time + transit_time
        next_time, next_day = self.__revised_time_day(current_time, current_day, total_time)

        # building the next state list
        next_state = [next_location, next_time, next_day]
        return next_state, wait_time, transit_time, ride_time

    # noinspection PyMethodMayBeStatic
    def __revised_time_day(self, time: int, day: int, ride_duration: int) -> tuple:
        """Calculate the revised day and time after transit"""
        ride_duration = int(ride_duration)
        if time + ride_duration < 24:
            # duration within the same day
            time = time + ride_duration  # day is unchanged here
        else:
            # duration spreading to the next day
            time = (time + ride_duration) % 24  # converting time to within 0-23 range
            num_days = (time + ride_duration) // 24  # getting number of days
            day = (day + num_days) % 7  # converting day to within 0-6 range
        return time, day

    def get_reward(self, wait_time: int, transit_time: int, ride_time: int) -> int:
        """Takes in wait_time, transit_tim and Time-ride_time and returns the reward"""
        # transit and wait time has no rewards/revenue. It results only in battery cost, hence they are idle time.
        idle_time = wait_time + transit_time
        # ride time is the passenger's travel time
        reward = (self.R * ride_time) - (self.C * (ride_time + idle_time))
        return reward

    def step(self, state: tuple, action: tuple, time_matrix) -> tuple:
        """Taking a ride to get rewards, next state and total time taken for the ride"""
        # getting the next state and the time involved
        next_state, wait_time, transit_time, ride_time = self.get_next_state(state, action, time_matrix)

        # calculating the reward for the ride taken and time involved
        reward = self.get_reward(wait_time, transit_time, ride_time)
        total_time = wait_time + transit_time + ride_time

        return reward, next_state, total_time

    def reset(self):
        return self.action_space, self.state_space, self.state_init
