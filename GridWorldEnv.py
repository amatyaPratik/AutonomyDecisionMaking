# Standard Library Import
from enum import Enum
import os
from typing import List, Optional

# Third-Party Imports
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import keyboard


class StateType(Enum):
    """
    Represents different state types in the environment.
    """
    HELL = 'hell'
    PATH = 'path'
    OBSTACLE = 'obstacle'
    HELPER = 'helper'
    GOAL = 'goal'


class SpecialAbilities(Enum):
    """
    Enumerates special abilities available to certain states.
    """
    RESET = "reset"
    FLY = "fly"
    CATCH_ANY_POKEMON = "catch_em_all"
    IGNORES_PROXIMITY_DAMAGE_IF_ASLEEP = "ignore_proximity_damage_if_asleep"


class ExtendedStateType(Enum):
    """
    Represents extended state types with additional properties like rewards, images, and special abilities.
    """
    PATH_SANDY_1 = (StateType.PATH, [0], "free-path-1.png", None)
    PATH_SANDY_2 = (StateType.PATH, [0], "free-path-2.png", None)
    PATH_GRASSY_1 = (StateType.PATH, [0], "free-path.png", None)
    PATH_GRASSY_2 = (StateType.PATH, [0], "bush-sm.png", None)

    OBSTACLE_FENCE = (StateType.OBSTACLE, [0], "fence-1.png", None)
    OBSTACLE_SIGN_BOARD = (StateType.OBSTACLE, [0], "sign-board.png", None)
    OBSTACLE_BUSH_BIG = (StateType.OBSTACLE, [0], "bush-lg.png", None)
    OBSTACLE_TREE_SMALL = (StateType.OBSTACLE, [0], "obstacle-2.png", None)
    OBSTACLE_TREE_BIG_TOP = (StateType.OBSTACLE, [0], "tree-2-top.png", None)
    OBSTACLE_TREE_BIG_BOTTOM = (
        StateType.OBSTACLE, [0], "tree-2-bottom.png", None)

    HELL_STATE_CACTUS = (StateType.HELL, [-1], "cactus-2.png", None)
    HELL_STATE_WATER_LEFT = (StateType.HELL, [-1], "puddle-left.png", None)
    HELL_STATE_WATER_RIGHT = (StateType.HELL, [-1], "puddle-right.png", None)
    HELL_STATE_POISONOUS_FLOWER = (
        StateType.HELL, [-2], "poisonous_flower.png", None)
    HELL_STATE_SNORLAX = (StateType.HELL, [-5, -3], "snorlax-sleeping.gif",
                          SpecialAbilities.IGNORES_PROXIMITY_DAMAGE_IF_ASLEEP)
    HELL_STATE_HAUNTER = (StateType.HELL, [-7, -3, -1], "haunter.gif", None)
    HELL_STATE_ABRA = (StateType.HELL, [-9, -4, -2], "abra.gif", None)
    HELL_STATE_MEWTWO = (
        StateType.HELL, [-8, -6], "mewtwo.gif", SpecialAbilities.RESET)

    HELPER_STATE_BIRD = (StateType.HELPER, [
                         1], "pidgeot.gif", SpecialAbilities.FLY)
    HELPER_STATE_MASTER_BALL = (
        StateType.HELPER, [3], "master_ball.png", SpecialAbilities.CATCH_ANY_POKEMON)

    GOAL = (StateType.GOAL, [20], "pikachu.gif", None)

    def __new__(cls, state_type: StateType, rewards: List[int], image: str, special_ability: Optional[SpecialAbilities]):
        obj = object.__new__(cls)
        obj._value_ = (state_type, rewards, image, special_ability)
        obj.state_type = state_type
        obj.rewards = rewards
        obj.image_location = image
        obj.special_ability = special_ability
        return obj


class State:
    """
    Represents a state in the Grid environment with its type and associated properties.
    """

    def __init__(self, state_type: ExtendedStateType):
        self.state_type = state_type

    def _get_state_info(self):
        """
        Retrieves information about the state.

        Returns:
            tuple: Contains state type, rewards, image location, and special ability.
        """
        return (
            self.state_type,
            self.state_type.rewards,
            self.state_type.image_location,
            self.state_type.special_ability
        )


class GridWorldEnv(gym.Env):
    """
    A class representing a simple DETERMINISTIC agent environment (which is
    extensible to create non-determinisim by giving certain abilities to
    the hell_states)

    Attributes:
        grid_size (int): The size of the grid for the environment.
        FREE_PASS_ONCE (bool): Flag denoting if agent is impervious to damage by any hellstate(Pokemon) until it captures one.
        agent_state (numpy.ndarray): The current state of the agent as coordinates [x, y].
        goal_state (numpy.ndarray): The coordinates [x, y] of the goal state.
        current_cumulative_reward (int): The current reward accumulated in a single step.
        # total_reward (int): The total reward accumulated across all steps.
        goal_reached (bool): A flag indicating if the goal has been reached.
        action_space (gym.spaces.Discrete): The action space of the environment.
        current_frame (int): The current frame number for animation purposes.
        ax (matplotlib.axes.Axes): The axes object for rendering.
        obstacles (list): A list of coordinates for obstacles.
        hell_states (list): A list of tuples containing coordinates, filenames, frames, rewards & special_damages for hell states.
        within_danger_zones (list): A list of tuples indicating danger zones affecting the agent.
        agent_frames (list): Frames for the agent's animation.
        goal_frames (list): Frames for the goal's animation.
        goal_reached_frames (list): Frames for the goal reached animation.
        walkable_path (str): The path to the walkable image file.
        pain_path (str): The path to the pain image file.
        warning_path (str): The path to the warning image file.
    """

    def __init__(self, grid_size=11):
        """
        Initialize the DeterministicEnv environment - properties, states, plots, assets.

        Parameters:
            grid_size (int): The size of the grid for the environment. Default is 11.
        """
        super().__init__()
        # Initialize a 11x11 grid with ExtendedStateType.PATH_GRASSY_1 as default state
        self.grid = self._create_grid(
            grid_size, grid_size, ExtendedStateType.PATH_GRASSY_1)

        # Matrix (position based) Map of the environment (NOT based on Rendering point of view !).
        grid_map = [
            ['PATH_SANDY_1', 'PATH_SANDY_1', 'PATH_SANDY_1', 'PATH_SANDY_1', 'PATH_SANDY_2',
                'HELL_STATE_CACTUS', '', 'OBSTACLE_TREE_SMALL', 'OBSTACLE_TREE_SMALL', '', 'HELPER_STATE_BIRD'],
            ['HELL_STATE_CACTUS', '', 'OBSTACLE_FENCE',
                'OBSTACLE_FENCE', '', '', '', '', 'OBSTACLE_TREE_SMALL', '', ''],
            ['OBSTACLE_SIGN_BOARD', '', '', '', '',
                'OBSTACLE_FENCE',  'OBSTACLE_FENCE', '', '', '', ''],
            ['', '', 'OBSTACLE_TREE_SMALL', 'OBSTACLE_FENCE', '',
                '', '', '', '', '', 'OBSTACLE_TREE_BIG_BOTTOM'],
            ['', '', '', '', '', 'OBSTACLE_TREE_BIG_BOTTOM', 'HELL_STATE_WATER_LEFT',
                'HELL_STATE_WATER_RIGHT', '', '', 'OBSTACLE_TREE_BIG_TOP'],
            ['', 'HELL_STATE_SNORLAX', '', '', 'HELL_STATE_POISONOUS_FLOWER', 'OBSTACLE_TREE_BIG_TOP', '', 'OBSTACLE_TREE_BIG_BOTTOM',
                'OBSTACLE_TREE_BIG_BOTTOM', 'HELL_STATE_HAUNTER', 'OBSTACLE_TREE_BIG_BOTTOM'],
            ['', '', '', 'OBSTACLE_BUSH_BIG', 'OBSTACLE_BUSH_BIG', 'OBSTACLE_TREE_SMALL', '',
                'OBSTACLE_TREE_BIG_TOP', 'OBSTACLE_TREE_BIG_TOP', '', 'OBSTACLE_TREE_BIG_TOP'],
            ['', 'OBSTACLE_TREE_SMALL', '', '', '',
                '', '', '', '', '', ''],
            ['', 'OBSTACLE_TREE_SMALL', 'OBSTACLE_TREE_SMALL', '', '', '', 'OBSTACLE_TREE_BIG_BOTTOM', '',
                'HELL_STATE_MEWTWO', '', 'HELL_STATE_POISONOUS_FLOWER'],
            ['', '', 'OBSTACLE_TREE_SMALL', 'OBSTACLE_TREE_SMALL', '', 'OBSTACLE_BUSH_BIG', 'OBSTACLE_TREE_BIG_TOP', '',
                'OBSTACLE_TREE_SMALL', '', ''],
            ['', 'HELPER_STATE_MASTER_BALL', '', '',
                'HELL_STATE_ABRA', '', '', '', '', '', 'GOAL']
        ]

        for i in range(grid_size):
            for j in range(grid_size):
                self._set_state(
                    j, i, self._get_enum_from_string(grid_map[i][j]))

        self.grid_size = grid_size
        self.FREE_PASS_ONCE = False
        self.agent_state = np.array([0, 0])
        self.goal_state = np.array([10, 10])
        self.current_cumulative_reward = 0
        # self.total_reward = 0
        self.goal_reached = False
        self.action_space = gym.spaces.Discrete(4)
        # self.action_mapping = {
        #     0: 'a',  # Move left
        #     1: 'A',  # Move left (capitalized)
        #     2: 's',  # Move down
        #     3: 'S',  # Move down (capitalized)
        #     4: 'd',  # Move right
        #     5: 'D',  # Move right (capitalized)
        #     6: 'w',  # Move up
        #     7: 'W',  # Move up (capitalized)
        # }

        self.current_frame = 0
        _, self.ax = plt.subplots()
        self._load_images()
        self.obstacles = self._generate_obstacles_list()
        self.hell_states = self._generate_hell_states_list()
        self.helper_states = self._generate_helper_states_list()
        self.walkable_states = self._generate_background()
        self.within_danger_zones = []
        plt.show(block=False)

    def _create_grid(self, rows, cols, default_state_type):
        """
        Create a (grid_size x grid_size) dimension grid with the default_state_type

        Parameters:
            rows (int): The number of rows the grid should have.
            cols (int): The number of columns the grid should have.
            default_state_type (ExtendedStateType): the StateType to set to all the states by default.

        """
        return [[State(default_state_type) for _ in range(cols)] for _ in range(rows)]

    def _set_state(self, row, col, state_type):
        """
        Set the state at a particular grid position to a certain StateType (ExtendedStateType).

        Parameters:
            row (int): The row index of the grid cell to reset.
            col (int): The column index of the grid cell to reset.
            state_type (ExtendedStateType): the StateType to set to the state
        """
        self.grid[row][col] = State(state_type)

    def _reset_state_to_normal(self, row, col):
        """
        Reset the state of a grid cell to the normal grassy path state.

        This method sets the state of the specified grid cell to a normal grassy path state
        (ExtendedStateType.PATH_GRASSY_1) and updates the lists of hell states and helper states
        accordingly.

        Parameters:
            row (int): The row index of the grid cell to reset.
            col (int): The column index of the grid cell to reset.
        """
        self.grid[row][col] = State(ExtendedStateType.PATH_GRASSY_1)
        self.hell_states = self._generate_hell_states_list()
        self.helper_states = self._generate_helper_states_list()

    def _load_images(self):
        """
        Load images for the agent, goal, and other elements of the environment.
        """
        self.agent_frames = imageio.mimread(
            os.path.join('.', 'pokemon', 'characters', 'ash.gif')
        )
        self.goal_frames = imageio.mimread(
            os.path.join('.', 'pokemon', 'characters', 'pikachu.gif')
        )
        self.goal_reached_frames = imageio.mimread(
            os.path.join('.', 'pokemon', 'assets', 'ash_finds_pikachu.gif')
        )
        self.pain_path = os.path.join(
            '.', 'pokemon', 'sprites', 'ouch.png'
        )
        self.warning_path = os.path.join(
            '.', 'pokemon', 'sprites', 'warning.png'
        )

    def _get_current_image_frame(self, img_frames):
        """
        Get the current frame of an image sequence based on the current frame number.

        Parameters:
            img_frames (list): A list containing frames of an image sequence.

        Returns:
            object: The current frame of the image sequence based on the current frame number.
        """
        return img_frames[self.current_frame % len(img_frames)]

    def _degree_of_effective_zone(self, point1, point2):
        """
        Calculate the distance of the surrounding single-cell-thick box (on which point1 lies), 
        from the center (point2) which it surrounds.

        Parameters:
            point1 (numpy.ndarray): The first point as [x, y].
            point2 (numpy.ndarray): The second point as [x, y].

        Returns:
            int: The degree of effect zone as the maximum of the absolute differences of the coordinates.
        """
        return max(abs(point1[0] - point2[0]), abs(point1[1] - point2[1]))

    def _get_enum_from_string(self, string):
        if not string:
            # Return PATH_SANDY_1 if the string is empty
            return ExtendedStateType.PATH_GRASSY_1
        try:
            return ExtendedStateType[string]
        except KeyError:
            return None  # Return None if the string does not match any enum member

    def _generate_obstacles_list(self):
        """
        Generate obstacles list from the pre-fixed grid.

        Returns:
            list: A list of coordinates & filename for the obstacles.
        """
        obstacles = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = self.grid[i][j]
                state_info = state._get_state_info()
                # Access the StateType of the ExtendedStateType of the State
                if (state_info[0].value[0] == StateType.OBSTACLE):
                    filename = state_info[2]
                    obstacles.append(([i, j], filename))
        return obstacles

    def _generate_hell_states_list(self):
        """
        Generate hell states with positions, filenames, frames, rewards and special_damage upon reaching the hell_state.

        Returns:
            list: A list of tuples containing coordinates, filenames, frames, rewards and special_damage for hell states.
        """

        # Description of Fields:
        #   1. Filename: The name of the image file associated with this state.
        #   2. Rewards: A list of rewards for different distances from the agent's position.
        #   3. Special Damage (Optional): Additional special effects or damages associated with this state.

        # Extract the 3rd item from special_ability list if it exists,
        # otherwise assign None to special_damage
        hell_states = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = self.grid[i][j]
                # state_type = state.state_type
                state_info = state._get_state_info()
                # Access the StateType of the ExtendedStateType of the State
                if (state_info[0].value[0] == StateType.HELL):
                    filename = state_info[2]
                    special_damage = state_info[3] if state_info[3] else None
                    rewards = state_info[1]
                    frames = imageio.mimread(os.path.join(
                        '.', 'pokemon', 'characters', state_info[2]))
                    hell_states.append(
                        ([i, j], filename, frames, rewards, special_damage))
        return hell_states

    def _generate_helper_states_list(self):
        helper_states = []

        # Extract the 3rd item from state_info for special_ability if it exists,
        # otherwise assign None to special_damage
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = self.grid[i][j]
                state_info = state._get_state_info()
                # Access the StateType of the ExtendedStateType of the State
                if (state_info[0].value[0] == StateType.HELPER):
                    filename = state_info[2]
                    special_ability = state_info[3] if state_info[3] else None
                    rewards = state_info[1]
                    frames = imageio.mimread(os.path.join(
                        '.', 'pokemon', 'characters', state_info[2]))
                    helper_states.append(
                        ([i, j], filename, frames, rewards, special_ability))
        return helper_states

    def _generate_background(self):
        walkable_states = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                state = self.grid[i][j]
                state_info = state._get_state_info()
                # Access the StateType of the ExtendedStateType of the State
                if (state_info[0].value[0] == StateType.PATH):
                    filename = state_info[2]
                    special_ability = state_info[3]
                    rewards = state_info[1]
                    frames = imageio.mimread(os.path.join(
                        '.', 'pokemon', 'sprites', state_info[2]))
                    walkable_states.append(
                        ([i, j], filename, frames, rewards, special_ability))
                elif (state_info[0].value[0] in [StateType.HELL, StateType.OBSTACLE, StateType.HELPER]):
                    frames = imageio.mimread(os.path.join(
                        '.', 'pokemon', 'sprites', ExtendedStateType.PATH_GRASSY_1.value[2]))
                    walkable_states.append(
                        ([i, j], filename, frames, rewards, special_ability))
        return walkable_states

    def _render_goal(self):
        """
        Render the goal image at the goal's position, centered within the cell.
        """
        goal_pos = self.goal_state + 0.5
        goal_frame = self._get_current_image_frame(self.goal_frames)
        self._display_image(goal_frame, goal_pos)

    def _render_obstacles(self):
        """
        Render image of the obstacle.
        """
        for obs in self.obstacles:
            obs_pos = [x + 0.5 for x in obs[0]]
            obstacle_path = os.path.join(
                '.', 'pokemon', 'sprites', obs[1]
            )
            obs_image = mpimg.imread(obstacle_path)
            self._display_image(obs_image, obs_pos)

    def _render_hell_states(self):
        """
        Render image of the hell_state.
        """
        for pos, _, hell_frames, _, _ in self.hell_states:
            hell_pos = [x + 0.5 for x in pos]
            hell_frame = self._get_current_image_frame(hell_frames)
            self._display_image(hell_frame, hell_pos)

    def _render_helper_states(self):
        """
        Render image of the helper_state.
        """
        for pos, _, helper_frames, _, _ in self.helper_states:
            helper_pos = [x + 0.5 for x in pos]
            helper_frames = self._get_current_image_frame(helper_frames)
            self._display_image(helper_frames, helper_pos)

    def _render_walkable_states(self):
        """
        Render image of the walkable_states.
        """
        for pos, filename, _, _, _ in self.walkable_states:
            self.walkable_path = os.path.join(
                '.', 'pokemon', 'sprites', filename
            )
            # Image for walkable path.
            walkable_image = mpimg.imread(self.walkable_path)
            self._display_image(walkable_image, [pos[0], pos[1]], True)

    def _render_agent(self):
        """
        Render image of the agent at the agent's position, centered within the cell.
        """
        agent_pos = self.agent_state + 0.5
        agent_frame = self._get_current_image_frame(self.agent_frames)
        self._display_image(agent_frame, agent_pos)

    def _render_goal_reached_gif(self):
        """
            Render success image frames when Goal is reached.
        """
        for frame in self.goal_reached_frames:
            self._display_image(frame)
            plt.pause(0.1)

    def _draw_cage_in_goal(self):
        """
            Draw vertical & horizontal bars at the Goal cell.
        """
        goal_pos = self.goal_state + 0.5
        # Draw vertical bars
        for i in np.linspace(goal_pos[0] - 0.5, goal_pos[0] + 0.5, num=4):
            self.ax.plot(
                [i, i], [goal_pos[1] - 0.5, goal_pos[1] + 0.5], color='gray', linewidth=2)

        # Draw horizontal bars
        for i in np.linspace(goal_pos[1] - 0.5, goal_pos[1] + 0.5, num=4):
            self.ax.plot([goal_pos[0] - 0.5, goal_pos[0] + 0.5],
                         [i, i], color='gray', linewidth=2)

    def _set_grid_dimensions(self):
        """
        Configure the axes limits and aspect ratio of the plot.

        This method sets the x and y axis limits to the size of the grid and ensures that the aspect ratio is equal.
        This ensures that the grid cells are square and the entire grid is visible within the plot.

        Sets:
            x-axis limit: 0 to grid_size
            y-axis limit: 0 to grid_size
            aspect ratio: equal (square cells)
        """
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect("equal")

    def _clear_grid(self):
        """
        Clear the current grid display.

        This method clears the current axes by removing all the plotted data and
        turns off the axis lines and labels, effectively resetting the display area.

        Steps:
        1. Clear the current axes using `self.ax.clear()`, which removes all elements
        from the axes.
        2. Turn off the axis lines and labels using `self.ax.axis('off')`, making the
        grid display area blank and free of any axis markers or labels.
        """
        self.ax.clear()
        self.ax.axis('off')

    def _draw_grid(self):
        """
        Draw a grid on the current axes.

        This method draws a grid on the current axes using black lines. The grid
        size is determined by `self.grid_size`. Vertical and horizontal lines are
        drawn to create the grid, with lines spaced one unit apart.

        Note:
        - This method assumes `self.ax` is a Matplotlib Axes object.
        - The grid lines are drawn using the `plot` method of the Axes object with
        a black color.
        """
        for x in range(self.grid_size + 1):
            self.ax.plot([x, x], [0, self.grid_size],
                         color='#88AE8A', linewidth=0.75)
        for y in range(self.grid_size + 1):
            self.ax.plot([0, self.grid_size], [y, y],
                         color='#88AE8A', linewidth=0.75)

    def _display_image(self, img_obj, pos=None, cover_entire_cell=False):
        """
        Display an image at a specified position.

        Parameters:
            img_obj (object): The image object to be displayed.
            pos (list, optional): The position to display the image. Defaults to None.
            cover_entire_cell (bool, optional): Whether to cover the entire cell. Defaults to False.
        """
        if pos is None:
            self.ax.imshow(img_obj)
        elif cover_entire_cell:
            self.ax.imshow(img_obj, extent=[
                           pos[0], pos[0] + 1, pos[1], pos[1] + 1])
        else:
            self.ax.imshow(img_obj, extent=[
                           pos[0] - 0.5, pos[0] + 0.5, pos[1] - 0.5, pos[1] + 0.5])

    def _clear_cell_image(self, pos):
        """
        Clear the image at the specified cell.

        Parameters:
            pos (list): The position of the cell to be cleared.
        """
        # Clear the image at the specified cell
        self.ax.imshow(np.zeros((1, 1)), extent=[
                       pos[0], pos[0] + 1, pos[1], pos[1] + 1])

    def _draw_colored_cell(self, pos, color):
        """
        Clear image in a cell & color it at the specified position.

        Parameters:
            pos (list): The position of the cell.
            color (str): The color to fill the cell with.
        """
        self._clear_cell_image(pos)
        # Add a red rectangle to indicate danger
        self.ax.add_patch(plt.Rectangle((pos[0], pos[1]), 1, 1, color=color))

    def _mark_danger_zone(self, center_pos, effect_radius, agent_pos):
        """
        Mark the danger zones around a specified center position.

        Parameters:
            center_pos (list): The center position of the danger zone.
            effect_radius (int): The bounding box's step count from the center_pos.
            agent_pos (list): The current position of the agent.
        """
        for i in range(
            center_pos[0] - effect_radius, center_pos[0] + effect_radius + 1
        ):
            for j in range(
                center_pos[1] - effect_radius, center_pos[1] +
                    effect_radius + 1
            ):
                if effect_radius == 0:
                    # display pain image when in hell_state
                    pain_image = mpimg.imread(self.pain_path)
                    self._display_image(pain_image, [i, j], True)
                    break
                if (not np.array_equal([i, j], center_pos)
                    and not np.array_equal([i, j], agent_pos)
                    and not any(np.array_equal([i, j], hs[0]) for hs in self.hell_states)
                        and not any(np.array_equal([i, j], obs[0]) for obs in self.obstacles)):
                    self._draw_colored_cell([i, j], 'orange')
                elif np.array_equal([i, j], agent_pos):
                    # display warning icon when in danger zone
                    warning_image = mpimg.imread(self.warning_path)
                    self._display_image(warning_image, agent_pos, True)

    def _get_obs(self):
        """
        Get the current observation of the environment.

        Returns:
            dict: A dictionary containing the current state of the agent and the goal state:
                - "agent" (np.ndarray): The current position of the agent.
                - "target" (np.ndarray): The position of the goal state.
        """
        return {"agent_pos": self.agent_state, "goal_pos": self.goal_state}

    def _get_info(self):
        """
        Get information about the current state of the environment.

        Returns:
            dict: A dictionary containing the following key-value pair:
                - "distance" (float): The Manhattan distance (L1 norm) between 
                the agent's current position and the goal state.
        """
        return {
            "distance_to_goal": np.linalg.norm(
                np.linalg.norm(self.goal_state - self.agent_state, ord=1)
            )
        }

    def reset(self):
        """
        Reset the Agent to the initial state.

        Returns:
            agent_state (NDArray[Any]): the position of the agent - i.e. reset to [0,0]
            observation (dict): The initial observation of the environment, including:
                - "agent" (np.ndarray): The initial position of the agent.
                - "target" (np.ndarray): The position of the goal state.
            info (dict): A dictionary containing the initial distance to the goal state, with:
                - "distance" (float): The Manhattan distance (L1 norm) between 
                the initial position of the agent and the goal state.
        """
        # self.total_reward = 0
        self.agent_state = np.array([0, 0])
        self.goal_reached = False
        self.current_frame = 0

        observation = self._get_obs()
        info = self._get_info()

        return self.agent_state, observation, info

    def transition_directly(self, pos):
        self.agent_state = np.array(pos)
        self.goal_reached = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Parameters:
            action (str): The action to be taken ('w', 's', 'a', 'd').

        Returns:
            tuple: A tuple containing the following elements:
                - agent_state: self.agent_state
                - observation (dict): The observation of the environment after the step, including:
                    - "agent" (np.ndarray): The new position of the agent.
                    - "target" (np.ndarray): The position of the goal state.
                - reward (float): The reward received from the environment after the step.
                - done (bool): A flag indicating whether the episode has terminated after this step.
                - info (dict): Additional information about the environment:
                    - "distance" (float): The Manhattan distance (L1 norm) between the agent's 
                    current position and the goal state.
        """
        print(f"action1: {action}")
        print(f"curren state: {self.agent_state}")
        if action >= 0 and action < 4:
            next_state = self.agent_state.copy()

            if action == 0:
                if self.agent_state[1] < self.grid_size - 1:  # up
                    next_state[1] += 1
            elif action == 2:
                if self.agent_state[1] > 0:  # down
                    next_state[1] -= 1
            elif action == 3:
                if self.agent_state[0] > 0:  # left
                    next_state[0] -= 1
            elif action == 1:
                if self.agent_state[0] < self.grid_size - 1:  # right
                    next_state[0] += 1

            if not any(np.array_equal(next_state, obs[0]) for obs in self.obstacles):
                self.agent_state = next_state
        else:
            raise ValueError(f"Invalid action: {action}")

        reward = 0
        self.current_cumulative_reward = 0

        done = False
        # if(np.array_equal(self.agent_state,[11,11])):
        reward = self.grid[self.agent_state[0]][self.agent_state[1]]._get_state_info()[
            1][0]  # Accessing the rewards from the info tuple
        print(f"reward: {reward}")
        stateType = self.grid[self.agent_state[0]][self.agent_state[1]]._get_state_info()[
            0].state_type
        print(f"stateType: {stateType}")
        if StateType.HELL == stateType:
            done = True
            print("END")
        if StateType.GOAL == stateType:
            done = True
            print("END")
            self.goal_reached = True
        self.current_cumulative_reward = reward
        # self.total_reward += reward

        if not done:
            self.within_danger_zones = []

            for pos, filename, _, rew, special_ability in self.helper_states:
                if (np.array_equal(self.agent_state, pos)):
                    reward = rew[0]  # Collect the reward of the helper_state.
                    self.current_cumulative_reward += reward
                    if special_ability == SpecialAbilities.FLY:
                        self._reset_state_to_normal(pos[0], pos[1])
                        print("Flying to [0,10]")
                        _, _ = self.transition_directly([0, 10])
                    elif special_ability == SpecialAbilities.CATCH_ANY_POKEMON:
                        # Remove the pokeball from the Grid since it is acquired.
                        self._reset_state_to_normal(pos[0], pos[1])
                        self.FREE_PASS_ONCE = True
                        self._clear_cell_image(pos)

            for pos, filename, _, rew, special_damage in self.hell_states:
                distance_from_hell_state = self._degree_of_effective_zone(
                    self.agent_state, pos)
                try:
                    # Get positive or negative rewards for agent based on distance from the hell_state.
                    reward = rew[distance_from_hell_state]
                    if reward < 0:
                        if (self.FREE_PASS_ONCE):
                            reward = 0
                        else:
                            # Populate active danger_zones list with (distance_from_hell_state, filename, pos)
                            # - useful for marking them when rendering.
                            self.within_danger_zones.append(
                                (distance_from_hell_state, filename, pos))
                        # Reset FREE_PASS_ONCE to False upon reaching a hellstate (Pokemon)
                        # and capturing it (effectively reseting the state to a default state).
                        if distance_from_hell_state == 0:
                            if (self.FREE_PASS_ONCE):
                                self._reset_state_to_normal(pos[0], pos[1])
                                self._render_hell_states()
                                if self.FREE_PASS_ONCE == True:
                                    self.FREE_PASS_ONCE = False
                                print(f"Caught Pokemon at {pos}")
                            else:
                                if special_damage == SpecialAbilities.RESET:
                                    _, _, _ = self.reset()

                    # Collect all rewards caused by all hell_states for a single step.
                    self.current_cumulative_reward += reward
                except IndexError:
                    pass

            # Add the current step's cumulative reward to global reward.
            # self.total_reward += self.current_cumulative_reward
        observation = self._get_obs()
        info = self._get_info()

        return self.agent_state, observation, self.current_cumulative_reward, done, info

    def render(self):
        """
        Render the current state of the environment after clearing previous.
        """

        self._clear_grid()
        if self.goal_reached:
            self._render_goal_reached_gif()
            plt.pause(0.7)
        else:
            self._draw_grid()
            self._render_walkable_states()
            self._render_goal()
            self._draw_cage_in_goal()
            self._render_obstacles()
            self._render_hell_states()
            self._render_helper_states()
            # Display the image of the agent first and then pain indicator on top if any
            self._render_agent()
            if self.within_danger_zones:
                for distance_from_danger, _, pos in self.within_danger_zones:
                    self._mark_danger_zone(
                        pos, distance_from_danger, self.agent_state)

            self._set_grid_dimensions()
            # Increment the current_frame for all animated GIF images.
            self.current_frame += 1
        plt.pause(0.05)

    def close(self):
        """
        Close the environment and the rendering window.
        """
        plt.close()

# Function 1: Create an instance of the environment
# -----------


def create_env():
    # Create the environment:
    # -----------------------
    env = GridWorldEnv()

    return env


# if __name__ == "__main__":
#     env = GridWorldEnv()
#     state = env.reset()
#     done = False
#     try:
#         while not done:
#             print(f"env.renderChoice: {env.renderChoice}")
#             if env.renderChoice:
#                 env.render()
#             action = input("Enter action (w=up, s=down, a=left, d=right): ")
#             try:
#                 observation, reward, done, info = env.step(action)
#                 print(f"Observation: {observation}, Reward: {
#                     reward}, Done: {done}, Info: {info}")
#                 if done:
#                     print("Agent reached the Goal!")
#                     print(f"Total Reward = {env.total_reward}")
#                     env.render()
#                     print("Press Enter to close the window")
#                     while True:
#                         if keyboard.is_pressed("enter"):
#                             env.close()
#                             break
#             except ValueError:
#                 print("Invalid input. Please enter a valid action.")
#     except KeyboardInterrupt:
#         pass
#     finally:
#         env.close()
