# satellite_avoidance_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy.time import Time
from astropy.constants import G
from astropy.coordinates import get_body_barycentric
from astropy.coordinates import solar_system_ephemeris
import logging
import os

# Use JPL ephemeris for better accuracy
solar_system_ephemeris.set('de430')

# Constants for orbit and gravitational forces
G_const = G.value  # Gravitational constant in m^3 kg^−1 s^−2
M_EARTH = Earth.mass.to(u.kg).value  # Mass of Earth in kg
EARTH_RADIUS = Earth.R.to(u.m).value  # Earth's radius in meters

class SatelliteAvoidanceEnv(gym.Env):
    """
    Custom Gym environment for satellite collision avoidance using reinforcement learning.

    Attributes:
        action_space (gym.spaces.Box): Continuous action space representing velocity changes in x, y, z directions.
        observation_space (gym.spaces.Box): Continuous observation space containing satellite state and debris positions.
        satellite_mass (float): Current mass of the satellite in kg.
        fuel_mass (float): Remaining fuel mass in kg.
        satellite_distance (float): Desired orbit altitude in meters (optional).
        init_angle (float): Inclination angle in degrees (optional).
        collision_course_probability (float): Probability to set the satellite on a collision course at initialization.
        current_time (astropy.time.Time): Current simulation time.
        debris_positions (list of np.ndarray): Positions of debris objects in meters.
        time_increment (float): Time step increment in seconds.
        enable_quantitative_metrics (bool): Flag to enable quantitative performance metrics tracking.
        enable_qualitative_evaluation (bool): Flag to enable qualitative trajectory recording.
        enable_advanced_evaluation (bool): Flag to enable advanced evaluation techniques (e.g., noise injection).
        enable_visualization_tools (bool): Flag to enable visualization tools for trajectory plotting.
        enable_comparative_analysis (bool): Flag to enable comparative analysis with baseline models.
        enable_robustness_testing (bool): Flag to enable robustness and generalization testing.
    """

    def __init__(
        self,
        debris_positions,
        max_debris=100,
        satellite_distance=None,  # Optional: Desired orbit altitude in meters
        init_angle=None,          # Optional: Inclination angle in degrees
        collision_course_probability=1.0,  # Default to always on collision course
        enable_quantitative_metrics=False,
        enable_qualitative_evaluation=False,
        enable_advanced_evaluation=False,
        enable_visualization_tools=False,
        enable_comparative_analysis=False,
        enable_robustness_testing=False
    ):
        """
        Initialize the Satellite Avoidance Environment.

        Args:
            debris_positions (list of array-like): Initial positions of debris objects in meters.
            max_debris (int): Maximum number of debris objects.
            satellite_distance (float, optional): Desired orbit altitude in meters. If None, collision course is set.
            init_angle (float, optional): Inclination angle in degrees. If None, collision course is set.
            collision_course_probability (float): Probability to set the satellite on a collision course each episode.
            enable_quantitative_metrics (bool): Enable tracking of quantitative performance metrics.
            enable_qualitative_evaluation (bool): Enable recording of satellite trajectories for qualitative analysis.
            enable_advanced_evaluation (bool): Enable advanced evaluation techniques like noise injection.
            enable_visualization_tools (bool): Enable visualization tools for trajectory plotting.
            enable_comparative_analysis (bool): Enable comparative analysis with baseline models.
            enable_robustness_testing (bool): Enable robustness and generalization testing.
        """
        super(SatelliteAvoidanceEnv, self).__init__()

        # Action space: changes in velocity in x, y, z directions (m/s^2)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)

        # Observation space: satellite position (3), velocity (3), fuel_mass (1), debris positions (max_debris * 3)
        self.max_debris = max_debris
        obs_space_size = 7 + self.max_debris * 3  # 3 position + 3 velocity + 1 fuel_mass + debris positions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32
        )

        # Satellite initial parameters
        self.satellite_mass = 1000.0  # kg
        self.fuel_mass = 500.0  # kg
        self.satellite_distance = satellite_distance  # Desired orbit altitude in meters
        self.init_angle = init_angle  # Inclination angle in degrees

        # Collision course probability
        self.collision_course_probability = collision_course_probability

        # Time settings
        self.current_time = Time.now()

        # Debris initialization
        self.debris_positions = [np.array(debris, dtype=np.float64) for debris in debris_positions]

        # Initialize the satellite orbit
        self._init_satellite_orbit()

        # Evaluation Flags
        self.enable_quantitative_metrics = enable_quantitative_metrics
        self.enable_qualitative_evaluation = enable_qualitative_evaluation
        self.enable_advanced_evaluation = enable_advanced_evaluation
        self.enable_visualization_tools = enable_visualization_tools
        self.enable_comparative_analysis = enable_comparative_analysis
        self.enable_robustness_testing = enable_robustness_testing

        # Initialize Evaluation Metrics
        if self.enable_quantitative_metrics:
            self.reset_evaluation_metrics()

        if self.enable_qualitative_evaluation:
            self.trajectory = []

        if self.enable_comparative_analysis:
            # Initialize baseline policy data
            self.baseline_collision_count = 0
            self.baseline_rewards = []
            self.baseline_delta_v = []
            self.baseline_min_distance = []

            # Logging Setup for Comparative Analysis
            logging.basicConfig(
                filename='baseline_comparative_analysis.log',
                level=logging.INFO,
                format='%(asctime)s:%(levelname)s:%(message)s'
            )
            logging.info("Comparative Analysis Started.")

    def _init_satellite_orbit(self):
        """
        Initialize the satellite's orbital parameters based on the desired altitude and inclination.
        Ensures that self.initial_orbit is always defined.
        """
        if self.satellite_distance is not None and self.init_angle is not None:
            # Set initial orbit with specified altitude and inclination
            self.initial_orbit = Orbit.circular(
                Earth,
                alt=self.satellite_distance * u.m,
                inc=self.init_angle * u.deg,
                epoch=self.current_time
            )
            self.satellite_position = self.initial_orbit.r.to(u.m).value
            self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value
        else:
            # Set to a default low Earth orbit (e.g., 700 km equatorial)
            self.initial_orbit = Orbit.circular(
                Earth,
                alt=700e3 * u.m,  # 700 km altitude
                inc=0 * u.deg,     # Equatorial orbit
                epoch=self.current_time
            )
            self.satellite_position = self.initial_orbit.r.to(u.m).value
            self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value

        # Calculate the orbital period
        self.orbital_period = self.initial_orbit.period.to(u.s).value  # Orbital period in seconds

    def reset_evaluation_metrics(self):
        """
        Reset quantitative performance metrics.
        """
        self.total_collisions = 0
        self.total_rewards = []
        self.total_steps = []
        self.total_delta_v = []
        self.min_distances = []
        self.min_distance = np.inf  # Initialize minimum distance

    def _generate_debris_positions(self, num_debris):
        """
        Generate debris positions in Low Earth Orbit (LEO).

        Args:
            num_debris (int): Number of debris to generate.

        Returns:
            list of np.ndarray: List of debris positions in meters.
        """
        debris_positions = []
        for _ in range(num_debris):
            # Randomly distribute debris in LEO (160 km to 2000 km altitude)
            altitude = np.random.uniform(160e3, 2000e3)  # meters
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = EARTH_RADIUS + altitude
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            debris_positions.append(np.array([x, y, z]))
        return debris_positions

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Args:
            seed (int, optional): Seed for random number generator.
            options (dict, optional): Additional options for resetting.

        Returns:
            tuple: Initial observation and an empty info dictionary.
        """
        super().reset(seed=seed)
        self.current_time = Time.now()
        self.satellite_mass = 1000.0
        self.fuel_mass = 500.0
        self.elapsed_time = 0.0

        # Determine if the satellite is on a collision course this episode
        self.collision_course = np.random.rand() < self.collision_course_probability

        # Reinitialize debris positions
        if self.enable_robustness_testing:
            # Vary debris density for robustness testing
            num_debris = np.random.randint(1, self.max_debris + 1)
        else:
            num_debris = np.random.randint(1, self.max_debris + 1)
        self.debris_positions = self._generate_debris_positions(num_debris)

        # Calculate time increment per step
        self.time_increment = self.orbital_period / 500  # Adjust the denominator to control steps per orbit

        if self.collision_course:
            # Select a random debris to collide with
            if len(self.debris_positions) == 0:
                # No debris to collide with; default to non-collision course
                self.collision_course = False
            else:
                target_debris = self.debris_positions[np.random.randint(len(self.debris_positions))]
                # Set satellite position slightly away from the debris
                offset_distance = 5000  # meters; distance to start from the debris
                direction_vector = target_debris / np.linalg.norm(target_debris)
                self.satellite_position = target_debris + direction_vector * offset_distance

                # Calculate velocity required to collide with the debris
                # Instead of setting velocity directly towards debris, add a small component in that direction
                base_velocity = np.linalg.norm(self.initial_orbit.v.to(u.m / u.s).value)  # m/s

                # Define a tangential direction
                # To ensure the orbit remains valid, compute a perpendicular vector
                # Here, we choose the cross product with the z-axis; handle edge cases where direction_vector is parallel to z-axis
                if not np.allclose(direction_vector, [0, 0, 1]):
                    tangential_dir = np.cross(direction_vector, [0, 0, 1])
                else:
                    tangential_dir = np.cross(direction_vector, [0, 1, 0])
                tangential_dir /= np.linalg.norm(tangential_dir)

                # Define a small delta_v towards the debris
                delta_v_magnitude = 0.05 * base_velocity  # 5% of base velocity

                # Update satellite_velocity with tangential and delta_v components
                self.satellite_velocity = (
                    tangential_dir * base_velocity +
                    direction_vector * delta_v_magnitude
                )

                # Define initial_orbit based on new position and velocity
                try:
                    self.initial_orbit = Orbit.from_vectors(
                        Earth,
                        r=self.satellite_position * u.m,          # Correct keyword argument
                        v=self.satellite_velocity * u.m / u.s,    # Correct keyword argument
                        epoch=self.current_time
                    )
                except ZeroDivisionError as e:
                    print(f"Error initializing orbit: {e}")
                    self.initial_orbit = Orbit.circular(
                        Earth,
                        alt=700e3 * u.m,  # Fallback to default altitude
                        inc=0 * u.deg,
                        epoch=self.current_time
                    )
                    self.satellite_position = self.initial_orbit.r.to(u.m).value
                    self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value

                # Recalculate orbital period
                self.orbital_period = self.initial_orbit.period.to(u.s).value
        else:
            if self.satellite_distance is not None and self.init_angle is not None:
                # Already initialized in _init_satellite_orbit
                pass
            else:
                # Set to default orbit
                self.initial_orbit = Orbit.circular(
                    Earth,
                    alt=700e3 * u.m,  # 700 km altitude
                    inc=0 * u.deg,     # Equatorial orbit
                    epoch=self.current_time
                )
                self.satellite_position = self.initial_orbit.r.to(u.m).value
                self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value
                # Recalculate orbital period
                self.orbital_period = self.initial_orbit.period.to(u.s).value

        # Reset Evaluation Metrics
        if self.enable_quantitative_metrics:
            self.reset_evaluation_metrics()

        # Reset Trajectory
        if self.enable_qualitative_evaluation:
            self.trajectory = [self.satellite_position.copy()]

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Get the current observation of the environment.

        Returns:
            np.ndarray: Concatenated array of satellite position, velocity, fuel mass, and debris positions.
        """
        debris_flat = np.array(self.debris_positions).flatten()
        if len(debris_flat) > self.max_debris * 3:
            debris_flat = debris_flat[:self.max_debris * 3]
        elif len(debris_flat) < self.max_debris * 3:
            debris_flat = np.pad(debris_flat, (0, self.max_debris * 3 - len(debris_flat)), mode='constant')

        obs = np.concatenate([
            self.satellite_position,
            self.satellite_velocity,
            [self.fuel_mass],
            debris_flat
        ])
        return obs

    def _apply_gravitational_force(self):
        """
        Apply gravitational forces from Earth, Moon, and Sun to the satellite's velocity.
        """
        r_vec = self.satellite_position
        r_norm = np.linalg.norm(r_vec)
        a_earth = -G_const * M_EARTH * r_vec / r_norm**3

        # Get Moon's position in Earth-centered frame
        moon_pos = get_body_barycentric('moon', self.current_time).get_xyz().to(u.m).value.flatten()
        earth_pos = get_body_barycentric('earth', self.current_time).get_xyz().to(u.m).value.flatten()
        moon_pos_earth_centered = moon_pos - earth_pos

        r_moon = moon_pos_earth_centered - self.satellite_position
        r_moon_norm = np.linalg.norm(r_moon)
        M_MOON = 7.34767309e22  # Mass of Moon in kg
        a_moon = G_const * M_MOON * r_moon / r_moon_norm**3

        # Get Sun's position in Earth-centered frame
        sun_pos = get_body_barycentric('sun', self.current_time).get_xyz().to(u.m).value.flatten()
        sun_pos_earth_centered = sun_pos - earth_pos

        r_sun = sun_pos_earth_centered - self.satellite_position
        r_sun_norm = np.linalg.norm(r_sun)
        M_SUN = 1.98847e30  # Mass of Sun in kg
        a_sun = G_const * M_SUN * r_sun / r_sun_norm**3

        a_total = a_earth + a_moon + a_sun
        self.satellite_velocity += a_total * self.time_increment

    def _apply_thrust(self, action):
        """
        Apply thrust to the satellite based on the agent's action.

        Args:
            action (np.ndarray): Velocity changes in x, y, z directions.
        """
        max_thrust = 0.1  # m/s^2 (acceleration)
        thrust_acceleration = action * max_thrust
        self.satellite_velocity += thrust_acceleration * self.time_increment

        specific_impulse = 300.0  # s
        g0 = 9.80665  # m/s^2
        delta_v = np.linalg.norm(thrust_acceleration * self.time_increment)
        if delta_v > 0:
            fuel_consumed = self.satellite_mass * (1 - np.exp(-delta_v / (specific_impulse * g0)))
        else:
            fuel_consumed = 0.0

        self.fuel_mass -= fuel_consumed
        self.satellite_mass -= fuel_consumed

        if self.fuel_mass <= 0:
            self.fuel_mass = 0
            print("Out of fuel!")

    def _calculate_reward(self):
        """
        Calculate the reward for the current state and action.

        Returns:
            tuple: (reward, done)
        """
        reward = 0.0
        done = False

        # 1. Positive Reward for Each Step Survived
        reward += 1.0

        # 2. Reward Proportional to Remaining Fuel
        # Encourages the agent to conserve fuel
        reward += 0.1 * self.fuel_mass

        # 3. Fuel Consumption Penalty
        # Penalizes fuel usage to promote efficiency
        fuel_consumed = 500.0 - self.fuel_mass
        reward -= fuel_consumed * 0.1

        # 4. Altitude Maintenance Penalty
        desired_altitude = self.satellite_distance if self.satellite_distance is not None else 700e3
        current_altitude = np.linalg.norm(self.satellite_position) - EARTH_RADIUS
        altitude_error = np.abs(current_altitude - desired_altitude)
        reward -= altitude_error * 0.01

        # 5. Bonus for Maintaining Desired Altitude within Threshold
        altitude_threshold = 100.0  # meters
        if altitude_error < altitude_threshold:
            reward += 10.0  # Bonus reward

        # 6. Eccentricity Penalty
        orbit_radius = np.linalg.norm(self.satellite_position)
        orbit_velocity = np.linalg.norm(self.satellite_velocity)
        specific_energy = 0.5 * orbit_velocity**2 - G_const * M_EARTH / orbit_radius
        specific_angular_momentum = np.cross(self.satellite_position, self.satellite_velocity)
        eccentricity_vector = np.cross(
            self.satellite_velocity, specific_angular_momentum
        ) / (G_const * M_EARTH) - self.satellite_position / orbit_radius
        eccentricity = np.linalg.norm(eccentricity_vector)
        reward -= eccentricity * 100.0

        # 7. Collision Avoidance Reward
        # Reward for maintaining safe distance from all debris
        collision_penalty_triggered = False
        safe_distance = 10e3  # meters
        for debris in self.debris_positions:
            distance_to_debris = np.linalg.norm(self.satellite_position - debris)
            if distance_to_debris < safe_distance:
                # Collision occurred
                reward -= 1000.0
                done = True
                collision_penalty_triggered = True
                break  # Exit loop if collision occurs
            else:
                # Safe distance maintained; small positive reward
                reward += 0.5

        if not collision_penalty_triggered and self.enable_quantitative_metrics:
            # Additional reward if all debris are safely distant
            reward += 0.1 * len(self.debris_positions)

        # 8. Penalty for Getting Too Close to Earth
        r_norm = np.linalg.norm(self.satellite_position)
        if r_norm < EARTH_RADIUS + 100e3:
            reward -= 1000.0
            done = True

        # 9. Penalty for Escaping Earth's Gravity
        elif r_norm > 1e8:
            reward -= 1000.0
            done = True

        # 10. Time-Based Efficiency Penalty
        reward -= 0.1

        # Log Quantitative Metrics
        if self.enable_quantitative_metrics:
            # Accumulate rewards
            self.total_rewards.append(reward)
            # Accumulate steps
            self.total_steps.append(1)  # Each step increments by 1
            # Accumulate delta-v usage
            delta_v = np.linalg.norm(self.satellite_velocity * self.time_increment)
            self.total_delta_v.append(delta_v)
            # Track minimum distance
            for debris in self.debris_positions:
                distance = np.linalg.norm(self.satellite_position - debris)
                if distance < self.min_distance:
                    self.min_distance = distance

        # Log Trajectory for Qualitative Evaluation
        if self.enable_qualitative_evaluation:
            self.trajectory.append(self.satellite_position.copy())

        return reward, done

    def step(self, action):
        """
        Perform a single step in the environment.

        Args:
            action (np.ndarray): Velocity changes in x, y, z directions.

        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        # Apply thrust
        self._apply_thrust(action)

        # Apply gravitational forces
        self._apply_gravitational_force()

        # Update satellite's position based on velocity and time increment
        self.satellite_position += self.satellite_velocity * self.time_increment

        # Update the current time and elapsed time
        self.current_time += self.time_increment * u.s
        self.elapsed_time += self.time_increment

        # Calculate reward and check for termination
        reward, done = self._calculate_reward()

        # Check if fuel is depleted
        if self.fuel_mass <= 0:
            done = True

        # Optionally, set 'done' to True after a full orbit
        if self.elapsed_time >= self.orbital_period:
            done = True

        truncated = False  # Can be set to True based on specific conditions

        return self._get_obs(), reward, done, truncated, {}

    def render(self, mode='human'):
        """
        Render the current state of the environment.

        Args:
            mode (str): The mode to render with. Currently supports 'human'.
        """
        print(f"Time: {self.current_time.iso}")
        print(f"Position (m): {self.satellite_position}")
        print(f"Velocity (m/s): {self.satellite_velocity}")
        print(f"Fuel Mass (kg): {self.fuel_mass}")

    def close(self):
        """
        Perform any necessary cleanup upon closing the environment.
        """
        if self.enable_comparative_analysis:
            logging.info("Comparative Analysis Ended.")
            logging.info(f"Total Collisions: {self.baseline_collision_count}")
            logging.info(f"Average Reward: {np.mean(self.baseline_rewards)}")
            logging.info(f"Average Delta-v Used: {np.mean(self.baseline_delta_v)}")
            logging.info(f"Average Minimum Distance to Debris: {np.mean(self.baseline_min_distance)}")
            logging.shutdown()

    def _baseline_policy(self, obs):
        """
        Simple rule-based policy for comparative analysis.

        Args:
            obs (np.ndarray): Current observation from the environment.

        Returns:
            np.ndarray: Action representing velocity changes in x, y, z directions.
        """
        # Extract satellite position and debris positions from observation
        satellite_pos = obs[:3]
        debris_pos = obs[4:]  
        debris_pos = debris_pos.reshape(-1, 3)

        # Define safe distance
        safe_distance = 1e4  # meters

        for debris in debris_pos:
            distance = np.linalg.norm(satellite_pos - debris)
            if distance < safe_distance:
                # Compute avoidance direction (away from debris)
                direction = satellite_pos - debris
                direction_norm = np.linalg.norm(direction)
                if direction_norm == 0:
                    direction = np.random.randn(3)
                    direction_norm = np.linalg.norm(direction)
                direction = direction / direction_norm
                # Apply a small delta-v in the avoidance direction
                action = direction * 0.1  # Scaling factor for maneuver strength
                return action

        # No maneuver needed
        return np.zeros(3)

    def evaluate_baseline(self, num_episodes=100):
        """
        Evaluate the baseline (rule-based) policy.

        Args:
            num_episodes (int): Number of episodes to run for baseline evaluation.

        Returns:
            dict: Dictionary containing baseline evaluation metrics.
        """
        collision_count = 0
        total_rewards = []
        total_steps = []
        total_delta_v = []
        min_distances = []
        trajectories = []

        for episode in range(num_episodes):
            obs, _ = self.reset()
            done = False
            cumulative_reward = 0.0
            steps = 0
            delta_v_total = 0.0
            min_distance = np.inf
            trajectory = []

            while not done:
                # Baseline action
                action = self._baseline_policy(obs)
                obs, reward, done, truncated, info = self.step(action)
                cumulative_reward += reward
                steps += 1

                # Track delta-v usage
                delta_v = np.linalg.norm(action * self.time_increment)
                delta_v_total += delta_v

                # Track minimum distance to debris
                distances = [np.linalg.norm(self.satellite_position - debris) for debris in self.debris_positions]
                current_min_distance = min(distances)
                if current_min_distance < min_distance:
                    min_distance = current_min_distance

                # Record trajectory
                if self.enable_qualitative_evaluation:
                    trajectory.append(self.satellite_position.copy())

                if done and reward <= -1000.0:
                    collision_count += 1

            total_rewards.append(cumulative_reward)
            total_steps.append(steps)
            total_delta_v.append(delta_v_total)
            min_distances.append(min_distance)
            trajectories.append(trajectory)

            # Save trajectory for visualization
            if self.enable_visualization_tools:
                self.save_trajectory(episode_num=episode+1)
                self.run_visualization(episode_num=episode+1)

            # Log baseline metrics
            if self.enable_comparative_analysis:
                self.baseline_rewards.append(cumulative_reward)
                self.baseline_delta_v.append(delta_v_total)
                self.baseline_min_distance.append(min_distance)
                logging.info(f"Episode {episode+1}: Reward={cumulative_reward}, Delta-v={delta_v_total}, Min Distance={min_distance}")

        # Calculate metrics
        collision_rate = collision_count / num_episodes * 100
        average_reward = np.mean(total_rewards)
        average_steps = np.mean(total_steps)
        average_delta_v = np.mean(total_delta_v)
        average_min_distance = np.mean(min_distances)

        print(f"\nBaseline Evaluation over {num_episodes} episodes:")
        print(f"Collision Rate: {collision_rate}%")
        print(f"Average Cumulative Reward: {average_reward:.2f}")
        print(f"Average Steps Taken: {average_steps}")
        print(f"Average Delta-v Used: {average_delta_v:.4f} m/s")
        print(f"Average Minimum Distance to Debris: {average_min_distance:.2f} meters")

        if self.enable_comparative_analysis:
            logging.info(f"Baseline Evaluation Summary:")
            logging.info(f"Collision Rate: {collision_rate}%")
            logging.info(f"Average Cumulative Reward: {average_reward:.2f}")
            logging.info(f"Average Delta-v Used: {average_delta_v:.4f} m/s")
            logging.info(f"Average Minimum Distance to Debris: {average_min_distance:.2f} meters")

        return {
            "collision_rate": collision_rate,
            "average_reward": average_reward,
            "average_steps": average_steps,
            "average_delta_v": average_delta_v,
            "average_min_distance": average_min_distance,
            "trajectories": trajectories
        }

    def save_trajectory(self, episode_num):
        """
        Save the satellite's trajectory for qualitative evaluation.

        Args:
            episode_num (int): The episode number.
        """
        if self.enable_qualitative_evaluation:
            trajectory = np.array(self.trajectory)
            np.save(f"trajectory_episode_{episode_num}.npy", trajectory)

    def run_visualization(self, episode_num):
        """
        Generate and save a 3D visualization of the satellite trajectory and debris positions.

        Args:
            episode_num (int): The episode number.
        """
        if self.enable_visualization_tools:
            import plotly.graph_objects as go

            # Load trajectory
            trajectory = np.load(f"trajectory_episode_{episode_num}.npy")

            fig = go.Figure()

            # Plot Earth
            u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
            x = EARTH_RADIUS * np.cos(u) * np.sin(v)
            y = EARTH_RADIUS * np.sin(u) * np.sin(v)
            z = EARTH_RADIUS * np.cos(v)
            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                colorscale='Blues',
                opacity=0.5,
                showscale=False,
                name='Earth'
            ))

            # Plot satellite trajectory
            fig.add_trace(go.Scatter3d(
                x=trajectory[:,0],
                y=trajectory[:,1],
                z=trajectory[:,2],
                mode='lines',
                line=dict(color='green', width=2),
                name='Satellite Trajectory'
            ))

            # Plot debris positions
            for debris in self.debris_positions:
                fig.add_trace(go.Scatter3d(
                    x=[debris[0]],
                    y=[debris[1]],
                    z=[debris[2]],
                    mode='markers',
                    marker=dict(size=2, color='red'),
                    name='Debris'
                ))

            fig.update_layout(
                title=f"Satellite Trajectory - Episode {episode_num}",
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)',
                    aspectmode='data'
                )
            )

            # Save the visualization as an HTML file
            fig.write_html(f"trajectory_episode_{episode_num}.html")
            print(f"Trajectory visualization saved for Episode {episode_num}.")

    def run_advanced_evaluation(self, num_episodes=100):
        """
        Perform advanced evaluation techniques such as noise injection to assess model resilience.

        Args:
            num_episodes (int): Number of episodes to run for advanced evaluation.
        """
        if self.enable_advanced_evaluation:
            for episode in range(num_episodes):
                obs, _ = self.reset()
                done = False
                cumulative_reward = 0.0

                while not done:
                    # Inject Gaussian noise into observations
                    noise = np.random.normal(0, 0.1, size=obs.shape)
                    noisy_obs = obs + noise

                    # Agent's action based on noisy observations
                    action = self._baseline_policy(noisy_obs)  # Using baseline for simplicity
                    obs, reward, done, truncated, info = self.step(action)
                    cumulative_reward += reward

            print("Advanced evaluation (noise injection) completed.")

    def evaluate_robustness(self, num_episodes=100):
        """
        Test model robustness by varying environment parameters such as fuel mass and satellite altitude.

        Args:
            num_episodes (int): Number of episodes to run for robustness testing.
        """
        if self.enable_robustness_testing:
            original_fuel_mass = self.fuel_mass
            original_satellite_mass = self.satellite_mass
            original_satellite_distance = self.satellite_distance

            for episode in range(num_episodes):
                # Randomize fuel mass between 400kg and 600kg
                self.fuel_mass = np.random.uniform(400.0, 600.0)
                # Randomize satellite distance between 600km and 800km
                self.satellite_distance = np.random.uniform(600e3, 800e3)
                # Reset the environment with new parameters
                obs, _ = self.reset()
                done = False
                cumulative_reward = 0.0

                while not done:
                    action = self._baseline_policy(obs)
                    obs, reward, done, truncated, info = self.step(action)
                    cumulative_reward += reward

            # Restore original parameters
            self.fuel_mass = original_fuel_mass
            self.satellite_mass = original_satellite_mass
            self.satellite_distance = original_satellite_distance

            print("Robustness and Generalization Testing completed.")
