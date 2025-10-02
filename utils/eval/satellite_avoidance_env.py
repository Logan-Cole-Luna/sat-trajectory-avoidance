# satellite_avoidance_env.py

import numpy as np
from typing import List, Optional
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

from astropy.constants import M_earth
M_EARTH = M_earth.to(u.kg).value  # Mass of Earth in kg


# Use JPL ephemeris for better accuracy
solar_system_ephemeris.set('de430')

# Constants for orbit and gravitational forces
G_const = G.value  # Gravitational constant in m^3 kg^−1 s^−2
EARTH_RADIUS = Earth.R.to(u.m).value  # Earth's radius in meters
# Distance thresholds (meters)
D_SAFE = 30e3
D_WARN = 20e3
D_COLLIDE = 10e3

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
            satellite_distance=None,
            init_angle=None,
            collision_course_probability=1.0,
            guarantee_collision=False,
            collision_ttc_steps=None,
            debris_radius_m: float = 0.0,
            # Control capability
            max_thrust: float = 0.05,
            # Dynamic debris parameters
            dynamic_debris: bool = False,
            debris_lateral_speed: float = 30.0,
            lateral_offset_scale: float = 5000.0,
            # Tunables
            collision_distance_m: float | None = None,
            approach_speed_scale: float = 1.0,
            start_offset_scale: float = 1.0,
            # Feature flags
            enable_quantitative_metrics=False,
            enable_qualitative_evaluation=False,
            enable_advanced_evaluation=False,
            enable_visualization_tools=False,
            enable_comparative_analysis=False,
            enable_robustness_testing=False,
            # Reward shaping coefficients (configurable)
            collision_penalty: float = 500.0,
            survival_reward: float = 0.02,
            distance_shaping_coeff: float = 0.0002,
            coast_bonus: float = 0.05,
            delta_v_linear_cost: float = 6.0,
            delta_v_quadratic_cost: float = 0.5,
            large_burn_penalty: float = 3.0,
            smoothness_penalty: float = 0.005,
            cumulative_dv_penalty: float = 0.2,
            projected_miss_reward: float = 0.0002,
            milestone_reward: float = 0.4,
            # Legacy parameters (for compatibility)
            dv_linear_cost: float = 10.0,
            dv_quadratic_cost: float = 2.0,
            large_burn_threshold: float = 0.4,
            large_burn_quadratic_cost: float = 12.0,
            coast_bonus_legacy: float = 0.5,
            danger_action_bonus: float = 4.0,
            action_smooth_coeff: float = 0.6,
            closing_speed_reward_scale: float = 3.0,
            projected_miss_reward_scale: float = 0.002,
            cumulative_delta_v_soft_cap: float = 60.0,
            cumulative_delta_v_exponent: float = 1.2,
    ):
        """Initialize the Satellite Avoidance Environment."""
        super().__init__()

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.max_debris = max_debris
        obs_space_size = 7 + self.max_debris * 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32)

        # Physical parameters
        self.satellite_mass = 1000.0
        self.fuel_mass = 500.0
        self.satellite_distance = satellite_distance
        self.init_angle = init_angle
        self.collision_course_probability = collision_course_probability
        self.guarantee_collision = guarantee_collision
        self.collision_ttc_steps = collision_ttc_steps
        self._disable_gravity_steps = 0
        self.debris_radius_m = float(debris_radius_m)
        self.collision_point = None
        self.max_thrust = float(max_thrust)  # m/s^2 per-axis acceleration at action=1.0

        # Dynamic debris / lateral avoidance helpers
        self.dynamic_debris = dynamic_debris
        self.debris_lateral_speed = float(debris_lateral_speed)
        self.lateral_offset_scale = float(lateral_offset_scale)
        self.debris_velocities = [np.zeros(3) for _ in debris_positions]

        # New tunables
        self.collision_distance_m = None if collision_distance_m is None else float(collision_distance_m)
        self.approach_speed_scale = float(approach_speed_scale)
        self.start_offset_scale = float(start_offset_scale)

        # New configurable reward shaping coefficients
        self.collision_penalty = float(collision_penalty)
        self.survival_reward = float(survival_reward)
        self.distance_shaping_coeff = float(distance_shaping_coeff)
        self.coast_bonus = float(coast_bonus)
        self.delta_v_linear_cost = float(delta_v_linear_cost)
        self.delta_v_quadratic_cost = float(delta_v_quadratic_cost)
        self.large_burn_penalty = float(large_burn_penalty)
        self.smoothness_penalty = float(smoothness_penalty)
        self.cumulative_dv_penalty = float(cumulative_dv_penalty)
        self.projected_miss_reward = float(projected_miss_reward)
        self.milestone_reward = float(milestone_reward)

        # Legacy reward shaping coefficients (for backward compatibility)
        self.dv_linear_cost = float(dv_linear_cost)
        self.dv_quadratic_cost = float(dv_quadratic_cost)
        self.large_burn_threshold = float(large_burn_threshold)
        self.large_burn_quadratic_cost = float(large_burn_quadratic_cost)
        self.coast_bonus_legacy = float(coast_bonus_legacy)
        self.danger_action_bonus = float(danger_action_bonus)
        self.action_smooth_coeff = float(action_smooth_coeff)
        self.closing_speed_reward_scale = float(closing_speed_reward_scale)
        self.projected_miss_reward_scale = float(projected_miss_reward_scale)
        self.cumulative_delta_v_soft_cap = float(cumulative_delta_v_soft_cap)
        self.cumulative_delta_v_exponent = float(cumulative_delta_v_exponent)
        self._prev_action = np.zeros(3, dtype=np.float32)
        self.action_change_mag = 0.0

        # Time and debris
        self.current_time = Time.now()
        self.debris_positions = [np.array(d, dtype=np.float64) for d in debris_positions]

        # Initialize orbit
        self._init_satellite_orbit()

        # Evaluation flags
        self.enable_quantitative_metrics = enable_quantitative_metrics
        self.enable_qualitative_evaluation = enable_qualitative_evaluation
        self.enable_advanced_evaluation = enable_advanced_evaluation
        self.enable_visualization_tools = enable_visualization_tools
        self.enable_comparative_analysis = enable_comparative_analysis
        self.enable_robustness_testing = enable_robustness_testing

        # Metrics containers
        self.total_delta_v = []
        self.min_distance = np.inf

        if self.enable_quantitative_metrics:
            self.reset_evaluation_metrics()
        if self.enable_qualitative_evaluation:
            self.trajectory = []
        if self.enable_comparative_analysis:
            self.baseline_collision_count = 0
            self.baseline_rewards = []
            self.baseline_delta_v = []
            self.baseline_min_distance = []
            logging.basicConfig(
                filename='baseline_comparative_analysis.log',
                level=logging.INFO,
                format='%(asctime)s:%(levelname)s:%(message)s',
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
        """Generate debris positions in Low Earth Orbit (LEO)."""
        debris_positions = []
        for _ in range(num_debris):
            altitude = np.random.uniform(160e3, 2000e3)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = EARTH_RADIUS + altitude
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            debris_positions.append(np.array([x, y, z]))
        return debris_positions
    def reset(self, seed=None, options=None):
        """Reset environment state; optionally set up a near head-on approach with optional lateral dynamics."""
        super().reset(seed=seed)

        # Basic episode state
        self.collision_point = None
        self.current_step = 0
        self.current_time = Time.now()
        self.satellite_mass = 1000.0
        self.fuel_mass = 500.0
        self.elapsed_time = 0.0

        # Decide forced collision
        self.collision_course = True if self.guarantee_collision else (np.random.rand() < self.collision_course_probability)
        self.collision_target_index = None

        # Debris field
        num_debris = np.random.randint(1, self.max_debris + 1)
        self.debris_positions = self._generate_debris_positions(num_debris)

        # Default integration step
        self.time_increment = self.orbital_period / 500

        # Scenario setup (feasibility-aware)
        if self.collision_course and len(self.debris_positions) > 0:
            self.collision_target_index = int(np.random.randint(len(self.debris_positions)))
            target_debris = self.debris_positions[self.collision_target_index]
            direction_vector = target_debris / max(np.linalg.norm(target_debris), 1e-9)

            base_offset = max(self.debris_radius_m + 2000.0, D_COLLIDE + 10000.0)
            start_offset = max(base_offset, 20000.0) * max(self.start_offset_scale, 1e-3)
            approach_dt = min(1.0, self.orbital_period / 2000.0)
            if self.collision_ttc_steps is not None:
                steps_to_hit = int(np.clip(int(self.collision_ttc_steps), 2, 1000))
                T = steps_to_hit * approach_dt
            else:
                T = 60.0
            self.time_increment = approach_dt

            relative_speed = (start_offset / max(T, 1e-6)) * self.approach_speed_scale
            v0 = -direction_vector * relative_speed

            # Feasibility adjustment: ensure max achievable lateral displacement can exceed clearance
            threshold = self.collision_distance_m if self.collision_distance_m is not None else max(D_COLLIDE, self.debris_radius_m)
            buffer = getattr(self, 'safe_zone_buffer_m', 3000.0)
            clearance_needed = threshold + buffer
            achievable_disp = 0.5 * self.max_thrust * (T ** 2)
            feasible = True
            if achievable_disp < (clearance_needed + 500.0):
                T_needed = np.sqrt(2 * (clearance_needed + 500.0) / max(self.max_thrust, 1e-9))
                T_needed = float(min(T_needed, 1200.0))
                if T_needed > T:
                    T = T_needed
                    relative_speed = start_offset / max(T, 1e-6) * self.approach_speed_scale
                    v0 = -direction_vector * relative_speed
                    steps_to_hit = int(np.clip(int(T / approach_dt), 2, 2000))
                    self.collision_ttc_steps = steps_to_hit
                else:
                    feasible = False
                achievable_disp = 0.5 * self.max_thrust * (T ** 2)
            self.achievable_lateral_disp = achievable_disp
            self.clearance_needed = clearance_needed
            self.scenario_feasible = feasible

            p0 = target_debris + direction_vector * start_offset

            if self.dynamic_debris:
                trial = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(trial, direction_vector)) > 0.95:
                    trial = np.array([0.0, 1.0, 0.0])
                lateral_vec = np.cross(direction_vector, trial)
                ln = np.linalg.norm(lateral_vec)
                if ln < 1e-9:
                    lateral_vec = np.array([1.0, 0.0, 0.0])
                else:
                    lateral_vec /= ln
                lateral_offset = lateral_vec * np.random.uniform(-self.lateral_offset_scale, self.lateral_offset_scale)
                p0 = target_debris + direction_vector * start_offset + lateral_offset
                self.debris_velocities = [np.zeros(3) for _ in self.debris_positions]
                self.debris_velocities[self.collision_target_index] = lateral_vec * self.debris_lateral_speed
            else:
                self.debris_velocities = [np.zeros(3) for _ in self.debris_positions]

            self.satellite_position = p0
            self.satellite_velocity = v0
        else:
            if not (self.satellite_distance is not None and self.init_angle is not None):
                self.initial_orbit = Orbit.circular(
                    Earth,
                    alt=700e3 * u.m,
                    inc=0 * u.deg,
                    epoch=self.current_time
                )
            self.satellite_position = self.initial_orbit.r.to(u.m).value
            self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value
            self.debris_velocities = [np.zeros(3) for _ in self.debris_positions]

        if self.enable_quantitative_metrics:
            self.reset_evaluation_metrics()
        if self.enable_qualitative_evaluation:
            self.trajectory = [self.satellite_position.copy()]

        self.cumulative_delta_v = 0.0
        self.first_burn_step = None
        self.first_burn_time = None
        self.safe_zone_entry_step = None
        self.safe_zone_entry_time = None
        self.previous_closing_speed = None
        self.safe_zone_buffer_m = 3000.0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self.action_change_mag = 0.0
        self.current_projected_miss_distance = None

        return self._get_obs(), {}

    def get_scenario_state(self):
        """Return a serializable snapshot of the current scenario (post-reset) so it can be replayed.

        Returns:
            dict: scenario components required to recreate identical initial conditions.
        """
        return {
            'satellite_position': np.array(self.satellite_position, dtype=float).copy(),
            'satellite_velocity': np.array(self.satellite_velocity, dtype=float).copy(),
            'fuel_mass': float(self.fuel_mass),
            'debris_positions': [np.array(d, dtype=float).copy() for d in self.debris_positions],
            'collision_course': bool(self.collision_course),
            'collision_target_index': None if self.collision_target_index is None else int(self.collision_target_index),
            'time_increment': float(self.time_increment),
            'debris_radius_m': float(self.debris_radius_m),
            'orbital_period': float(getattr(self, 'orbital_period', 0.0)),
        }

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

    def _gravity_accel_at(self, pos_vec):
        """Compute total gravitational acceleration (Earth + Moon + Sun) at an arbitrary position."""
        r_vec = pos_vec
        r_norm = np.linalg.norm(r_vec)
        a_earth = -G_const * M_EARTH * r_vec / max(r_norm**3, 1e-12)

        moon_pos = get_body_barycentric('moon', self.current_time).get_xyz().to(u.m).value.flatten()
        earth_pos = get_body_barycentric('earth', self.current_time).get_xyz().to(u.m).value.flatten()
        moon_pos_earth_centered = moon_pos - earth_pos
        r_moon = moon_pos_earth_centered - pos_vec
        r_moon_norm = np.linalg.norm(r_moon)
        M_MOON = 7.34767309e22
        a_moon = G_const * M_MOON * r_moon / max(r_moon_norm**3, 1e-12)

        sun_pos = get_body_barycentric('sun', self.current_time).get_xyz().to(u.m).value.flatten()
        sun_pos_earth_centered = sun_pos - earth_pos
        r_sun = sun_pos_earth_centered - pos_vec
        r_sun_norm = np.linalg.norm(r_sun)
        M_SUN = 1.98847e30
        a_sun = G_const * M_SUN * r_sun / max(r_sun_norm**3, 1e-12)

        return a_earth + a_moon + a_sun

    def _apply_thrust(self, action):
        """Apply thrust to the satellite based on the agent's action.

        Args:
            action (np.ndarray): Velocity changes in x, y, z directions (normalized -1..1).
        """
        max_thrust = self.max_thrust
        a = np.clip(action, -1.0, 1.0)
        prev = self._prev_action
        self.action_change_mag = float(np.linalg.norm(a - prev))
        self._prev_action = a.astype(np.float32)
        thrust_acceleration = a * max_thrust
        self.satellite_velocity += thrust_acceleration * self.time_increment

        specific_impulse = 300.0  # s
        g0 = 9.80665  # m/s^2
        delta_v = np.linalg.norm(thrust_acceleration * self.time_increment)
        self.last_delta_v = float(delta_v)
        if delta_v > 0:
            fuel_consumed = self.satellite_mass * (1 - np.exp(-delta_v / (specific_impulse * g0)))
        else:
            fuel_consumed = 0.0
        self.fuel_mass -= fuel_consumed
        self.satellite_mass -= fuel_consumed
        if self.fuel_mass <= 0:
            self.fuel_mass = 0
            pass

    def _calculate_reward(self):
        """Fuel-efficient collision avoidance reward.

        Principles:
          1. Strong collision penalty.
          2. Dense shaping for increasing projected miss distance & reducing closing speed.
          3. Aggressive per-step delta-v cost (linear + quadratic) except in immediate danger.
          4. Penalty for large impulsive burns & for jittery control (action change).
          5. Soft cap on cumulative delta-v with accelerating penalty beyond cap.
          6. Small survival/time reward so zero-action isn't trivially optimal if collision looms.
        """
        # Unified small-scale reward (keep value targets ~ O(1-10))
        reward = 0.0
        done = False

        current_delta_v = float(getattr(self, 'last_delta_v', 0.0))
        if len(self.debris_positions) > 0:
            d_min = min(np.linalg.norm(self.satellite_position - d) for d in self.debris_positions)
        else:
            d_min = np.inf

        custom_collide = self.collision_distance_m if self.collision_distance_m is not None else None
        base_collide = max(D_COLLIDE, self.debris_radius_m)
        threshold = custom_collide if custom_collide is not None else base_collide
        extended_safe_threshold = threshold + getattr(self, 'safe_zone_buffer_m', 3000.0)

        # 1. Collision terminal
        if d_min <= threshold:
            # Configurable collision penalty
            reward -= self.collision_penalty
            done = True
            if hasattr(self, 'collision_point'):
                self.collision_point = self.satellite_position.copy()
            return reward, done

        # 2. Survival baseline
        reward += self.survival_reward  # configurable survival tick

        # 3. Distance shaping
        if d_min > extended_safe_threshold:
            dist_excess = d_min - extended_safe_threshold
            # Softly cap distance shaping to avoid runaway values
            capped = min(dist_excess, 8000.0)
            reward += self.distance_shaping_coeff * capped
            if current_delta_v < 0.005:
                reward += self.coast_bonus  # configurable coast bonus
        elif d_min < extended_safe_threshold:
            gap = extended_safe_threshold - d_min
            capped_gap = min(gap, 8000.0)
            reward -= 0.00015 * capped_gap  # 8 km deficit => -1.2

        # 4. Closing speed & projected miss shaping
        closing_speed = getattr(self, 'current_closing_speed', None)
        proj_miss = getattr(self, 'current_projected_miss_distance', None)
        if closing_speed is not None:
            prev = self.previous_closing_speed
            if prev is not None and prev > 0 and closing_speed < prev:
                improvement = (prev - closing_speed) / max(prev, 1e-6)
                reward += 0.5 * improvement  # bounded ~[0,0.5]
            self.previous_closing_speed = closing_speed
        if proj_miss is not None and proj_miss > threshold:
            gain = proj_miss - threshold
            reward += self.projected_miss_reward * min(gain, 15000.0)  # configurable projected miss reward

        # 5. Delta-v cost (context-sensitive)
        if current_delta_v > 0.0:
            # Configurable delta-v costs
            dv_cost = self.delta_v_linear_cost * current_delta_v + self.delta_v_quadratic_cost * (current_delta_v ** 2)
            if d_min <= extended_safe_threshold:
                dv_cost *= 0.6  # allow maneuvering when near
            reward -= dv_cost
            if current_delta_v > 0.2:
                reward -= self.large_burn_penalty * (current_delta_v - 0.2) ** 2
            if d_min < 1.15 * threshold and current_delta_v > 0.05:
                reward += 0.3 * min(1.0, current_delta_v / 0.15)

        # 6. Action smoothness penalty (penalize jitter)
        if self.action_change_mag > 0.0 and current_delta_v < 0.25:
            reward -= self.smoothness_penalty * (self.action_change_mag ** 2)

        # 7. Soft cap on cumulative delta-v
        if self.cumulative_delta_v > self.cumulative_delta_v_soft_cap:
            over = self.cumulative_delta_v - self.cumulative_delta_v_soft_cap
            reward -= self.cumulative_dv_penalty * (over ** 1.3)

        # 8. Long episode bonus (encourage survival without burns)
        if getattr(self, 'current_step', 0) % 100 == 0 and self.current_step > 0:
            reward += self.milestone_reward

        # Metrics tracking
        self.total_delta_v.append(current_delta_v)
        for debris in self.debris_positions:
            dist = np.linalg.norm(self.satellite_position - debris)
            if dist < self.min_distance:
                self.min_distance = dist

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
        # Increment step counter
        if not hasattr(self, 'current_step'):
            self.current_step = 0
        self.current_step += 1
        
        # Apply thrust
        self._apply_thrust(action)

        # Track cumulative delta-v & first burn
        dv_step = float(getattr(self, 'last_delta_v', 0.0))
        self.cumulative_delta_v += dv_step
        if dv_step > 0.02 and self.first_burn_step is None:
            self.first_burn_step = int(self.current_step)
            self.first_burn_time = float(self.elapsed_time)

        # Apply gravitational forces (optionally disabled for guaranteed-collision setup)
        if self._disable_gravity_steps > 0:
            self._disable_gravity_steps -= 1
        else:
            self._apply_gravitational_force()

        # Update satellite's position based on velocity and time increment
        self.satellite_position += self.satellite_velocity * self.time_increment

        # Update debris positions if dynamic model enabled
        if self.dynamic_debris and len(self.debris_velocities) == len(self.debris_positions):
            for i, vel in enumerate(self.debris_velocities):
                self.debris_positions[i] = self.debris_positions[i] + vel * self.time_increment

        # Update the current time and elapsed time
        self.current_time += self.time_increment * u.s
        self.elapsed_time += self.time_increment

        # Compute relative kinematics to target debris (if any) for logging & shaping
        relative_closing_speed = None
        projected_miss_distance = None
        # Ensure debris_velocities length matches debris_positions length (can desync on scenario replay)
        if len(self.debris_velocities) != len(self.debris_positions):
            self.debris_velocities = [np.zeros(3) for _ in self.debris_positions]
        if self.collision_target_index is not None and 0 <= self.collision_target_index < len(self.debris_positions):
            target_pos = self.debris_positions[self.collision_target_index]
            rel_pos = target_pos - self.satellite_position
            # Base relative velocity assumes debris static unless dynamic debris enabled
            if self.dynamic_debris and self.collision_target_index < len(self.debris_velocities):
                base_debris_vel = self.debris_velocities[self.collision_target_index]
            else:
                base_debris_vel = np.zeros(3)
            rel_vel = (base_debris_vel - self.satellite_velocity)
            r_norm = np.linalg.norm(rel_pos)
            if r_norm > 1e-6:
                radial_speed = -np.dot(rel_pos, rel_vel) / r_norm  # >0 means closing
                relative_closing_speed = float(radial_speed)
                # Linear projection time to closest approach (if closing)
                if radial_speed > 1e-3:
                    t_ca = r_norm / radial_speed
                    proj_rel = rel_pos + rel_vel * t_ca
                    projected_miss_distance = float(np.linalg.norm(proj_rel))
                else:
                    projected_miss_distance = float(r_norm)
            self.current_closing_speed = relative_closing_speed
            self.current_projected_miss_distance = projected_miss_distance

        # Calculate reward and check for termination
        reward, done = self._calculate_reward()
        termination_reason = None

        # Check if fuel is depleted
        if self.fuel_mass <= 0:
            done = True
            termination_reason = termination_reason or "fuel_depleted"

        # Optionally, set 'done' to True after a full orbit
        if self.elapsed_time >= self.orbital_period:
            done = True
            termination_reason = termination_reason or "orbit_complete"

        truncated = False  # Can be set to True based on specific conditions

        # Provide info for metrics and debugging
        d_min = None
        if len(self.debris_positions) > 0:
            d_min = float(min(np.linalg.norm(self.satellite_position - d) for d in self.debris_positions))
        custom_collide = self.collision_distance_m if getattr(self, 'collision_distance_m', None) is not None else None
        threshold = custom_collide if custom_collide is not None else max(D_COLLIDE, self.debris_radius_m)
        if d_min is not None and d_min <= threshold:
            termination_reason = termination_reason or "collision"
            # Record approximate collision point
            self.collision_point = self.satellite_position.copy()
        info = {
            "fuel_mass": float(self.fuel_mass),
            "elapsed_time": float(self.elapsed_time),
            "min_distance": d_min,
            "collision_occurred": d_min is not None and d_min <= threshold,
            "last_delta_v": float(getattr(self, 'last_delta_v', 0.0)),
            "termination_reason": termination_reason,
            "cumulative_delta_v": float(self.cumulative_delta_v),
            "first_burn_step": None if self.first_burn_step is None else int(self.first_burn_step),
            "safe_zone_entry_step": None if self.safe_zone_entry_step is None else int(self.safe_zone_entry_step),
            "relative_closing_speed": relative_closing_speed,
            "projected_miss_distance": projected_miss_distance,
            "scenario_feasible": getattr(self, 'scenario_feasible', True),
            "achievable_lateral_disp": float(getattr(self, 'achievable_lateral_disp', 0.0)),
            "clearance_needed": float(getattr(self, 'clearance_needed', 0.0)),
        }
        return self._get_obs(), reward, done, truncated, info

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
        # Leverage internal state to derive relative kinematics instead of relying on the
        # padded observation array (which may include dummy debris entries).
        sat_pos = np.asarray(self.satellite_position, dtype=float)
        sat_vel = np.asarray(self.satellite_velocity, dtype=float)

        if len(self.debris_positions) == 0:
            return np.zeros(3, dtype=np.float32)

        # Collision threshold used by the environment (meters).
        base_threshold = self.collision_distance_m
        if base_threshold is None:
            base_threshold = max(D_COLLIDE, self.debris_radius_m)

        # React well before impact to give the rule-based policy a fighting chance.
        caution_distance = max(2.5 * base_threshold, 3000.0)
        danger_distance = max(4.0 * base_threshold, 6000.0)
        ttc_limit = 240.0  # seconds until closest approach to trigger avoidance
        ttc_critical = 120.0

        best_score = 0.0
        best_action: Optional[np.ndarray] = None

        debris_velocities: List[np.ndarray]
        if len(getattr(self, "debris_velocities", [])) == len(self.debris_positions):
            debris_velocities = [np.asarray(v, dtype=float) for v in self.debris_velocities]
        else:
            debris_velocities = [np.zeros(3, dtype=float) for _ in self.debris_positions]

        for idx, debris in enumerate(self.debris_positions):
            debris_pos = np.asarray(debris, dtype=float)
            rel_pos = debris_pos - sat_pos
            distance = float(np.linalg.norm(rel_pos))
            if not np.isfinite(distance) or distance <= 1e-6:
                continue

            rel_vel = debris_velocities[idx] - sat_vel
            rel_speed = float(np.linalg.norm(rel_vel))

            # Closing speed projection (>0 indicates approaching).
            radial_speed = -float(np.dot(rel_pos, rel_vel) / distance)
            if radial_speed <= 0.0 and distance > caution_distance:
                # Debris is receding or already distant -> low risk.
                continue

            if radial_speed > 1e-5:
                time_to_closest = distance / radial_speed
            else:
                time_to_closest = np.inf

            # Risk score blends distance and time-to-collision pressure.
            proximity_score = max(0.0, (danger_distance - distance) / danger_distance)
            timing_score = 0.0 if not np.isfinite(time_to_closest) else max(0.0, (ttc_limit - time_to_closest) / ttc_limit)
            risk_score = proximity_score + timing_score
            if radial_speed > 0.0 and time_to_closest < ttc_critical:
                risk_score += 0.5

            if risk_score <= 0.0:
                continue

            # Base thrust away from the debris.
            away_dir = -rel_pos / distance

            if radial_speed > 0.0 and rel_speed > 1e-6:
                # Add component opposing the relative velocity to bleed closing speed.
                vel_dir = -rel_vel / (rel_speed + 1e-8)
                away_dir = away_dir + 0.5 * vel_dir

            norm = float(np.linalg.norm(away_dir))
            if norm > 1e-6:
                away_dir = away_dir / norm
            else:
                away_dir = np.zeros(3)

            # Scale burns more aggressively the higher the perceived risk.
            thrust_mag = float(np.clip(0.25 + 0.75 * risk_score, 0.0, 1.0))
            candidate = away_dir * thrust_mag

            if risk_score > best_score and np.any(candidate):
                best_score = risk_score
                best_action = candidate

        if best_action is not None:
            return best_action.astype(np.float32)

        return np.zeros(3, dtype=np.float32)

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
            # Save trajectory & optionally visualize
            if self.enable_visualization_tools:
                self.save_trajectory(episode_num=episode+1)
                if self.dynamic_debris:
                    self.run_visualization(episode_num=episode+1)

            # Log baseline metrics per episode if comparative analysis enabled
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
