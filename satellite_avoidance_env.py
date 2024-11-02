# satellite_avoidance_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from astropy import units as u
from poliastro.bodies import Earth, Moon, Sun
from poliastro.twobody import Orbit
from astropy.time import Time
from poliastro.ephem import build_ephem_interpolant
from astropy.constants import G

# Constants for orbit and gravitational forces
G_const = G.value  # Gravitational constant in m^3 kg^−1 s^−2
M_EARTH = Earth.mass.to(u.kg).value  # Mass of Earth in kg
EARTH_RADIUS = Earth.R.to(u.m).value  # Earth's radius in meters
M_MOON = Moon.mass.to(u.kg).value  # Mass of Moon in kg
M_SUN = Sun.mass.to(u.kg).value  # Mass of Sun in kg

class SatelliteAvoidanceEnv(gym.Env):
    def __init__(self, debris_positions, max_debris=100, satellite_distance=700e3):
        super(SatelliteAvoidanceEnv, self).__init__()

        # Action space: changes in velocity in x, y, z directions
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)

        # Observation space: satellite position, velocity, fuel mass, and debris positions
        self.max_debris = max_debris
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7 + self.max_debris * 3,), dtype=np.float32
        )

        # Satellite initial parameters
        self.initial_orbit = Orbit.circular(Earth, alt=satellite_distance * u.m)
        self.satellite_mass = 1000.0  # kg
        self.fuel_mass = 500.0  # kg
        self.satellite_position = self.initial_orbit.r.to(u.m).value
        self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value

        # Time settings
        self.time_step = 10.0  # seconds
        self.current_time = Time.now()

        # Debris initialization
        self.debris_positions = [np.array(debris, dtype=np.float64) for debris in debris_positions]

        # Ephemeris for Moon and Sun
        self.ephem_moon = build_ephem_interpolant(
            Moon, 28 * u.day, (self.current_time, self.current_time + 28 * u.day), rtol=1e-2
        )
        self.ephem_sun = build_ephem_interpolant(
            Sun, 365 * u.day, (self.current_time, self.current_time + 365 * u.day), rtol=1e-2
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.initial_orbit = Orbit.circular(Earth, alt=700e3 * u.m, epoch=Time.now())
        self.satellite_position = self.initial_orbit.r.to(u.m).value
        self.satellite_velocity = self.initial_orbit.v.to(u.m / u.s).value
        self.satellite_mass = 1000.0
        self.fuel_mass = 500.0
        self.current_time = self.initial_orbit.epoch

        # Reinitialize debris positions
        self.debris_positions = [np.random.randn(3) * 10000 for _ in range(self.max_debris)]

        return self._get_obs(), {}

    def _get_obs(self):
        debris_flat = np.array(self.debris_positions).flatten()
        if len(self.debris_positions) > self.max_debris:
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
        r_vec = self.satellite_position
        r_norm = np.linalg.norm(r_vec)
        a_earth = -G_const * M_EARTH * r_vec / r_norm**3

        moon_pos = np.asarray(self.ephem_moon(self.current_time).get_xyz().to(u.m).value.flatten(), dtype=np.float64)

        print("Moon Position Type:", type(moon_pos), "Values:", moon_pos)

        r_moon = moon_pos - self.satellite_position
        r_moon_norm = np.linalg.norm(r_moon)
        a_moon = G_const * M_MOON * r_moon / r_moon_norm**3

        sun_pos = self.ephem_sun(self.current_time).get_xyz().to(u.m).value.flatten()
        r_sun = sun_pos - self.satellite_position
        r_sun_norm = np.linalg.norm(r_sun)
        a_sun = G_const * M_SUN * r_sun / r_sun_norm**3

        a_total = a_earth + a_moon + a_sun
        self.satellite_velocity += a_total * self.time_step

    def _apply_thrust(self, action):
        max_thrust = 0.1  # N/kg (acceleration)
        thrust_acceleration = action * max_thrust
        self.satellite_velocity += thrust_acceleration * self.time_step

        specific_impulse = 300.0  # s
        g0 = 9.80665  # m/s^2
        delta_v = np.linalg.norm(thrust_acceleration * self.time_step)
        fuel_consumed = self.satellite_mass * (1 - np.exp(-delta_v / (specific_impulse * g0)))

        self.fuel_mass -= fuel_consumed
        self.satellite_mass -= fuel_consumed

        if self.fuel_mass < 0:
            self.fuel_mass = 0
            print("Out of fuel!")

    def step(self, action):
        self._apply_thrust(action)
        self._apply_gravitational_force()
        self.satellite_position += self.satellite_velocity * self.time_step
        self.current_time += self.time_step * u.s

        reward, done = self._calculate_reward()
        truncated = False
        if self.fuel_mass <= 0:
            done = True

        return self._get_obs(), reward, done, truncated, {}

    def _calculate_reward(self):
        reward = 0.0
        done = False

        # Fuel consumption penalty
        reward -= (500.0 - self.fuel_mass) * 0.1

        # Altitude maintenance penalty
        desired_altitude = 700e3
        current_altitude = np.linalg.norm(self.satellite_position) - EARTH_RADIUS
        altitude_error = np.abs(current_altitude - desired_altitude)
        reward -= altitude_error * 0.01

        # Eccentricity penalty
        orbit_radius = np.linalg.norm(self.satellite_position)
        orbit_velocity = np.linalg.norm(self.satellite_velocity)
        specific_energy = 0.5 * orbit_velocity**2 - G_const * M_EARTH / orbit_radius
        specific_angular_momentum = np.cross(self.satellite_position, self.satellite_velocity)
        eccentricity_vector = np.cross(
            self.satellite_velocity, specific_angular_momentum
        ) / (G_const * M_EARTH) - self.satellite_position / orbit_radius
        eccentricity = np.linalg.norm(eccentricity_vector)
        reward -= eccentricity * 100.0

        # Collision penalty
        for debris in self.debris_positions:
            distance_to_debris = np.linalg.norm(self.satellite_position - debris)
            if distance_to_debris < 10e3:
                reward -= 1000.0
                done = True

        # Earth impact or escape penalty
        r_norm = np.linalg.norm(self.satellite_position)
        if r_norm < EARTH_RADIUS + 100e3:
            reward -= 1000.0
            done = True
        elif r_norm > 1e8:
            reward -= 1000.0
            done = True

        return reward, done

    def render(self, mode='human'):
        print(f"Time: {self.current_time.iso}")
        print(f"Position (m): {self.satellite_position}")
        print(f"Velocity (m/s): {self.satellite_velocity}")
        print(f"Fuel Mass (kg): {self.fuel_mass}")

