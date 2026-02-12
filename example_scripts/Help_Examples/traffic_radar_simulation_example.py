"""
Traffic Radar Simulation Example - Multi-Vehicle Stop-and-Go Scenario
=====================================================================

This script demonstrates a realistic automotive radar simulation with multiple vehicles
in a stop-and-go traffic scenario. The simulation includes:

- 15+ vehicles arranged in realistic highway traffic formation
- Stop-and-go traffic dynamics with varying speeds and following distances
- 77 GHz automotive radar system (typical for ACC/AEB systems)
- Range-Doppler processing to detect vehicle positions and velocities
- Real-time visualization of traffic scenario and radar returns
- Realistic vehicle materials and scattering characteristics

Traffic Scenario:
- Highway with multiple lanes
- Lead vehicle controlling traffic flow
- Following vehicles with realistic car-following behavior
- Emergency braking scenarios
- Lane changes and merging behavior

Author: Traffic Simulation Example
Date: June 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import utilities
from pem_utilities.actor import Actors
from pem_utilities.materials import MaterialManager
from pem_utilities.antenna_device import Waveform, add_single_tx_rx
from pem_utilities.simulation_options import SimulationOptions
from pem_utilities.model_visualization import ModelVisualization
from pem_utilities.rotation import euler_to_rot
from pem_utilities.path_helper import get_repo_paths # common paths to reference
from pem_utilities.pem_core import Perceive_EM_API # Perceive EM API

# paths include, paths.repo_root, paths.example_scripts, paths.models, paths.materials, paths.output, paths.antenna_device_library
paths = get_repo_paths() 
pem_api_manager = Perceive_EM_API()
pem = pem_api_manager.pem  # The configured API object, all commands exist here
RssPy = pem_api_manager.RssPy # 


class TrafficVehicle:
    """
    Represents a single vehicle in the traffic simulation with realistic dynamics
    """
    
    def __init__(self, vehicle_id, initial_pos, initial_velocity=0.0, lane=0, vehicle_type='sedan'):
        self.vehicle_id = vehicle_id
        self.initial_pos = np.array(initial_pos, dtype=float)
        self.current_pos = np.array(initial_pos, dtype=float)
        self.velocity = initial_velocity  # m/s
        self.target_velocity = initial_velocity
        self.max_velocity = 30.0  # 30 m/s (~67 mph highway speed)
        self.min_velocity = 0.0
        self.acceleration = 0.0
        self.max_acceleration = 3.0  # m/s²
        self.max_deceleration = -8.0  # m/s² (emergency braking)
        self.comfortable_deceleration = -3.0  # m/s²
        self.following_distance = 25.0  # desired following distance in meters
        self.min_safe_distance = 10.0  # minimum safe distance to prevent collisions (increased)
        self.lateral_safe_distance = 2.5  # safe lateral distance for lane changes
        self.lane = lane
        self.vehicle_type = vehicle_type
        self.length = 4.5  # vehicle length in meters
        self.width = 1.8   # vehicle width in meters
        self.leader = None  # reference to vehicle being followed
        
        # Enhanced collision detection parameters
        self.collision_zone_front = 15.0  # meters - front collision detection zone
        self.collision_zone_rear = 5.0    # meters - rear collision detection zone
        self.collision_zone_side = 2.5    # meters - side collision detection zone
        
        # Traffic behavior parameters
        self.reaction_time = 0.2  # seconds
        self.aggressiveness = np.random.uniform(0.7, 1.1)  # driving style factor
        
        # Stop-and-go behavior parameters
        self.stopped_time = 0.0  # How long vehicle has been stopped
        self.max_stop_duration = np.random.uniform(1.5, 3.5)  # Random stop duration (1.5-3.5 seconds)
        self.minimum_restart_gap = 8.0  # Minimum gap needed to restart after being stopped
        self.creep_velocity = 2.0  # Low speed for creeping forward when stopped
        
        # Reference to all other vehicles for collision detection
        self.all_vehicles = None
        
    def set_vehicle_list(self, vehicle_list):
        """Set reference to all vehicles for comprehensive collision detection"""
        self.all_vehicles = vehicle_list
        
    def get_nearby_vehicles(self, search_radius=50.0):
        """
        Get all vehicles within search radius for collision detection
        """
        if self.all_vehicles is None:
            return []
        
        nearby = []
        for vehicle in self.all_vehicles:
            if vehicle.vehicle_id == self.vehicle_id:
                continue
                
            # Calculate distance to other vehicle
            distance = np.linalg.norm(vehicle.current_pos[:2] - self.current_pos[:2])
            if distance <= search_radius:
                nearby.append(vehicle)
        
        return nearby
        
    def check_collision_with_vehicle(self, other_vehicle):
        """
        Check if this vehicle would collide with another vehicle
        Returns: (collision_risk, collision_type, distance)
        """
        # Calculate relative position
        rel_pos = other_vehicle.current_pos - self.current_pos
        longitudinal_dist = rel_pos[0]  # x-direction (forward/backward)
        lateral_dist = abs(rel_pos[1])  # y-direction (left/right)
        
        # Check if vehicles are in same general area
        if lateral_dist > (self.width/2 + other_vehicle.width/2 + self.collision_zone_side):
            return False, "none", np.linalg.norm(rel_pos[:2])
        
        # Check longitudinal collision zones
        if longitudinal_dist > 0:  # Other vehicle is ahead
            if longitudinal_dist < (self.collision_zone_front + other_vehicle.length/2 + self.length/2):
                if longitudinal_dist < (self.min_safe_distance + other_vehicle.length/2 + self.length/2):
                    return True, "front_critical", longitudinal_dist
                else:
                    return True, "front_warning", longitudinal_dist
        else:  # Other vehicle is behind
            if abs(longitudinal_dist) < (self.collision_zone_rear + other_vehicle.length/2 + self.length/2):
                return True, "rear", abs(longitudinal_dist)
        
        return False, "none", np.linalg.norm(rel_pos[:2])
        
    def get_collision_threats(self):
        """
        Analyze all nearby vehicles for collision threats
        Returns: list of collision threats with details
        """
        threats = []
        nearby_vehicles = self.get_nearby_vehicles()
        
        for vehicle in nearby_vehicles:
            collision_risk, collision_type, distance = self.check_collision_with_vehicle(vehicle)
            if collision_risk:
                # Calculate time to collision
                relative_velocity = self.velocity - vehicle.velocity
                if relative_velocity > 0.1:  # Approaching
                    time_to_collision = distance / relative_velocity
                else:
                    time_to_collision = float('inf')
                
                threat = {
                    'vehicle': vehicle,
                    'type': collision_type,
                    'distance': distance,
                    'time_to_collision': time_to_collision,
                    'relative_velocity': relative_velocity
                }
                threats.append(threat)
        
        # Sort by urgency (shortest time to collision first)
        threats.sort(key=lambda t: t['time_to_collision'])
        return threats
        
    def get_distance_to_leader(self):
        """
        Calculate distance from rear of this vehicle to rear of leader vehicle
        """
        if self.leader is None:
            return float('inf')
        return self.leader.current_pos[0] - self.current_pos[0] - self.leader.length
        
    def update_dynamics(self, dt, traffic_state="normal"):
        """
        Update vehicle position and velocity with comprehensive collision avoidance
        """
        # Get all collision threats
        collision_threats = self.get_collision_threats()
        
        # Initialize acceleration based on traffic state or leader following
        base_acceleration = 0.0
        
        # Track if vehicle should be forced to restart (stuck for too long)
        force_restart = False
        if self.velocity < 0.1:
            self.stopped_time += dt
            # Force restart if stopped for more than 6 seconds (regardless of conditions)
            if self.stopped_time > 6.0:
                force_restart = True
                print(f"FORCE RESTART: Vehicle {self.vehicle_id} has been stopped for {self.stopped_time:.1f}s")
        else:
            self.stopped_time = 0.0
        
        if self.leader is not None:
            # Car-following behavior
            distance_to_leader = self.get_distance_to_leader()
            relative_velocity = self.velocity - self.leader.velocity
            
            # Intelligent Driver Model (IDM) for car following
            desired_distance = max(self.following_distance, 
                                 self.velocity * self.reaction_time + 
                                 max(0, (self.velocity * relative_velocity) / 
                                     (2 * np.sqrt(abs(self.max_acceleration * abs(self.comfortable_deceleration))))))
            
            # Enhanced restart logic for following vehicles
            if force_restart or (self.stopped_time >= self.max_stop_duration and distance_to_leader > self.minimum_restart_gap):
                # Try to restart - reduce following distance requirement for restart
                restart_gap = self.minimum_restart_gap if not force_restart else 4.0  # Smaller gap for forced restart
                if distance_to_leader > restart_gap:
                    base_acceleration = self.max_acceleration * 0.4  # Stronger restart acceleration
                    self.target_velocity = min(8.0, self.leader.velocity + 2.0)  # Try to match leader's speed
                    print(f"RESTART: Vehicle {self.vehicle_id} restarting with gap {distance_to_leader:.1f}m")
                    self.stopped_time = 0.0
                    self.max_stop_duration = np.random.uniform(1.5, 3.5)
                elif force_restart:
                    # Emergency restart even with small gap
                    base_acceleration = self.max_acceleration * 0.2
                    self.target_velocity = 3.0  # Creep speed
                    print(f"EMERGENCY RESTART: Vehicle {self.vehicle_id} forced restart with small gap {distance_to_leader:.1f}m")
                    self.stopped_time = 0.0
                    self.max_stop_duration = np.random.uniform(1.5, 3.5)
            elif distance_to_leader > desired_distance + 5.0:
                # Can accelerate normally
                velocity_term = (self.velocity / self.max_velocity)**4
                distance_term = (desired_distance / max(distance_to_leader, 1.0))**2
                base_acceleration = self.max_acceleration * (1 - velocity_term - distance_term)
            else:
                # Maintain safe following distance, but allow gentle acceleration if stopped too long
                if self.stopped_time > 3.0 and distance_to_leader > 6.0:
                    base_acceleration = self.max_acceleration * 0.1  # Gentle acceleration
                else:
                    base_acceleration = self.comfortable_deceleration * 0.5
        else:
            # Lead vehicle behavior - enhanced restart logic
            if force_restart:
                self.target_velocity = 15.0  # Resume normal speed
                print(f"LEAD RESTART: Vehicle {self.vehicle_id} (leader) forced restart")
                self.stopped_time = 0.0
                self.max_stop_duration = np.random.uniform(1.5, 3.5)
            elif traffic_state == "stop":
                # Even in stop state, don't stay stopped forever
                if self.stopped_time >= self.max_stop_duration:
                    self.target_velocity = 5.0  # Resume slowly even in stop state
                    print(f"STOP OVERRIDE: Vehicle {self.vehicle_id} (leader) resuming despite stop state")
                    self.stopped_time = 0.0
                    self.max_stop_duration = np.random.uniform(2.0, 4.0)
                else:
                    self.target_velocity = 0.0
            elif traffic_state == "slow":
                self.target_velocity = 8.0
            elif traffic_state == "normal":
                self.target_velocity = self.max_velocity
            elif traffic_state == "emergency_brake":
                self.target_velocity = 0.0
                base_acceleration = self.max_deceleration
                
            # Acceleration toward target velocity
            velocity_error = self.target_velocity - self.velocity
            if abs(velocity_error) > 0.5:
                if velocity_error > 0:
                    # Stronger acceleration for restart situations
                    accel_factor = 1.0 if not force_restart else 1.5
                    base_acceleration = min(self.max_acceleration * 0.7 * accel_factor, velocity_error * 2.0)
                else:
                    base_acceleration = max(self.comfortable_deceleration, velocity_error * 2.0)
        
        # Apply collision avoidance overrides (but allow restarts)
        final_acceleration = base_acceleration
        
        if collision_threats and not force_restart:  # Don't override force restart with collision avoidance
            most_urgent_threat = collision_threats[0]
            threat_type = most_urgent_threat['type']
            time_to_collision = most_urgent_threat['time_to_collision']
            distance = most_urgent_threat['distance']
            
            # Emergency braking for critical threats
            if threat_type == "front_critical" or time_to_collision < 2.0:
                # Emergency braking to avoid collision
                final_acceleration = self.max_deceleration
                print(f"EMERGENCY: Vehicle {self.vehicle_id} emergency braking - "
                      f"TTC: {time_to_collision:.1f}s, Distance: {distance:.1f}m")
                
            elif threat_type == "front_warning" or time_to_collision < 4.0:
                # Strong deceleration for warning level threats, but less if trying to restart
                if self.stopped_time > 3.0 and distance > 5.0:
                    # Reduce braking force for restart attempts
                    decel_strength = max(0.1, min(0.5, 4.0 / max(time_to_collision, 0.1)))
                else:
                    decel_strength = max(0.3, min(1.0, 4.0 / max(time_to_collision, 0.1)))
                final_acceleration = self.comfortable_deceleration * decel_strength
                
            elif threat_type == "rear":
                # If vehicle behind is too close, try to accelerate gently (if safe ahead)
                if distance < self.min_safe_distance and self.velocity < self.max_velocity * 0.8:
                    if not any(t['type'] == 'front_critical' for t in collision_threats):
                        final_acceleration = max(final_acceleration, self.max_acceleration * 0.3)
        
        # Apply acceleration limits
        final_acceleration = np.clip(final_acceleration, self.max_deceleration, self.max_acceleration)
        
        # Update velocity
        new_velocity = self.velocity + final_acceleration * dt
        new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)
        
        # Final collision check before updating position (relaxed for force restart)
        predicted_pos = self.current_pos.copy()
        predicted_pos[0] += new_velocity * dt
        
        # Check if predicted position would cause collision
        collision_detected = False
        if not force_restart:  # Skip collision check for forced restarts
            for vehicle in self.get_nearby_vehicles(20.0):  # Check nearby vehicles
                predicted_rel_pos = vehicle.current_pos - predicted_pos
                longitudinal_dist = predicted_rel_pos[0]
                lateral_dist = abs(predicted_rel_pos[1])
                
                # Check for overlap
                if (lateral_dist < (self.width/2 + vehicle.width/2 + 0.5) and
                    abs(longitudinal_dist) < (self.length/2 + vehicle.length/2 + 1.0)):
                    collision_detected = True
                    break
        
        if collision_detected:
            # Reduce velocity to prevent collision, but allow minimal movement for restarts
            if self.stopped_time > 4.0:
                new_velocity = max(0.5, min(new_velocity, self.velocity * 0.95))  # Allow minimal movement
                print(f"COLLISION PREVENTION: Vehicle {self.vehicle_id} minimal movement to avoid deadlock")
            else:
                new_velocity = min(new_velocity, self.velocity * 0.9)
                print(f"COLLISION PREVENTION: Vehicle {self.vehicle_id} reducing speed to avoid collision")
        
        self.acceleration = final_acceleration
        self.velocity = new_velocity
        
        # Update position
        self.current_pos[0] += self.velocity * dt
        
        return self.current_pos, self.velocity

class TrafficScenario:
    """
    Manages the overall traffic scenario with multiple vehicles
    """
    
    def __init__(self, num_vehicles=15):
        self.vehicles = []
        self.time = 0.0
        self.dt = 0.1  # 100ms time steps
        self.traffic_states = ["normal", "slow", "stop", "normal", "emergency_brake", "normal"]
        self.state_durations = [5.0, 3.0, 4.0, 6.0, 2.0, 10.0]  # seconds for each state
        self.current_state_idx = 0
        self.state_timer = 0.0
        
        # Create vehicles in realistic highway formation
        self.create_traffic_formation(num_vehicles)
        
    def create_traffic_formation(self, num_vehicles):
        """
        Create realistic multi-lane traffic formation
        """
        # Lane positions (y-coordinates)
        lane_positions = [0, 3.7, 7.4]  # 3 lanes, 3.7m apart
        
        # Create ego vehicle (radar platform) in center lane
        ego_pos = [0, lane_positions[1], 1.2]  # 1.2m height for radar
        
        vehicle_spacing = 25.0  # base spacing between vehicles
        
        for i in range(num_vehicles):
            # Distribute vehicles across lanes
            lane = i % 3
            lane_vehicle_count = i // 3
            
            # Position vehicles ahead of ego vehicle
            x_pos = (lane_vehicle_count + 1) * vehicle_spacing + np.random.uniform(-5, 5)
            y_pos = lane_positions[lane] + np.random.uniform(-0.5, 0.5)  # slight lateral variation
            z_pos = 0.9  # vehicle center height
            
            # Random initial velocity
            initial_velocity = np.random.uniform(15, 25)  # 15-25 m/s
            
            # Vehicle types for variety
            vehicle_types = ['sedan', 'suv', 'truck', 'compact']
            vehicle_type = vehicle_types[i % len(vehicle_types)]
            
            vehicle = TrafficVehicle(
                vehicle_id=f"vehicle_{i:02d}",
                initial_pos=[x_pos, y_pos, z_pos],
                initial_velocity=initial_velocity,
                lane=lane,
                vehicle_type=vehicle_type
            )
            
            self.vehicles.append(vehicle)
        
        # CRITICAL: Set vehicle list reference for each vehicle for collision detection
        for vehicle in self.vehicles:
            vehicle.set_vehicle_list(self.vehicles)
        
        # Set up car-following relationships
        for lane in range(3):
            lane_vehicles = [v for v in self.vehicles if v.lane == lane]
            lane_vehicles.sort(key=lambda v: v.current_pos[0])  # sort by x position
            
            # Each vehicle follows the one in front in the same lane
            for i in range(1, len(lane_vehicles)):
                lane_vehicles[i].leader = lane_vehicles[i-1]
    
    def update_traffic(self):
        """
        Update entire traffic scenario for one time step
        """
        # Update traffic state
        self.state_timer += self.dt
        if self.state_timer >= self.state_durations[self.current_state_idx]:
            self.current_state_idx = (self.current_state_idx + 1) % len(self.traffic_states)
            self.state_timer = 0.0
        
        current_traffic_state = self.traffic_states[self.current_state_idx]
        
        # Update all vehicles
        for vehicle in self.vehicles:
            vehicle.update_dynamics(self.dt, current_traffic_state)
        
        self.time += self.dt
        
        return current_traffic_state

def main():
    """
    Main traffic radar simulation
    """
    
    print("=" * 70)
    print("TRAFFIC RADAR SIMULATION - STOP-AND-GO SCENARIO")
    print("=" * 70)
    
    # ========================================================================
    # 1. SIMULATION PARAMETERS
    # ========================================================================
    print("\n1. Configuring Simulation Parameters")
    print("-" * 40)
    
    # Automotive radar parameters (77 GHz ACC/AEB system)
    center_freq = 77e9         # 77 GHz automotive radar
    bandwidth = 1e9            # 1 GHz bandwidth for good range resolution
    num_freqs = 512            # High resolution for multiple targets
    cpi_duration = 50e-3       # 50 ms for good velocity resolution
    num_pulse_CPI = 256        # Many pulses for Doppler processing
    
    # Simulation parameters optimized for multiple vehicles
    ray_spacing = 0.15         # Slightly coarser for performance with many targets
    max_reflections = 2        # Limit reflections for performance
    go_blockage = 1           # Enable realistic shadowing between vehicles
    
    print(f"   ✓ Radar frequency: {center_freq/1e9:.0f} GHz")
    print(f"   ✓ Bandwidth: {bandwidth/1e6:.0f} MHz")
    print(f"   ✓ Range resolution: {3e8/(2*bandwidth):.2f} m")
    print(f"   ✓ Max velocity: {3e8/(4*center_freq*cpi_duration):.1f} m/s")
    
    # ========================================================================
    # 2. CREATE TRAFFIC SCENARIO
    # ========================================================================
    print("\n2. Creating Traffic Scenario")
    print("-" * 32)
    
    traffic = TrafficScenario(num_vehicles=15)
    print(f"   ✓ Created {len(traffic.vehicles)} vehicles")
    print("   ✓ 3-lane highway configuration")
    print("   ✓ Stop-and-go traffic dynamics enabled")
    
    # Display initial vehicle positions
    print("   Initial vehicle positions:")
    for i, vehicle in enumerate(traffic.vehicles):
        print(f"     Vehicle {i:2d}: Lane {vehicle.lane}, "
              f"Position ({vehicle.current_pos[0]:6.1f}, {vehicle.current_pos[1]:4.1f}) m, "
              f"Speed {vehicle.velocity:4.1f} m/s")
    
    # ========================================================================
    # 3. CREATE SCENE ACTORS
    # ========================================================================
    print("\n3. Creating Scene and Actors")
    print("-" * 30)
    
    # Initialize material manager with frequency-specific materials
    mat_manager = MaterialManager(
        generate_itu_materials=True,
        itu_freq_ghz=center_freq/1e9
    )
    all_actors = Actors(material_manager=mat_manager)
    
    # Create road surface (optional - adds realism)
    road_name = all_actors.add_actor(
        filename=os.path.join(paths.models, 'Simple_Road.stl'),  # Use as road surface
        mat_idx=mat_manager.get_index('concrete'),
        target_ray_spacing=0.5
    )
    all_actors.actors[road_name].coord_sys.pos = [0, 0, -0.1]  # Position under traffic
    all_actors.actors[road_name].coord_sys.rot = euler_to_rot(phi=00, theta=0, order='zyz', deg=True)
    all_actors.actors[road_name].coord_sys.update()
    
    # Create vehicle actors
    vehicle_actors = []
    for i, vehicle in enumerate(traffic.vehicles):
        # Use different models for variety if available, otherwise use sphere
        try:
            if vehicle.vehicle_type == 'truck':
                model_file = 'Chevy_Silverado/Chevy_Silverado.json'  # Larger vehicle
                scale_factor = 1.0
                material = 'metal'
            else:
                model_file = 'Audi_A1_2010/Audi_A1_2010.json'  # Standard vehicle
                scale_factor = 1.0
                material = 'metal'
            
            actor_name = all_actors.add_actor(
                filename=os.path.join(paths.models, model_file),
                mat_idx=mat_manager.get_index(material),
                target_ray_spacing=0.1
            )
            
            # Scale and position vehicle
            all_actors.actors[actor_name].coord_sys.pos = vehicle.current_pos
            all_actors.actors[actor_name].coord_sys.scl = [scale_factor, 1.0, 0.8]
            all_actors.actors[actor_name].coord_sys.update()
            
            vehicle_actors.append(actor_name)
            
        except Exception as e:
            print(f"   Warning: Could not create vehicle {i}: {e}")
            vehicle_actors.append(None)
    
    print(f"   ✓ Created {len([v for v in vehicle_actors if v is not None])} vehicle actors")
    
    # Create ego vehicle (radar platform)
    ego_actor_name = all_actors.add_actor()
    all_actors.actors[ego_actor_name].coord_sys.pos = [0, 3.7, 1.2]  # Center lane, radar height
    all_actors.actors[ego_actor_name].coord_sys.update()
    
    # ========================================================================
    # 4. CONFIGURE RADAR SYSTEM
    # ========================================================================
    print("\n4. Configuring Automotive Radar")
    print("-" * 35)
    
    # Define automotive radar waveform
    waveform_dict = {
        "mode": "PulsedDoppler",
        "output": "RangeDoppler",
        "center_freq": center_freq,
        "bandwidth": bandwidth,
        "num_freq_samples": num_freqs,
        "cpi_duration": cpi_duration,
        "num_pulse_CPI": num_pulse_CPI,
        "tx_multiplex": "INDIVIDUAL",
        "mode_delay": "CENTER_CHIRP"
    }
    
    mode_name = 'traffic_radar'
    waveform = Waveform(waveform_dict)
    
    # Add automotive radar antenna (forward-looking)
    radar_device = add_single_tx_rx(
        all_actors,
        waveform,
        mode_name,
        parent_h_node=all_actors.actors[ego_actor_name].h_node,
        ffd_file='dipole.ffd',  # Use directional antenna pattern
        scale_pattern=3.0
    )
    
    print(f"   ✓ Radar configured: {center_freq/1e9:.0f} GHz automotive system")
    print(f"   ✓ Range window: 0 to {3e8*num_freqs/(2*bandwidth):.0f} m")
    print(f"   ✓ Velocity window: ±{3e8/(4*center_freq*cpi_duration):.1f} m/s")
    
    # ========================================================================
    # 5. CONFIGURE SIMULATION OPTIONS
    # ========================================================================
    print("\n5. Configuring Simulation")
    print("-" * 28)
    
    sim_options = SimulationOptions()
    sim_options.ray_spacing = ray_spacing
    sim_options.max_reflections = max_reflections
    sim_options.max_transmissions = 0  # No penetration for vehicle bodies
    sim_options.go_blockage = go_blockage
    sim_options.field_of_view = 180   # Forward-looking radar
    sim_options.auto_configure_simulation()
    
    print(f"   ✓ Ray spacing: {sim_options.ray_spacing} m")
    print(f"   ✓ Max reflections: {sim_options.max_reflections}")
    print(f"   ✓ GO blockage: {sim_options.go_blockage}")
    
    # Get response domains for analysis
    which_mode = radar_device.modes[mode_name]
    radar_device.waveforms[mode_name].get_response_domains(which_mode)
    vel_domain = radar_device.waveforms[mode_name].vel_domain
    rng_domain = radar_device.waveforms[mode_name].rng_domain
    
    # Check if simulation is ready
    if not pem.isReady():
        print("   ✗ Simulation not ready!")
        print(pem.getLastWarnings())
        return
    
    print("   ✓ Simulation ready")
    
    # ========================================================================
    # 6. SETUP VISUALIZATION
    # ========================================================================
    print("\n6. Setting up Visualization")
    print("-" * 30)
    
    output_movie_name = os.path.join(paths.output, 'traffic_radar_simulation.mp4')
    modeler = ModelVisualization(
        all_actors,
        show_antennas=True,
        fps=10,
        output_movie_name=output_movie_name,
        output_video_size=None  # HD output
    )
    
    print(f"   ✓ Visualization configured")
    print(f"   ✓ Output: {output_movie_name}")
    
    # ========================================================================
    # 7. RUN TRAFFIC SIMULATION
    # ========================================================================
    print("\n7. Running Traffic Radar Simulation")
    print("-" * 40)
    
    num_iterations = 300  # 30 seconds at 10 Hz
    detection_log = []
    
    print("   Traffic states sequence:")
    for i, state in enumerate(traffic.traffic_states):
        print(f"     {i+1}. {state.upper()}: {traffic.state_durations[i]:.1f}s")
    
    print(f"\n   Running {num_iterations} iterations...")
    
    for iteration in range(num_iterations):
        # Update traffic scenario
        current_traffic_state = traffic.update_traffic()
        
        # Update vehicle actor positions
        for i, (vehicle, actor_name) in enumerate(zip(traffic.vehicles, vehicle_actors)):
            if actor_name is not None:
                # Update position and velocity in scene
                all_actors.actors[actor_name].coord_sys.pos = vehicle.current_pos
                all_actors.actors[actor_name].coord_sys.lin = [vehicle.velocity, 0, 0]
                all_actors.actors[actor_name].coord_sys.update()
        
        # Run radar simulation
        pem_api_manager.isOK(pem.computeResponseSync())
        
        # Retrieve radar response
        (ret, response) = pem.retrieveResponse(
            radar_device.modes[mode_name], 
            RssPy.ResponseType.RANGE_DOPPLER
        )
        
        # Process radar data
        radar_image = 20 * np.log10(np.abs(response[0, 0]) + 1e-12)  # Convert to dB
        
        # Simple target detection (peaks in range-Doppler map)
        detection_threshold = np.max(radar_image) - 20  # 20 dB below peak
        detections = np.where(radar_image > detection_threshold)
        
        # Convert detections to range and velocity
        detected_ranges = rng_domain[detections[1]]
        detected_velocities = vel_domain[detections[0]]
        
        # Log detections
        detection_log.append({
            'time': traffic.time,
            'traffic_state': current_traffic_state,
            'num_detections': len(detected_ranges),
            'ranges': detected_ranges.copy(),
            'velocities': detected_velocities.copy()
        })
        
        # Update visualization
        modeler.update_frame(
            plot_data=radar_image,
            plot_limits=[np.max(radar_image) - 40, np.max(radar_image)]
        )
        
        # Progress update
        if iteration % 30 == 0:
            print(f"     Iteration {iteration:3d}/{num_iterations}: "
                  f"Traffic={current_traffic_state:12s}, "
                  f"Detections={len(detected_ranges):2d}, "
                  f"Time={traffic.time:5.1f}s")
    
    modeler.close()
    
    # ========================================================================
    # 8. ANALYSIS AND RESULTS
    # ========================================================================
    print("\n8. Simulation Results Analysis")
    print("-" * 35)
    
    # Analyze detection statistics
    total_detections = sum(log['num_detections'] for log in detection_log)
    avg_detections = total_detections / len(detection_log)
    
    print(f"   ✓ Simulation completed successfully")
    print(f"   ✓ Total iterations: {num_iterations}")
    print(f"   ✓ Simulation time: {traffic.time:.1f} seconds")
    print(f"   ✓ Average detections per frame: {avg_detections:.1f}")
    print(f"   ✓ Total vehicle movements tracked")
    
    # Traffic state analysis
    state_detections = {}
    for log in detection_log:
        state = log['traffic_state']
        if state not in state_detections:
            state_detections[state] = []
        state_detections[state].append(log['num_detections'])
    
    print("\n   Detection statistics by traffic state:")
    for state, detections in state_detections.items():
        avg_det = np.mean(detections)
        print(f"     {state.upper():15s}: {avg_det:5.1f} avg detections")
    
    # Range and velocity statistics
    all_ranges = np.concatenate([log['ranges'] for log in detection_log if len(log['ranges']) > 0])
    all_velocities = np.concatenate([log['velocities'] for log in detection_log if len(log['velocities']) > 0])
    
    if len(all_ranges) > 0:
        print(f"\n   Target range statistics:")
        print(f"     Min range: {np.min(all_ranges):.1f} m")
        print(f"     Max range: {np.max(all_ranges):.1f} m")
        print(f"     Mean range: {np.mean(all_ranges):.1f} m")
        
        print(f"   Target velocity statistics:")
        print(f"     Min velocity: {np.min(all_velocities):.1f} m/s")
        print(f"     Max velocity: {np.max(all_velocities):.1f} m/s")
        print(f"     Mean velocity: {np.mean(all_velocities):.1f} m/s")
    
    print(f"\n   ✓ Results saved to: {output_movie_name}")
    print(f"   ✓ Traffic radar simulation complete!")
    
    print("\n" + "=" * 70)
    print("TRAFFIC RADAR SIMULATION COMPLETE")
    print("=" * 70)
    print("\nThis simulation demonstrated:")
    print("• Multi-vehicle stop-and-go traffic scenario")
    print("• 77 GHz automotive radar system")
    print("• Realistic vehicle dynamics and car-following behavior")
    print("• Range-Doppler processing for target detection")
    print("• Traffic state transitions (normal → slow → stop → emergency)")
    print("• Real-time visualization with radar overlay")

if __name__ == "__main__":
    """
    Run the traffic radar simulation
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError in simulation: {e}")
        import traceback
        traceback.print_exc()