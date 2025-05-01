#!/usr/bin/env python3
import os
import csv
import queue
import weakref
import carla
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

class CarlaSyncMode:
    """Fixed version for CARLA 0.9.8 with proper cleanup"""
    def __init__(self, world, *sensors, fps=30):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self._queues = []
        self._callbacks = []
        
    def __enter__(self):
        # World tick queue
        self._queues.append(queue.Queue())
        self._callbacks.append(self.world.on_tick(self._queues[-1].put))
        
        # Sensor queues
        for sensor in self.sensors:
            q = queue.Queue()
            self._queues.append(q)
            self._callbacks.append(sensor.listen(q.put))
            
        return self
    
    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data if hasattr(x, 'frame'))
        return data
    
    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if hasattr(data, 'frame'):
                if data.frame == self.frame:
                    return data
            else:  # IMU data
                return data
    
    def __exit__(self, *args, **kwargs):
        # Proper cleanup for CARLA 0.9.8
        for callback in self._callbacks:
            try:
                if hasattr(callback, 'stop'):
                    callback.stop()
                elif hasattr(callback, '__del__'):
                    callback.__del__()
            except:
                pass
                
        for sensor in self.sensors:
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except:
                pass

def process_image(image):
    """Convert CARLA image to PIL Image"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Remove alpha
    array = array[:, :, ::-1]  # BGR to RGB
    return Image.fromarray(array)

def process_depth(image):
    """Convert CARLA depth to grayscale PIL Image"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array.astype(np.float32)
    normalized_depth = (array[:, :, 0] + array[:, :, 1] * 256) / (256**2 - 1)
    return Image.fromarray((normalized_depth * 255).astype(np.uint8))

def main():
    try:
        # Setup output directories
        os.makedirs("output/rgb", exist_ok=True)
        os.makedirs("output/depth", exist_ok=True)
        
        # Connect to CARLA with longer timeout
        client = carla.Client("localhost", 2000)
        client.set_timeout(30.0)  # Increased from 10.0
        world = client.get_world()
        
        # Setup vehicle
        blueprint = world.get_blueprint_library().filter("model3")[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(blueprint, spawn_point)
        vehicle.set_autopilot(True)
        
        # Setup sensors
        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "360")
        camera_bp.set_attribute("fov", "90")
        
        depth_bp = world.get_blueprint_library().find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", "640")
        depth_bp.set_attribute("image_size_y", "360")
        depth_bp.set_attribute("fov", "90")
        
        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        
        # Attach sensors
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=vehicle)
        
        depth = world.spawn_actor(
            depth_bp,
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=vehicle)
        
        imu = world.spawn_actor(
            imu_bp,
            carla.Transform(),
            attach_to=vehicle)
        
        # Create CSV file
        csv_path = "output/driving_log.csv"
        columns = ["frame", "timestamp", "rgb_path", "depth_path", 
                  "accel_x", "accel_y", "accel_z", 
                  "gyro_x", "gyro_y", "gyro_z",
                  "steering", "throttle", "brake", "speed"]
        
        # Main collection loop
        with open(csv_path, "w", newline="") as csv_file, \
             CarlaSyncMode(world, camera, depth, imu, fps=10) as sync_mode:
            
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            
            for frame in tqdm(range(1000)):  # Collect 1000 frames
                try:
                    # Get synchronized data
                    _, rgb_data, depth_data, imu_data = sync_mode.tick(timeout=2.0)
                    
                    # Process and save images
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    rgb_path = f"output/rgb/{timestamp}.jpg"
                    depth_path = f"output/depth/{timestamp}.png"
                    
                    process_image(rgb_data).save(rgb_path, quality=95)
                    process_depth(depth_data).save(depth_path)
                    
                    # Get vehicle state
                    control = vehicle.get_control()
                    velocity = vehicle.get_velocity()
                    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
                    
                    # Write to CSV
                    writer.writerow({
                        "frame": frame,
                        "timestamp": timestamp,
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "accel_x": imu_data.accelerometer.x,
                        "accel_y": imu_data.accelerometer.y,
                        "accel_z": imu_data.accelerometer.z,
                        "gyro_x": imu_data.gyroscope.x,
                        "gyro_y": imu_data.gyroscope.y,
                        "gyro_z": imu_data.gyroscope.z,
                        "steering": control.steer,
                        "throttle": control.throttle,
                        "brake": control.brake,
                        "speed": speed
                    })
                    
                except Exception as e:
                    print(f"Skipping frame {frame} due to error: {e}")
                    continue
                    
    except KeyboardInterrupt:
        print("Collection stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Manual cleanup
        print("Cleaning up actors...")
        for actor in [camera, depth, imu, vehicle]:
            if actor and actor.is_alive:
                try:
                    actor.destroy()
                except:
                    pass
        print("Done.")

if __name__ == "__main__":
    main()