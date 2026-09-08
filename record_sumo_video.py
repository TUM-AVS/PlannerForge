#!/usr/bin/env python3
"""
Record SUMO Simulation as Video
Runs SUMO simulation and creates a video visualization
"""

import os
import sys
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
from collections import defaultdict

def create_sumo_config_with_output(scenario_folder, output_prefix="video"):
    """Create SUMO config with FCD output for video generation"""
    scenario_path = Path(scenario_folder)
    
    # Find network and route files
    net_file = None
    route_file = None
    
    for file in scenario_path.iterdir():
        if file.suffix == '.xml':
            if '.net.xml' in file.name:
                net_file = file.name
            elif '.vehicles.rou.xml' in file.name:
                route_file = file.name
    
    if not net_file or not route_file:
        raise FileNotFoundError(f"Could not find network or route file in {scenario_folder}")
    
    print(f"✅ Found scenario files:")
    print(f"   Network: {net_file}")
    print(f"   Routes:  {route_file}")
    
    # Create output folder
    output_folder = scenario_path / "video_output"
    output_folder.mkdir(exist_ok=True)
    
    # Create SUMO config
    config_file = scenario_path / f"{output_prefix}.sumocfg"
    fcd_output = output_folder / "fcd_output.xml"
    
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_file}"/>
        <route-files value="{route_file}"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="0.1"/>
    </time>
    <output>
        <fcd-output value="{fcd_output.relative_to(scenario_path)}"/>
    </output>
    <processing>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created SUMO config: {config_file.name}")
    
    return str(config_file), str(fcd_output), str(scenario_path / net_file)


def run_sumo_headless(config_file):
    """Run SUMO in command-line mode"""
    print("\n" + "="*80)
    print("RUNNING SUMO SIMULATION")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            ['sumo', '-c', config_file, '--start'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print("SUMO stderr:", result.stderr)
            raise RuntimeError(f"SUMO exited with code {result.returncode}")
        
        print("✅ SUMO simulation completed")
        return True
        
    except Exception as e:
        print(f"❌ Error running SUMO: {e}")
        return False


def load_network_bounds(net_file):
    """Load network boundaries from .net.xml"""
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        # Get network bounds from location element
        location = root.find('location')
        if location is not None:
            bounds = location.get('convBoundary')
            if bounds:
                x_min, y_min, x_max, y_max = map(float, bounds.split(','))
                return x_min, y_min, x_max, y_max
        
        # Fallback: calculate from edges
        x_coords = []
        y_coords = []
        
        for edge in root.findall('.//edge'):
            for lane in edge.findall('lane'):
                shape = lane.get('shape')
                if shape:
                    for point in shape.split():
                        x, y = map(float, point.split(','))
                        x_coords.append(x)
                        y_coords.append(y)
        
        if x_coords and y_coords:
            margin = 50  # Add margin
            return (min(x_coords) - margin, min(y_coords) - margin,
                    max(x_coords) + margin, max(y_coords) + margin)
        
        # Default bounds if nothing found
        return -500, -500, 500, 500
        
    except Exception as e:
        print(f"⚠️  Could not load network bounds: {e}")
        return -500, -500, 500, 500


def load_network_edges(net_file):
    """Load edge geometry from .net.xml"""
    edges = {}
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        for edge in root.findall('.//edge'):
            edge_id = edge.get('id')
            if edge_id and not edge_id.startswith(':'):  # Skip internal edges
                lanes = []
                for lane in edge.findall('lane'):
                    shape = lane.get('shape')
                    if shape:
                        points = []
                        for point in shape.split():
                            x, y = map(float, point.split(','))
                            points.append((x, y))
                        lanes.append(points)
                if lanes:
                    edges[edge_id] = lanes
        
        print(f"✅ Loaded {len(edges)} edges from network")
        return edges
        
    except Exception as e:
        print(f"⚠️  Could not load network edges: {e}")
        return {}


def load_fcd_data(fcd_file):
    """Load vehicle trajectories from FCD output"""
    print("\n" + "="*80)
    print("LOADING FCD DATA")
    print("="*80 + "\n")
    
    if not os.path.exists(fcd_file):
        raise FileNotFoundError(f"FCD output not found: {fcd_file}")
    
    tree = ET.parse(fcd_file)
    root = tree.getroot()
    
    # Store data by timestep
    timesteps = []
    
    for timestep_elem in root.findall('timestep'):
        time = float(timestep_elem.get('time'))
        vehicles = []
        
        for vehicle in timestep_elem.findall('vehicle'):
            vid = int(vehicle.get('id'))
            x = float(vehicle.get('x'))
            y = float(vehicle.get('y'))
            angle = float(vehicle.get('angle'))
            speed = float(vehicle.get('speed'))
            
            vehicles.append({
                'id': vid,
                'x': x,
                'y': y,
                'angle': angle,
                'speed': speed
            })
        
        timesteps.append({
            'time': time,
            'vehicles': vehicles
        })
    
    print(f"✅ Loaded {len(timesteps)} timesteps")
    print(f"   Duration: {timesteps[-1]['time']:.1f}s")
    print(f"   Max vehicles: {max(len(ts['vehicles']) for ts in timesteps)}")
    
    return timesteps


def create_video(fcd_file, net_file, output_path, target_vehicles=[16, 17], fps=10):
    """Create video from FCD data"""
    print("\n" + "="*80)
    print("CREATING VIDEO")
    print("="*80 + "\n")
    
    # Load data
    timesteps = load_fcd_data(fcd_file)
    edges = load_network_edges(net_file)
    x_min, y_min, x_max, y_max = load_network_bounds(net_file)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    
    # Draw road network
    print("Drawing road network...")
    for edge_id, lanes in edges.items():
        for lane_points in lanes:
            if len(lane_points) > 1:
                xs = [p[0] for p in lane_points]
                ys = [p[1] for p in lane_points]
                ax.plot(xs, ys, 'gray', linewidth=1, alpha=0.3, zorder=1)
    
    # Prepare vehicle artists
    vehicle_markers = {}
    vehicle_texts = {}
    
    # Title
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=16, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    print(f"Creating animation with {len(timesteps)} frames at {fps} FPS...")
    
    def init():
        """Initialize animation"""
        return []
    
    def update(frame_idx):
        """Update frame"""
        if frame_idx >= len(timesteps):
            return []
        
        timestep_data = timesteps[frame_idx]
        time = timestep_data['time']
        vehicles = timestep_data['vehicles']
        
        # Update time display
        time_text.set_text(f'Time: {time:.1f}s | Frame: {frame_idx}/{len(timesteps)}')
        
        # Get current vehicle IDs
        current_ids = {v['id'] for v in vehicles}
        
        # Remove vehicles that are no longer present
        for vid in list(vehicle_markers.keys()):
            if vid not in current_ids:
                vehicle_markers[vid].remove()
                vehicle_texts[vid].remove()
                del vehicle_markers[vid]
                del vehicle_texts[vid]
        
        # Update or create vehicles
        for v in vehicles:
            vid = v['id']
            
            # Choose color and size based on vehicle ID
            if vid in target_vehicles:
                color = 'red'
                size = 200
                zorder = 10
            else:
                color = 'blue'
                size = 100
                zorder = 5
            
            # Determine if stopped
            if v['speed'] < 0.1:
                color = 'orange' if vid in target_vehicles else 'lightblue'
            
            if vid in vehicle_markers:
                # Update existing vehicle
                vehicle_markers[vid].set_offsets([[v['x'], v['y']]])
                vehicle_markers[vid].set_color(color)
                vehicle_texts[vid].set_position((v['x'], v['y'] + 3))
            else:
                # Create new vehicle
                marker = ax.scatter([v['x']], [v['y']], 
                                   c=color, s=size, marker='o',
                                   edgecolors='black', linewidths=1,
                                   zorder=zorder)
                text = ax.text(v['x'], v['y'] + 3, str(vid),
                              fontsize=8, ha='center', va='bottom',
                              fontweight='bold' if vid in target_vehicles else 'normal',
                              zorder=zorder+1)
                vehicle_markers[vid] = marker
                vehicle_texts[vid] = text
        
        return list(vehicle_markers.values()) + list(vehicle_texts.values()) + [time_text]
    
    # Create animation
    # Sample every Nth frame for reasonable video length
    sample_rate = max(1, len(timesteps) // (60 * fps))  # Cap at 60 seconds
    frames = range(0, len(timesteps), sample_rate)
    
    print(f"Sampling every {sample_rate} frame(s) -> {len(list(frames))} output frames")
    
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                        blit=False, repeat=True, interval=1000/fps)
    
    # Save video
    print(f"\nSaving video to: {output_path}")
    
    # Try MP4 first (requires ffmpeg)
    if output_path.endswith('.mp4'):
        try:
            writer = FFMpegWriter(fps=fps, bitrate=2000, 
                                 extra_args=['-vcodec', 'libx264'])
            anim.save(output_path, writer=writer, dpi=100)
            print(f"✅ Video saved as MP4: {output_path}")
        except Exception as e:
            print(f"⚠️  Could not save as MP4: {e}")
            print("Trying GIF format instead...")
            output_path = output_path.replace('.mp4', '.gif')
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer, dpi=80)
            print(f"✅ Video saved as GIF: {output_path}")
    else:
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=80)
        print(f"✅ Video saved as GIF: {output_path}")
    
    plt.close(fig)
    
    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 record_sumo_video.py <scenario_folder> [output_file] [fps]")
        print("\nExample:")
        print("  python3 record_sumo_video.py Scenarios/DEU_Weimar-71_1_T-4/Modified/TEST_DEBUG")
        print("  python3 record_sumo_video.py Scenarios/DEU_Weimar-71_1_T-4/Modified/TEST_DEBUG simulation.mp4 10")
        sys.exit(1)
    
    scenario_folder = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "sumo_simulation.mp4"
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    if not os.path.exists(scenario_folder):
        print(f"❌ Scenario folder not found: {scenario_folder}")
        sys.exit(1)
    
    print("="*80)
    print("SUMO VIDEO RECORDER")
    print("="*80)
    print(f"\n📁 Scenario: {scenario_folder}")
    print(f"🎬 Output: {output_file}")
    print(f"🎥 FPS: {fps}\n")
    
    try:
        # Step 1: Create SUMO config
        config_file, fcd_file, net_file = create_sumo_config_with_output(
            scenario_folder, output_prefix="video"
        )
        
        # Step 2: Run SUMO simulation
        success = run_sumo_headless(config_file)
        if not success:
            print("\n❌ SUMO simulation failed")
            sys.exit(1)
        
        # Step 3: Create video from FCD output
        # Determine output path
        if not os.path.isabs(output_file):
            output_path = os.path.join(scenario_folder, output_file)
        else:
            output_path = output_file
        
        create_video(fcd_file, net_file, output_path, target_vehicles=[16, 17], fps=fps)
        
        print("\n" + "="*80)
        print("✅ VIDEO CREATION COMPLETE!")
        print("="*80)
        print(f"\n📹 Video saved to: {output_path}")
        print(f"💡 You can now watch this video to see the simulation")
        print(f"🎯 Vehicles 16 and 17 are highlighted in RED")
        print(f"🔵 Other vehicles are shown in BLUE")
        print(f"🟠 Stopped vehicles (speed < 0.1 m/s) are shown in ORANGE/LIGHT BLUE")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

