# This code contains helper functions for dealing with the road network in relation to the ego vehicle's position

from pathlib import Path
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point

# TODO: Fix turning priorities nomenclature if desired
traffic_sign_dict = {
    "102": "right_before_left_rule",
    "205": "yield",
    "206": "stop",
    "301": "right_of_way",
    "306": "priority_road",
    # Turning priority roads are currently classified into three categories based on the direction for which the ego lane has priority; this is enough for the DB search (the internal logic corresponding to the size of the intersection remains) --> but is it enough for modification?
    "1002-10": "turning_priority_left",
    "1002-11": "turning_priority_wait",
    "1002-12": "turning_priority_left",
    "1002-13": "turning_priority_left",
    "1002-14": "turning_priority_wait",
    "1002-20": "turning_priority_right",
    "1002-21": "turning_priority_wait",
    "1002-22": "turning_priority_right",
    "1002-23": "turning_priority_right",
    "1002-24": "turning_priority_wait",
    "274": "speed_limit",
    "275": "required_speed",
    "276": "no_overtaking",
    "720": "green_arrow_sign",
    "310": "town_sign",
    "260": "ban_on_motorcycles_and_multi_lane_vehicles",
    "272": "no_u_turn"
}

# Simple function to load in an XML file and transform it into a string
def load_xml(file_path_str:str):
    file_path = Path(file_path_str)
    xml_str = file_path.read_text()
    return xml_str

def lanelets_to_polygons(xml_string:str):
    root = ET.fromstring(xml_string)
    polygon_dict = dict()

    for lanelet in root.findall('lanelet'):
        lanelet_id = lanelet.attrib['id']

        left_bound_points = []
        right_bound_points = []

        for element in lanelet.findall('leftBound'):
            for point in element.findall('point'):
                left_bound_points.append([float(point.find("x").text), float(point.find("y").text)])

        for element in lanelet.findall('rightBound'):
            for point in element.findall('point'):
                right_bound_points.append([float(point.find("x").text), float(point.find("y").text)])

        if not left_bound_points or not right_bound_points:
            continue  # Skip this lanelet if any boundary is missing

        # Reverse the right bound to ensure the correct clockwise or counter-clockwise order
        right_bound_points.reverse()

        # Combine leftBound and reversed rightBound to form a loop
        polygon_points = left_bound_points + right_bound_points

        # Create a Shapely Polygon from the points
        try:
            polygon = Polygon(polygon_points)
            # Ensure the polygon is valid (may raise ValueError for incomplete or degenerate shapes)
            if not polygon.is_valid:
                raise ValueError(f"Polygon is invalid for lanelet {lanelet_id}")

            # Add the polygon to the dictionary with the corresponding ID
            polygon_dict[lanelet_id] = polygon
        except (ValueError, Exception) as e:
            print(f"Error creating a polygon for lanelet {lanelet_id}: {e}")

    # Return the dictionary of lanelet ID to Polygon mappings
    return polygon_dict

# Function to check lanelets for a single point
def check_point_in_polygons(x, y, polygons):

    current_point = Point(x, y)
    for polygon_id, polygon in polygons.items():
        if polygon.contains(current_point):
            return polygon_id
    return None

# Uses Polygon approach to determine the lanelet belonging to the ego
def find_ego_lanelet(xml_string:str):
    root = ET.fromstring(xml_string)

    ego_point = [
        float(root.find("planningProblem").find("initialState").find("position").find("point").find("x").text),
        float(root.find("planningProblem").find("initialState").find("position").find("point").find("y").text)
    ]

    polygons = lanelets_to_polygons(xml_string)

    ego_lanelet_id = check_point_in_polygons(ego_point[0], ego_point[1], polygons)

    return ego_lanelet_id

# Construct and return egoLanelet as en ET.Element
# Considers traffic lights and traffic signs on the ego lanelet currently
# Also returns information on the adjacent lanes left and right
def characterize_ego_lanelet(xml_string:str) -> ET.Element:
    ego_lanelet_id = find_ego_lanelet(xml_string)
    root = ET.fromstring(xml_string)

    new_element = ET.Element("egoLane")
    new_element.attrib['id'] = ego_lanelet_id

    ego_lanelet = None

    for lanelet in root.findall('lanelet'):
        if lanelet.attrib['id'] == ego_lanelet_id:
            ego_lanelet = lanelet
            break
    if ego_lanelet is None:
        raise ValueError(f"Ego lanelet with ID {ego_lanelet_id} not found")

    # Traffic signs
    new_traffic_signs = ET.SubElement(new_element, "trafficSigns")
    # Create lookup dictionary for traffic signs by ID
    traffic_signs = {sign.attrib['id']: sign for sign in root.findall('trafficSign')}
    # Iterate over trafficSignRef elements in ego_lanelet
    for traffic_sign_ref in ego_lanelet.findall('trafficSignRef'):
        ref_id = traffic_sign_ref.attrib['ref']
        traffic_sign = traffic_signs.get(ref_id)

        if traffic_sign is None:
            raise ValueError(
                f"The traffic sign with ID {ref_id} referenced in the planning problem is not present in the file.")

        # Figure out which type a given traffic sign is
        ts_type = traffic_sign.find("trafficSignElement").find("trafficSignID").text
        ts_type = traffic_sign_dict.get(ts_type)

        # If the sign is a speed limit or required speed sign, add its value as a sub-element
        if ts_type == "speed_limit" or ts_type == "required_speed":
            value = traffic_sign.find("trafficSignElement").find("additionalValue").text
            element = ET.SubElement(new_traffic_signs, ts_type)
            element.attrib['id'] = ref_id
            element.text = value
        # If the sign is a different category, ignore the value sub-element
        else:
            ET.SubElement(new_traffic_signs, ts_type).attrib['id'] = ref_id

    # Traffic lights
    new_traffic_lights = ET.SubElement(new_element, "trafficLights")
    for traffic_light_ref in ego_lanelet.findall('trafficLightRef'):
        ref_id = traffic_light_ref.attrib['ref']
        ET.SubElement(new_traffic_lights, "trafficLight").attrib['ref'] = ref_id

    # LEFT LANES
    # adjacent_left = ego_lanelet.find("adjacentLeft")
    # left_lanes = ET.SubElement(new_element, "leftLanes")
    # while adjacent_left is not None:
    #     al = ET.SubElement(left_lanes, "adjacentLeft")
    #     al.attrib["id"] = adjacent_left.attrib["ref"]
    #     al.attrib["drivingDir"] = adjacent_left.attrib["drivingDir"]

    #     next_adjacent = None
    #     for lanelet in root.findall('lanelet'):
    #         if lanelet.attrib.get('id') == adjacent_left.attrib['ref']:
    #             next_adjacent = lanelet.find("adjacentLeft")
      #           break
     #    adjacent_left = next_adjacent

    # RIGHT LANES
    # adjacent_right = ego_lanelet.find("adjacentRight")
    # right_lanes = ET.SubElement(new_element, "rightLanes")
    # while adjacent_right is not None:
    #   ar = ET.SubElement(right_lanes, "lane")
      #   ar.attrib["id"] = adjacent_right.attrib["ref"]
        # ar.attrib["drivingDir"] = adjacent_right.attrib["drivingDir"]

        # next_adjacent = None
        # for lanelet in root.findall('lanelet'):
          #   if lanelet.attrib.get('id') == adjacent_right.attrib['ref']:
            #     next_adjacent = lanelet.find("adjacentRight")
              #   break
        # adjacent_right = next_adjacent

    return new_element

# Creates a summary of start and end edges of dynamic obstacles based on the CR file
def dynamic_summary_from_cr(file_path_str: str) -> str:
    xml = load_xml(file_path_str)
    polygons = lanelets_to_polygons(xml)

    root = ET.fromstring(xml)
    dynamic_obstacles = root.findall("dynamicObstacle")

    summary_lines = []

    for dynamic_obstacle in dynamic_obstacles:
        obstacle_id = dynamic_obstacle.attrib["id"]

        # Starting point and lanelet
        starting_point = [
            float(dynamic_obstacle.find("initialState").find("position").find("point").find("x").text),
            float(dynamic_obstacle.find("initialState").find("position").find("point").find("y").text)
        ]
        starting_lanelet = check_point_in_polygons(starting_point[0], starting_point[1], polygons)

        # End point and lanelet
        trajectory = dynamic_obstacle.find("trajectory")
        final_state = trajectory.findall("state")[-1]
        end_point = [
            float(final_state.find("position").find("point").find("x").text),
            float(final_state.find("position").find("point").find("y").text)
        ]
        end_lanelet = check_point_in_polygons(end_point[0], end_point[1], polygons)

        summary_lines.append(f"Vehicle {obstacle_id}: {starting_lanelet} -> {end_lanelet}")

    return "\n".join(summary_lines)

# Calculate goal position from a lanelet/edge ID
# Returns the center point at the specified position (beginning/middle/final) of the lanelet with proper orientation
def calculate_goal_from_lanelet(xml_string: str, lanelet_id: str, position: str = "final") -> dict:
    """
    Calculate goal position coordinates from a lanelet/edge ID.
    
    Args:
        xml_string: CommonRoad XML as string
        lanelet_id: Target lanelet/edge ID (as string)
        position: Position in lanelet (default: "final")
                 - "beginning" (0%)
                 - "quarter" (25%)
                 - "middle" (50%)
                 - "three_quarter" (75%)
                 - "final" (100%)
    
    Returns:
        dict with keys: x, y, orientation, or None if lanelet not found
    """
    root = ET.fromstring(xml_string)
    
    # Find the target lanelet
    target_lanelet = None
    for lanelet in root.findall('lanelet'):
        if lanelet.attrib['id'] == str(lanelet_id):
            target_lanelet = lanelet
            break
    
    if target_lanelet is None:
        print(f"Warning: Lanelet {lanelet_id} not found in scenario")
        return None
    
    # Get the center line points (average of left and right bounds)
    left_bound_points = []
    right_bound_points = []
    
    for element in target_lanelet.findall('leftBound'):
        for point in element.findall('point'):
            left_bound_points.append([float(point.find("x").text), float(point.find("y").text)])
    
    for element in target_lanelet.findall('rightBound'):
        for point in element.findall('point'):
            right_bound_points.append([float(point.find("x").text), float(point.find("y").text)])
    
    if not left_bound_points or not right_bound_points:
        print(f"Warning: Lanelet {lanelet_id} has incomplete boundary points")
        return None
    
    # Calculate center line points
    min_points = min(len(left_bound_points), len(right_bound_points))
    center_points = []
    for i in range(min_points):
        center_x = (left_bound_points[i][0] + right_bound_points[i][0]) / 2
        center_y = (left_bound_points[i][1] + right_bound_points[i][1]) / 2
        center_points.append([center_x, center_y])
    
    num_points = len(center_points)
    if num_points < 2:
        print(f"Warning: Lanelet {lanelet_id} has insufficient center points")
        return None
    
    import math
    
    # Select the goal point based on position parameter
    if position.lower() == "beginning":
        # Beginning of lanelet (first point - 0%)
        goal_x = center_points[0][0]
        goal_y = center_points[0][1]
        
        # Calculate orientation from first few points
        orientation_calc_points = min(max(2, int(num_points * 0.1)), num_points)
        dx = center_points[orientation_calc_points - 1][0] - center_points[0][0]
        dy = center_points[orientation_calc_points - 1][1] - center_points[0][1]
        orientation = math.atan2(dy, dx)
        
    elif position.lower() == "quarter":
        # Quarter of lanelet (25% point)
        quarter_idx = num_points // 4
        goal_x = center_points[quarter_idx][0]
        goal_y = center_points[quarter_idx][1]
        
        # Calculate orientation around quarter point
        before_idx = max(0, quarter_idx - max(1, int(num_points * 0.05)))
        after_idx = min(num_points - 1, quarter_idx + max(1, int(num_points * 0.05)))
        dx = center_points[after_idx][0] - center_points[before_idx][0]
        dy = center_points[after_idx][1] - center_points[before_idx][1]
        orientation = math.atan2(dy, dx)
        
    elif position.lower() == "middle":
        # Middle of lanelet (50% point)
        mid_idx = num_points // 2
        goal_x = center_points[mid_idx][0]
        goal_y = center_points[mid_idx][1]
        
        # Calculate orientation around middle point (use points before and after)
        before_idx = max(0, mid_idx - max(1, int(num_points * 0.05)))
        after_idx = min(num_points - 1, mid_idx + max(1, int(num_points * 0.05)))
        dx = center_points[after_idx][0] - center_points[before_idx][0]
        dy = center_points[after_idx][1] - center_points[before_idx][1]
        orientation = math.atan2(dy, dx)
        
    elif position.lower() == "three_quarter":
        # Three quarter of lanelet (75% point)
        three_quarter_idx = (num_points * 3) // 4
        goal_x = center_points[three_quarter_idx][0]
        goal_y = center_points[three_quarter_idx][1]
        
        # Calculate orientation around three quarter point
        before_idx = max(0, three_quarter_idx - max(1, int(num_points * 0.05)))
        after_idx = min(num_points - 1, three_quarter_idx + max(1, int(num_points * 0.05)))
        dx = center_points[after_idx][0] - center_points[before_idx][0]
        dy = center_points[after_idx][1] - center_points[before_idx][1]
        orientation = math.atan2(dy, dx)
        
    else:  # "final" or default
        # End of lanelet (last point - 100%)
        goal_x = center_points[-1][0]
        goal_y = center_points[-1][1]
        
        # Calculate orientation from the last few points for stability
        orientation_calc_points = max(2, int(num_points * 0.1))  # Use last 10% of points, minimum 2
        start_idx = num_points - orientation_calc_points
        
        dx = center_points[-1][0] - center_points[start_idx][0]
        dy = center_points[-1][1] - center_points[start_idx][1]
        orientation = math.atan2(dy, dx)
    
    return {
        "x": goal_x,
        "y": goal_y,
        "orientation": orientation,
        "lanelet_id": lanelet_id,
        "position": position
    }

# Modify the goal state in the planning problem
def modify_goal_state(xml_string: str, new_goal: dict, goal_length: float = 6.0, goal_width: float = 2.0) -> str:
    """
    Modify the goal state position in the planning problem.
    
    Args:
        xml_string: CommonRoad XML as string
        new_goal: dict with keys x, y, orientation (from calculate_goal_from_lanelet)
        goal_length: length of goal rectangle (default 6.0 meters)
        goal_width: width of goal rectangle (default 2.0 meters)
    
    Returns:
        Modified XML string
    """
    root = ET.fromstring(xml_string)
    
    # Find the planning problem
    planning_problem = root.find('planningProblem')
    if planning_problem is None:
        print("Warning: No planning problem found in scenario")
        return xml_string
    
    # Find the goal state
    goal_state = planning_problem.find('goalState')
    if goal_state is None:
        print("Warning: No goal state found in planning problem")
        return xml_string
    
    # Find the position rectangle
    position = goal_state.find('position')
    if position is None:
        print("Warning: No position element in goal state")
        return xml_string
    
    # Replace whatever the goal position currently is — a <rectangle>, a
    # <lanelet ref="..."> reference, a polygon, or a multi-shape group — with a
    # single <rectangle> at the requested position. The previous version only
    # updated an EXISTING <rectangle> and returned the file unchanged for
    # lanelet/polygon goals, so the goal silently never moved.
    for child in list(position):
        position.remove(child)
    rectangle = ET.SubElement(position, 'rectangle')
    ET.SubElement(rectangle, 'length').text = str(goal_length)
    ET.SubElement(rectangle, 'width').text = str(goal_width)
    ET.SubElement(rectangle, 'orientation').text = str(new_goal['orientation'])
    center = ET.SubElement(rectangle, 'center')
    ET.SubElement(center, 'x').text = str(new_goal['x'])
    ET.SubElement(center, 'y').text = str(new_goal['y'])

    # Convert back to string
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding='unicode')

#print(dynamic_summary_from_cr("<repo>/Scenarios/ZAM_Tjunction-1_60_T-1/Original/ZAM_Tjunction-1_60_T-1.cr.xml"))