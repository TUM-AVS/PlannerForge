"""Single source of truth for CommonRoad scenario styling.

Every scenario picture the app produces — the database preview PNG, the
database/modified scenario GIF, and the OSM-generated scenario PNG/GIF —
must look the same: identical lanelet-ID font, vehicle-ID font, traffic
signs, static-obstacle colours and figure geometry. Before this module
existed each renderer set its own draw parameters, so a generated
scenario had no lanelet IDs and no traffic signs at all, and its labels
came out at matplotlib's default size instead of the Frenetix sizes.

Consumers:
  * Frenetix-Motion-Planner/main_without_ego.py  (database + modified GIF)
  * osm_pipeline/gif_cr.py                       (generated GIF)
  * osm_pipeline/bev_cr.py                       (generated PNG)
  * commonroad_interface/create_png.py           (database preview PNG)

Anything that draws a scenario should call `apply_rcparams()`,
`install_icons()` and `build_draw_params()` and render through
`CustomRenderer` — never a bare MPRenderer with hand-rolled parameters.
"""
from __future__ import annotations

# === Standard library ===
import math
from typing import Tuple

# === Third-party libraries ===
import numpy as np
import shapely.geometry
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import text  # noqa: F401  (enables matplotlib.text.Text below)

# === CommonRoad ===
from commonroad.common.common_lanelet import LineMarking
from commonroad.common.util import Interval
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.traffic_light import TrafficLightState

# === CommonRoad Visualization ===
from commonroad.visualization.draw_params import (
    LaneletNetworkParams,
    MPDrawParams,
    OptionalSpecificOrAllDrawParams,
)
from commonroad.visualization.mp_renderer import MPRenderer, ZOrders
from commonroad.visualization.util import (
    LineCollectionDataUnits,
    LineDataUnits,
    collect_center_line_colors,
    colormap_idx,
    get_arrow_path_at,
    get_tangent_angle,
    get_vehicle_direction_triangle,
    line_marking_to_linestyle,
    traffic_light_color_dict,
)

# --- Canonical figure geometry and font sizes --------------------------------
# Lanelet labels are drawn by CustomRenderer at LANELET_LABEL_FONTSIZE points;
# vehicle/static IDs are drawn by stock MPRenderer code, which reads
# matplotlib's rcParams, hence apply_rcparams(). Font sizes are in points, so
# the on-screen look depends on FIGSIZE — keep the figure size identical
# everywhere and the labels come out the same relative size at any DPI.
FIGSIZE: Tuple[int, int] = (20, 10)
DPI: int = 300
#: Milliseconds per GIF frame at a one-timestep stride. imageio >= 2.28 reads
#: `duration` in MILLISECONDS; passing seconds (0.2) writes a 0 ms delay, which
#: makes the viewer fall back to its own default rate — that is why the database
#: GIFs used to play at a different speed from the generated ones.
FRAME_DURATION_MS: int = 200
#: savefig crop, applied by every consumer so the scenario fills the frame
#: identically (an uncropped canvas makes labels look larger by comparison).
BBOX_INCHES = "tight"
PAD_INCHES: float = 0.1
#: Font size of the lanelet-ID / traffic-sign text drawn by CustomRenderer.
LANELET_LABEL_FONTSIZE: int = 5
#: rcParams font size, which is what stock MPRenderer uses for vehicle IDs.
LABEL_FONT_SIZE: int = 8

# Lanelet labels must sit under the goal region, not over it.
ZOrders.LANELET_LABEL = 25.0


def apply_rcparams() -> None:
    """Set the matplotlib font sizes that stock MPRenderer uses for the
    vehicle-ID and static-obstacle-ID labels. Idempotent."""
    plt.rcParams["font.size"] = LABEL_FONT_SIZE
    plt.rcParams["axes.labelsize"] = LABEL_FONT_SIZE
    plt.rcParams["xtick.labelsize"] = LABEL_FONT_SIZE
    plt.rcParams["ytick.labelsize"] = LABEL_FONT_SIZE


def install_icons() -> None:
    """Register the extra obstacle icons (motorcycle, taxi, priority
    vehicle) so an obstacle type renders identically in every picture.

    Without this a taxi is a plain orange car in the database GIF but a
    yellow roof-signed taxi in a generated one. Import is done lazily and
    failures are swallowed: the icon pack lives in osm_pipeline, which is
    not importable from every environment that draws a scenario, and
    missing icons degrade to stock CommonRoad shapes rather than an error.
    """
    try:
        from osm_pipeline import cr_icons
    except Exception:
        return
    try:
        cr_icons.install()
    except Exception:
        pass

# --- Canonical renderer ------------------------------------------------------
# MPRenderer subclass that draws lanelet labels at a fixed small font size,
# offset off the centerline, without the stock opaque bbox behind the text.
class CustomRenderer(MPRenderer):
    def draw_lanelet_network(
            self, obj: LaneletNetwork, draw_params: OptionalSpecificOrAllDrawParams[LaneletNetworkParams] = None
    ) -> None:
        from matplotlib.path import Path
        import matplotlib.collections as collections
        """
        Draws a lanelet network

        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict that
            recreates the structure of an object,
        :return: None
        """
        if draw_params is None:
            draw_params = self.draw_params.lanelet_network
        elif isinstance(draw_params, MPDrawParams):
            draw_params = draw_params.lanelet_network

        traffic_lights = obj.traffic_lights
        traffic_signs = obj.traffic_signs
        intersections = obj.intersections
        lanelets = obj.lanelets

        time_begin = draw_params.time_begin
        if traffic_lights is not None:
            draw_traffic_lights = draw_params.traffic_light.draw_traffic_lights

            traffic_light_colors = draw_params.traffic_light
        else:
            draw_traffic_lights = False

        if traffic_signs is not None:
            draw_traffic_signs = draw_params.traffic_sign.draw_traffic_signs
            show_traffic_sign_label = draw_params.traffic_sign.show_label
        else:
            draw_traffic_signs = show_traffic_sign_label = False

        if intersections is not None and len(intersections) > 0:
            draw_intersections = draw_params.intersection.draw_intersections
        else:
            draw_intersections = False

        if draw_intersections is True:
            draw_incoming_lanelets = draw_params.intersection.draw_incoming_lanelets
            incoming_lanelets_color = draw_params.intersection.incoming_lanelets_color
            draw_crossings = draw_params.intersection.draw_crossings
            crossings_color = draw_params.intersection.crossings_color
            draw_successors = draw_params.intersection.draw_successors
            successors_left_color = draw_params.intersection.successors_left_color
            successors_straight_color = draw_params.intersection.successors_straight_color
            successors_right_color = draw_params.intersection.successors_right_color
            show_intersection_labels = draw_params.intersection.show_label
        else:
            draw_incoming_lanelets = draw_crossings = draw_successors = show_intersection_labels = False

        left_bound_color = draw_params.lanelet.left_bound_color
        right_bound_color = draw_params.lanelet.right_bound_color
        center_bound_color = draw_params.lanelet.center_bound_color
        unique_colors = draw_params.lanelet.unique_colors
        draw_stop_line = draw_params.lanelet.draw_stop_line
        stop_line_color = draw_params.lanelet.stop_line_color
        draw_line_markings = draw_params.lanelet.draw_line_markings
        show_label = draw_params.lanelet.show_label
        draw_border_vertices = draw_params.lanelet.draw_border_vertices
        draw_left_bound = draw_params.lanelet.draw_left_bound
        draw_right_bound = draw_params.lanelet.draw_right_bound
        draw_center_bound = draw_params.lanelet.draw_center_bound
        draw_start_and_direction = draw_params.lanelet.draw_start_and_direction
        draw_linewidth = draw_params.lanelet.draw_linewidth
        fill_lanelet = draw_params.lanelet.fill_lanelet
        facecolor = draw_params.lanelet.facecolor
        antialiased = draw_params.antialiased
        lanelet_zorder = draw_params.lanelet.zorder

        draw_lanlet_ids = draw_params.draw_ids

        colormap_tangent = draw_params.lanelet.colormap_tangent

        # Collect lanelets
        incoming_lanelets = set()
        incomings_left = {}
        incomings_id = {}
        crossings = set()
        all_successors = set()
        successors_left = set()
        successors_straight = set()
        successors_right = set()
        if draw_intersections:
            # collect incoming lanelets
            if draw_incoming_lanelets:
                incomings: List[set] = []
                inc_2_intersections = obj.map_inc_lanelets_to_intersections
                for intersection in intersections:
                    for incoming in intersection.incomings:
                        incomings.append(incoming.incoming_lanelets)
                        for l_id in incoming.incoming_lanelets:
                            incomings_left[l_id] = incoming.left_of
                            incomings_id[l_id] = incoming.incoming_id
                incoming_lanelets: Set[int] = set.union(*incomings)

            if draw_crossings:
                tmp_list: List[set] = [intersection.crossings for intersection in intersections]
                crossings: Set[int] = set.union(*tmp_list)

            if draw_successors:
                tmp_list: List[set] = [
                    incoming.successors_left for intersection in intersections for incoming in intersection.incomings
                ]
                successors_left: Set[int] = set.union(*tmp_list)
                tmp_list: List[set] = [
                    incoming.successors_straight
                    for intersection in intersections
                    for incoming in intersection.incomings
                ]
                successors_straight: Set[int] = set.union(*tmp_list)
                tmp_list: List[set] = [
                    incoming.successors_right for intersection in intersections for incoming in intersection.incomings
                ]
                successors_right: Set[int] = set.union(*tmp_list)
                all_successors = set.union(successors_straight, successors_right, successors_left)

        # select unique colors from colormap for each lanelet's center_line

        incoming_vertices_fill = list()
        crossing_vertices_fill = list()
        succ_left_paths = list()
        succ_straight_paths = list()
        succ_right_paths = list()

        vertices_fill = list()
        coordinates_left_border_vertices = []
        coordinates_right_border_vertices = []
        direction_list = list()
        center_paths = list()
        left_paths = list()
        right_paths = list()

        if draw_traffic_lights:
            center_line_color_dict = collect_center_line_colors(obj, traffic_lights, time_begin)

        cmap_lanelet = colormap_idx(len(lanelets))

        # collect paths for drawing
        for i_lanelet, lanelet in enumerate(lanelets):
            if isinstance(draw_lanlet_ids, list) and lanelet.lanelet_id not in draw_lanlet_ids:
                continue

            # project lanelet vertices to xy-plane as we make a 2D plot
            center_vertices_2d = lanelet.center_vertices[:, :2]
            left_vertices_2d = lanelet.left_vertices[:, :2]
            right_vertices_2d = lanelet.right_vertices[:, :2]

            def _draw_bound(vertices, line_marking, paths, coordinate_border_vertices):
                if draw_border_vertices:
                    coordinate_border_vertices.append(vertices)

                if (
                        draw_line_markings
                        and line_marking is not LineMarking.UNKNOWN
                        and line_marking is not LineMarking.NO_MARKING
                ):
                    linestyle, dashes, linewidth_metres = line_marking_to_linestyle(line_marking)
                    if lanelet.distance[-1] <= linewidth_metres:
                        paths.append(Path(right_vertices_2d, closed=False))
                    else:
                        tmp_vertices = vertices.copy()
                        line_string = shapely.geometry.LineString(tmp_vertices)
                        max_dist = line_string.project(shapely.geometry.Point(*vertices[-1])) - linewidth_metres / 2

                        if line_marking in (LineMarking.DASHED, LineMarking.BROAD_DASHED):
                            # In Germany, dashed lines are 6m long and 12m apart.
                            distances_start = np.arange(linewidth_metres / 2, max_dist, 12.0 + 6.0)
                            distances_end = distances_start + 6.0
                            # Cut off the last dash if it is too long.
                            distances_end[-1] = min(distances_end[-1], max_dist)
                            p_start = [line_string.interpolate(s).coords for s in distances_start]
                            p_end = [line_string.interpolate(s).coords for s in distances_end]
                            pts = np.squeeze(np.stack((p_start, p_end), axis=1), axis=2)
                            collection = LineCollectionDataUnits(
                                pts,
                                zorder=ZOrders.RIGHT_BOUND,
                                linewidth=linewidth_metres,
                                alpha=1.0,
                                color=right_bound_color,
                            )
                            self.static_collections.append(collection)
                        else:
                            # Offset, start and end of the line marking, to make them aligned with the lanelet.
                            tmp_vertices[0, :] = line_string.interpolate(linewidth_metres / 2).coords
                            tmp_vertices[-1, :] = line_string.interpolate(max_dist).coords
                            self.static_artists.append(
                                LineDataUnits(
                                    tmp_vertices[:, 0],
                                    tmp_vertices[:, 1],
                                    zorder=ZOrders.RIGHT_BOUND,
                                    linewidth=linewidth_metres,
                                    alpha=1.0,
                                    color=right_bound_color,
                                    linestyle=linestyle,
                                    dashes=dashes,
                                )
                            )
                else:
                    paths.append(Path(vertices, closed=False))

            # left bound
            if (draw_border_vertices or draw_left_bound) and (
                    lanelet.adj_left is None or not lanelet.adj_left_same_direction
            ):
                _draw_bound(
                    left_vertices_2d, lanelet.line_marking_left_vertices, left_paths, coordinates_left_border_vertices
                )

            # right bound
            if draw_border_vertices or draw_right_bound:
                _draw_bound(
                    right_vertices_2d,
                    lanelet.line_marking_right_vertices,
                    right_paths,
                    coordinates_right_border_vertices,
                )

            # stop line
            if draw_stop_line and lanelet.stop_line:
                # project stop line to xy-plane for 2D plot
                stop_line = np.vstack([lanelet.stop_line.start[:2], lanelet.stop_line.end[:2]])
                linestyle, dashes, linewidth_metres = line_marking_to_linestyle(lanelet.stop_line.line_marking)
                # cut off in the beginning, because linewidth_metres is added
                # later
                vec = stop_line[1, :] - stop_line[0, :]
                tangent = vec / np.linalg.norm(vec)
                stop_line[0, :] += linewidth_metres * tangent / 2
                stop_line[1, :] -= linewidth_metres * tangent / 2
                line = LineDataUnits(
                    stop_line[:, 0],
                    stop_line[:, 1],
                    zorder=ZOrders.STOP_LINE,
                    linewidth=linewidth_metres,
                    alpha=1.0,
                    color=stop_line_color,
                    linestyle=linestyle,
                    dashes=dashes,
                )
                self.static_artists.append(line)

            if unique_colors:
                # set center bound color to unique value
                center_bound_color = cmap_lanelet(i_lanelet)

            # direction arrow
            if draw_start_and_direction:
                center = center_vertices_2d[0]
                orientation = math.atan2(*(center_vertices_2d[1] - center)[::-1])
                lanelet_width = np.linalg.norm(right_vertices_2d[0] - left_vertices_2d[0])
                arrow_width = min(lanelet_width, 1.5)
                path = get_arrow_path_at(*center, orientation, arrow_width)
                if unique_colors:
                    direction_list.append(
                        matplotlib.patches.PathPatch(
                            path,
                            color=center_bound_color,
                            lw=0.5,
                            zorder=ZOrders.DIRECTION_ARROW,
                            antialiased=antialiased,
                        )
                    )
                else:
                    direction_list.append(path)

            # visualize traffic light state through colored center bound
            has_traffic_light = draw_traffic_lights and lanelet.lanelet_id in center_line_color_dict
            if has_traffic_light:
                light_state = center_line_color_dict[lanelet.lanelet_id]

                if light_state is not TrafficLightState.INACTIVE:
                    linewidth_metres = 0.75
                    # dashed line for red_yellow
                    linestyle = "--" if light_state == TrafficLightState.RED_YELLOW else "-"
                    dashes = (5, 5) if linestyle == "--" else (None, None)

                    # cut off in the beginning, because linewidth_metres is added later
                    tmp_center = center_vertices_2d.copy()
                    if lanelet.distance[-1] > linewidth_metres:
                        tmp_center[0, :] = lanelet.interpolate_position(linewidth_metres)[0][:2]
                    zorder = (
                        ZOrders.LIGHT_STATE_GREEN
                        if light_state == TrafficLightState.GREEN
                        else ZOrders.LIGHT_STATE_OTHER
                    )
                    line = LineDataUnits(
                        tmp_center[:, 0],
                        tmp_center[:, 1],
                        zorder=zorder,
                        linewidth=linewidth_metres,
                        alpha=0.7,
                        color=traffic_light_color_dict(light_state, traffic_light_colors),
                        linestyle=linestyle,
                        dashes=dashes,
                    )
                    self.dynamic_artists.append(line)

            # draw colored center bound. Hierarchy or colors: successors > usual
            # center bound
            is_successor = draw_intersections and draw_successors and lanelet.lanelet_id in all_successors
            if is_successor:
                if lanelet.lanelet_id in successors_left:
                    succ_left_paths.append(Path(center_vertices_2d, closed=False))
                elif lanelet.lanelet_id in successors_straight:
                    succ_straight_paths.append(Path(center_vertices_2d, closed=False))
                else:
                    succ_right_paths.append(Path(center_vertices_2d, closed=False))

            elif draw_center_bound:
                if unique_colors:
                    center_paths.append(
                        mpl.patches.PathPatch(
                            Path(center_vertices_2d, closed=False),
                            edgecolor=center_bound_color,
                            facecolor="none",
                            lw=draw_linewidth,
                            zorder=ZOrders.CENTER_BOUND,
                            antialiased=antialiased,
                        )
                    )
                elif colormap_tangent:
                    relative_angle = draw_params.relative_angle
                    points = center_vertices_2d.reshape(-1, 1, 2)
                    angles = get_tangent_angle(points[:, 0, :], relative_angle)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(0, 360)
                    lc = collections.LineCollection(
                        segments,
                        cmap="hsv",
                        norm=norm,
                        lw=draw_linewidth,
                        zorder=ZOrders.CENTER_BOUND,
                        antialiased=antialiased,
                    )
                    lc.set_array(angles)
                    self.static_collections.append(lc)

            is_incoming_lanelet = (
                    draw_intersections and draw_incoming_lanelets and (lanelet.lanelet_id in incoming_lanelets)
            )
            is_crossing = draw_intersections and draw_crossings and (lanelet.lanelet_id in crossings)

            # Draw lanelet area
            if fill_lanelet:
                if not is_incoming_lanelet and not is_crossing:
                    vertices_fill.append(np.concatenate((right_vertices_2d, np.flip(left_vertices_2d, 0))))

            # collect incoming lanelets in separate list for plotting in
            # different color
            if is_incoming_lanelet:
                incoming_vertices_fill.append(np.concatenate((right_vertices_2d, np.flip(left_vertices_2d, 0))))
            elif is_crossing:
                crossing_vertices_fill.append(np.concatenate((right_vertices_2d, np.flip(left_vertices_2d, 0))))

            # Draw labels
            if show_label or show_intersection_labels or draw_traffic_signs:
                strings = []
                if show_label:
                    strings.append(str(lanelet.lanelet_id))
                if is_incoming_lanelet and show_intersection_labels:
                    strings.append(f"int_id: {inc_2_intersections[lanelet.lanelet_id].intersection_id}")
                    strings.append("inc_id: " + str(incomings_id[lanelet.lanelet_id]))
                    strings.append("inc_left: " + str(incomings_left[lanelet.lanelet_id]))
                if draw_traffic_signs and show_traffic_sign_label:
                    traffic_signs_tmp = [obj._traffic_signs[id] for id in lanelet.traffic_signs]
                    if traffic_signs_tmp:
                        # add as text to label
                        str_tmp = "sign: "
                        add_str = ""
                        for sign in traffic_signs_tmp:
                            for el in sign.traffic_sign_elements:
                                # TrafficSignIDGermany(
                                # el.traffic_sign_element_id).name would give
                                # the
                                # name
                                str_tmp += add_str + el.traffic_sign_element_id.value
                                add_str = ", "

                        strings.append(str_tmp)

                label_string = ", ".join(strings)
                if len(label_string) > 0:
                    # compute normal angle of label box
                    clr_positions = lanelet.interpolate_position(0.5 * lanelet.distance[-1])
                    # project to xy-plane (last tuple element is an index, so we do not need to project it)
                    clr_positions = (clr_positions[0][:2], clr_positions[1][:2], clr_positions[2][:2], clr_positions[3])
                    normal_vector = np.array(clr_positions[1]) - np.array(clr_positions[2])
                    angle = np.rad2deg(math.atan2(normal_vector[1], normal_vector[0])) - 90
                    angle = angle if Interval(-90, 90).contains(angle) else angle - 180

                    # Custom modification for better view

                    # center_pos = np.array(clr_positions[0])
                    # left_pos = np.array(clr_positions[1])
                    # right_pos = np.array(clr_positions[2])

                    center_pos = np.array(clr_positions[0])  # label anchor point

                    # Get direction of travel (tangent) from centerline
                    centerline = lanelet.center_vertices
                    idx = clr_positions[3]

                    # Compute direction of travel (tangent)
                    if 0 < idx < len(centerline) - 1:
                        p_prev = np.array(centerline[idx - 1][:2])
                        p_next = np.array(centerline[idx + 1][:2])
                        tangent = p_next - p_prev
                        tangent /= np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0 else 1.0
                    else:
                        tangent = np.array([1.0, 0.0])  # fallback

                    # Compute normal vector pointing to the RIGHT of travel direction
                    normal = np.array([tangent[1], -tangent[0]])  # right-hand normal

                    # Apply shift
                    label_shift = 2.0  # meters; adjust this value as needed
                    offset_pos = center_pos + label_shift * normal
                    x, y = offset_pos[0], offset_pos[1]

                    # Unit vector from right to left
                    # normal = left_pos - right_pos
                    # normal /= np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else 1.0

                    # Pick a shift direction — you can use lanelet ID, left/right rule, or metadata
                    # Example 1: Push label slightly toward left boundary
                    # label_shift = 0.5  # meters; adjust as needed
                    # offset_pos = center_pos + label_shift * normal

                    # Now use this offset position for placing the label
                    # x, y = offset_pos[0], offset_pos[1]

                    self.static_artists.append(
                        matplotlib.text.Text(
                            # clr_positions[0][0],
                            # clr_positions[0][1],
                            offset_pos[0],
                            offset_pos[1],
                            label_string,
                            fontsize=LANELET_LABEL_FONTSIZE,
                            color="black",
                            alpha=0.75,
                            # bbox={"facecolor": center_bound_color, "pad": 2},
                            horizontalalignment="center",
                            verticalalignment="center",
                            rotation=angle,
                            zorder=ZOrders.LANELET_LABEL,
                        )
                    )

        # draw paths and collect axis handles
        if draw_right_bound:
            self.static_collections.append(
                collections.PathCollection(
                    right_paths,
                    edgecolor=right_bound_color,
                    facecolor="none",
                    lw=draw_linewidth,
                    zorder=lanelet_zorder + 0.1,
                    antialiased=antialiased,
                )
            )
        if draw_left_bound:
            self.static_collections.append(
                collections.PathCollection(
                    left_paths,
                    edgecolor=left_bound_color,
                    facecolor="none",
                    lw=draw_linewidth,
                    zorder=lanelet_zorder + 0.1,
                    antialiased=antialiased,
                )
            )
        if unique_colors:
            if draw_center_bound:
                if draw_center_bound:
                    self.static_collections.append(
                        collections.PatchCollection(
                            center_paths, match_original=True, zorder=ZOrders.CENTER_BOUND, antialiased=antialiased
                        )
                    )
                if draw_start_and_direction:
                    self.static_collections.append(
                        collections.PatchCollection(
                            direction_list, match_original=True, zorder=ZOrders.DIRECTION_ARROW, antialiased=antialiased
                        )
                    )

        elif not colormap_tangent:
            if draw_center_bound:
                self.static_collections.append(
                    collections.PathCollection(
                        center_paths,
                        edgecolor=center_bound_color,
                        facecolor="none",
                        lw=draw_linewidth,
                        zorder=ZOrders.CENTER_BOUND,
                        antialiased=antialiased,
                    )
                )
            if draw_start_and_direction:
                self.static_collections.append(
                    collections.PathCollection(
                        direction_list,
                        color=center_bound_color,
                        lw=0.5,
                        zorder=ZOrders.DIRECTION_ARROW,
                        antialiased=antialiased,
                    )
                )

        if successors_left:
            self.static_collections.append(
                collections.PathCollection(
                    succ_left_paths,
                    edgecolor=successors_left_color,
                    facecolor="none",
                    lw=draw_linewidth * 3.0,
                    zorder=ZOrders.SUCCESSORS,
                    antialiased=antialiased,
                )
            )
        if successors_straight:
            self.static_collections.append(
                collections.PathCollection(
                    succ_straight_paths,
                    edgecolor=successors_straight_color,
                    facecolor="none",
                    lw=draw_linewidth * 3.0,
                    zorder=ZOrders.SUCCESSORS,
                    antialiased=antialiased,
                )
            )
        if successors_right:
            self.static_collections.append(
                collections.PathCollection(
                    succ_right_paths,
                    edgecolor=successors_right_color,
                    facecolor="none",
                    lw=draw_linewidth * 3.0,
                    zorder=ZOrders.SUCCESSORS,
                    antialiased=antialiased,
                )
            )

        # fill lanelets with facecolor
        self.static_collections.append(
            collections.PolyCollection(
                vertices_fill,
                zorder=lanelet_zorder,
                facecolor=facecolor,
                edgecolor="none",
                antialiased=antialiased,
            )
        )
        if incoming_vertices_fill:
            self.static_collections.append(
                collections.PolyCollection(
                    incoming_vertices_fill,
                    facecolor=incoming_lanelets_color,
                    edgecolor="none",
                    zorder=ZOrders.INCOMING_POLY,
                    antialiased=antialiased,
                )
            )
        if crossing_vertices_fill:
            self.static_collections.append(
                collections.PolyCollection(
                    crossing_vertices_fill,
                    facecolor=crossings_color,
                    edgecolor="none",
                    zorder=ZOrders.CROSSING_POLY,
                    antialiased=antialiased,
                )
            )

        # draw_border_vertices
        if draw_border_vertices:
            coordinates_left_border_vertices = np.concatenate(coordinates_left_border_vertices, axis=0)
            # left vertices
            self.static_collections.append(
                collections.EllipseCollection(
                    np.ones([coordinates_left_border_vertices.shape[0], 1]) * 1.5,
                    np.ones([coordinates_left_border_vertices.shape[0], 1]) * 1.5,
                    np.zeros([coordinates_left_border_vertices.shape[0], 1]),
                    offsets=coordinates_left_border_vertices,
                    color=left_bound_color,
                    zorder=ZOrders.LEFT_BOUND + 0.1,
                )
            )

            coordinates_right_border_vertices = np.concatenate(coordinates_right_border_vertices, axis=0)
            # right_vertices
            self.static_collections.append(
                collections.EllipseCollection(
                    np.ones([coordinates_right_border_vertices.shape[0], 1]) * 1.5,
                    np.ones([coordinates_right_border_vertices.shape[0], 1]) * 1.5,
                    np.zeros([coordinates_right_border_vertices.shape[0], 1]),
                    offsets=coordinates_right_border_vertices,
                    color=right_bound_color,
                    zorder=ZOrders.LEFT_BOUND + 0.1,
                )
            )

        if draw_traffic_signs:
            # draw actual traffic sign
            for sign in traffic_signs:
                sign.draw(self, draw_params.traffic_sign)

        if draw_traffic_lights:
            # draw actual traffic light
            for light in traffic_lights:
                light.draw(self, draw_params.traffic_light)


# --- Canonical draw parameters ----------------------------------------------
def build_draw_params(timestep: int = 0, show_agent_labels: bool = True,
                      show_lanelet_labels: bool = True,
                      show_static_labels: bool = True) -> MPDrawParams:
    """Build the canonical MPDrawParams for drawing a scenario at `timestep`.

    The defaults are the ones the database/modified GIF has always used, so
    passing nothing reproduces that look exactly. Callers should not tweak the
    returned object — change it here instead, or every picture drifts apart
    again.
    """

    # Main draw parameters
    obs_params = MPDrawParams()
    obs_params.dynamic_obstacle.time_begin = timestep
    obs_params.dynamic_obstacle.draw_icon = True
    obs_params.dynamic_obstacle.show_label = show_agent_labels  # Configurable agent labels
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    obs_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

    # Static obstacles - Gray (not red)
    obs_params.static_obstacle.show_label = show_static_labels  # Configurable static obstacle labels
    obs_params.static_obstacle.occupancy.shape.facecolor = "#808080"
    obs_params.static_obstacle.occupancy.shape.edgecolor = "#404040"

    # Enable traffic elements for realistic visualization
    obs_params.traffic_light.draw_traffic_lights = True
    obs_params.traffic_sign.draw_traffic_signs = True
    obs_params.lanelet_network.traffic_light.draw_traffic_lights = True
    obs_params.lanelet_network.traffic_sign.draw_traffic_signs = True
    obs_params.lanelet_network.traffic_light.show_label = False  # Don't clutter with IDs
    obs_params.lanelet_network.traffic_sign.show_label = False
    
    # Set time_begin for traffic lights to show correct state at each timestep
    obs_params.lanelet_network.traffic_light.time_begin = timestep
    obs_params.time_begin = timestep
    
    obs_params.lanelet_network.lanelet.show_label = show_lanelet_labels  # Configurable lanelet labels


    # FORCE disable ALL intersection drawing (same as working visualization.py)
    obs_params.lanelet_network.intersection.draw_intersections = False  # Disable entire intersection drawing
    obs_params.lanelet_network.intersection.draw_crossings = False  # Disable crossings specifically
    obs_params.lanelet_network.intersection.draw_incoming_lanelets = False
    obs_params.lanelet_network.intersection.draw_successors = False
    obs_params.lanelet_network.intersection.show_label = False

    obs_params.axis_visible = False

    return obs_params
