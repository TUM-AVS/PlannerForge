import json
import os.path
import threading
import queue
from pathlib import Path

from db_wrapper import ScenarioDBWrapper
from llm_wrapper import CommercialLLMWrapper, OllamaLLMWrapper, QwenLLMWrapper
from utils.xml_parsing_utils import requires_trajectory_generation, generate_exit_trajectories_without_llm, update_benchmark_id, add_traffic_lights
from config import SessionConfig

from utils.ego_lanelet_utils import dynamic_summary_from_cr, calculate_goal_from_lanelet, modify_goal_state
from utils.format_handling_utils import generate_modified_scenario_name, extract_clean_yaml, build_batch_options
from utils.cost_handling_utils import get_min_cost, format_analysis
from utils.data_utils import (
    make_combined_bar_plot,
    make_batch_summary_string,
    make_performance_plot,
    make_success_rate_plot,
    make_cost_analysis_plot,
    make_batch_dataframe
)

import subprocess

from dotenv import load_dotenv
import os, subprocess

load_dotenv()

conda_bin = os.getenv("CONDA_BIN", "conda")
# Name of the conda env holding the CommonRoad/SUMO toolchain (see setup/install.md)
cr_conda_env = os.getenv("CR_CONDA_ENV", "cr37")

# Class running the search loop in an extra thread
class ProcessEngine:
    def __init__(self):
        # LLM setup
        self.db = ScenarioDBWrapper(persist_dir="chroma")
        # self.llm = LLMWrapper(scenario_db=self.db)
        self.llm = None

        # Debug log
        # Provides abstract and specifically formatted information
        # Mainly important for the query part
        self.debug_log = "Debugger Active. Any part of the dialogue can be skipped by pressing enter without adding any text.\n###################"
        self.query_counter = 1

        # Uses the literal console output for closer analysis and to not clutter the debug log
        self.console_log = ""

        # Search setup
        self.extracted_location_json = {}
        self.after_location_ids = []
        self.location_unavailable = None  # set when an explicit location has 0 DB matches
        self.extracted_tags = []
        self.after_tags_ids = []
        self.after_tags_docs = []
        self.extracted_road_net_json = {}
        self.after_road_net_ids = []
        self.after_road_net_docs = []
        self.extracted_obstacles_json = {}
        self.after_obstacles_ids = []
        self.after_obstacles_docs = []
        self.extracted_velocity_json = {}
        self.after_velocity_ids = []
        self.after_velocity_docs = []

        self.modifications = ["Base Scenario - No Modification"]

        # Keeping track of the current top results for the visualization output
        self.current_step = 0
        self.current_results = []

        # The final result at the end of the query part
        self.result = ""
        self.file_path = ""

        # The GIFs corresponding to the selected scenario
        self.gif_displays = []

        # Maps a modified scenario name -> the scenario it was derived from
        # (its immediate parent), so "what changed?" can diff per-modification.
        self.scenario_parents = {}

        # The GIFs of the scenarios run through the motion planner
        self.planner_displays = []

        # Keeping track of run simulations
        self.simulations = ""
        self.batch_simulations = ""

        self.plot = None
        self.plot_performance = None
        self.plot_success = None
        self.plot_cost = None
        self.batch_data_df = None

        # Setup motion planners here
        self.planners = {
            "FRENETIX": {
                # Path to the venv where the planner is set up
                "venv_path": str(Path("Frenetix-Motion-Planner/venv/bin/activate").resolve()),
                # Path to the main.py method used for single simulation
                "script_path": str(Path("Frenetix-Motion-Planner/main_batch.py").resolve()),
                # Path to a directory ending in /logs
                # The scenario-bound information should be stored in a folder like ".../logs/<scenario_name>/"
                # The same goes for the GIF - it should be stored in ".../logs/<scenario_name>/<scenario_name>.gif"
                "log_dir": str(Path("Frenetix-Motion-Planner/logs").resolve()),
                # If cost functions of a defined style (see Frenetix JSON cost logs) are supported, enable this (y = yes, n = no)
                "cost_support": "y",
                # If any types of weights can be adjusted via a YAML file, put it here
                "weights_file": str(Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml").resolve()),
            },
            "MP-RBFN": {
                # Python interpreter of the MP-RBFN environment (see setup/install.md)
                "venv_path": os.getenv("MPRBFN_PYTHON", "python3"),
                "script_path": str(Path("RBFN-Motion-Primitives/scripts/run_cr_simulation_batch.py").resolve()),
                "log_dir": str(Path("RBFN-Motion-Primitives/logs").resolve()),
                "cost_support": "n",
                "weights_file": "",
                # Extra environment applied only to MP-RBFN subprocess launches.
                # ground_truth prediction is the mode that actually completes; the
                # default semantic_cv hangs per scenario (see run_cr_simulation_batch).
                # Skipping plots keeps batch runs fast/headless.
                "env": {
                    "RBFN_PREDICTION_MODE": "ground_truth",
                    "RBFN_SKIP_PLOTTING": "1",
                },
            }
        }
        self.selected_planner = "FRENETIX"

        # Threading setup
        self.input_queue = queue.Queue()
        self.result_ready = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def set_llm(self, llm_parameters: dict):
        if self.llm is None and llm_parameters["mode"] == "Commercial":
            self.llm = CommercialLLMWrapper(scenario_db=self.db, model=llm_parameters["model"], api_key=llm_parameters["key"])
        elif self.llm is None and llm_parameters["mode"] == "Ollama":
            self.llm = OllamaLLMWrapper(scenario_db=self.db, model=llm_parameters["ollama_model"], base_url=llm_parameters["ollama_url"])
        elif self.llm is None and llm_parameters["mode"] == "Qwen":
            self.llm = QwenLLMWrapper(
                scenario_db=self.db,
                model=llm_parameters["qwen_model"],
                api_key=llm_parameters["qwen_key"],
                base_url=llm_parameters["qwen_base_url"],
            )

    def _run(self):
        """
        Background loop that waits for (step, input) pairs
        and calls the corresponding handler.
        """
        handlers = {
            "1": self._handle_location,
            "2": self._handle_tags,
            "3": self._handle_road_net,
            "4": self._handle_obstacles,
            "5": self._handle_velocity
        }
        # self.db = ScenarioDBWrapper(persist_dir="chroma")
        # self.llm = LLMWrapper(
        #     scenario_db=self.db,
        # )

        while self.running:
            try:
                step, user_input = self.input_queue.get(timeout=1)
                step = str(step)
                if step == "-1":
                    if len(self.after_velocity_ids) > 0:
                        self.result = self.after_velocity_ids
                        self.file_path = self.db.get_file_path(self.after_velocity_ids[0])
                    break
                if step in handlers:
                    # If a location was requested but has NO scenarios in the DB,
                    # keep the result empty for the remaining steps instead of
                    # letting tags fall back to an all-DB search or velocity fall
                    # back to a semantic search (which re-populated USA/ZAM).
                    if self.location_unavailable and step in ("2", "3", "4", "5"):
                        self.current_step = int(step)
                        self.current_results = []
                        continue
                    handlers[step](user_input)
            except queue.Empty:
                continue

        self._finalize()

    # TODO: still need to decide how to handle requests for specific cities (e.g. Schwetzingen, Munich)
    def _handle_location(self, input:str):
        print(f"[Processor] Running Step 1 with input: {input}")
        self.location_unavailable = None  # reset each time location is (re)processed

        # Check if LLM call necessary
        if len(input) == 0:
            self.debug_log += "\nSkipping location.\n###################"
            return

        self.extracted_location_json = self.llm.extract_location(input)
        # LLM call was necessary, but user still might have supplied unspecific information
        if not self.extracted_location_json:
            self.debug_log += "\nEither the user desires no specific location or the LLM was unable to extract a description. Skipping this step.\n###################"
            return

        self.after_location_ids = self.db.find_best_location(self.extracted_location_json)

        if len(self.after_location_ids) == 0:
            # The user named a location that has ZERO scenarios in the DB. Do NOT
            # silently continue (the search would ignore location and return
            # scenarios from other countries). Flag it so the UI can say so.
            self.debug_log += "\nThe desired location is not available in our data base. It is recommended to run a new search to receive a scenario relevant to the query. To restart the search immediately, please press the restart button.\n###################"
            self.location_unavailable = self._location_unavailable_message(self.extracted_location_json)
            self.current_step = 1
            self.current_results = []
        else:
            self.location_unavailable = None
            self.debug_log += f"\nLocation processed successfully. Extracted JSON object: {self.extracted_location_json}\nRemaining number of candidates: {len(self.after_location_ids)}\n###################"

            self.current_step = 1 # We use an approach of setting the current step to a number instead of +1, since steps may be skipped
            self.current_results = self.after_location_ids

    def _location_unavailable_message(self, location: dict) -> str:
        """User-facing message when a requested location has no scenarios."""
        cc = location.get("country_code")
        spec = location.get("specifier")
        asked = spec or cc or "that location"
        try:
            avail = self.db.available_countries()
            top = ", ".join(f"{c} ({n})" for c, n in avail[:10])
        except Exception:
            top = "several countries"
        return (
            f"❌ **No scenarios found for `{asked}`.**\n\n"
            f"The scenario database does not contain any scenarios for this location.\n\n"
            f"**Available countries** (by 3-letter code, count): {top}, …\n\n"
            f"You can:\n"
            f"   • **Type one of the available countries/locations** above to search it\n"
            f"   • Use the **Generate** tab to build a scenario for `{asked}` from OpenStreetMap\n"
            f"   • Press **Restart** to start over"
        )

    def _handle_tags(self, input:str):
        self.extracted_tags = self.llm.extract_tags(input)

        # Note that tags can handle the case of being supplied an empty list of location ids --> this means location will be ignored for the remainder of the run
        self.after_tags_ids, self.after_tags_docs = self.db.find_best_tags(self.after_location_ids, self.extracted_tags)

        if len(input) == 0: # This means the user  did not supply any tags and the LLM made a random choice
            self.debug_log += f"\nTags processed successfully. Tags were chosen by the LLM, since none were supplied: {self.extracted_tags}\nRemaining number of candidates: {len(self.after_tags_ids)}\n###################"
        else:
            self.debug_log += f"\nTags processed successfully. Extracted tags: {self.extracted_tags}\nRemaining number of candidates: {len(self.after_tags_ids)}\n###################"

            self.current_step = 2
            self.current_results = self.after_tags_ids

    def _handle_road_net(self, input:str):
        if len(input) == 0:
            self.debug_log += "\nSkipping road network.\n##################"
            self.after_road_net_ids, self.after_road_net_docs = self.after_tags_ids, self.after_tags_docs

            self.current_step = 3
            self.current_results = self.after_road_net_ids

            return

        self.extracted_road_net_json = self.llm.extract_road_net(input)
        if not self.extracted_road_net_json:
            self.after_road_net_ids, self.after_road_net_docs = self.after_tags_ids, self.after_tags_docs
            self.debug_log += "\nEither the user desires no specific road network or the LLM was unable to extract a description. Skipping this step.\n###################"
            return

        self.after_road_net_ids, self.after_road_net_docs = self.db.find_best_road_net(self.after_tags_ids, self.after_tags_docs, self.extracted_road_net_json)

        if len(self.after_road_net_ids) == 0:
            self.debug_log += f"\nThere are no results in our data base after specifying the desired road network. A relaxed search will be run. The current lookup failed with the following extracted parameters: {self.extracted_road_net_json}\nPlease press the restart button to start a new query.\n###################"

            # If no results were found, run a search with relaxed parameters
            # self.run_relaxation()
        else:
            self.debug_log += f"\nRoad network processed successfully. Extracted JSON object: {self.extracted_road_net_json}\nRemaining number of candidates: {len(self.after_road_net_ids)}\n###################"

            self.current_step = 3
            self.current_results = self.after_road_net_ids

    def _handle_obstacles(self, input:str):
        if len(input) == 0:
            self.debug_log += "\nSkipping obstacles.\n##################"
            self.after_obstacles_ids, self.after_obstacles_docs = self.after_road_net_ids, self.after_road_net_docs

            self.current_step = 4
            self.current_results = self.after_obstacles_ids

            return

        self.extracted_obstacles_json = self.llm.extract_obstacles(input)
        if not self.extracted_obstacles_json:
            self.after_obstacles_ids, self.after_obstacles_docs = self.after_road_net_ids, self.after_road_net_docs
            self.debug_log += "\nEither the user desires no specific obstacles or the LLM was unable to extract a description. Skipping this step.\n###################"
            return

        self.after_obstacles_ids, self.after_obstacles_docs = self.db.find_best_obstacles(self.after_road_net_ids, self.after_road_net_docs, self.extracted_obstacles_json)

        if len(self.after_obstacles_ids) == 0:
            self.debug_log += f"\nThere are no results in our data base after specifying the desired obstacles. A relaxed search will be run. The current lookup failed with the following extracted parameters: {self.extracted_obstacles_json}\nPlease press the restart button to start a new query.\n###################"

            # If no results were found, run a search with relaxed parameters
            # self.run_relaxation()
        else:
            self.debug_log += f"\nObstacles processed successfully. Extracted JSON object: {self.extracted_obstacles_json}\nRemaining number of candidates: {len(self.after_obstacles_ids)}\n###################"

            self.current_step = 4
            self.current_results = self.after_obstacles_ids

    def _handle_velocity(self, input:str):
        if len(input) == 0:
            self.debug_log += "\nSkipping velocity.\n###############"
            self.after_velocity_ids, self.after_velocity_docs = self.after_obstacles_ids, self.after_obstacles_docs

            self.current_step = 5
            self.current_results = self.after_velocity_ids

            return

        self.extracted_velocity_json = self.llm.extract_velocity(input)
        if not self.extracted_velocity_json:
            self.after_velocity_ids, self.after_velocity_docs = self.after_obstacles_ids, self.after_obstacles_docs
            self.debug_log += "\nEither the user desires no specific velocity or the LLM was unable to extract a description. Skipping this step.\n###################"
            return

        self.after_velocity_ids, self.after_velocity_docs = self.db.find_best_velocity(self.after_obstacles_ids, self.after_obstacles_docs, self.extracted_velocity_json)

        if len(self.after_velocity_ids) == 0:
            self.debug_log += f"\nThere are no results in our data base after specifying the desired velocity. Falling back to semantic similarity search...\n"
            
            # Fallback to semantic search
            query_parts = []
            if self.extracted_location_json:
                query_parts.append(f"location in {self.extracted_location_json}")
            if self.extracted_tags:
                query_parts.append(f"scenario with {', '.join(self.extracted_tags)}")
            if self.extracted_road_net_json:
                query_parts.append(f"road features: {self.extracted_road_net_json}")
            if self.extracted_obstacles_json:
                query_parts.append(f"obstacles: {self.extracted_obstacles_json}")
            if self.extracted_velocity_json:
                query_parts.append(f"velocity: {self.extracted_velocity_json}")
            
            semantic_query = " ".join(query_parts) if query_parts else "driving scenario"
            self.debug_log += f"Semantic query: {semantic_query}\n"
            
            # Use semantic search as fallback
            self.after_velocity_ids = self.db.semantic_search(semantic_query, n_results=20)
            self.debug_log += f"✓ Semantic search returned {len(self.after_velocity_ids)} scenarios\n###################"
            
            self.current_step = 5
            self.current_results = self.after_velocity_ids
        else:
            self.debug_log += f"\nVelocity processed successfully. Extracted JSON object: {self.extracted_velocity_json}\nRemaining number of candidates: {len(self.after_velocity_ids)}\n###################"

            self.current_step = 5
            self.current_results = self.after_velocity_ids

    def handle_input(self, step:int, user_input:str):
        """
        Called by UI logic to provide input for a specific step.
        """

        # If we are in modification mode, different behavior will be desired
        if step >= 6:
            return

        self.input_queue.put((step, user_input))

    # Returns the final result at the end as well as the corresponding file path
    def end_and_wait(self):
        """
        Signals the processor that input is complete and waits for result.
        """
        self.input_queue.put((-1, None))
        self.result_ready.wait() # Waits until processing is done
        return self.result, self.file_path

    def _finalize(self):
        print("[Processor] Finalizing result...")
        self.result_ready.set()

    def get_debug_log(self):
        return self.debug_log[-3000:]  # Only return last N chars

    # TODO: for handling file paths everywhere, try to make sure they work on all OS's
    # Returns a list of file paths as strings; does not allow possibility for image db outside of current working directory
    # Returns the step after which the images were created
    def get_images(self):
        output = []

        for result in self.current_results:
            new_name = result.removesuffix(".xml").removesuffix(".cr")
            image_path = Path(f"Scenarios/{new_name}/Original/{new_name}.png").resolve()

            if not image_path.exists():
                self.debug_log += f"\nCould not find image. Rendering started.\n###################"
                proc = subprocess.Popen(
                    [
                        conda_bin, "run", "-n", cr_conda_env,
                        "python", "commonroad_interface/create_png_subprocess.py",
                        "--input-file", f"Scenarios/{new_name}/Original/{result}",
                        "--target-name", new_name,
                        "--output-folder", f"Scenarios/{new_name}/Original",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                    universal_newlines=True
                )

                # Read output line by line as it arrives
                for stdout_line in proc.stdout:
                    print(stdout_line, end='')  # print live to console
                    self.console_log += stdout_line  # append to debug_log live

                # Also read stderr line by line (optional, or you can merge it into stdout)
                for stderr_line in proc.stderr:
                    print(stderr_line, end='')  # print live to console
                    self.console_log += stderr_line

                proc.stdout.close()
                proc.stderr.close()

                return_code = proc.wait()

                if return_code != 0:
                    raise RuntimeError("PNG creation failed.")

                self.debug_log += f"\nPNG creation complete.\n###################"

            output.append(str(image_path))

        return output, self.current_step

    def get_selected_images(self, selected_step: int):
        output = []

        # Helper function for processing image IDs
        def process_image_ids(image_ids):
            for result in image_ids:
                new_name = result.removesuffix(".xml").removesuffix(".cr")
                image_path = Path(f"Scenarios/{new_name}/Original/{new_name}.png").resolve()

                if not image_path.exists():
                    self.debug_log += f"\nCould not find image. Rendering started.\n###################"
                    proc = subprocess.Popen(
                        [
                            conda_bin, "run", "-n", cr_conda_env,
                            "python", "commonroad_interface/create_png_subprocess.py",
                            "--input-file", f"Scenarios/{new_name}/Original/{result}",
                            "--target-name", new_name,
                            "--output-folder", f"Scenarios/{new_name}/Original",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,  # Line buffered
                        universal_newlines=True
                    )

                    # Read output line by line as it arrives
                    for stdout_line in proc.stdout:
                        print(stdout_line, end='')  # print live to console
                        self.console_log += stdout_line

                    # Also read stderr line by line (optional, or you can merge it into stdout)
                    for stderr_line in proc.stderr:
                        print(stderr_line, end='')  # print live to console
                        self.console_log += stderr_line

                    proc.stdout.close()
                    proc.stderr.close()

                    return_code = proc.wait()

                    if return_code != 0:
                        raise RuntimeError("PNG creation failed.")

                    self.debug_log += f"\nPNG creation complete.\n###################"

                output.append(str(image_path))

        if selected_step == 1:
            process_image_ids(self.after_location_ids)
            return output, 1
        elif selected_step == 2:
            process_image_ids(self.after_tags_ids)
            return output, 2
        elif selected_step == 3:
            process_image_ids(self.after_road_net_ids)
            return output, 3
        elif selected_step == 4:
            process_image_ids(self.after_obstacles_ids)
            return output, 4
        elif selected_step == 5:
            process_image_ids(self.after_obstacles_ids)
            return output, 5
        else:
            return self.gif_displays, 7

    def select_initial_gif(self, selected_candidate: str) -> list[str]:
        """
        Selects or generates a GIF for the given scenario candidate.

        The function assumes a predefined folder structure where the scenario is located under:
            Scenarios/<ScenarioName>/Original/<ScenarioName>.xml

        If the corresponding GIF does not already exist, the function will call the
        Frenetix motion planner (via a subprocess) to generate it. The function also
        adds console and debug log information and summarizes vehicle start/end edges
        if a CommonRoad .cr file (.cr.xml) is present.

        Parameters:
        ----------
        selected_candidate : str
            Path to the selected scenario file (either .xml or .png). This is used to
            extract the base scenario name.

        Returns:
        -------
        list[str] or None
            Returns a list with the file path to the rendered GIF (as a string), or
            None if an error occurred during GIF generation.
        """

        # Remove all possible suffixes (.png, .xml, .gif) to get clean scenario name
        file_name = Path(selected_candidate).name.removesuffix(".png").removesuffix(".xml").removesuffix(".gif")
        folder_path = Path("Scenarios").resolve() / file_name
        original_path = Path(folder_path) / "Original"
        gif_path = (Path(original_path) / file_name).with_suffix(".gif")

        print("GIF should be at", gif_path)
        self.console_log += f"\nGIF should be at {gif_path}"

        if not gif_path.exists():
            scenario_xml = original_path / f"{file_name}.xml"
            self.debug_log += "\nGIF does not exist and is being rendered now.\n###################"

            # Path to the virtual environment activation script
            venv_python = Path("Frenetix-Motion-Planner/venv/bin/activate").resolve()

            # Path to the Frenetix rendering script
            script_path = Path("Frenetix-Motion-Planner/main_without_ego.py").resolve()

            # Bash command to activate the virtual environment and run the GIF creation script
            command = f"source {venv_python} && python {script_path} --input-file {str(scenario_xml)} --output-dir {str(original_path)}"

            try:
                # Launch subprocess with line-buffered stdout/stderr
                # This renders the GIF based on a Frenetix simulation (without an ego vehicle)
                proc = subprocess.Popen(
                    ["bash", "-c", command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                # Stream stdout live into console log
                for stdout_line in proc.stdout:
                    print(stdout_line, end='')
                    self.console_log += stdout_line

                # Stream stderr live into console log
                for stderr_line in proc.stderr:
                    print(stderr_line, end='')
                    self.console_log += stderr_line

                proc.stdout.close()
                proc.stderr.close()

                return_code = proc.wait()

                if return_code != 0:
                    # Subprocess failed — append error and return fallback value
                    error_msg = f"\n❌ Frenetix subprocess exited with code {return_code}.\nCheck input XML or dependencies.\n"
                    self.console_log += error_msg
                    return None

                self.debug_log += "\nGIF was successfully rendered.\n###################"

            except Exception as e:
                # Unexpected exception — log and return fallback value
                error_msg = f"\n❌ Exception during GIF rendering subprocess: {str(e)}\n"
                self.console_log += error_msg
                return None

        else:
            print("GIF already exists.")

        # Try to summarize vehicle dynamics from the CommonRoad .cr.xml file if available
        cr_xml_path = original_path / f"{file_name}.cr.xml"
        if os.path.exists(cr_xml_path):
            summary = dynamic_summary_from_cr(str(cr_xml_path))
        else:
            summary = dynamic_summary_from_cr(str(original_path / f"{file_name}.xml"))

        self.debug_log += (
            "\nHere is a summary of the start and end edges of the dynamic vehicles in this simulation:"
            f"\n{summary}\n##################"
        )

        self.gif_displays = [str(gif_path)]
        self.console_log += f"\nGIF is at {gif_path}"

        return self.gif_displays

    def load_generated_scenario(self, name: str) -> list[str] | None:
        """Promote a freshly-generated scenario (produced by the Generate
        tab via osm_pipeline) into the chat workflow at step 7.

        Expects ``Scenarios/<name>/Original/<name>.xml`` to already exist
        — the Generate tab writes the planning-problem CR XML there. This
        method renders the GIF via the existing Frenetix-without-ego
        subprocess (same path used by direct scenario-name lookup) and
        returns the GIF list so the caller can populate Gradio state.
        """
        xml_path = Path("Scenarios").resolve() / name / "Original" / f"{name}.xml"
        if not xml_path.exists():
            self.console_log += (
                f"\nload_generated_scenario: expected {xml_path} to exist "
                "but it does not. Did the Generate tab write the file?"
            )
            return None

        self.current_results = [f"{name}.xml"]
        self.current_step = 7
        return self.select_initial_gif(name)

    def _repair_invalid_routes(self, modified_route_path, net_path) -> str:
        """Make every vehicle route in a SUMO route file VALID so the SUMO
        re-simulation cannot crash. A route is invalid if it references an edge
        missing from the net, or has a disconnected hop — both are common after
        the CommonRoad->SUMO conversion (renamed/dropped edges left in the
        routes) and from LLM-guessed add/reroute paths. Each invalid route is
        replaced by a connectivity-valid path between its first and last
        *existing* edges (BFS over the net's <connection> graph), falling back to
        the longest valid connected walk. Valid routes are left untouched — the
        LLM still chooses intent (what/where); this only guarantees validity.
        Returns a short summary string."""
        import re, xml.etree.ElementTree as ET
        from collections import deque
        try:
            net_txt = open(net_path).read()
        except Exception:
            return ""

        edges_in_net = {e for e in re.findall(r'<edge id="([^"]+)"', net_txt)
                        if not e.startswith(":")}
        graph = {}
        for m in re.finditer(r'<connection from="([^"]+)" to="([^"]+)"', net_txt):
            f, t = m.group(1), m.group(2)
            if f.startswith(":") or t.startswith(":"):
                continue
            graph.setdefault(f, set()).add(t)

        def bfs(start, end):
            if start == end:
                return [start]
            seen = {start}
            q = deque([[start]])
            while q:
                path = q.popleft()
                for nxt in graph.get(path[-1], ()):
                    if nxt == end:
                        return path + [nxt]
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(path + [nxt])
            return None

        def is_valid(edges):
            return (all(e in edges_in_net for e in edges)
                    and all(edges[i + 1] in graph.get(edges[i], ())
                            for i in range(len(edges) - 1)))

        def repair(edges):
            existing = [e for e in edges if e in edges_in_net]
            if not existing:
                return None
            # 1) shortest connected path between the intended endpoints
            p = bfs(existing[0], existing[-1])
            if p:
                return p
            # 2) fallback: longest connected walk, bridging short gaps
            out = [existing[0]]
            for nxt in existing[1:]:
                if nxt == out[-1]:
                    continue
                if nxt in graph.get(out[-1], ()):
                    out.append(nxt)
                else:
                    bridge = bfs(out[-1], nxt)
                    if bridge and len(bridge) <= 10:
                        out.extend(bridge[1:])
                    else:
                        break
            return out

        try:
            tree = ET.parse(modified_route_path)
        except Exception:
            return ""
        root = tree.getroot()
        fixed, failed, changed = [], [], False
        for veh in root.findall("vehicle"):
            route_el = veh.find("route")
            if route_el is None:
                continue
            edges = (route_el.get("edges") or "").split()
            if not edges or is_valid(edges):
                continue
            new = repair(edges)
            vid = veh.get("id")
            if new and is_valid(new):
                route_el.set("edges", " ".join(new))
                fixed.append((vid, edges, new))
                changed = True
            else:
                failed.append(vid)
        if changed:
            tree.write(modified_route_path, encoding="utf-8", xml_declaration=True)
        parts = [f"veh {vid}: [{' '.join(o)}] -> [{' '.join(n)}]" for vid, o, n in fixed]
        if failed:
            parts.append(f"could not repair: {failed}")
        return "; ".join(parts)

    def _handle_modification(self, user_input: str, scenario: str, mod_type: str):
        """
        Modifies the provided CommonRoad scenario based on the user's prompt using a language model.
        The method:
          1. Converts the scenario to SUMO format (if not already available).
          2. Optionally repairs broken vehicle routes.
          3. Prompts an LLM to modify the scenario's route file based on user input.
          4. Simulates the modified scenario.
          5. Generates a visual GIF of the new scenario.
          6. Updates internal state to track the new GIF and returns it.

        Parameters:
            user_input (str): Instruction or modification request for the LLM.
            scenario (str): Path to the original scenario GIF file.

        Returns:
            tuple: (List of paths to all GIFs including the new one, Index of the new GIF)
                   If an error occurs, returns (None, 0)
        """
        self.debug_log += "\n\nMODIFICATION STARTED\n"

        scenario_pre = Path(scenario).resolve().name.removesuffix(".gif").removesuffix(".png").removesuffix(
            "_without_ego")
        full_path = str(Path(scenario).resolve().parent)

        repaired_config = Path(full_path) / f"{scenario_pre}_repaired.vehicles.rou.xml"
        regular_config = Path(full_path) / f"{scenario_pre}.vehicles.rou.xml"

        if repaired_config.exists():
            self.debug_log += f"A repaired SUMO configuration of base scenario exists at {full_path}\n###################"
            used_base_config = Path(full_path) / f"{scenario_pre}_repaired"
        elif regular_config.exists():
            self.debug_log += f"\nA SUMO configuration of base scenario exists at {full_path}/SUMO\n###################"
            used_base_config = Path(full_path) / f"{scenario_pre}"
        else:
            used_base_config = Path(full_path) / f"{scenario_pre}"
            self.debug_log += f"\nNo SUMO configuration for the base scenario exists. Conversion process started.\n###################"

            try:
                proc = subprocess.Popen(
                    [
                        conda_bin, "run", "-n", cr_conda_env,
                        "python", "commonroad_interface/convert_to_sumo_subprocess.py",
                        "--input-file", f"{full_path}/{scenario_pre}.xml",
                        "--target-name", scenario_pre,
                        "--sumo-root", f"{full_path}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                for stdout_line in proc.stdout:
                    print(stdout_line, end='')
                    self.console_log += stdout_line
                for stderr_line in proc.stderr:
                    print(stderr_line, end='')
                    self.console_log += stderr_line

                proc.stdout.close()
                proc.stderr.close()

                if proc.wait() != 0:
                    raise RuntimeError("SUMO Conversion subprocess failed.")
            except Exception as e:
                self.debug_log += f"\nError during SUMO conversion subprocess: {e}\n"
                return None, 0

            self.debug_log += "\nSUMO Conversion process of base scenario finished successfully.\n###################"

        base_net_file = f"{used_base_config}.net.xml"
        base_route_file = f"{used_base_config}.vehicles.rou.xml"

        requires_repair, broken_vehicles, exit_edges = requires_trajectory_generation(base_net_file,
                                                                                      base_route_file)
        if requires_repair and SessionConfig.get_use_repair_module():
            self.console_log += "\n🔧 The scenario contains broken trajectories. Repair will be run automatically. 🔧\n"
            generate_exit_trajectories_without_llm(base_net_file, base_route_file, broken_vehicles, exit_edges)
        elif requires_repair:
            self.console_log += (
                "\n⚠️  The scenario contains broken trajectories that need repair.\n"
                "   Repair module is currently DISABLED.\n"
                "   \n"
                "   To fix this scenario:\n"
                "   1. Close the application (Ctrl+C in terminal)\n"
                "   2. Re-run: python interface.py\n"
                "   3. When prompted, select 'y' (yes) to enable repair module\n"
                "   4. Re-do your modification\n"
                "   \n"
                "   Without repair, the motion planner may crash or produce incorrect results.\n"
            )
            self.debug_log += "\nWARNING: Broken trajectories detected. Repair module is disabled. User needs to re-run app with repair enabled.\n###################"

        scenario_post = generate_modified_scenario_name(scenario_pre, mod_type)
        self.scenario_parents[scenario_post] = scenario   # parent, for per-modification diff

        if Path(full_path).name == "Original":
            folder_path = Path(full_path).parent / "Modified" / scenario_post
        else:
            folder_path = Path(full_path).parent / scenario_post
        os.makedirs(folder_path, exist_ok=True)

        self.debug_log += "\nThe folder for the modified SUMO files has been created. Conversion process started.\n###################"

        try:
            proc = subprocess.Popen(
                [
                    conda_bin, "run", "-n", cr_conda_env,
                    "python", "commonroad_interface/convert_to_sumo_subprocess.py",
                    "--input-file", f"{used_base_config}.xml",
                    "--target-name", scenario_post,
                    "--sumo-root", f"{folder_path}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for stdout_line in proc.stdout:
                print(stdout_line, end='')
                self.debug_log += stdout_line
            for stderr_line in proc.stderr:
                print(stderr_line, end='')
                self.debug_log += stderr_line

            proc.stdout.close()
            proc.stderr.close()

            if proc.wait() != 0:
                raise RuntimeError("SUMO Conversion subprocess failed.")
        except Exception as e:
            self.debug_log += f"\nError during modified SUMO conversion: {e}\n"
            return None, 0

        self.debug_log += "\nSUMO Conversion process terminated successfully.\n###################"
        self.debug_log += "\nThe LLM is now being prompted.\n###################"

        try:
            modified_rou_file = self.llm.modify(
                net_file_path=base_net_file,
                rou_file_path=base_route_file,
                user_prompt=user_input
            )
        except Exception as e:
            self.debug_log += f"\nError during LLM modification: {e}\n"
            return None, 0

        self.debug_log += "\nThe LLM has responded.\n###################"

        # Validate the modified route file before writing
        if not modified_rou_file or len(modified_rou_file.strip()) == 0:
            self.debug_log += "\n❌ Error: LLM returned empty route file!\n"
            self.console_log += "\n❌ LLM modification failed: Empty route file returned\n"
            return None, 0
        
        if "<routes" not in modified_rou_file:
            self.debug_log += "\n❌ Error: LLM returned invalid route file (no <routes> tag)!\n"
            self.console_log += "\n❌ LLM modification failed: Invalid XML format\n"
            return None, 0

        try:
            with open(f"{folder_path}/{scenario_post}.vehicles.rou.xml", "w") as dst:
                dst.write(modified_rou_file)
            self.debug_log += f"\n✅ Route file written: {len(modified_rou_file)} bytes\n"
        except Exception as e:
            self.debug_log += f"\n❌ Error writing modified route file: {e}\n"
            self.console_log += f"\n❌ Failed to write route file: {e}\n"
            return None, 0

        # Route-finder: make EVERY route valid before re-simulating. The LLM is
        # unreliable at connecting edges (add/reroute), AND the CommonRoad->SUMO
        # conversion can leave preserved vehicles with routes referencing
        # missing edges — any one broken route makes SUMO fail and every
        # modification silently falls back to the original. Repairing all routes
        # (BFS over the net's <connection> graph) keeps the simulation running.
        repaired = self._repair_invalid_routes(
            f"{folder_path}/{scenario_post}.vehicles.rou.xml",
            f"{folder_path}/{scenario_post}.net.xml")
        if repaired:
            msg = f"🛣️  Repaired invalid route(s) so the simulation can run — {repaired}"
            self.console_log += f"\n{msg}\n"
            self.debug_log += f"\n{msg}\n"

        self.debug_log += "\nRoute file has been modified.\n###################"
        self.debug_log += "\nRunning simulation of newly created scenario.\n###################"

        try:
            proc = subprocess.Popen(
                [
                    conda_bin, "run", "-n", cr_conda_env,
                    "python", "commonroad_interface/simulate_sumo_subprocess.py",
                    "--folder_path", str(folder_path),
                    "--preserve_trajectories",  # NEW: Preserve exact trajectories for unmodified vehicles
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for stdout_line in proc.stdout:
                print(stdout_line, end='')
                self.console_log += stdout_line
            for stderr_line in proc.stderr:
                print(stderr_line, end='')
                self.console_log += stderr_line

            proc.stdout.close()
            proc.stderr.close()

            if proc.wait() != 0:
                raise RuntimeError("Simulation subprocess failed.")
        except Exception as e:
            self.debug_log += f"\nError during simulation subprocess: {e}\n"
            return None, 0

        new_file = update_benchmark_id(str(folder_path / f"{scenario_post}.xml"), scenario_post)

        self.debug_log += "\nSimulation finished.\n###################"
        self.debug_log += "\nGIF creation started.\n###################"

        venv_python = Path("Frenetix-Motion-Planner/venv/bin/activate").resolve()
        script_path = Path("Frenetix-Motion-Planner/main_without_ego.py").resolve()
        command = f"source {venv_python} && python {script_path} --input-file {new_file} --output-dir {folder_path}"

        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for stdout_line in proc.stdout:
                print(stdout_line, end='')
                self.console_log += stdout_line
            for stderr_line in proc.stderr:
                print(stderr_line, end='')
                self.console_log += stderr_line

            proc.stdout.close()
            proc.stderr.close()

            if proc.wait() != 0:
                raise RuntimeError("Frenetix simulation subprocess failed.")
        except Exception as e:
            self.debug_log += f"\nError during GIF creation subprocess: {e}\n"
            return None, 0

        self.debug_log += "\nGIF creation finished.\n##################"

        gif_path = folder_path / (scenario_post + ".gif")
        self.gif_displays.append(str(gif_path))

        self.debug_log += "\n\nMODIFICATION FINISHED\n"

        summary = dynamic_summary_from_cr(new_file)
        self.debug_log += f"\nHere is a summary of the start and end edges of the dynamic vehicles in this simulation:\n{summary}\n##################"

        # Construct list of modifications to keep track of changes
        # Indices align with gif list, so it is always clear which scenario had which modification
        self.modifications.append(f"Modified from: {scenario} - Modification: {user_input}")

        return self.gif_displays, len(self.gif_displays) - 1

    def _handle_goal_modification(self, user_input: str, scenario: str):
        """
        Modifies the ego vehicle's goal state in the planning problem.
        
        Parameters:
            user_input (str): User's request to change the goal position
            scenario (str): Path to the current scenario GIF file
            
        Returns:
            tuple: (List of paths to all GIFs including the new one, Index of the new GIF)
                   If an error occurs, returns (None, 0)
        """
        self.debug_log += "\n\nGOAL MODIFICATION STARTED\n"
        
        scenario_pre = Path(scenario).resolve().name.removesuffix(".gif").removesuffix(".png").removesuffix("_without_ego")
        full_path = str(Path(scenario).resolve().parent)
        
        # Load the current CommonRoad XML
        cr_xml_path = Path(full_path) / f"{scenario_pre}.xml"
        if not cr_xml_path.exists():
            self.debug_log += f"\nError: CommonRoad XML file not found at {cr_xml_path}\n"
            return None, 0
        
        with open(cr_xml_path, 'r') as f:
            xml_string = f.read()
        
        # Prompt LLM to extract target edge ID and position
        self.debug_log += "\nPrompting LLM to extract target edge ID and position...\n"
        
        goal_json = self.llm.modify_goal(user_input, xml_string)
        
        target_edge = goal_json.get("target_edge")
        position = goal_json.get("position", "final")  # Default to "final" if not specified
        
        if target_edge is None:
            error_msg = goal_json.get("description", "No target edge specified")
            self.debug_log += f"\nError: {error_msg}\n"
            return None, 0
        
        self.debug_log += f"\nTarget edge/lanelet: {target_edge}\n"
        self.debug_log += f"\nPosition in lanelet: {position}\n"
        
        # Calculate new goal position from the target edge/lanelet
        self.debug_log += f"\nCalculating goal position for edge {target_edge} at {position}...\n"
        new_goal = calculate_goal_from_lanelet(xml_string, target_edge, position)
        
        if new_goal is None:
            self.debug_log += f"\nError: Could not find lanelet {target_edge} in scenario\n"
            return None, 0
        
        self.debug_log += f"\nNew goal position: x={new_goal['x']:.2f}, y={new_goal['y']:.2f}, orientation={new_goal['orientation']:.2f}\n"
        
        # Modify the XML with the new goal
        modified_xml = modify_goal_state(xml_string, new_goal)
        
        # Generate new scenario name
        scenario_post = generate_modified_scenario_name(scenario_pre, "G")
        self.scenario_parents[scenario_post] = scenario   # parent, for per-modification diff
        
        if Path(full_path).name == "Original":
            folder_path = Path(full_path).parent / "Modified" / scenario_post
        else:
            folder_path = Path(full_path).parent / scenario_post
        os.makedirs(folder_path, exist_ok=True)
        
        # Save the modified XML
        new_xml_path = folder_path / f"{scenario_post}.xml"
        with open(new_xml_path, 'w') as f:
            f.write(modified_xml)
        
        self.debug_log += f"\nModified XML saved to {new_xml_path}\n"
        
        # Regenerate the GIF visualization
        self.debug_log += "\nRegenerating scenario visualization...\n"
        try:
            proc = subprocess.Popen(
                [
                    conda_bin, "run", "-n", cr_conda_env,
                    "python", "Frenetix-Motion-Planner/main_without_ego.py",
                    "--input-file", str(new_xml_path),
                    "--output-dir", str(folder_path)
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for stdout_line in proc.stdout:
                print(stdout_line, end='')
                self.console_log += stdout_line
            for stderr_line in proc.stderr:
                print(stderr_line, end='')
                self.console_log += stderr_line
            
            proc.stdout.close()
            proc.stderr.close()
            
            if proc.wait() != 0:
                raise RuntimeError("Visualization subprocess failed.")
        except Exception as e:
            self.debug_log += f"\nError during visualization: {e}\n"
            return None, 0
        
        self.debug_log += "\nVisualization finished.\n"
        
        gif_path = folder_path / (scenario_post + ".gif")
        self.gif_displays.append(str(gif_path))
        
        self.debug_log += "\n\nGOAL MODIFICATION FINISHED\n"
        
        # Track the modification
        self.modifications.append(f"Modified from: {scenario} - Goal modification: {user_input}")

        return self.gif_displays, len(self.gif_displays) - 1

    def _handle_cost_param_change(self, user_input: str):
        # Store the old, unedited version for safety purposes
        cost_file_path = Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml").resolve()
        old_cost_file_path = Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml").resolve().parent / "cost_old.yaml"
        if not os.path.exists(old_cost_file_path):
            with open(str(cost_file_path), "r") as f:
                cost_file = f.read()
            with open(str(old_cost_file_path), "w") as dst:
                dst.write(cost_file)

        new_cost_file = self.llm.update_frenetix_cost(user_input, str(cost_file_path))

        # TODO: this check does not work entirely and is maybe more complicated than necessary
        # if not check_yaml_structure(cost_file, new_cost_file):
            # return False, new_cost_file

        cleaned_cost_file = extract_clean_yaml(new_cost_file)

        if cleaned_cost_file is None:
            return False, new_cost_file

        # Snapshot the weights as they are RIGHT NOW (before this update) so a
        # later "which weights changed?" can diff this specific change.
        # cost_old.yaml stays the original baseline; cost_prev.yaml tracks the
        # immediately-previous state.
        prev_cost_file_path = cost_file_path.parent / "cost_prev.yaml"
        try:
            with open(str(cost_file_path), "r") as src:
                with open(str(prev_cost_file_path), "w") as dst:
                    dst.write(src.read())
        except OSError:
            pass

        with open(str(cost_file_path), "w") as dst:
            dst.write(cleaned_cost_file)

        return True, cleaned_cost_file

    def _compare_cost_weights(self) -> str:
        """Diff the current planner cost weights against the previous (or, if no
        previous snapshot exists, the original default) values and report which
        weights changed. Backs the 'param_compare' action ("which weights
        changed?")."""
        import yaml
        base = Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner").resolve()
        cur_p = base / "cost.yaml"
        prev_p = base / "cost_prev.yaml"
        orig_p = base / "cost_old.yaml"

        if not cur_p.exists():
            return "There are no planner parameters to compare (cost.yaml not found)."
        if prev_p.exists():
            ref_p, ref_label = prev_p, "the previous values (before your last change)"
        elif orig_p.exists():
            ref_p, ref_label = orig_p, "the planner defaults"
        else:
            return ("No earlier parameter snapshot to compare against yet — change the "
                    "cost weights once, then ask again and I'll show exactly what changed.")

        def _load(p):
            with open(str(p), "r") as f:
                return yaml.safe_load(f) or {}

        try:
            cur, ref = _load(cur_p), _load(ref_p)
        except Exception as e:
            return f"Could not read the parameter files for comparison: {type(e).__name__}: {e}"

        def _num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None

        sections = []
        for section in ("cost_weights", "external_cost_weights"):
            c = cur.get(section) if isinstance(cur.get(section), dict) else {}
            r = ref.get(section) if isinstance(ref.get(section), dict) else {}
            rows = []
            for k in sorted(set(c) | set(r)):
                cv, rv = c.get(k), r.get(k)
                if k not in r:
                    rows.append(f"  + {k}: (new) → {cv}")
                elif k not in c:
                    rows.append(f"  − {k}: {rv} → (removed)")
                else:
                    cn, rn = _num(cv), _num(rv)
                    if cn is None or rn is None:
                        if cv != rv:
                            rows.append(f"  • {k}: {rv} → {cv}")
                    elif abs(cn - rn) > 1e-9:
                        arrow = "↑" if cn > rn else "↓"
                        rows.append(f"  • {k}: {rv} → {cv}   ({arrow} {abs(cn - rn):.3g})")
            if rows:
                sections.append(f"{section}:\n" + "\n".join(rows))

        if not sections:
            return f"No cost weights differ from {ref_label}."
        return (f"Here's what changed vs {ref_label}:\n\n" + "\n\n".join(sections))

    def _diff_scenarios(self, before_path, after_path) -> str:
        """Diff two CommonRoad scenarios for a before/after view: vehicles
        added/removed, per-vehicle speed changes, and ego-goal moves."""
        from commonroad.common.file_reader import CommonRoadFileReader
        import numpy as np
        try:
            b_sc, b_pp = CommonRoadFileReader(str(before_path)).open()
            a_sc, a_pp = CommonRoadFileReader(str(after_path)).open()
        except Exception as e:
            return f"Could not read the scenarios to compare: {type(e).__name__}: {e}"

        def vmap(sc):
            d = {}
            for o in sc.dynamic_obstacles:
                sts = [o.initial_state] + (o.prediction.trajectory.state_list if o.prediction else [])
                vs = [s.velocity for s in sts if getattr(s, "velocity", None) is not None]
                d[o.obstacle_id] = (float(np.mean(vs)) if vs else 0.0,
                                    float(np.max(vs)) if vs else 0.0)
            return d

        bm, am = vmap(b_sc), vmap(a_sc)
        removed = sorted(set(bm) - set(am))
        added = sorted(set(am) - set(bm))
        speed_changes = []
        for vid in sorted(set(bm) & set(am)):
            (bmean, bmax), (amean, amax) = bm[vid], am[vid]
            if abs(amax - bmax) > 0.3 or abs(amean - bmean) > 0.3:
                speed_changes.append((vid, bmean, bmax, amean, amax))

        lines = [f"**Before → after:** {len(bm)} → {len(am)} vehicles"]
        if removed:
            lines.append(f"➖ Removed vehicle(s): {removed}")
        if added:
            lines.append(f"➕ Added vehicle(s): {added}")
        for vid, bmean, bmax, amean, amax in speed_changes:
            lines.append(f"🚗 Vehicle {vid} speed: mean {bmean:.1f}→{amean:.1f}, "
                         f"max {bmax:.1f}→{amax:.1f} m/s")
        # Route (edge/lanelet) changes — from the SUMO route files, if present.
        import re as _re
        def route_map(scen_path):
            p = Path(scen_path)
            nm = p.name
            for suf in (".cr.xml", ".xml"):
                if nm.endswith(suf):
                    nm = nm[:-len(suf)]
                    break
            rou = p.parent / f"{nm}.vehicles.rou.xml"
            if not rou.exists():
                return {}
            out = {}
            for mm in _re.finditer(r'<vehicle id="(\d+)".*?<route edges="([^"]*)"',
                                   rou.read_text(), _re.S):
                out[int(mm.group(1))] = mm.group(2).strip()
            return out
        br, ar = route_map(before_path), route_map(after_path)
        for vid in sorted(set(br) & set(ar)):
            if br[vid] != ar[vid]:
                lines.append(f"🛣️ Vehicle {vid} route: [{br[vid]}] → [{ar[vid]}]")

        # Ego-goal move, described by lanelet + position.
        try:
            def goal_desc(sc, pp):
                for p in pp.planning_problem_dict.values():
                    g = p.goal.state_list[0]
                    pos = getattr(g, "position", None)
                    c = getattr(pos, "center", None) if pos is not None else None
                    if c is None:
                        continue
                    xy = (round(float(c[0]), 1), round(float(c[1]), 1))
                    found = sc.lanelet_network.find_lanelet_by_position([np.array(c)])
                    lid = found[0][0] if found and found[0] else None
                    return xy, lid
                return None, None
            (bxy, blid), (axy, alid) = goal_desc(b_sc, b_pp), goal_desc(a_sc, a_pp)
            if bxy and axy and (bxy != axy or blid != alid):
                bd = f"lanelet {blid} @ {bxy}" if blid is not None else f"{bxy}"
                ad = f"lanelet {alid} @ {axy}" if alid is not None else f"{axy}"
                lines.append(f"🎯 Ego goal moved: {bd} → {ad}")
        except Exception:
            pass

        if len(lines) == 1:
            return "No differences detected — vehicles, speeds, routes and goal are unchanged."
        return "\n".join(lines)

    def _scenario_diff_message(self, gifs, index) -> str:
        """Diff the current scenario against its IMMEDIATE parent (the scenario
        it was modified from); fall back to the original base scenario if no
        parent was tracked. The 'what changed before vs after?' view."""
        import re
        if not gifs or not (0 <= index < len(gifs)):
            return "No scenario is loaded yet — pick or modify a scenario first."

        def cr_file(gif_path):
            p = Path(gif_path)
            nm = p.name.removesuffix(".gif").removesuffix("_without_ego")
            return next((c for c in (p.parent / f"{nm}.cr.xml", p.parent / f"{nm}.xml")
                         if c.exists()), None)

        cur = Path(gifs[index])
        scen = cur.name.removesuffix(".gif").removesuffix("_without_ego")
        after_path = cr_file(gifs[index])
        if after_path is None:
            return "Couldn't find the current scenario's CommonRoad file to compare."

        # Immediate parent (what this scenario was modified from), if tracked.
        before_path, before_label = None, None
        parent_gif = self.scenario_parents.get(scen)
        if parent_gif:
            before_path = cr_file(parent_gif)
            before_label = Path(parent_gif).name.removesuffix(".gif").removesuffix("_without_ego")

        # Fall back to the original base scenario.
        if before_path is None:
            m = re.match(r"(.+_T-\d+)_[A-Za-z]+\d+$", scen)
            base = m.group(1) if m else scen
            if base == scen:
                return ("This is the original, unmodified scenario — make a change first "
                        "(behavior / reroute / add / remove / goal), then ask again and I'll "
                        "show exactly what changed.")
            parts = cur.resolve().parts
            root = Path(*parts[:parts.index("Modified")]) if "Modified" in parts else cur.parent.parent
            cand = root / "Original" / f"{base}.xml"
            if cand.exists():
                before_path, before_label = cand, base

        if before_path is None:
            return "Couldn't find a previous scenario to compare against."
        return (f"Here's what changed in **{scen}** vs **{before_label}**:\n\n"
                + self._diff_scenarios(before_path, after_path))

    def _scenario_vehicle_ids(self, gif_path: str):
        """Best-effort list of dynamic-obstacle (vehicle) IDs in the scenario
        backing the given GIF, parsed straight from its CommonRoad XML."""
        import re
        try:
            p = Path(gif_path)
            scen = p.name.removesuffix(".gif").removesuffix("_without_ego")
            for cand in (p.parent / f"{scen}.cr.xml", p.parent / f"{scen}.xml"):
                if cand.exists():
                    ids = sorted({int(m) for m in re.findall(
                        r'dynamicObstacle id="(\d+)"', cand.read_text())})
                    if ids:
                        return ids
        except Exception:
            pass
        return []

    def _behaviour_styles(self) -> list:
        """Driving-style preset names, parsed from the behavior-modification
        prompt so the help reflects whatever presets are actually defined
        (instead of a hard-coded list)."""
        import re
        try:
            txt = Path("modification_prompt_files/manipulation_prompt.txt").read_text()
            m = re.search(r"Available Behavior Presets(.*?)Few-Shot Examples", txt, re.S)
            section = m.group(1) if m else txt
            styles = []
            for lab in re.findall(r"^\s*([A-Z][A-Za-z][^<>:\n]*?):\s*$", section, re.M):
                w = re.match(r"[A-Za-z]+", lab.strip())
                if w and w.group(0).lower() not in styles:
                    styles.append(w.group(0).lower())
            return styles
        except Exception:
            return []

    def _goal_positions(self) -> list:
        """Goal position keywords (where along the lanelet), parsed from the
        goal-modification prompt."""
        import re
        try:
            txt = Path("modification_prompt_files/goal_modification_prompt.txt").read_text()
            out = []
            for v in re.findall(r'position:\s*"(\w+)"', txt):
                v = {"final": "end"}.get(v, v).replace("_", "-")
                if v not in out:
                    out.append(v)
            return out
        except Exception:
            return []

    def _modify_help(self, gifs, index) -> str:
        """General introduction shown when the user wants to modify but gave no
        specifics: explains the capabilities in plain language, the actual
        driving-style and goal-position options (parsed from the prompts), and
        this scenario's vehicles — and invites follow-up questions."""
        ids = self._scenario_vehicle_ids(gifs[index]) if gifs and 0 <= index < len(gifs) else []
        styles = self._behaviour_styles()
        positions = self._goal_positions()

        def _join(xs, last="or"):
            xs = [str(x) for x in xs]
            if not xs:
                return ""
            if len(xs) == 1:
                return xs[0]
            return ", ".join(xs[:-1]) + f", {last} " + xs[-1]

        if ids:
            who = (f"target a specific vehicle by its ID — this scenario has **{len(ids)}** of "
                   f"them (IDs: {', '.join(map(str, ids))}) — or say **all vehicles**")
        else:
            who = "target a specific vehicle by its ID, or say **all vehicles**"
        styles_txt = _join(styles) if styles else "cautious, aggressive, faster"
        pos_txt = _join(positions) if positions else "beginning, middle, or end"
        ex_id = ids[0] if ids else 8

        return (
            "Sure — I can change this scenario in a few ways. Just describe what you'd like in "
            "plain language (there's no fixed syntax), and say which vehicle: "
            f"{who}.\n\n"
            "**What I can do:**\n"
            f"• **Driving style** — make a vehicle drive differently. Available styles: "
            f"*{styles_txt}* (or just say *more cautious* / *more aggressive* / *faster* and "
            "I'll match the closest one).\n"
            "• **Add or remove a vehicle** — remove a specific car, or add one on a chosen lanelet.\n"
            "• **Reroute a vehicle** — send it onto a different lanelet.\n"
            f"• **Move the ego goal** — to a target lanelet, and optionally where along it: "
            f"*{pos_txt}* (default: end).\n\n"
            f"**Examples:** *make vehicle {ex_id} more cautious* · *make all vehicles aggressive* · "
            "*remove the third car* · *move the ego goal to the middle of lanelet 42*\n\n"
            "You can also just ask — e.g. *what driving styles are available?* or "
            "*where on the lanelet can the goal go?*"
        )

    def _handle_traffic_light_modification(self, user_input: str, scenario: str):
        traffic_lights = self.llm.modify_traffic_lights(user_input)

        scenario_pre = Path(scenario).resolve().name.removesuffix(".gif").removesuffix(".png").removesuffix(
            "_without_ego")
        full_path = str(Path(scenario).resolve().parent)

        # TODO: update mod_type to support traffic lights as well ("T" --> for now is treated as a trajectory modification)
        scenario_post = generate_modified_scenario_name(scenario_pre, "T")
        self.scenario_parents[scenario_post] = scenario   # parent, for per-modification diff

        # TODO: 1) create new folder 2) create temporary? file 3) start conversion

        # Folder path for the modified scenario
        if Path(full_path).name == "Original":
            folder_path = Path(full_path).parent / "Modified" / scenario_post
        else:
            folder_path = Path(full_path).parent / scenario_post
        os.makedirs(folder_path, exist_ok=True)

        # Temporary file which will be modified used as the input for the SUMO conversion
        temp_file = folder_path / f"{scenario_post}.xml"

        # Adds light to file, but does not show up in frenetix -> we might need the position parameter? Easy fix: use edge point of target lanelet as position
        add_traffic_lights(file_path_str=f"{full_path}/{scenario_pre}.xml", target_path_str=str(temp_file), new_traffic_lights=traffic_lights["traffic_lights"])

        return

    def _execute_batch_simulation(self, batch_action: dict):

        def generate_csv_based_on_ids(ids: list[str]) -> str:
            csv_string = ""
            for _id in ids:
                _id = Path(_id).name.removesuffix(".gif").removesuffix(".png").removesuffix(".xml")
                csv_string += f"{_id}\n"
            with open("scenario_batch_list.csv", "w") as f:
                f.write(csv_string)
            return str(Path("scenario_batch_list.csv").resolve())

        if batch_action is None or "fail" in batch_action.keys():
            return
        # User has chosen a base preset for evaluation
        elif "base" in batch_action.keys():
            scenarios_csv = f"batch_{batch_action['base']}_scenarios.csv"
            scenarios_csv_path = str(Path(f"Frenetix-Motion-Planner/batch_presets/{scenarios_csv}").resolve())
            batch_type = f"Base {batch_action['base']} Scenarios"
        # User has chosen to evaluate based on a step from the query process
        elif "query" in batch_action.keys():
            step = batch_action["query"]
            batch_type = f"Query Step {step}"
            if step == "1":
                scenarios_csv_path = generate_csv_based_on_ids(self.after_location_ids)
            elif step == "2":
                scenarios_csv_path = generate_csv_based_on_ids(self.after_tags_ids)
            elif step == "3":
                scenarios_csv_path = generate_csv_based_on_ids(self.after_road_net_ids)
            elif step == "4":
                scenarios_csv_path = generate_csv_based_on_ids(self.after_obstacles_ids)
            elif step == "5":
                scenarios_csv_path = generate_csv_based_on_ids(self.after_velocity_ids)
            else:
                raise Exception("Query approach, but no step could be matched.")

        else:
            raise Exception("Neither fail, nor base, nor query recognized.")

        # Absolute path to the virtual environment's Python interpreter
        planner_cfg = self.planners[self.selected_planner]
        venv_path = planner_cfg["venv_path"]
        # Path to the main script
        script_path = planner_cfg["script_path"]

        # Check if venv_path is an activate script or python executable
        if venv_path.endswith("activate"):
            command = f"source {venv_path} && python {script_path} --input-file {scenarios_csv_path} --batch"
        else:
            # Direct python executable path (e.g., conda environment)
            command = f"{venv_path} {script_path} --input-file {scenarios_csv_path} --batch"

        # Planner-specific environment (e.g. MP-RBFN needs ground_truth prediction
        # to avoid per-scenario hangs). Inherit the current env and overlay.
        run_env = os.environ.copy()
        run_env.update(planner_cfg.get("env", {}))

        print(f"[batch] planner={self.selected_planner} scenarios={scenarios_csv_path}")
        self.console_log += f"\n[batch] planner={self.selected_planner} scenarios={scenarios_csv_path}\n"

        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=run_env,
            )
            for stdout_line in proc.stdout:
                print(stdout_line, end='')
                self.console_log += stdout_line
            for stderr_line in proc.stderr:
                print(stderr_line, end='')
                self.console_log += stderr_line

            proc.stdout.close()
            proc.stderr.close()

            if proc.wait() != 0:
                raise RuntimeError(f"{self.selected_planner} batch simulation subprocess failed.")
        except Exception as e:
            self.debug_log += f"\nError during batch simulation subprocess: {e}\n"
            return None, 0

        # If weights file does not exist, provide a fictional path
        if not self.planners[self.selected_planner]["weights_file"]:
            weights_file = "non_existent"
        else:
            weights_file = self.planners[self.selected_planner]["weights_file"]
        
        # Generate all batch visualizations
        logs_path = Path(self.planners[self.selected_planner]["log_dir"])
        weights_path = Path(weights_file).resolve()
        
        self.plot = make_combined_bar_plot(logs_path, weights_path)
        self.plot_performance = make_performance_plot(logs_path, weights_path)
        self.plot_success = make_success_rate_plot(logs_path)
        self.plot_cost = make_cost_analysis_plot(logs_path, weights_path)
        self.batch_data_df = make_batch_dataframe(logs_path)
        
        self.batch_simulations += make_batch_summary_string(logs_path, weights_path, batch_type) + "\n"

        return

    def _planner_from_text(self, text: str):
        """Detect a planner named in a free-text batch request and switch to it.

        The batch runner picks the planner from self.selected_planner (normally the
        GUI radio). But users also say e.g. "run batch with MP-RBFN" in chat, which
        previously had no effect (the batch silently used the dropdown's Frenetix).
        Here we honor the chat: if the text names a known planner we set
        self.selected_planner and return its key; otherwise return None (keep the
        current selection). Matching is case-insensitive and alias-aware."""
        t = (text or "").lower()
        # "rbfn" covers mp-rbfn / mprbfn / mp rbfn / rbfn; check it before frenetix.
        if "rbfn" in t and "MP-RBFN" in self.planners:
            self.selected_planner = "MP-RBFN"
            return "MP-RBFN"
        if ("frenetix" in t or "frenetic" in t) and "FRENETIX" in self.planners:
            self.selected_planner = "FRENETIX"
            return "FRENETIX"
        return None

    # This function first uses the LLM to decide which action the user wants to take. Based on the user's input, the LLM will transform the received information into JSON objects (or similar structures), which will call the corresponding method. For now, the only allowed action is the modification.
    def handle_batch_request(self, user_input: str, allow_generic: bool = True):
        """Answer batch-related requests (options QA, run, analysis) WITHOUT a
        selected scenario — batch simulations run on the base-scenario presets,
        so they must work from the search stage too (react_to_request requires
        an open scenario and would raise). Returns the response string, or
        None if the classifier says the request is not batch-related (the
        caller then continues its normal flow).

        allow_generic: also answer generic qa/fail classifications. Set False
        for sticky-context follow-ups, where a non-batch classification should
        fall back to the normal search flow instead of a canned error."""
        action_json, _ = self.llm.find_desired_reaction(user_input)

        if "batch_qa" in action_json.keys():
            batch_options = build_batch_options(loc=self.extracted_location_json, loc_ids=self.after_location_ids, tags=self.extracted_tags, tags_ids=self.after_tags_ids, road=self.extracted_road_net_json, road_ids=self.after_road_net_ids, obs=self.extracted_obstacles_json, obs_ids=self.after_obstacles_ids, vel=self.extracted_velocity_json, vel_ids=self.after_velocity_ids)
            return self.llm.batch_run_qa(user_input, batch_options)

        if "batch_simulation" in action_json.keys():
            named = self._planner_from_text(user_input)
            batch_action, response = self.llm.batch_simulation(user_input)
            if batch_action and "fail" in batch_action.keys():
                response = batch_action["fail"]
            else:
                self._execute_batch_simulation(batch_action=batch_action)
                if named:
                    response = f"▶️ Running batch with **{named}**.\n\n{response or ''}".rstrip()
            return response

        if "batch_analysis" in action_json.keys():
            return self.llm.analyze_batch(user_prompt=user_input, batch_simulations=self.batch_simulations, log_path_str=self.planners[self.selected_planner]["log_dir"] + "/score_overview.csv")

        if allow_generic and "qa" in action_json.keys():
            return action_json["qa"]

        if allow_generic and "fail" in action_json.keys():
            return action_json["fail"]

        return None

    def react_to_request(self, user_input: str, gifs: [str], index: int):

        if len(gifs) == 0 or len(gifs) < index + 1:
            raise ValueError(f"Provided list of GIFs was empty. (Or possibly the index does not match: {len(gifs)} GIFs, index = {index})")

        action_json, raw_response = self.llm.find_desired_reaction(user_input)
        if "fail" in action_json.keys():
            return gifs, index, action_json["fail"]

        if "vehicle_mod" in action_json.keys():
            mod_type = action_json["vehicle_mod"]
            
            # Check if goal modification is requested
            if "G" in mod_type:
                # Handle goal modification first
                gifs, index = self._handle_goal_modification(user_input, gifs[index])
                if gifs is None:
                    return self.gif_displays, index, "Error: Goal modification failed. Please check the target edge/lanelet ID."
                
                # Remove "G" from mod_type to handle remaining modifications
                mod_type = mod_type.replace("G", "")
            
            # Handle vehicle modifications (T, B, P) if any remain
            if mod_type:
                gifs, index = self._handle_modification(user_input, gifs[index], mod_type)

        elif "param_mod" in action_json.keys():
            param_success, output = self._handle_cost_param_change(user_input)
            if param_success:
                return gifs, index, f"The parameters were updated successfully. Here are the new parameters:\n```yaml\n{output}\n```"
            else:
                return gifs, index, output

        elif "param_qa" in action_json.keys():
            cost_file_path = Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml").resolve()
            with open(str(cost_file_path), "r") as f:
                output = f.read()

            answer = self.llm.param_qa(user_input, output)

            return gifs, index, f"{answer}\n```yaml\n{output}\n```"

        elif "param_compare" in action_json.keys():
            return gifs, index, self._compare_cost_weights()

        elif "modify_help" in action_json.keys():
            return gifs, index, self._modify_help(gifs, index)

        elif "scenario_diff" in action_json.keys():
            return gifs, index, self._scenario_diff_message(gifs, index)

        elif "qa" in action_json.keys():
            return gifs, index, action_json["qa"]

        elif "analysis" in action_json.keys():
            analysis = self.llm.analyze_simulation(user_input, self.simulations)
            return gifs, index, analysis

        elif "batch_qa" in action_json.keys():
            batch_options = build_batch_options(loc=self.extracted_location_json, loc_ids=self.after_location_ids, tags=self.extracted_tags, tags_ids=self.after_tags_ids, road=self.extracted_road_net_json, road_ids=self.after_road_net_ids, obs=self.extracted_obstacles_json, obs_ids=self.after_obstacles_ids, vel=self.extracted_velocity_json, vel_ids=self.after_velocity_ids)
            response = self.llm.batch_run_qa(user_input, batch_options)
            return gifs, index, response

        elif "batch_simulation" in action_json.keys():
            named = self._planner_from_text(user_input)
            batch_action, response = self.llm.batch_simulation(user_input)
            # If there's a "fail" key, extract the message from JSON
            if batch_action and "fail" in batch_action.keys():
                response = batch_action["fail"]
            else:
                self._execute_batch_simulation(batch_action=batch_action)
                if named:
                    response = f"▶️ Running batch with **{named}**.\n\n{response or ''}".rstrip()
            return gifs, index, response

        elif "batch_analysis" in action_json.keys():
            response = self.llm.analyze_batch(user_prompt=user_input, batch_simulations=self.batch_simulations, log_path_str=self.planners[self.selected_planner]["log_dir"] + "/score_overview.csv")
            return gifs, index, response

        return gifs, index, ""

    # A function that takes in the list of gifs as well as the index to determine the selected candidate and start a simulation with the Frenetix motion planner
    # Updates the list for the planner output; returns this list and the index pointing towards the latest element
    # TODO: define where and how the resulting simulation GIF + logs will be stored; for now they end up in the default folder of the Frenetix module
    def run_with_frenetix(self, gifs: [str], index: int):
        if len(gifs) == 0 or len(gifs) < index + 1:
            raise ValueError(f"Provided list of GIFs was empty. (Or possibly the index does not match: {len(gifs)} GIFs, index = {index})")

        scenario_gif_path = gifs[index]
        scenario = Path(scenario_gif_path).name.removesuffix(".gif").removesuffix("_without_ego") # Consider adding .removesuffix(".xml")
        scenario_xml_path = str(Path(scenario_gif_path).parent) + "/" + scenario + ".xml"

        self.debug_log += "\n\nFRENETIX SIMULATION STARTED\n"
        self.debug_log += f"\nThe provided GIF was: {scenario_gif_path}\n"
        self.debug_log += f"\nThe extracted scenario is: {scenario_xml_path}\n"

        # Absolute path to the virtual environment's Python interpreter
        venv_python = Path("Frenetix-Motion-Planner/venv/bin/activate").resolve()

        # Path to the Frenetix script
        script_path = Path("Frenetix-Motion-Planner/main.py").resolve()

        # Shell command: activate venv, then run script
        command = f"source {venv_python} && python {script_path} --input-file {str(scenario_xml_path)}"

        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Read output line by line as it arrives
        for stdout_line in proc.stdout:
            print(stdout_line, end='')  # print live to console
            self.console_log += stdout_line  # append to debug_log live

        # Also read stderr line by line (optional, or you can merge it into stdout)
        for stderr_line in proc.stderr:
            print(stderr_line, end='')  # print live to console
            self.console_log += stderr_line

        proc.stdout.close()
        proc.stderr.close()

        return_code = proc.wait()

        final_output = "\n".join(self.console_log.strip().splitlines()[-12:])

        if return_code != 0:
            raise RuntimeError("Frenetix simulation subprocess failed.")

        self.debug_log += "\nFRENETIX SIMULATION FINISHED\n"

        # TODO: adjust this, once a proper storage/naming scheme has been implemented
        log_dir = Path("Frenetix-Motion-Planner/logs").resolve() / scenario

        cost_dir = log_dir / "cost"
        cost = get_min_cost(str(cost_dir))

        weight_file_path = Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml").resolve()
        with open(str(weight_file_path), "r") as f:
            weights = f.read()

        # TODO: add modification field to format_analysis
        if index < len(self.modifications):
            modification = self.modifications[index]
        else:
            modification = self.modifications[0]
        self.simulations += format_analysis(scenario_name=scenario, cost=cost, weights=weights, final_output=final_output)

        gif_files = list(log_dir.glob("*.gif"))

        if not gif_files:
            # No GIF means the planner plotted no frames — almost always because
            # it found no kinematically feasible trajectory (see the planner log
            # for "No Kinematic Feasible and Optimal Trajectory Available!").
            raise RuntimeError(
                f"The planner produced no animation for '{scenario}': it found "
                f"no kinematically feasible trajectory for this scenario "
                f"(planner log: {log_dir}). Try a different scenario.")

        gif_path = str(gif_files[0])

        self.planner_displays.append(gif_path)

        return self.planner_displays, len(self.planner_displays) - 1

    def run_with_planner(self, gifs: [str], index: int):
        if len(gifs) == 0 or len(gifs) < index + 1:
            raise ValueError(
                f"Provided list of GIFs was empty. (Or possibly the index does not match: {len(gifs)} GIFs, index = {index})")
        scenario_gif_path = gifs[index]
        scenario = Path(scenario_gif_path).name.removesuffix(".gif").removesuffix(
            "_without_ego")  # Consider adding .removesuffix(".xml")
        scenario_xml_path = str(Path(scenario_gif_path).parent) + "/" + scenario + ".xml"

        self.debug_log += "\n\nMOTION PLANNER SIMULATION STARTED\n"
        self.debug_log += f"\nThe provided GIF was: {scenario_gif_path}\n"
        self.debug_log += f"\nThe extracted scenario is: {scenario_xml_path}\n"

        # Absolute path to the virtual environment's Python interpreter
        planner_cfg = self.planners[self.selected_planner]
        venv_path = planner_cfg["venv_path"]
        # Path to the main script
        script_path = planner_cfg["script_path"]

        # Check if venv_path is an activate script or python executable
        if venv_path.endswith("activate"):
            # Shell command: activate venv, then run script
            command = f"source {venv_path} && python {script_path} --input-file {str(scenario_xml_path)}"
        else:
            # Direct python executable path (e.g., conda environment)
            command = f"{venv_path} {script_path} --input-file {str(scenario_xml_path)}"

        # Planner-specific environment (MP-RBFN needs ground_truth prediction to
        # avoid hanging; see the planners config). Same treatment as batch runs.
        run_env = os.environ.copy()
        run_env.update(planner_cfg.get("env", {}))

        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            env=run_env,
        )

        # Read output line by line as it arrives
        for stdout_line in proc.stdout:
            print(stdout_line, end='')  # print live to console
            self.console_log += stdout_line  # append to debug_log live

        # Also read stderr line by line (optional, or you can merge it into stdout)
        for stderr_line in proc.stderr:
            print(stderr_line, end='')  # print live to console
            self.console_log += stderr_line

        proc.stdout.close()
        proc.stderr.close()

        return_code = proc.wait()

        final_output = "\n".join(self.console_log.strip().splitlines()[-12:])

        if return_code != 0:
            raise RuntimeError("Frenetix simulation subprocess failed.")

        self.debug_log += "\nFRENETIX SIMULATION FINISHED\n"

        log_dir = Path(self.planners[self.selected_planner]["log_dir"]) / scenario

        cost = -1.0
        weights = "No weights supported."
        if self.planners[self.selected_planner]["cost_support"] == "y":
            cost_dir = log_dir / "cost"
            cost = get_min_cost(str(cost_dir))
        if self.planners[self.selected_planner]["weights_file"]:
            with open(self.planners[self.selected_planner]["weights_file"], "r") as f:
                weights = f.read()

        # TODO: add simulator field for cross-planner comparison to analysis part
        if index < len(self.modifications):
            modification = self.modifications[index]
        else:
            modification = self.modifications[0]
        self.simulations += format_analysis(scenario_name=scenario, cost=cost, weights=weights,
                                            final_output=final_output, mod_info=modification)

        # TODO: this requires a structure
        gif_files = list(log_dir.glob("*.gif"))

        if not gif_files:
            # No GIF means the planner plotted no frames — almost always because
            # it found no kinematically feasible trajectory (see the planner log
            # for "No Kinematic Feasible and Optimal Trajectory Available!").
            raise RuntimeError(
                f"The planner produced no animation for '{scenario}': it found "
                f"no kinematically feasible trajectory for this scenario "
                f"(planner log: {log_dir}). Try a different scenario.")

        gif_path = str(gif_files[0])

        self.planner_displays.append(gif_path)

        return self.planner_displays, len(self.planner_displays) - 1

    # Loads the JSON file containing dimensions
    # Assumed to be in the same folder as the image/GIF
    def load_plot_dimensions_json(self, image_path:str) -> dict:
        folder_path = Path(image_path).parent
        file_path = folder_path / f"{Path(image_path).stem}.json"
        with open(file_path) as f:
            metadata = json.load(f)
        return metadata

    def set_planner(self, planner: str):
        self.selected_planner = planner
    
    def find_scenario_by_name(self, scenario_name: str):
        """
        Find a scenario by its name and return the path to its PNG visualization.
        Searches in the Scenarios directory for the scenario.
        
        Args:
            scenario_name: The scenario name (e.g., 'DEU_Weimar-71_1_T-4')
            
        Returns:
            str: Path to the scenario's PNG file if found, None otherwise
        """
        import os, re
        from pathlib import Path

        scenarios_dir = Path("Scenarios")

        if not scenarios_dir.exists():
            print(f"   ❌ Scenarios directory not found: {scenarios_dir}")
            return None

        raw = scenario_name.strip().replace('.xml', '')
        # Pull a CommonRoad scenario-ID token out of the message, so the user can
        # wrap it in extra text — e.g. "DEU_Weimar-71_1_T-4, let's use this one".
        m = re.search(r"[A-Za-z]{3}_[A-Za-z]+(?:-\d+)+_\d+_[A-Za-z]+-\d+", raw)
        candidates = [c for c in (m.group(0) if m else None, raw) if c]

        scenario_dirs = [d for d in scenarios_dir.iterdir() if d.is_dir()]

        def _png(d):
            p = d / "Original" / f"{d.name}.png"
            return str(p) if p.exists() else None

        # 1) Case-insensitive exact directory-name match on any candidate.
        for cand in candidates:
            cl = cand.lower()
            for d in scenario_dirs:
                if d.name.lower() == cl and _png(d):
                    print(f"   ✅ Found exact match: {_png(d)}")
                    return _png(d)

        # 2) Substring match in EITHER direction (input contains the id, or the
        #    id contains the input) — preferring the longest (most specific) dir.
        print(f"   🔍 Searching for partial match of: {candidates[0]}")
        best = None
        for cand in candidates:
            cl = cand.lower()
            if not cl:
                continue
            for d in scenario_dirs:
                dn = d.name.lower()
                if (cl in dn or dn in cl) and _png(d):
                    if best is None or len(d.name) > len(best[1]):
                        best = (_png(d), d.name)
        if best:
            print(f"   ✅ Found partial match: {best[0]}")
            return best[0]

        print(f"   ❌ No scenario found matching: {candidates[0]}")
        return None

    # Restart everything in order but keep the previous log
    def restart(self):
        # CRITICAL FIX: Stop old thread before creating a new one to prevent race conditions
        # Without this, multiple threads run simultaneously causing unpredictable behavior
        if hasattr(self, 'running') and self.running:
            print("   🛑 Stopping old thread...")
            self.running = False  # Signal old thread to stop
            
            # Wait for old thread to finish (with timeout to prevent hanging)
            if hasattr(self, 'thread') and self.thread.is_alive():
                self.thread.join(timeout=2.0)
                if self.thread.is_alive():
                    print("   ⚠️  Old thread did not stop cleanly, but continuing...")
                else:
                    print("   ✅ Old thread stopped successfully")
        
        # self.end_and_wait()
        self.query_counter += 1
        self.debug_log += f"\nSuccessfully reset the DB search engine. This is query number {self.query_counter} of this session.\n###################"

        self.db = ScenarioDBWrapper("chroma")
        self.llm = None

        self.extracted_location_json = {}
        self.after_location_ids = []
        self.location_unavailable = None  # set when an explicit location has 0 DB matches
        self.extracted_tags = []
        self.after_tags_ids = []
        self.after_tags_docs = []
        self.extracted_road_net_json = {}
        self.after_road_net_ids = []
        self.after_road_net_docs = []
        self.extracted_obstacles_json = {}
        self.after_obstacles_ids = []
        self.after_obstacles_docs = []
        self.extracted_velocity_json = {}
        self.after_velocity_ids = []
        self.after_velocity_docs = []

        # Uses the literal console output for closer analysis and to not clutter the debug log
        self.console_log = ""

        self.current_step = 0
        self.current_results = []

        self.result = ""
        self.file_path = ""

        # The GIFs corresponding to the selected scenario
        self.gif_displays = []

        # Maps a modified scenario name -> the scenario it was derived from
        # (its immediate parent), so "what changed?" can diff per-modification.
        self.scenario_parents = {}

        # The GIFs of the scenarios run through the motion planner
        self.planner_displays = []

        # Create NEW thread with fresh queue and event
        self.input_queue = queue.Queue()
        self.result_ready = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("   ✅ New thread started")

# engine.run_with_planner(["<repo>/Scenarios/GRC_NeaSmyrni-98_1_T-9/Original/GRC_NeaSmyrni-98_1_T-9.gif"], 0)
