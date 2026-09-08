import re
import time
from pathlib import Path
import gradio as gr

from process_engine import ProcessEngine
from config import SessionConfig, DATA_OSM_ROOT
from interface_generate import build_generate_tab, GIF_VIEWER_URL

import os
from dotenv import load_dotenv

load_dotenv()
DEFAULT_API_KEY = os.getenv("DEFAULT_API_KEY", "").strip()
DEFAULT_API_MODEL = os.getenv("DEFAULT_API_MODEL", "").strip()

DEFAULT_OLLAMA_URL = os.getenv("DEFAULT_OLLAMA_URL", "").strip()
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "").strip()

DEFAULT_MODE = os.getenv("DEFAULT_MODE", "").strip()

# Qwen (OpenAI-compatible token-plan MaaS endpoint)
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "").strip()
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "").strip()
QWEN_MODEL = os.getenv("QWEN_MODEL", "").strip()

global_configs = {
    "key": DEFAULT_API_KEY if DEFAULT_API_KEY else None,
    "model": DEFAULT_API_MODEL if DEFAULT_API_MODEL else None,
    "ollama_url": DEFAULT_OLLAMA_URL if DEFAULT_OLLAMA_URL else None,
    "ollama_model": DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL else None,
    "qwen_key": QWEN_API_KEY if QWEN_API_KEY else None,
    "qwen_base_url": QWEN_BASE_URL if QWEN_BASE_URL else None,
    "qwen_model": QWEN_MODEL if QWEN_MODEL else None,
    "mode": DEFAULT_MODE if DEFAULT_MODE else None
}

class ChatHandler:
    def __init__(self):
        self.engine = ProcessEngine()
        self.step = 1
        # True after a message was answered by the batch detour: follow-ups
        # ("run the 100 scenario simulation", "yes start it") stay in batch
        # mode until a message is classified as not batch-related.
        self.batch_context = False

    def chat(self, user_input:str, image_list, index):
        response = ""
        gifs = []

        self.engine.set_llm(llm_parameters=global_configs)

        # Batch requests (options QA / run / analysis) do not need a selected
        # scenario — answer them directly at ANY stage instead of forcing the
        # input through the scenario-search steps (which would reply with
        # "please describe the scenario location ...").
        # Trigger on: the word "batch", a batch-size phrasing ("100 scenario
        # simulation"), or a follow-up while in batch context ("yes run it").
        lowered = (user_input or "").lower()
        mentions_batch = "batch" in lowered
        mentions_size = re.search(r"\b\d+\s*scenario", lowered) is not None
        if (mentions_batch or mentions_size or self.batch_context) and self.step != 8:
            try:
                # qa/fail answers are only trusted when the message is explicitly
                # batch-shaped; in sticky-context mode a non-batch classification
                # returns None and the normal search flow continues.
                batch_response = self.engine.handle_batch_request(
                    user_input, allow_generic=(mentions_batch or mentions_size))
            except Exception as e:
                print(f"⚠️  Batch detour failed, continuing with normal flow: {e}")
                batch_response = None
            if batch_response:
                self.batch_context = True
                images, image_step = self.engine.get_images()
                # Keep self.step unchanged so the search/selection continues where it was
                return batch_response, gifs, index, self.step, images, image_step
            self.batch_context = False

        # Handle input for processing steps (1-5)
        # Step 6+ are for scenario selection and modification
        if self.step <= 5:
            self.engine.handle_input(self.step, user_input)

        if self.step == 1: # TODO: immediate fail if location not present in DB
            # NEW: Check if user entered a scenario name directly at the beginning
            if user_input and user_input.strip():
                # Try to detect if this looks like a scenario name (has typical CommonRoad naming pattern)
                scenario_input = user_input.strip().replace('.xml', '')
                
                # Check if it looks like a scenario name (contains country code or typical pattern)
                # Typical format: DEU_CityName-##_#_T-# or similar
                is_scenario_name = '_' in scenario_input and ('-' in scenario_input or scenario_input.count('_') >= 2)
                
                if is_scenario_name:
                    # Try to find this scenario
                    found_scenario = self.engine.find_scenario_by_name(scenario_input)
                    
                    if found_scenario:
                        print(f"✅ Direct scenario selection at start: {scenario_input}")
                        
                        # Load the scenario directly and jump to step 7
                        gifs = self.engine.select_initial_gif(found_scenario)
                        
                        if gifs is not None:
                            # Check repair module status
                            repair_status = SessionConfig.get_use_repair_module()
                            repair_info = (
                                f"\n\nℹ️  **Repair module:** {'ENABLED ✓' if repair_status else 'DISABLED ⚠️'}\n"
                                f"   ({'Broken trajectories will be fixed automatically during modifications' if repair_status else 'Restart app and select y to enable repair for modifications'})"
                            )
                            
                            response = (
                                f"✅ **Scenario found: `{scenario_input}`**\n\n"
                                f"Loaded scenario directly. You can now:\n"
                                f"   • **Modify** the scenario\n"
                                f"   • **Run with Motion Planner**\n"
                                f"   • Press **Restart** to search for a different scenario"
                                f"{repair_info}"
                            )
                            self.step = 7  # Jump to modification/planner step
                            index = 0
                            images, image_step = self.engine.get_images()
                            return response, gifs, index, self.step, images, image_step
                        else:
                            response = (
                                f"❌ **Could not load scenario:** `{scenario_input}`\n\n"
                                f"The scenario was found but could not be loaded.\n"
                                f"Please try:\n"
                                f"   • A different scenario name\n"
                                f"   • Or describe your requirements to search"
                            )
                            self.step = 0  # Reset to start
                            images, image_step = self.engine.get_images()
                            return response, [], index, self.step + 1, images, image_step
                    # If not found, continue with normal query (treat as location input)
                    print(f"   ℹ️  '{scenario_input}' not found as direct scenario, treating as query input")

            response = "Please provide a general description of the scenario and its content (e.g., highway situations, merging lanes, traffic jams, …)"

        elif self.step == 2:
            response = "Great! Now, please give a more detailed description of the road on which the ego is driving (traffic signs, traffic lights)" # Missing more complex information like geometry/intersections

        elif self.step == 3:
            response = "Please describe your desired obstacles next. Obstacles include both static (parked cars, construction zones, ...) and dynamic ones (cars, trucks, pedestrians, ...)"

        elif self.step == 4:
            response = "You now have the option of specifying an initial velocity for the ego vehicle."

        elif self.step == 5:
            top_results, file_path = self.engine.end_and_wait()
            
            if len(top_results) > 0:
                # Format top 5 scenarios for display
                top_5 = top_results[0:5]
                scenario_list = "\n".join([f"   {i+1}. `{s.replace('.xml', '')}`" for i, s in enumerate(top_5)])
                
                response = (
                    f"**Search Complete!**\n\n"
                    f"Found **{len(top_results)} matching scenario{'s' if len(top_results) > 1 else ''}**\n\n"
                    f"**Top {min(5, len(top_results))} match{'es' if min(5, len(top_results)) > 1 else ''}:**\n"
                    f"{scenario_list}\n\n"
                    f"**Next steps:**\n"
                    f"   • **Type scenario name** (e.g., `DEU_Weimar-71_1_T-4`) for direct selection ⚡\n"
                    f"   • **Or** browse with ◀ Previous / Next ▶ buttons and press **Enter**"
                )
                self.step = 6 # Increase step, so that at the end step 7 is returned -> Enter modification
            elif self.engine.location_unavailable:
                # The requested location has zero scenarios in the DB — say so
                # clearly (with available countries) instead of the generic note.
                response = self.engine.location_unavailable
                # Force a fresh search (same as the generic no-match path).
            else:
                response = (
                    f"**No matching scenarios found**\n\n"
                    f"**Try adjusting your search:**\n"
                    f"   • Use more general descriptions\n"
                    f"   • Try different locations or road types\n"
                    f"   • Reduce specific requirements\n\n"
                    f"Press **Restart** to start a new search"
                )
                # Returns step 6 at the end, so that user is forced to start a new search

        elif self.step == 6:
            # Note: Step 6 is typically skipped (step 5 sets step=6, then increment makes it 7)
            # This is here for edge cases
            self.step = 5
            response = "Press the restart button to query another scenario."

        # At step 7, the logic changes. After a scenario has been found, the GIF should be displayed and the modification should be entered. This should update the image_list to only contain the GIF (the list is updated in the process engine class).
        elif self.step == 7:
            # Check if we already have GIFs loaded (scenario already selected)
            # If so, skip scenario name matching and treat input as modification request
            has_gifs_loaded = len(image_list) > 0 and any(img.endswith('.gif') for img in image_list)
            
            # NEW: Check if user input is a scenario name for direct selection
            # BUT only if we don't have GIFs loaded yet (still at search results stage)
            if user_input and user_input.strip() and not has_gifs_loaded:
                # Get current image list (search results)
                current_images, _ = self.engine.get_images()
                
                # Only do scenario name matching if we have .png search results
                has_search_results = len(current_images) > 0 and any(img.endswith('.png') for img in current_images)
                
                if has_search_results:
                    print(f"🔍 Direct selection attempt: '{user_input.strip()}'")
                    print(f"   Available scenarios: {len(current_images)}")
                    
                    # Clean user input (remove .xml suffix if present)
                    user_scenario = user_input.strip().replace('.xml', '')
                    
                    # Search for matching scenario in the results
                    matched_index = -1
                    for i, img_path in enumerate(current_images):
                        scenario_name = img_path.split("/")[-1].replace(".png", "")
                        print(f"   Comparing with [{i}]: {scenario_name}")
                        # Support partial matching (case-insensitive)
                        if user_scenario.lower() in scenario_name.lower() or scenario_name.lower() in user_scenario.lower():
                            matched_index = i
                            print(f"   ✅ MATCH FOUND at index {i}: {scenario_name}")
                            break
                    
                    if matched_index >= 0:
                        # Found a matching scenario! Select it
                        print(f"📍 Direct scenario selection: index={matched_index}, path={current_images[matched_index]}")
                        index = matched_index
                        image_list = current_images
                        print(f"   Calling select_initial_gif with: {image_list[index]}")
                        # Continue to regular selection logic below
                    else:
                        # No match found - show available scenarios
                        scenario_names = [img.split("/")[-1].replace(".png", "") for img in current_images]
                        scenario_list = "\n".join([f"   {i+1}. `{name}`" for i, name in enumerate(scenario_names[:10])])
                        response = (
                            f"❌ **Scenario not found:** `{user_scenario}`\n\n"
                            f"**Available scenarios:**\n{scenario_list}\n\n"
                            f"💡 **Tip:** Type the full scenario name or use ◀ Previous / Next ▶ buttons to browse, then press **Enter**."
                        )
                        self.step = 6  # Stay at selection step
                        images, image_step = self.engine.get_images()
                        return response, [], index, self.step + 1, images, image_step
            
            # If GIFs are already loaded and user typed something, treat as modification request
            if has_gifs_loaded and user_input and user_input.strip():
                print(f"🔧 User has GIF loaded, treating '{user_input.strip()}' as modification request")
                # Advance to step 8 for modification handling
                self.step = 7  # Will be incremented to 8 at the end
            
            # Add repair module status info
            repair_status = SessionConfig.get_use_repair_module()
            repair_note = f" (Repair: {'✓ enabled' if repair_status else '⚠️ disabled'})"
            response = f"This is the GIF of the chosen scenario. Let me know if you wish to modify it{repair_note}, or start running the motion planner."

            print("image_list", image_list)
            print("index", index)

            # If the user has not yet looked at a single image, two options are possible: 1) No results were found (should be caught and avoided before step 7 is ever reached) 2) The corresponding GIF was not found
            # 3) The user has simply decided not to look at any images --> perform an extra check here by loading images for the final result and choosing the first one
            if len(image_list) == 0:
                image_list = self.engine.get_images()[0] # get_images() returns a tuple of images, step
                if len(image_list) == 0:
                    raise ValueError("There are no images available.")
                index = 0

            gifs = self.engine.select_initial_gif(image_list[index])

            if gifs is None:
                response = "Sorry, I could not create a GIF for this scenario. Please press the restart button to query another scenario."
                self.step = 6
                images, image_step = self.engine.get_images()

                return response, [], index, 5, images, image_step

            index = 0

        # In this step, we get the input from the user, for what he wants to do. Depending on this, different methods might be called (modification, motion planner execution, etc.)
        # TODO: the logic here changes - we can no longer simply use pre-written dialogue, because LLM output matters --> need to change logic so that from this point onwards, the responses are created by the process engine class
        elif self.step == 8:
            # Supply images and index so that processor knows which one desired
            gifs, index, string_output = self.engine.react_to_request(user_input, image_list, index)
            images, image_step = self.engine.get_images()

            if gifs is None:
                response = "Sorry, something went wrong in the modification process. Please press the restart button to query another scenario."
                self.step = 6

                return response, [], index, 5, images, image_step

            # If relevant LLM output has been generated during the process, it will be displayed instead of a scripted answer
            # Unlike the failure from above, we stay in the modification phase
            if len(string_output) > 0:
                response = string_output
                self.step = 8
                return response, gifs, index, self.step, images, image_step

            # Reset step, so that we remain in a loop for step 8
            self.step = 7

            print("\n\nGIFs of the chosen scenario: ", gifs)

            response = "Here is your modification. Would you like to proceed?"

        self.step += 1

        images, image_step = self.engine.get_images()

        return response, gifs, index, self.step, images, image_step


handler = ChatHandler()

planner_options = list(handler.engine.planners.keys())

def chatbot_stream(user_input: str, history, image_list, index):
    response, gifs, index, step, images, image_step = handler.chat(user_input, image_list, index)
    if not history:
        history = []
    history.append({"role": "user", "content": user_input})

    # Placeholder
    assistant_message = {"role": "assistant", "content": ""}
    history.append(assistant_message)

    # This means that we are not yet in the second phase (no GIFs are loaded in) -> we just update images to the latest ones
    if len(gifs) == 0 and len(images) > 0:
        # Simulate streaming response
        for partial_text in simulated_streaming_response(response):
            assistant_message["content"] = partial_text

            # Why are we returning history twice? --> easiest fix for our format, which allows us to start the chat with a bot message

            # The index should not jump around, if the selected image is not removed from the candidate list between two steps
            # If it is removed, the index should be set to 0 to guarantee being in bounds
            if index >= len(image_list):
                index = 0

            if len(image_list) > 0:
                current_image = image_list[index]
                filename = current_image.split("/")[-1].removesuffix(".png")
                label = f"**{filename}**  {index + 1}/{len(image_list)} candidates after step {image_step}"
            else:
                label = ""

            if image_step == 1:
                selected = "1. Location"
            elif image_step == 2:
                selected = "2. Tags"
            elif image_step == 3:
                selected = "3. Road Network"
            elif image_step == 4:
                selected = "4. Obstacles"
            else:
                selected = "5. Velocity"

            yield history, history, gr.update(value=""), gr.update(value=images[index]), images, index, gr.update(value=label, visible=True), gr.update(value=selected), image_step, gr.update(), gr.update()

    elif len(gifs) == 0 and len(images) == 0:
        # Simulate streaming response
        for partial_text in simulated_streaming_response(response):
            assistant_message["content"] = partial_text

            # Why are we returning history twice? --> easiest fix for our format, which allows us to start the chat with a bot message

            yield history, history, gr.update(value=""), gr.update(), images, 0, gr.update(), gr.update(), step - 1, gr.update(), gr.update()

    else:
        # Simulate streaming response
        for partial_text in simulated_streaming_response(response):
            assistant_message["content"] = partial_text

            filename = gifs[index].split("/")[-1].removesuffix(".gif")
            caption = f"**{filename}**  {index + 1}/{len(gifs)} plots"

            yield history, history, gr.update(value=""), gr.update(value=gifs[index]), gifs, index, caption, gr.update(visible=False), step, gr.update(visible=False), gr.update(value="Run with Motion Planner", visible=True)

# output format of corr. gradio function: [history, chatbot, user_input, image_output, image_list_state, index_state, image_caption, step_selector, selected_step_state, image_update, run_frenetix]


def simulated_streaming_response(full_text, delay=0.02):
    current_text = ""
    for char in full_text:
        current_text += char
        yield current_text
        time.sleep(delay)

def get_debug_info():
    return handler.engine.get_debug_log()

# Currently it is a little unclean with the order in which the restart happens and debug information is displayed
def restart():
    handler.step = 1
    handler.batch_context = False
    handler.engine.restart()

    history = [{"role": "assistant", "content": "Hi, welcome to the scenario generation. To start the process, please provide me with a location for the scenario, or let me know if you do not have any specific place in mind.\n\n**💡 Quick Tip:** If you already know a specific scenario name (e.g., `DEU_Weimar-71_1_T-4`), you can type it directly to skip the search and go straight to modification/motion planner."}]

    return history, history, gr.update(value=None, visible=True), gr.update(value=""), [], 0, gr.update(value=""), gr.update(choices=["1. Location", "2. Tags", "3. Road Network", "4. Obstacles", "5. Velocity"], value=None, visible=True), 5, gr.update(value=None, visible=False), gr.update(value=None, visible=False), gr.update(value=None, visible=False)
# Output order: [history, chatbot, image_output, user_input, image_list_state, index_state, image_caption, step_selector, selected_step_state, image_update, run_frenetix, planner_output]

# Loads in the newest batch of images
# Returns the new list of file paths, resets the index to 0 and displays the first of the new images
def load_images():

    images, step = handler.engine.get_images()

    if len(images) == 0:
        return images, 0, gr.update(value=None), gr.update(value=None), step, gr.update(value=None)

    filename = images[0].split("/")[-1].removesuffix(".png")
    label = f"**{filename}**  1/{len(images)} candidates after step {step}"

    return images, 0, gr.update(value=images[0]), gr.update(value=label), step, gr.update(value=None,choices=["1. Location", "2. Tags", "3. Road Network", "4. Obstacles", "5. Velocity"])

# Returns the updated index, as well as the  file path of the next image to be displayed
def show_next_image(images, index, step):
    number_images = len(images)

    if number_images == 0:
        return index, None, gr.update(value=None)

    index = (index + 1) % number_images
    current_image = images[index]

    # This means we still are in the query mode
    if step < 7:
        # Extract file name
        filename = current_image.split("/")[-1].removesuffix(".png")
        label = f"**{filename}**  {index + 1}/{number_images} candidates after step {step}"
    # Else we are in the GIF mode and no longer should consider candidates & steps
    else:
        filename = current_image.split("/")[-1].removesuffix(".gif")
        label = f"**{filename}**  {index + 1}/{number_images} plots"

    return index, current_image, gr.update(value=label, visible=True)

def show_previous_image(images, index, step):
    number_images = len(images)

    if number_images == 0:
        return index, None, gr.update(value=None)

    index = index - 1 if index > 0 else number_images - 1
    current_image = images[index]

    if step < 7:
        filename = current_image.split("/")[-1].removesuffix(".png")
        label = f"**{filename}**  {index + 1}/{number_images} candidates after step {step}"
    else:
        filename = current_image.split("/")[-1].removesuffix(".gif")
        label = f"**{filename}**  {index + 1}/{number_images} plots"

    return index, current_image, gr.update(value=label, visible=True)

# TODO: there is some bug with the step selector and the input from above
def select_step(selected_step):
    if selected_step == "1. Location":
        step = 1
    elif selected_step == "2. Tags":
        step = 2
    elif selected_step == "3. Road Network":
        step = 3
    elif selected_step == "4. Obstacles":
        step = 4
    elif selected_step == "5. Velocity":
        step = 5
    else:
        step = 7
    images, step = handler.engine.get_selected_images(step)

    print(images)

    if len(images) == 0:
        return images, 0, gr.update(value=None), gr.update(value=None), step

    filename = images[0].split("/")[-1].removesuffix(".png")
    label = f"**{filename}**  1/{len(images)} candidates after step {step}"

    return images, 0, gr.update(value=images[0]), gr.update(value=label), step

def run_frenetix_planner(gifs, index):
    try:
        planner_gifs, planner_index = handler.engine.run_with_planner(gifs, index)
    except Exception as e:
        # e.g. the planner found no feasible trajectory -> no GIF. Surface it as
        # a toast instead of crashing the event handler.
        gr.Warning(f"⚠️ {e}")
        return gr.update(visible=False)
    if not planner_gifs:
        gr.Warning("⚠️ The planner produced no output for this scenario.")
        return gr.update(visible=False)
    return gr.update(value=planner_gifs[planner_index], visible=True)
# Return order: [planner_output]

# Method that takes in clicks on images and processes them
# Used to map a goal area chosen by a user to the coordinates of the CR plot
def image_click(image_list, index, event: gr.SelectData):
    px, py = event.index  # pixel click

    # Helper function that transforms a clicked pixel into the dimensions from the CR plot
    def pixel_to_world(px, py, plot_limits, img_width, img_height):
        x_min, x_max, y_min, y_max = plot_limits
        x_world = x_min + px / img_width * (x_max - x_min)
        y_world = y_max - py / img_height * (y_max - y_min)
        return x_world, y_world

    # Load in the information from the JSON object storing the specific dimensions of a graph
    metadata = handler.engine.load_plot_dimensions_json(image_list[index])

    # The mathematical formula for this is:
    # x = x_min + px / img_width * (x_max - x_min)
    # y = y_max - py / img_height * (y_max - y_min)
    img_w, img_h = int(metadata["dpi"] * metadata["figsize"][0]), int(metadata["dpi"] * metadata["figsize"][1])
    coords = pixel_to_world(px, py, metadata["plot_limits"], img_w, img_h)

    print(f"\n SELECTED COORDS IN CR DIMENSIONS: x={coords[0]}, y={coords[1]}\n\n")

# Allows the user to activate the goal-region choice mode
def toggle_interactivity(current_state):
    return gr.update(interactive=not current_state), not current_state

def poll_console_output():
    return handler.engine.console_log

def poll_plot():
    return handler.engine.plot

def poll_plot_performance():
    return handler.engine.plot_performance

def poll_plot_success():
    return handler.engine.plot_success

def poll_plot_cost():
    return handler.engine.plot_cost

def poll_batch_dataframe():
    return handler.engine.batch_data_df

# Sets the motion planner from the engine to the planner selected in the Radio
def select_motion_planner(planner: str):
    handler.engine.set_planner(planner)

def enlarge_image(images, index):
    """Open the enlarged (zoom/pan) view. The viewer iframe mirrors the current
    image_output image live, so we only need to toggle the modal here."""
    if not images or len(images) == 0:
        return gr.update(visible=False)
    return gr.update(visible=True)

def close_enlarged_view():
    """Close the enlarged image view"""
    return gr.update(visible=False)

def open_enlarged_view():
    """Open an enlarged-view modal. The viewer iframe mirrors its source image
    live, so we only toggle the modal visible here."""
    return gr.update(visible=True)

def _fields_for_mode(mode):
    """Single source of truth for the Setup tab's two text fields per provider.
    Returns (cred_value, cred_label, cred_placeholder, model_value, model_label,
    model_placeholder). The first field doubles as API key (Commercial/Qwen) or
    URL (Ollama)."""
    if mode == "Commercial":
        return (DEFAULT_API_KEY, "API Key (Commercial)", "Enter your API key",
                DEFAULT_API_MODEL, "Model Name", "e.g., gemini-2.5-flash")
    if mode == "Qwen":
        return (QWEN_API_KEY, "API Key (Qwen)", "Enter your Qwen API key",
                QWEN_MODEL, "Qwen Model Name", "e.g., qwen3.6-flash")
    # Ollama
    return (DEFAULT_OLLAMA_URL, "Ollama URL", "e.g., http://localhost:11434",
            DEFAULT_OLLAMA_MODEL, "Ollama Model Name", "e.g., qwen2.5:14b")

def update_fields_for_mode(mode):
    """Update input fields based on selected mode (Commercial/Ollama/Qwen)"""
    cred_v, cred_l, cred_p, model_v, model_l, model_p = _fields_for_mode(mode)
    return (
        gr.update(value=cred_v, label=cred_l, placeholder=cred_p),
        gr.update(value=model_v, label=model_l, placeholder=model_p),
    )

def save_settings(api_or_url, model_name, mode_choice):
    # Mode: always take what the user selected (Commercial/Ollama/Qwen)
    global_configs["mode"] = mode_choice if mode_choice else global_configs.get("mode")

    # Qwen mode: api_or_url is the Qwen API key; base_url stays fixed from .env
    if global_configs["mode"] == "Qwen":
        if api_or_url.strip():
            global_configs["qwen_key"] = api_or_url.strip()
        elif not global_configs.get("qwen_key"):
            return (
                "❌ No Qwen API key found. Please enter one.",
                gr.update(selected=0),     # Stay on Setup tab
                gr.update(visible=False),
                gr.update(visible=False),
            )
        if model_name.strip():
            global_configs["qwen_model"] = model_name.strip()
        return (
            f"✅ Qwen mode set! Model: {global_configs['qwen_model'] or '(default)'} | Endpoint: {global_configs['qwen_base_url']}\n\n🎉 Setup complete! Switching to Main App...",
            gr.update(selected=1),     # Switch to Main App tab (id=1)
            gr.update(visible=True),
            gr.update(visible=True),
        )

    # API key / URL: prefer user input, otherwise keep existing (default from .env)
    if api_or_url.strip():
        global_configs["key"] = api_or_url.strip()
    elif not global_configs.get("key"):
        # no user input and no default from .env
        return (
            "❌ No API key or URL found. Please enter one.",
            gr.update(selected=0),     # Stay on Setup tab
            gr.update(visible=False),  # Don't show chatbot
            gr.update(visible=False),  # Don't show user input
        )

    # Model: prefer user input, otherwise keep existing (default from .env)
    if model_name.strip():
        global_configs["model"] = model_name.strip()

    # If we're in Ollama mode, map correctly
    if global_configs["mode"] == "Ollama":
        if api_or_url.strip():
            global_configs["ollama_url"] = api_or_url.strip()
        elif not global_configs.get("ollama_url"):
            return (
                "❌ No Ollama URL found. Please enter one.",
                gr.update(selected=0),     # Stay on Setup tab
                gr.update(visible=False),
                gr.update(visible=False),
            )

        if model_name.strip():
            global_configs["ollama_model"] = model_name.strip()

        return (
            f"✅ Ollama mode set! URL: {global_configs['ollama_url']} | Model: {global_configs['ollama_model'] or '(default)'}\n\n🎉 Setup complete! Switching to Main App...",
            gr.update(selected=1),     # Switch to Main App tab (id=1)
            gr.update(visible=True),   # Show chatbot
            gr.update(visible=True),   # Show user input
        )

    # Otherwise, Commercial mode
    return (
        f"✅ Commercial mode set! Key: {'(using default)' if not api_or_url.strip() else '(custom)'} | Model: {global_configs['model'] or '(default)'}\n\n🎉 Setup complete! Switching to Main App...",
        gr.update(selected=1),     # Switch to Main App tab (id=1)
        gr.update(visible=True),   # Show chatbot
        gr.update(visible=True),   # Show user input
    )


with gr.Blocks(theme=gr.themes.Soft(), css=".debug-output { font-family: monospace; background-color: #f9f9f9; border: 1px solid #ccc; padding: 8px; } .cr-img-source { display: none !important; }") as demo:
    with gr.Tabs(selected=0) as tabs:
        with gr.Tab("Setup", id=0) as setup_tab:
            gr.Markdown("## Welcome to PlannerForge")
            gr.Markdown(
                "*LLM agents for scenario-based testing of motion planners in autonomous "
                "driving.*\n\n"
                "PlannerForge unifies the full testing pipeline — **generate, retrieve, "
                "modify, test, and analyze** driving scenarios — behind a single chatbot, "
                "driven by off-the-shelf LLMs (no domain-specific fine-tuning). Configure "
                "your LLM below to begin, or expand **About** to see how it works."
            )

            with gr.Accordion("ℹ️ About PlannerForge — how it works", open=False):
                gr.Markdown(
                    "PlannerForge extends the classic six-stage scenario-based-testing "
                    "pipeline with two LLM-era stages (ADS Enhancement and ADS Benchmarking), "
                    "and exposes two open-source motion planners — **Frenetix** "
                    "(sampling-based) and **MP-RBFN** (learning-based) — under one interface.\n\n"
                    "**The three tabs**\n"
                    "- **Setup** — choose your LLM provider (Commercial API, local Ollama, or "
                    "Qwen) and model.\n"
                    "- **Generate** — build a CommonRoad scenario from scratch: describe it in "
                    "plain language, fetch real roads from OpenStreetMap, and simulate traffic "
                    "with SUMO (*Stage 1 — map & traffic synthesis*); then pick an ego vehicle "
                    "and goal region (*Stage 2 — planning-problem synthesis*).\n"
                    "- **Main App** — chat to retrieve scenarios from the CommonRoad database, "
                    "modify them (trajectory · behaviour · add/remove vehicles · goal), run a "
                    "motion planner, and get an LLM analysis of the outcome (collision · "
                    "timeout · kinematic infeasibility).\n\n"
                    "There's no fixed order — a two-stage **Module Router** classifies each "
                    "request and calls the matching module on demand, so you can loop freely "
                    "between modify → test → analyze."
                )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Choose Your LLM Mode")
                    commercial_or_os = gr.Radio(
                        choices=["Commercial", "Ollama", "Qwen"],
                        value=DEFAULT_MODE if DEFAULT_MODE else "Commercial",
                        label="LLM Provider"
                    )

                    gr.Markdown("---")

                    _init_cred_v, _init_cred_l, _init_cred_p, _init_model_v, _init_model_l, _init_model_p = _fields_for_mode(DEFAULT_MODE if DEFAULT_MODE else "Commercial")

                    api_input = gr.Textbox(
                        label=_init_cred_l,
                        type="password",
                        value=_init_cred_v,
                        placeholder=_init_cred_p
                    )
                    model_input = gr.Textbox(
                        label=_init_model_l,
                        type="text",
                        value=_init_model_v,
                        placeholder=_init_model_p
                    )
                    
                    # Connect mode change to update input fields
                    commercial_or_os.change(
                        update_fields_for_mode,
                        inputs=[commercial_or_os],
                        outputs=[api_input, model_input]
                    )
                    
                    confirm_btn = gr.Button("✅ Confirm & Start", variant="primary", size="lg")
                    status = gr.Textbox(label="Status", interactive=False, lines=2)

        # Generate tab sits between Setup and Main App. It's declared empty here
        # to lock its visual position, then re-entered (`with generate_tab:`)
        # below — once the Main App components it wires to (chatbot, history, …)
        # exist. id=2 is kept so existing `selected=` switching logic is unchanged.
        with gr.Tab("Generate", id=2) as generate_tab:
            pass

        with gr.Tab("Main App", id=1) as main_tab:
            history = gr.State(value = [ {"role": "assistant", "content": "Hi, welcome to the scenario generation. To start the process, please provide me with a location for the scenario, or let me know if you do not have any specific place in mind.\n\n**💡 Quick Tip:** If you already know a specific scenario name (e.g., `DEU_Weimar-71_1_T-4`), you can type it directly to skip the search and go straight to modification/motion planner."} ])

            # Scenario / image state (invisible) — defined before the layout so
            # image_output.select() below can reference them.
            image_list_state = gr.State([])
            index_state = gr.State(0)
            selected_step_state = gr.State(5)
            # TODO: currently invisible --> still needs to be clicked twice, if step selector has been used before
            image_update = gr.Button("Update Images", visible=False)

            # Main layout — LEFT column: chat on top, scenario Gif window below;
            # RIGHT column: Debug Output (small) over Console Output (large).
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(history.value, type="messages", height=480, visible=False)
                    user_input = gr.Textbox(label="Your Message", visible=False)
                    with gr.Row():
                        send_btn = gr.Button("Send", scale=3)
                        # Restart Logic, remove if bad
                        restart_btn = gr.Button("Restart", scale=1)

                    # Single scenario window: this gr.Image is the hidden live
                    # SOURCE (Gradio keeps its <img> src updated); the iframe is
                    # the one visible window — scroll to zoom, drag to pan, +/−.
                    image_output = gr.Image(
                        elem_id="image_output",
                        elem_classes=["cr-img-source"],  # hidden via CSS; mirrored by the iframe
                        type="filepath",
                        interactive=False,
                        visible=True,
                        show_download_button=False,
                        show_share_button=False,
                        height=600
                    )
                    image_output.select(fn=image_click, inputs=[image_list_state, index_state], outputs=[])

                    gr.HTML(value=(
                        f'<iframe src="/{GIF_VIEWER_URL}" '
                        f'data-src-selector="#image_output img" data-src-attr="src" '
                        f'data-empty-text="Pick a scenario — it appears here. Scroll to zoom, drag to pan, or use the +/− buttons." '
                        f'style="width:100%; height:360px; border:1px solid #ccc; '
                        f'border-radius:6px;"></iframe>'
                    ))
                    image_caption = gr.Markdown(value="", visible=True)
                    with gr.Row():
                        image_previous = gr.Button("◀ Previous", scale=1)
                        image_next = gr.Button("Next ▶", scale=1)

                with gr.Column(scale=1):
                    debug_output = gr.Textbox(
                        label="Debug Output",
                        lines=10,
                        interactive=False,
                        every=3,
                        value=get_debug_info,
                        elem_classes=["debug-output"],
                        show_copy_button=True
                    )
                    console_output = gr.Textbox(
                        label="Console Output",
                        lines=26,
                        interactive=False,
                        every=3,
                        value=poll_console_output,
                        elem_classes=["debug-output"],
                        show_copy_button=True
                    )

            # The logic here is this: image_update should always override the step selection and show the latest images
            # The step_selector is there for a comparison with the previous steps
            step_selector = gr.Radio(["1. Location", "2. Tags", "3. Road Network", "4. Obstacles", "5. Velocity"], label="Select Step")
            step_selector.change(fn=select_step, inputs=[step_selector], outputs=[image_list_state, index_state, image_output, image_caption, selected_step_state])

            # Button that becomes visible as soon as the first GIF is chosen and runs the selected scenario with Frenetix
            run_frenetix = gr.Button("Run with Motion Planner", scale=1, visible=False)

            # TODO
            select_planner = gr.Radio(choices=planner_options, label="Select Planner")
            select_planner.change(fn=select_motion_planner, inputs=[select_planner], outputs=[])

            # Output for the scenarios run through a planner
            planner_output = gr.Image(
                type="filepath", 
                visible=False,
                show_download_button=True,
                height=600
            )
            
            # Batch Analysis — collapsed by default to keep the view clean.
            with gr.Accordion("📊 Batch Simulation Analysis", open=False):
                with gr.Tabs() as detailed_batch_tabs:
                    with gr.Tab("⚙️ Driving Mode Parameters"):
                        gr.Markdown("### Cost weights across the 5 driving modes (Default, Comfort, Balanced, Efficiency-Sporty, Safety-Conservative) vs. your current model")
                        plot_performance = gr.Plot(
                            label="Driving Mode Configuration",
                            every=3,
                            value=poll_plot_performance
                        )

                    with gr.Tab("🎯 Success Analysis"):
                        gr.Markdown("### Success/failure patterns, timestep distribution, and failure reasons")
                        plot_success = gr.Plot(
                            label="Success Rate & Timestep Analysis",
                            every=3,
                            value=poll_plot_success
                        )

                    with gr.Tab("💰 Cost & Duration"):
                        gr.Markdown("### Top 20 longest scenarios with success/failure status")
                        plot_cost = gr.Plot(
                            label="Duration Analysis",
                            every=3,
                            value=poll_plot_cost
                        )

                    with gr.Tab("📋 Raw Data"):
                        gr.Markdown("### Complete simulation results table")
                        batch_data_table = gr.DataFrame(
                            value=poll_batch_dataframe,
                            every=3,
                            label="Simulation Results",
                            interactive=False,
                            wrap=True
                        )

            # Button that lets the user update images to the latest search result
            # Also resets the step selector to not have anything selected
            # TODO: has to be clicked twice, before it overrides the selected step --> why is this the case? --> Probably same issue as with Restart button - we need to set a value in the gr.update()
            image_update.click(fn=load_images, inputs=[], outputs=[image_list_state, index_state, image_output, image_caption, selected_step_state, step_selector])

            # Buttons to click through the images
            image_next.click(fn=show_next_image, inputs=[image_list_state, index_state, selected_step_state], outputs=[index_state, image_output, image_caption])
            image_previous.click(fn=show_previous_image, inputs=[image_list_state, index_state, selected_step_state], outputs=[index_state, image_output, image_caption])
            
            # Enlarge/zoom functionality

            send_btn.click(chatbot_stream, [user_input, history, image_list_state, index_state], [history, chatbot, user_input, image_output, image_list_state, index_state, image_caption, step_selector, selected_step_state, image_update, run_frenetix])
            user_input.submit(chatbot_stream, [user_input, history, image_list_state, index_state], [history, chatbot, user_input, image_output, image_list_state, index_state, image_caption, step_selector, selected_step_state, image_update, run_frenetix])

            # Only allows clean restart, if chatbot is currently not writing text (some kind of yield/streaming conflict)
            restart_btn.click(restart, inputs=[], outputs=[history, chatbot, image_output, user_input, image_list_state, index_state, image_caption, step_selector, selected_step_state, image_update, run_frenetix, planner_output])

            run_frenetix.click(fn=run_frenetix_planner, inputs=[image_list_state, index_state], outputs=[planner_output])

        # Re-enter the Generate tab (declared empty above, between Setup and Main
        # App) now that all the Main App components passed below have been created.
        with generate_tab:
            build_generate_tab(
                handler=handler,
                tabs=tabs,
                main_app_tab_id=1,
                history=history,
                chatbot=chatbot,
                image_output=image_output,
                image_list_state=image_list_state,
                index_state=index_state,
                image_caption=image_caption,
                run_frenetix=run_frenetix,
                user_input=user_input,
            )

    confirm_btn.click(
        save_settings,
        [api_input, model_input, commercial_or_os],  # inputs
        [status, tabs, chatbot, user_input]  # outputs (tabs controls which tab is selected)
    )


# demo.launch()

def prompt_user_for_bool():
    while True:
        answer = input("Use the scenario repair module this session? (y/n): ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        elif answer in {"n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")

if __name__ == "__main__":


    use_special = prompt_user_for_bool()
    SessionConfig.set_use_repair_module(use_special)

    # allowed_paths lets the Generate tab's <iframe>s load files through
    # Gradio's /gradio_api/file= endpoint: `static/` for map.html +
    # gif_viewer.html, and `data/osm/` so the GIF viewer can fetch the
    # produced CommonRoad sim GIF (and other BEV outputs) by absolute path.
    demo.launch(allowed_paths=[
        str(Path(__file__).parent / "static"),
        str(DATA_OSM_ROOT),
    ])