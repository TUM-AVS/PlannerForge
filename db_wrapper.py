import chromadb
import re
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from utils.xml_parsing_utils import *
import os
from pathlib import Path
from multiprocessing import Pool

# --- Helper for parallel workers for building DB ---
def process_scenario(file_path_str: str):
    """Processes a scenario file into metadata and content.
    Returns None if an error occurred, else a tuple with (scenario_name, scenario_content, mapping).
    """
    scenario_content, scenario_name, tags, location = add_metadata_to_xml(file_path_str)

    if scenario_content == "ERROR":
        return None

    mapping = tags | location | {"file_path": file_path_str}

    # Prepare filesystem
    file_name = Path(file_path_str).stem.removesuffix(".xml").removesuffix(".cr")
    scenario_path = (Path("Scenarios") / file_name).resolve()

    if not os.path.exists(scenario_path):
        os.makedirs(scenario_path / "Original")
        target_file_path = (scenario_path / "Original" / f"{file_name}.xml")
        with target_file_path.open("w") as f:
            f.write(scenario_content)
    else:
        print(f"Scenario {scenario_name} already in folder.")

    return (scenario_name, scenario_content, mapping)


class ScenarioDBWrapper:
    def __init__(self, persist_dir: str, embedding_function: str | None = None):
        self.persist_dir = persist_dir

        # embed with sentence transformers; remember: eventually embedding method should be changeable via parameter for experiments
        self.embedding_function = embedding_function or SentenceTransformerEmbeddingFunction()
        # using Chroma raw (instead of LangChain wrapper), since granular filter control is desired
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        self.collection = self.chroma_client.get_or_create_collection(
            name="scenarios",
            embedding_function=self.embedding_function,
            metadata={
                "description": "Scenario Dataset",
            }
        )

        self.tag_values = [
            "comfort",
            "critical",
            "emergency_breaking",
            "evasive",
            "highway",
            "illegal_cut_in",
            "intersection",
            "interstate",
            "lane_change",
            "lane_following",
            "merging_lanes",
            "multi_lane",
            "no_oncoming_traffic",
            "oncoming_traffic",
            "parallel_lanes",
            "race_track",
            "roundabout",
            "rural",
            "simulated",
            "single_lane",
            "slip_road",
            "speed_limit",
            "traffic_jam",
            "turn_left",
            "turn_right",
            "two_lane",
            "urban"
        ]

    # Store a single scenario to the DB
    # Adds a scenario to chroma, as well as to the folder directory
    def save_scenario(self, file_path_str: str) -> bool:
        result = process_scenario(file_path_str)
        if result is None:
            return False

        scenario_name, scenario_content, mapping = result
        self.collection.add(ids=scenario_name,
                            documents=scenario_content,
                            metadatas=[mapping])
        print(f"Successfully saved scenario {scenario_name}")
        return True


    def save_folder(self, folder_path_str: str, num_workers: int = 4):
        folder = Path(folder_path_str).resolve()
        files = [str(folder / file) for file in os.listdir(folder)]

        results = []
        with Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(process_scenario, files):
                if result is not None:
                    results.append(result)

        # Unpack results for bulk insert
        if results:
            ids, documents, metadatas = zip(*results)
            self.collection.add(ids=list(ids),
                                documents=list(documents),
                                metadatas=list(metadatas))

        print(f"Successfully saved {len(results)} scenarios out of {len(files)}")

    # Find the file path of a scenario, given its id (which is the name of the xml file, including the ending)
    def get_file_path(self, id_:str) -> str:
        meta = self.collection.get(ids=[id_], include=["metadatas"])
        file_path = meta.get("metadatas")[0].get("file_path")

        if file_path is None:
            print("File could not be found. Make sure it is still in its original location. This does not refer to the Chroma DB, since the file is needed in its original XML forma.")
            return "File Not Found"

        return file_path

    def find_best_location(self, location:dict) -> list[str]:
        """Filter the DB by location metadata. Chroma `$eq` is case-
        sensitive, so we first try an exact match; if zero results come
        back AND a specifier was provided, retry with a case-insensitive
        scan over loaded metadatas. This recovers from LLM extractions
        like 'tyler' / 'TYLER' / 'Tyler ' that don't match the DB's
        canonical 'Tyler'."""
        # Pass 1: exact $eq match.
        eq_clauses = [{k: {"$eq": v}} for k, v in location.items()]
        if not eq_clauses:
            return []
        where = eq_clauses[0] if len(eq_clauses) == 1 else {"$and": eq_clauses}
        query_result = self.collection.get(where=where, limit=400,
                                              include=["metadatas"])
        if query_result["ids"]:
            return query_result["ids"]

        # Pass 2: case-insensitive scan. Only fall back here when the
        # exact match returned 0 — keeps the fast path fast.
        cc = location.get("country_code")
        spec = location.get("specifier")
        if cc is None:
            return []
        # Narrow by country first (always exact-case for the 3-letter
        # ISO code) then case-fold the specifier comparison.
        country_only = self.collection.get(
            where={"country_code": {"$eq": cc}}, limit=400,
            include=["metadatas"])
        if not spec:
            return country_only["ids"]
        # Fold to lowercase alphanumerics so multi-word extractions match
        # CamelCase specifiers ('New York' / 'new-york' → 'newyork'), in the
        # same spirit as the case-insensitive recovery above.
        def _fold(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())
        spec_fold = _fold(str(spec))
        out = []
        for sid, meta in zip(country_only["ids"], country_only["metadatas"]):
            ms = meta.get("specifier", "")
            if isinstance(ms, str) and _fold(ms) == spec_fold:
                out.append(sid)
        return out

    def available_countries(self) -> list:
        """(country_code, count) pairs present in the collection, most common
        first — used to tell the user which locations ARE available when the
        one they asked for has no scenarios."""
        from collections import Counter
        try:
            res = self.collection.get(include=["metadatas"], limit=100000)
        except Exception:
            return []
        counts = Counter(
            m.get("country_code") for m in res.get("metadatas", [])
            if m and m.get("country_code"))
        return counts.most_common()

    # TODO: Tags are immediately looped, until a result is found. This needs to change if we focus more on interactive scenarios, since these (possibly?) tend to be sparsely annotated.
    def find_best_tags(self, after_location_ids:list[str], tags:list[str]):
        filtered_ids = []
        filtered_docs = []

        eq_tags = [{tag: {"$eq": 1}} for tag in tags]

        iter_counter = 0
        while len(filtered_ids) == 0:

            # If the user provides no tags, no results will be returned
            if len(eq_tags) == 0:
                print("---\nNo fitting scenario exists for the specified location. Tags were iteratively removed until only 1 left. Should inform the user and restart the search process in the interface.py file. Use the fact that filtered_ids is empty for this behavior.\n---")
                break

            if len(eq_tags) == 1:
                where = eq_tags[0]
            else:
                where = {"$and": eq_tags}

            if len(after_location_ids) == 0: # If this list is empty, we do not consider location, because the user skipped it
                query_result = self.collection.get(where=where, limit = 400, include=["documents"])
            else:
                query_result = self.collection.get(ids=after_location_ids, where=where, limit=400, include=["documents"])

            filtered_ids = query_result["ids"]
            filtered_docs = query_result["documents"]

            eq_tags.pop() # Here for counting when 0 tags are left

            iter_counter += 1

        # Returns ids (= names) and documents of found results
        return filtered_ids, filtered_docs

    def find_best_road_net(self, after_tags_ids:list[str], after_tags_docs:list[str], road_net_from_user:dict):
        filtered_ids = []
        filtered_documents = []

        for id_, doc in zip(after_tags_ids, after_tags_docs):
            from_xml = get_ego_lane_from_xml(doc)
            add_to_filtered = True

            # to_be_added = {} # Stores all the traffic signs/lights that should be added to the XML file

            for k, v in road_net_from_user.items():
                if k not in from_xml:
                    if 0 in v:
                        continue
                    else: # TODO: enter modification call for adding new traffic sign

                        # Do not forget to update from_xml and the lists after receiving the modified file
                        # to_be_added.add({k: v})

                        add_to_filtered = False
                        break

                if not self.in_range(from_xml[k], v): # TODO: enter modification call for changing value of existing sign
                    add_to_filtered = False
                    break

            # if not add_to_filtered:
                # modified_id, modified_doc, debug_info = add_traffic_sign_or_light(id_, doc, to_be_added)
                # filtered_ids.append(modified_id)
                # filtered_docs.append(modified_doc)

            if add_to_filtered:
                filtered_ids.append(id_)
                filtered_documents.append(doc)

        return filtered_ids, filtered_documents

    # Helper for range matches.
    # `value` is an exact value (from XML), `bounds` is a list with one
    # or two elements (from the user / LLM extraction).
    # - 2-element bounds [lo, hi]: inclusive range check.
    # - 1-element bounds [v]: treat as a SOFT match with tolerance. The
    #   previous exact-equality semantics (`value == bounds[0]`) failed
    #   on tiny floating-point differences (e.g. user said 7.0 m/s,
    #   scenario stored 6.9141 → equality fails). With tolerance we
    #   match when the difference is within `tol` (absolute) or 20 %
    #   (relative), whichever is larger. Floats use ±2.0 m/s as a
    #   reasonable velocity tolerance; ints use ±3 as a count tolerance.
    def in_range(self, value, bounds, *, abs_tol_float=2.0, abs_tol_int=3):
        if len(bounds) == 1:
            target = bounds[0]
            # Binary presence features (traffic_light, stop, traffic signs) are
            # 0/1. The count tolerance below would wrongly match a present
            # feature (value 1) against a [0] ("none"/"without") constraint —
            # e.g. "without traffic light" keeping a scenario that has one — so
            # require an exact match when both sides are binary.
            if target in (0, 1) and value in (0, 1):
                return value == target
            try:
                diff = abs(float(value) - float(target))
            except (TypeError, ValueError):
                return value == target
            tol = abs_tol_int if isinstance(target, int) and isinstance(value, int) \
                else abs_tol_float
            return diff <= max(tol, 0.2 * abs(float(target)))
        elif len(bounds) == 2:
            return bounds[0] <= value <= bounds[1]
        else:
            raise ValueError("Bounds must be a list of 1 or 2 elements.")

    def find_best_obstacles(self, after_l1_ids: list[str], after_l1_docs: list[str], obs_from_user: dict[str, dict]):
        filtered_ids = []
        filtered_documents = []

        for id_, doc in zip(after_l1_ids, after_l1_docs): # If 0 stop signs are specified in the step before and then 0 pedestrians later, this loop is unreachable
            from_xml = get_obstacles_from_xml(xml_string=doc)
            add_to_filtered = True

            for k, v in obs_from_user.items():
                if k not in from_xml:
                    if 0 in v: continue # Example: if the user wants 0 cars, the value is 0. But if a scenario contains no cars, the count will not be 0 - there simply will not be a car-object present. This is why the document can be added to the results, even though an obstacle key desired by the user is not present in the XML file.
                    else:
                        add_to_filtered = False
                        break
                if not self.in_range(from_xml[k], obs_from_user[k]):
                    add_to_filtered = False
                    break

            if add_to_filtered:
                filtered_ids.append(id_)
                filtered_documents.append(doc)

        return filtered_ids, filtered_documents

    def find_best_velocity(self, after_obstacles_ids: list[str], after_obstacles_docs: list[str], initial_velocity_from_user: dict[str, list[float]]):
        filtered_ids = []
        filtered_documents = []

        for id_, doc in zip(after_obstacles_ids, after_obstacles_docs):
            from_xml = get_velocity_from_xml(xml_string=doc)

            if from_xml == -1.0: continue
            if self.in_range(from_xml, initial_velocity_from_user.get("initial_velocity")):
                filtered_ids.append(id_)
                filtered_documents.append(doc)

        return filtered_ids, filtered_documents

    def semantic_search(self, query_text: str, n_results: int = 5, where: dict = None):
        """
        Perform semantic similarity search using embedded scenario content.
        Falls back to this when metadata filtering returns insufficient results.
        
        Args:
            query_text: Natural language description of desired scenario
            n_results: Number of results to return
            where: Optional metadata filters to apply
            
        Returns:
            List of scenario IDs ranked by semantic similarity
        """
        try:
            if where:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where
                )
            else:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results
                )
            return results['ids'][0] if results['ids'] else []
        except Exception as e:
            print(f"Semantic search failed: {e}")
            return []
