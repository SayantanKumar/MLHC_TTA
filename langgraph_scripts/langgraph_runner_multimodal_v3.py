import re
import json
import random
import argparse
from typing import List, Tuple, Dict, Any, Optional
from langchain_community.llms import LlamaCpp
import os
from datetime import datetime
from langchain_core.output_parsers import BaseOutputParser
import requests

from mimic_struct_utils import get_topk_struct_events

class BSVParser(BaseOutputParser):
    """Base BSV (Bar-Separated Values) parser that converts BSV text to dictionaries"""
    def parse(self, text: str) -> List[Dict[str, str]]:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            return []
        
        # Split header and rows
        header = [h.strip() for h in lines[0].split("|")]
        rows = [line.split("|") for line in lines[1:]]
        
        # Create dictionary for each row
        return [dict(zip(header, [val.strip() for val in row])) for row in rows]

class CentralEventsBSVParser(BSVParser):
    """Parser for central events in BSV format"""
    def parse(self, text: str) -> Dict[str, Any]:
        parsed = super().parse(text)
        if not parsed:
            return {"central_events": []}
        
        # Expect format: event
        return {"central_events": [row["event"] for row in parsed if "event" in row]}

class TimeDistancesBSVParser(BSVParser):
    """Parser for time distances in BSV format"""
    def parse(self, text: str) -> Dict[str, Any]:
        parsed = super().parse(text)
        if not parsed:
            return {"time_distances": []}
        
        # Expect format: event1|event2|e2_minus_e1|confidence
        time_distances = []
        for row in parsed:
            try:
                time_distances.append({
                    "event1": row["event1"],
                    "event2": row["event2"],
                    "e2_minus_e1": float(row["e2_minus_e1"]),  # Changed from hours_diff
                    "confidence": int(row["confidence"])
                })
            except (KeyError, ValueError):
                continue
        return {"time_distances": time_distances}

class NonCentralEventsBSVParser(BSVParser):
    """Parser for non-central events in BSV format"""
    def parse(self, text: str) -> Dict[str, Any]:
        parsed = super().parse(text)
        if not parsed:
            return {"non_central_events": []}
        
        # Expect format: event|central_event|relative_time|confidence
        non_central_events = []
        for row in parsed:
            try:
                non_central_events.append({
                    "event": row["event"],
                    "central_event": row["central_event"],
                    "relative_time": float(row["relative_time"]),
                    "confidence": int(row["confidence"])
                })
            except (KeyError, ValueError):
                continue
        return {"non_central_events": non_central_events}
    
class UpdatedCentralTimelineBSVParser(BSVParser):
    """Parser for updated central timeline events in BSV format"""
    def parse(self, text: str) -> Dict[str, Any]:
        parsed = super().parse(text)
        if not parsed:
            return {"updated_central_timeline": []}
        
        # Expect format: event|time
        updated_central_timeline = []
        for row in parsed:
            try:
                updated_central_timeline.append({
                    "event": row["event"],
                    "time": float(row["time"])
                })
            except (KeyError, ValueError):
                continue
        return {"updated_central_timeline": updated_central_timeline}

class TimelineBSVParser(BSVParser):
    """Parser for timeline events in BSV format"""
    def parse(self, text: str) -> Dict[str, Any]:
        parsed = super().parse(text)
        if not parsed:
            return {"timeline": []}
        
        # Expect format: event|time
        timeline = []
        for row in parsed:
            try:
                timeline.append({
                    "event": row["event"],
                    "time": float(row["time"])
                })
            except (KeyError, ValueError):
                continue
        return {"timeline": timeline}

class UpdatedTimelineBSVParser(BSVParser):
    """Parser for timeline events in BSV format"""
    def parse(self, text: str) -> Dict[str, Any]:
        parsed = super().parse(text)
        if not parsed:
            return {"updated_timeline": []}
        
        # Expect format: event|time
        timeline = []
        for row in parsed:
            try:
                timeline.append({
                    "event": row["event"],
                    "time": float(row["time"])
                })
            except (KeyError, ValueError):
                continue
        return {"updated_timeline": timeline}
    
class State:
    def __init__(self, discharge_summary: str, hadm_id: int, llm: Any = None, llm_endpoint: str = None):
        self.discharge_summary = discharge_summary
        self.hadm_id = hadm_id
        self.central_events: List[str] = []
        self.time_distances: List[Tuple[str, str, float, int]] = []  # (event1, event2, e2_minus_e1, confidence)
        self.non_central_events: List[Tuple[str, str, float, int]] = []  # (non_central_event, central_event, relative_time, confidence)
        self.central_timeline: List[Tuple[str, float]] = []  # (event, time)
        self.central_timeline_with_topk_rows: List[Tuple[str, float, List[Tuple[str, str, float, float]]]] = []  # (event, time, topk_rows)
        self.updated_central_timeline: List[Tuple[str, float]] = []  # (event, time)
        self.timeline: List[Tuple[str, float]] = []  # (event, time)
        self.timeline_with_topk_rows: List[Tuple[str, float, List[Tuple[str, str, float, float]]]] = []  # (event, time, topk_rows)
        self.updated_timeline: List[Tuple[str, float]] = []  # (event, time)
        self.llm = llm
        self.llm_endpoint = llm_endpoint


OUTPUT_FILE_TO_STATE_ATTR = {
    "central_events.json": "central_events",
    "time_distances.json": "time_distances",
    "central_timeline.json": "central_timeline",
    "central_timeline_with_topk_rows.json": "central_timeline_with_topk_rows",
    "updated_central_timeline.json": "updated_central_timeline",
    "non_central_events.json": "non_central_events",
    "timeline.json": "timeline",
    "timeline_with_topk_rows.json": "timeline_with_topk_rows",
    "updated_timeline.json": "updated_timeline",
}

class TimelineAgent:
    def __init__(self, llm=None, llm_endpoint=None, omit_tags: List[str] = None):
        self.llm = llm
        self.llm_endpoint = llm_endpoint
        self.template_dir = "templates"
        # Store the omit_tags, defaulting to ["think"] if None
        self.omit_tags = omit_tags if omit_tags is not None else ["think"]
        
        # Create BSV parsers
        self.central_events_parser = CentralEventsBSVParser()
        self.time_distances_parser = TimeDistancesBSVParser()
        self.non_central_events_parser = NonCentralEventsBSVParser()
        self.updated_central_timeline_parser = UpdatedCentralTimelineBSVParser()
        self.timeline_parser = TimelineBSVParser()
        self.updated_timeline_parser = UpdatedTimelineBSVParser()

    def _invoke_llm(self, prompt: str) -> str:
        """Handle LLM invocation either locally or via endpoint"""
        if self.llm_endpoint:
            try:
                # Format the request for llama-server's chat completion API
                request_data = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 16000,
                    "temperature": 0.6
                }
                
                response = requests.post(
                    f"{self.llm_endpoint}/v1/chat/completions",
                    json=request_data,
                    timeout=6000
                )
                response.raise_for_status()
                
                # Extract the content from the response
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Error calling LLM endpoint: {e}")
                raise
        elif self.llm:
            return self.llm.invoke(prompt)
        else:
            raise ValueError("No LLM or endpoint configured")
        
    def _load_template(self, template_name: str) -> str:
        """Load a prompt template from file."""
        template_path = os.path.join(self.template_dir, f"{template_name}.template")
        try:
            with open(template_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: Template file not found - {template_path}")
            return ""
        
    def extract_bsv(self, response: str, parser: BaseOutputParser) -> Optional[Dict[str, Any]]:
        # Use self.omit_tags to remove content inside those tags
        cleaned_response = response
        for tag in self.omit_tags:
            pattern = re.compile(rf"<{tag}>.*?</{tag}>", re.DOTALL)
            cleaned_response = pattern.sub("", cleaned_response)

        # Now try parsing the cleaned response
        try:
            return parser.parse(cleaned_response)
        except Exception as e:
            print(f"---Cleaned input:---\n{cleaned_response}\n")
            print(f"Error during parsing: {e}")
            return None

    def run_with_retries(self, prompt: str, parser: BaseOutputParser, n_retries: int = 3) -> Optional[Dict[str, Any]]:
        for _ in range(n_retries):
            response = self._invoke_llm(prompt)  # Changed from llm.invoke
            bsv_data = self.extract_bsv(response, parser)
            if bsv_data:
                return bsv_data
        return None

    def extract_central_events(self, state: State, n_retries: int = 3) -> State:
        template = self._load_template("extract_central_events")
        prompt = template.format(
            discharge_summary=state.discharge_summary,
            state=state
        )
        
        for attempt in range(n_retries):
            response = self._invoke_llm(prompt)
            parsed = self.extract_bsv(response, self.central_events_parser)
            if parsed and parsed["central_events"]:  # Check for non-empty results
                state.central_events = parsed["central_events"]
                return state
            else:
                print(f"\tFailed to parse this:\n\t{parsed}\n\t------\n\t{response}\n")
            print(f"Attempt {attempt + 1}: Empty central events result, retrying...")
        
        print(f"Failed to extract central events after {n_retries} attempts")
        return state

    def compute_time_distances(self, state: State, n_retries: int = 3) -> State:
        template = self._load_template("compute_time_distances")
        prompt = template.format(
            central_events="|".join(state.central_events),
            discharge_summary=state.discharge_summary
        )
        
        for _ in range(n_retries):
            response = self._invoke_llm(prompt)
            parsed = self.extract_bsv(response, self.time_distances_parser)
            if parsed:
                state.time_distances = [
                    (item['event1'], item['event2'], item['e2_minus_e1'], item['confidence'])  # Changed from hours_diff
                    for item in parsed["time_distances"]
                ]
                return state
        
        print(f"Failed to compute time distances after {n_retries} attempts")
        return state

    def reconstruct_central_timeline(self, state: State, n_retries: int = 3) -> State:
        template = self._load_template("reconstruct_central_timeline")
        
        # Format time distances for display
        time_distances_str = "\n".join(
            f"{td[0]} | {td[1]} | {td[2]} hours | confidence {td[3]}"
            for td in state.time_distances
        )
        
        prompt = template.format(
            central_events="\n".join(state.central_events),
            time_distances=time_distances_str,
            discharge_summary=state.discharge_summary
        )
        
        for attempt in range(n_retries):
            response = self._invoke_llm(prompt)
            parsed = self.extract_bsv(response, self.timeline_parser)
            if parsed and parsed["timeline"]:  # Check for non-empty results
                # Convert to list of tuples and sort by time
                central_timeline = sorted(
                    [(item['event'], item['time']) for item in parsed["timeline"]],
                    key=lambda x: x[1]
                )
                state.central_timeline = central_timeline
                return state
            print(f"Attempt {attempt + 1}: Empty central timeline result, retrying...")
        
        print(f"Failed to reconstruct central timeline after {n_retries} attempts")
        return state

    def extract_non_central_events(self, state: State, n_retries: int = 3) -> State:
        template = self._load_template("extract_non_central_events")
        prompt = template.format(
            central_events="|".join(state.central_events),
            discharge_summary=state.discharge_summary,
            state=state
        )
        
        for _ in range(n_retries):
            response = self._invoke_llm(prompt)  # Changed from llm.invoke
            parsed = self.extract_bsv(response, self.non_central_events_parser)
            if parsed:
                state.non_central_events = [
                    (item['event'], item['central_event'], 
                    item['relative_time'], item['confidence'])
                    for item in parsed["non_central_events"]
                ]
                return state
        
        print(f"Failed to extract non-central events after {n_retries} attempts")
        return state

    def update_central_timeline(self, state: State, n_retries: int = 3) -> State:
        # Reuse previously saved retrieval results when available.
        if not state.central_timeline_with_topk_rows:
            state.central_timeline_with_topk_rows = []
            for event, time in state.central_timeline:
                topk_rows = get_topk_struct_events(hadm_id=state.hadm_id, query=event, topk=5, max_times=5)
                state.central_timeline_with_topk_rows.append((event, time, topk_rows))

        template = self._load_template("update_central_timeline")

        format_strs = []
        for i, (event, time, topk_rows) in enumerate(state.central_timeline_with_topk_rows):
            entry_str = f"Event {i+1}: {event}, time: {time:.2f}\nTop-10 EHR rows:\nname | value | time | similarity"
            for td in topk_rows:
                time_str = " ".join([f"{t:.2f}" for t in td[2]])
                entry_str += f"\n{td[0]} | {td[1]} | {time_str} | {td[3]:.3f}"
            format_strs.append(entry_str)
        central_timeline_with_topk_rows_str = "\n\n".join(format_strs)

        prompt = template.format(
            central_timeline_with_topk_rows=central_timeline_with_topk_rows_str,
            discharge_summary=state.discharge_summary
        )
        
        for attempt in range(n_retries):
            response = self._invoke_llm(prompt)
            parsed = self.extract_bsv(response, self.updated_central_timeline_parser)
            if parsed and parsed["updated_central_timeline"]:
                state.updated_central_timeline = [
                    (item['event'], item['time'])
                    for item in parsed["updated_central_timeline"]
                ]
                return state
            print(f"Attempt {attempt + 1}: Empty updated central timeline result, retrying...")
        
        print(f"Failed to update central timeline after {n_retries} attempts")
        return state

    def reconstruct_timeline(self, state: State, n_retries: int = 3) -> State:
        template = self._load_template("reconstruct_timeline")
        
        # Prepare central timeline string for the prompt
        central_timeline_str = "\n".join(
            f"{event} at {time} hours" 
            for event, time in state.updated_central_timeline
        )
        
        prompt = template.format(
            discharge_summary=state.discharge_summary,
            central_timeline=central_timeline_str,
            non_central_events="\n".join(
                f"{e[0]} occurs {e[2]} hours {'before' if e[2] < 0 else 'after'} {e[1]} (confidence {e[3]})"
                for e in state.non_central_events
            ),
            state=state
        )
        
        for attempt in range(n_retries):
            response = self._invoke_llm(prompt)
            parsed = self.extract_bsv(response, self.timeline_parser)
            if parsed and parsed["timeline"]:  # Check for non-empty results
                state.timeline = [
                    (item['event'], item['time'])
                    for item in parsed["timeline"]
                ]
                return state
            print(f"Attempt {attempt + 1}: Empty timeline result, retrying...")
        
        print(f"Failed to reconstruct timeline after {n_retries} attempts")
        return state

    def update_timeline(self, state: State, n_retries: int = 3) -> State:
        # Reuse previously saved retrieval results when available.
        if not state.timeline_with_topk_rows:
            state.timeline_with_topk_rows = []
            for event, time in state.timeline:
                topk_rows = get_topk_struct_events(hadm_id=state.hadm_id, query=event, topk=5, max_times=5)
                state.timeline_with_topk_rows.append((event, time, topk_rows))

        template = self._load_template("update_central_timeline")

        format_strs = []
        for i, (event, time, topk_rows) in enumerate(state.timeline_with_topk_rows):
            entry_str = f"Event {i+1}: {event}, time: {time:.2f}\nTop-10 EHR rows:\nname | value | time | similarity"
            for td in topk_rows:
                time_str = " ".join([f"{t:.2f}" for t in td[2]])
                entry_str += f"\n{td[0]} | {td[1]} | {time_str} | {td[3]:.3f}"
            format_strs.append(entry_str)
        timeline_with_topk_rows_str = "\n\n".join(format_strs)

        prompt = template.format(
            central_timeline_with_topk_rows=timeline_with_topk_rows_str,
            discharge_summary=state.discharge_summary
        )
        
        for attempt in range(n_retries):
            response = self._invoke_llm(prompt)
            parsed = self.extract_bsv(response, self.updated_timeline_parser)
            if parsed and parsed["updated_timeline"]:
                state.updated_timeline = [
                    (item['event'], item['time'])
                    for item in parsed["updated_timeline"]
                ]
                return state
            print(f"Attempt {attempt + 1}: Empty updated timeline result, retrying...")
        
        print(f"Failed to update timeline after {n_retries} attempts")
        return state

def print_outputs(state: State):
    for i, ((event, time, topk_rows), (event2, time2)) in enumerate(zip(state.timeline_with_topk_rows, state.updated_timeline)):
        entry_str = f"Event {i+1}: {event}, time: {time:.2f}\nTop-10 EHR rows:\nname | value | time | similarity"
        for td in topk_rows:
            time_str = " ".join([f"{t:.2f}" for t in td[2]])
            entry_str += f"\n{td[0]} | {td[1]} | {time_str} | {td[3]:.3f}"
        print(entry_str)
        print(f"-> Updated to: {event2}: {time2:.2f} hours\n")


def save_outputs(state: State, log_folder: str, only_files: Optional[List[str]] = None,
                 skip_empty: bool = False, print_summary: bool = True,
                 overwrite_non_empty_existing: bool = False):
    """Save selected outputs to log files in the specified folder.

    By default, existing non-empty JSON files are preserved and will not be overwritten.
    Only missing or empty or unreadable files are written.
    """
    os.makedirs(log_folder, exist_ok=True)

    output_map = {
        "central_events.json": state.central_events,
        "time_distances.json": state.time_distances,
        "central_timeline.json": state.central_timeline,
        "non_central_events.json": state.non_central_events,
        "central_timeline_with_topk_rows.json": state.central_timeline_with_topk_rows,
        "updated_central_timeline.json": state.updated_central_timeline,
        "timeline.json": state.timeline,
        "timeline_with_topk_rows.json": state.timeline_with_topk_rows,
        "updated_timeline.json": state.updated_timeline,
    }

    files_to_write = only_files if only_files is not None else list(output_map.keys())
    for filename in files_to_write:
        content = output_map[filename]
        if skip_empty and not is_non_empty_content(content):
            continue

        existing = load_json_output(log_folder, filename)
        if existing is not None and not overwrite_non_empty_existing:
            continue

        with open(os.path.join(log_folder, filename), 'w') as f:
            f.write(json.dumps(content, indent=2))

    if print_summary:
        print(f"Outputs saved to: {log_folder}")
        print("Reconstructed Timeline:")
        for event in state.updated_timeline:
            print(f"{event[0]}: {event[1]} hours")

        print("\nDetailed Timeline with Top-K EHR Rows:")
        print_outputs(state)

def main():
    parser = argparse.ArgumentParser(description="Run timeline reconstruction from discharge summaries")
    parser.add_argument('--model_path', 
                       default="/data/weissjc/.cache/gguf/DeepSeek-V3-0324-IQ1/DeepSeek-V3-0324-UD-IQ1_S-00001-of-00004.gguf",
                       help="Path to the GGUF model file")
    parser.add_argument('--llm_endpoint',
                       type=str,
                       help="Endpoint URL for LLM server (e.g., http://localhost:8080)")
    parser.add_argument('--model_basename',
                       type=str,
                       help="Required when using --llm_endpoint to identify the model")
    parser.add_argument('--summary_file', 
                       type=str,
                       help="Path to a single discharge summary text file (.txt)")
    parser.add_argument('--hadm_id', 
                       type=int,
                       help="MIMIC-III or IV HADM_ID corresponding to the discharge summary")
    parser.add_argument('--summary_directory', 
                       type=str,
                       help="Path to a directory containing multiple summary files")
    parser.add_argument('--summary_ext',
                       type=str,
                       default=".txt",
                       help="File extension for summary files (default: .txt, can be .txt.gz for gzipped files)")
    parser.add_argument('--log_folder',
                       type=str,
                       help="Folder to save output logs")
    parser.add_argument('--debug',
                       action='store_true',
                       help="Enable debug mode with timestamped logs")
    parser.add_argument('--omit_tags',
                       type=str,
                       default="think",
                       help="Comma-separated list of tags to omit during parsing (default: 'think')")
    # Add n_retries argument
    parser.add_argument('--n_retries',
                       type=int,
                       default=3,
                       help="Number of retries for each processing step (default: 3)")
    args = parser.parse_args()

    # Validate input arguments
    if not args.summary_file and not args.summary_directory:
        raise ValueError("Must provide either --summary_file or --summary_directory")
    if args.summary_file and args.summary_directory:
        raise ValueError("Cannot provide both --summary_file and --summary_directory")

    if args.llm_endpoint:
        if not args.model_basename:
            print("Error: When using --llm_endpoint, you must specify --model_basename")
            print("This helps identify which model is being used in the logs")
            exit(1)
        llm = None
        llm_endpoint = args.llm_endpoint
        model_basename = args.model_basename
    else:
        llm = LlamaCpp(
            model_path=args.model_path,
            n_gpu_layers=9999,
            n_ctx=16000
        )
        llm_endpoint = None
        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]

    omit_tags = [tag.strip() for tag in args.omit_tags.split(',') if tag.strip()]

    if args.summary_file:
        # Process single file
        basename = os.path.splitext(os.path.basename(args.summary_file))[0]
        if args.summary_file.endswith('.gz'):
            basename = os.path.splitext(basename)[0]  # Remove .gz extension too

        try:
            if args.summary_file.endswith('.gz'):
                import gzip
                try:
                    with gzip.open(args.summary_file, 'rt', encoding='utf-8') as f:
                        summary = f.read()
                except UnicodeDecodeError:
                    with gzip.open(args.summary_file, 'rt', encoding='unicode_escape') as f:
                        summary = f.read()
            else:
                try:
                    with open(args.summary_file, 'r', encoding='utf-8') as f:
                        summary = f.read()
                except UnicodeDecodeError:
                    with open(args.summary_file, 'r', encoding='unicode_escape') as f:
                        summary = f.read()
        except Exception as e:
            print(f"Error reading file {args.summary_file}: {e}")
            exit(1)

        log_folder = args.log_folder if args.log_folder else \
                     f"logs/{basename}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" if args.debug else \
                     f"logs/{basename}_{model_basename}"
        os.makedirs(log_folder, exist_ok=True)
        
        # Skip if outputs already exist
        if not should_skip_processing(log_folder):
            run_graph(summary, args.hadm_id, llm, n_retries=args.n_retries, log_folder=log_folder,
                      llm_endpoint=llm_endpoint, omit_tags=omit_tags)
        else:
            print(f"Skipping processing as outputs already exist in {log_folder}")
    else:
        # Process directory of files
        if not os.path.isdir(args.summary_directory):
            raise ValueError(f"Directory not found: {args.summary_directory}")
        
        # Get all files with specified extension
        files = [f for f in os.listdir(args.summary_directory) 
                 if f.endswith(args.summary_ext) or 
                    (args.summary_ext == '.txt' and f.endswith('.txt.gz'))]
        random.shuffle(files)  # Process files in random order
        
        for filename in files:
            filepath = os.path.join(args.summary_directory, filename)
            try:
                if filename.endswith('.gz'):
                    import gzip
                    try:
                        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                            summary = f.read()
                    except UnicodeDecodeError:
                        with gzip.open(filepath, 'rt', encoding='unicode_escape') as f:
                            summary = f.read()
                else:
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            summary = f.read()
                    except UnicodeDecodeError:
                        with open(filepath, 'r', encoding='unicode_escape') as f:
                            summary = f.read()
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue
            
            basename = os.path.splitext(filename)[0]
            hadm_id = int(re.findall(r"\d+", basename)[-1])
            log_folder = args.log_folder if args.log_folder else \
                         f"logs/{basename}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" if args.debug else \
                         f"logs/{basename}_{model_basename}"
            os.makedirs(log_folder, exist_ok=True)
            
            # Skip if outputs already exist
            if not should_skip_processing(log_folder):
                print(f"\nProcessing: {filename}")
                run_graph(summary, hadm_id, llm, n_retries=args.n_retries, log_folder=log_folder,
                          llm_endpoint=llm_endpoint, omit_tags=omit_tags)
            else:
                print(f"\nSkipping {filename} as outputs already exist in {log_folder}")

def is_non_empty_content(content: Any) -> bool:
    return content not in (None, "", [], {})


def load_json_output(log_folder: str, filename: str) -> Optional[Any]:
    file_path = os.path.join(log_folder, filename)
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            content = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Ignoring unreadable JSON file {file_path}: {e}")
        return None
    if not is_non_empty_content(content):
        return None
    return content


def load_existing_outputs_into_state(state: State, log_folder: str):
    for filename, attr_name in OUTPUT_FILE_TO_STATE_ATTR.items():
        content = load_json_output(log_folder, filename)
        if content is not None:
            setattr(state, attr_name, content)


PIPELINE_STEPS = [
    {
        "name": "central_events",
        "primary_output": "central_events.json",
        "outputs": ["central_events.json"],
        "dependencies": [],
        "runner": "extract_central_events",
    },
    {
        "name": "time_distances",
        "primary_output": "time_distances.json",
        "outputs": ["time_distances.json"],
        "dependencies": ["central_events.json"],
        "runner": "compute_time_distances",
    },
    {
        "name": "central_timeline",
        "primary_output": "central_timeline.json",
        "outputs": ["central_timeline.json"],
        "dependencies": ["central_events.json", "time_distances.json"],
        "runner": "reconstruct_central_timeline",
    },
    {
        "name": "non_central_events",
        "primary_output": "non_central_events.json",
        "outputs": ["non_central_events.json"],
        "dependencies": ["central_events.json"],
        "runner": "extract_non_central_events",
    },
    {
        "name": "updated_central_timeline",
        "primary_output": "updated_central_timeline.json",
        "outputs": ["central_timeline_with_topk_rows.json", "updated_central_timeline.json"],
        "dependencies": ["central_timeline.json"],
        "runner": "update_central_timeline",
    },
    {
        "name": "timeline",
        "primary_output": "timeline.json",
        "outputs": ["timeline.json"],
        "dependencies": ["updated_central_timeline.json", "non_central_events.json"],
        "runner": "reconstruct_timeline",
    },
    {
        "name": "updated_timeline",
        "primary_output": "updated_timeline.json",
        "outputs": ["timeline_with_topk_rows.json", "updated_timeline.json"],
        "dependencies": ["timeline.json"],
        "runner": "update_timeline",
    },
]


def should_skip_processing(log_folder: str) -> bool:
    """Skip only when every output file is present and non-empty."""
    for filename in OUTPUT_FILE_TO_STATE_ATTR:
        if load_json_output(log_folder, filename) is None:
            return False
    return True


def run_graph(discharge_summary: str, hadm_id: int, llm: Any, n_retries: int = 3,
              log_folder: str = None, llm_endpoint: str = None, omit_tags: List[str] = None) -> State:
    agent = TimelineAgent(llm=llm, llm_endpoint=llm_endpoint, omit_tags=omit_tags)
    state = State(discharge_summary, hadm_id=hadm_id, llm=llm, llm_endpoint=llm_endpoint)

    os.makedirs(log_folder, exist_ok=True)
    load_existing_outputs_into_state(state, log_folder)

    for step in PIPELINE_STEPS:
        primary_output = step["primary_output"]
        primary_content = load_json_output(log_folder, primary_output)
        if primary_content is not None:
            print(f"Skipping {step['name']} because {primary_output} already exists and is non-empty")
            setattr(state, OUTPUT_FILE_TO_STATE_ATTR[primary_output], primary_content)
            for extra_output in step["outputs"]:
                extra_content = load_json_output(log_folder, extra_output)
                if extra_content is not None:
                    setattr(state, OUTPUT_FILE_TO_STATE_ATTR[extra_output], extra_content)
            continue

        missing_dependencies = [
            dep for dep in step["dependencies"]
            if load_json_output(log_folder, dep) is None and not is_non_empty_content(getattr(state, OUTPUT_FILE_TO_STATE_ATTR[dep]))
        ]
        if missing_dependencies:
            print(f"Cannot run {step['name']} because these dependency files are missing or empty: {missing_dependencies}")
            break

        print(f"Running {step['name']} because {primary_output} is missing or empty")
        state = getattr(agent, step["runner"])(state, n_retries)
        save_outputs(state, log_folder, only_files=step["outputs"], skip_empty=True, print_summary=False)
        load_existing_outputs_into_state(state, log_folder)

    return state

if __name__ == "__main__":
    main()