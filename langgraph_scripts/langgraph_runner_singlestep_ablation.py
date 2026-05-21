import re
import json
import random
import argparse
from typing import List, Tuple, Dict, Any, Optional
import os
from datetime import datetime
from collections import defaultdict

import requests
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.llms import LlamaCpp

from mimic_struct_utils import get_topk_struct_events


class BSVParser(BaseOutputParser):
    """Base BSV (bar-separated values) parser.

    Expects a header row somewhere in the response, but downstream parsers may
    also recover rows heuristically when the model omits the header.
    """

    def parse(self, text: str) -> List[Dict[str, str]]:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            return []

        header = [h.strip() for h in lines[0].split("|")]
        rows = [line.split("|") for line in lines[1:]]
        return [dict(zip(header, [val.strip() for val in row])) for row in rows]


class TimelineBSVParser(BaseOutputParser):
    """Robust parser for single-step extraction output.

    Preferred format:
      event|time|confidence

    Also tolerates:
      event|time
      leading prose / code fences / missing header
      numeric strings embedded in text like "24 hours" or "confidence=7"
    """

    @staticmethod
    def _extract_first_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(m.group(0)) if m else None

    @staticmethod
    def _extract_first_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        m = re.search(r"\d+", str(value))
        return int(m.group(0)) if m else None

    def parse(self, text: str) -> Dict[str, Any]:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            return {"timeline": []}

        header_idx = None
        for i, line in enumerate(lines):
            normalized = line.lower().replace(" ", "")
            if normalized in {"event|time|confidence", "event|time"}:
                header_idx = i
                break

        if header_idx is not None:
            data_lines = lines[header_idx + 1 :]
            header = [h.strip().lower() for h in lines[header_idx].split("|")]
        else:
            data_lines = lines
            header = None

        timeline = []
        for line in data_lines:
            if "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if header is not None:
                row = dict(zip(header, parts))
                event = row.get("event")
                time_val = self._extract_first_number(row.get("time"))
                conf_val = self._extract_first_int(row.get("confidence"))
            else:
                if len(parts) >= 3:
                    event = "|".join(parts[:-2]).strip()
                    time_val = self._extract_first_number(parts[-2])
                    conf_val = self._extract_first_int(parts[-1])
                elif len(parts) == 2:
                    event = parts[0].strip()
                    time_val = self._extract_first_number(parts[1])
                    conf_val = None
                else:
                    continue

            if not event or time_val is None:
                continue

            if conf_val is None:
                conf_val = 5
            conf_val = max(1, min(9, int(conf_val)))

            timeline.append(
                {
                    "event": event,
                    "time": float(time_val),
                    "confidence": conf_val,
                }
            )

        return {"timeline": timeline}


class UpdatedTimelineBSVParser(BaseOutputParser):
    """Robust parser for updated timeline output.

    Preferred format:
      event|time

    Also tolerates:
      event|time|confidence
      missing header / leading prose / code fences

    Any confidence returned here is ignored later; confidence is always copied
    from the original extracted timeline.
    """

    @staticmethod
    def _extract_first_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(m.group(0)) if m else None

    @staticmethod
    def _extract_first_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        m = re.search(r"\d+", str(value))
        return int(m.group(0)) if m else None

    def parse(self, text: str) -> Dict[str, Any]:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        if not lines:
            return {"updated_timeline": []}

        header_idx = None
        for i, line in enumerate(lines):
            normalized = line.lower().replace(" ", "")
            if normalized in {"event|time", "event|time|confidence"}:
                header_idx = i
                break

        if header_idx is not None:
            data_lines = lines[header_idx + 1 :]
            header = [h.strip().lower() for h in lines[header_idx].split("|")]
        else:
            data_lines = lines
            header = None

        updated = []
        for line in data_lines:
            if "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if header is not None:
                row = dict(zip(header, parts))
                event = row.get("event")
                time_val = self._extract_first_number(row.get("time"))
                conf_val = self._extract_first_int(row.get("confidence"))
            else:
                if len(parts) >= 3:
                    event = "|".join(parts[:-2]).strip()
                    time_val = self._extract_first_number(parts[-2])
                    conf_val = self._extract_first_int(parts[-1])
                elif len(parts) == 2:
                    event = parts[0].strip()
                    time_val = self._extract_first_number(parts[1])
                    conf_val = None
                else:
                    continue

            if not event or time_val is None:
                continue

            updated.append(
                {
                    "event": event,
                    "time": float(time_val),
                    "confidence": conf_val,
                }
            )

        return {"updated_timeline": updated}


class State:
    def __init__(self, discharge_summary: str, hadm_id: int, llm: Any = None, llm_endpoint: str = None):
        self.discharge_summary = discharge_summary
        self.hadm_id = hadm_id

        # (event, time, confidence)
        self.timeline: List[Tuple[str, float, int]] = []

        # (event, time, confidence, topk_rows)
        self.timeline_with_topk_rows: List[
            Tuple[str, float, int, List[Tuple[str, str, List[float], float]]]
        ] = []

        # (event, time, confidence)
        self.updated_timeline: List[Tuple[str, float, int]] = []

        self.llm = llm
        self.llm_endpoint = llm_endpoint


OUTPUT_FILE_TO_STATE_ATTR = {
    "timeline.json": "timeline",
    "timeline_with_topk_rows.json": "timeline_with_topk_rows",
    "updated_timeline.json": "updated_timeline",
}


class TimelineAgent:
    def __init__(self, llm=None, llm_endpoint=None, model_basename: Optional[str] = None, omit_tags: List[str] = None):
        self.llm = llm
        self.llm_endpoint = llm_endpoint
        self.model_basename = model_basename or "default"
        self.template_dir = "templates"
        self.omit_tags = omit_tags if omit_tags is not None else ["think"]

        self.timeline_parser = TimelineBSVParser()
        self.updated_timeline_parser = UpdatedTimelineBSVParser()

    def _invoke_llm(self, prompt: str) -> str:
        """Handle LLM invocation either locally or via endpoint."""
        if self.llm_endpoint:
            model_key = (self.model_basename or "").lower()
            request_data: Dict[str, Any] = {
                "model": self.model_basename,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful clinical information extraction assistant. "
                            "Return only raw bar-separated values (BSV) matching the requested schema. "
                            "Do not add explanations, markdown, or prose before or after the table."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 4096,
                "temperature": 0.2,
            }

            if "qwen3.5" in model_key or "qwen35" in model_key:
                request_data["top_p"] = 0.95
                request_data["presence_penalty"] = 0.0
                request_data["extra_body"] = {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                }

            response = requests.post(
                f"{self.llm_endpoint}/v1/chat/completions",
                json=request_data,
                timeout=6000,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        if self.llm:
            return self.llm.invoke(prompt)

        raise ValueError("No LLM or endpoint configured")

    def _load_template(self, template_name: str) -> str:
        template_path = os.path.join(self.template_dir, f"{template_name}.template")
        try:
            with open(template_path, "r") as f:
                return f.read()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Template file not found: {template_path}") from e

    def _strip_omit_tags(self, text: str) -> str:
        cleaned = text

        for tag in self.omit_tags:
            pattern = re.compile(rf"<{tag}>.*?</{tag}>", re.DOTALL | re.IGNORECASE)
            cleaned = pattern.sub("", cleaned)

        cleaned = re.sub(r"```(?:[a-zA-Z0-9_-]+)?", "", cleaned)
        cleaned = cleaned.replace("```", "")

        lines = [line.rstrip() for line in cleaned.splitlines()]
        start_idx = 0
        for i, line in enumerate(lines):
            if "|" in line:
                start_idx = i
                break
        cleaned = "\n".join(lines[start_idx:]).strip()

        return cleaned

    def extract_bsv(self, response: str, parser: BaseOutputParser) -> Optional[Dict[str, Any]]:
        cleaned = self._strip_omit_tags(response)
        try:
            return parser.parse(cleaned)
        except Exception as e:
            print(f"---Cleaned input:---\n{cleaned}\n")
            print(f"Error during parsing: {e}")
            return None

    def run_with_retries(self, prompt: str, parser: BaseOutputParser, n_retries: int = 3) -> Optional[Dict[str, Any]]:
        for _ in range(n_retries):
            response = self._invoke_llm(prompt)
            bsv_data = self.extract_bsv(response, parser)
            if bsv_data:
                return bsv_data
        return None

    def extract_events_with_time(self, state: State, n_retries: int = 3) -> State:
        """Single-step extraction of all events with timestamps and confidence.

        Template required: templates/extract_events_with_time.template
        Intended output:
            event|time|confidence
        """
        template = self._load_template("extract_events_with_time")
        prompt = template.format(discharge_summary=state.discharge_summary)

        for attempt in range(n_retries):
            response = self._invoke_llm(prompt)
            bsv_data = self.extract_bsv(response, self.timeline_parser)
            if bsv_data and bsv_data.get("timeline"):
                state.timeline = [
                    (item["event"], item["time"], item["confidence"])
                    for item in bsv_data["timeline"]
                ]
                return state
            print(f"Attempt {attempt + 1}: failed to extract timeline, retrying...")

        print(f"Failed to extract timeline after {n_retries} attempts")
        return state

    @staticmethod
    def _merge_confidence_from_original(
        original_timeline: List[Tuple[str, float, int]],
        updated_rows: List[Dict[str, Any]],
    ) -> List[Tuple[str, float, int]]:
        """Attach confidence to updated rows by always inheriting it from the original timeline.

        The structured-data update step is only used to revise timestamps. Confidence is kept
        from the original text-extracted timeline and matched by event occurrence order.
        """
        original_conf_by_event: Dict[str, List[int]] = defaultdict(list)
        for event, _time, confidence in original_timeline:
            original_conf_by_event[event].append(confidence)

        used_counts: Dict[str, int] = defaultdict(int)
        merged: List[Tuple[str, float, int]] = []

        for row in updated_rows:
            event = row["event"]
            time = row["time"]

            idx = used_counts[event]
            used_counts[event] += 1

            if idx < len(original_conf_by_event[event]):
                confidence = original_conf_by_event[event][idx]
            else:
                confidence = 5

            merged.append((event, time, int(confidence)))

        return merged

    def update_timeline(self, state: State, n_retries: int = 3, template_name: str = "update_central_timeline") -> State:
        """Use structured retrieval to update timestamps.

        Reuses the existing update template. The prompt includes the original confidence,
        but the updated output only needs event|time. Confidence is always copied from
        the original timeline.
        """
        if not state.timeline:
            print("Cannot update timeline because extracted timeline is empty")
            return state

        if not state.timeline_with_topk_rows:
            state.timeline_with_topk_rows = []
            for event, time, confidence in state.timeline:
                topk_rows = get_topk_struct_events(
                    hadm_id=state.hadm_id,
                    query=event,
                    topk=5,
                    max_times=5,
                )
                state.timeline_with_topk_rows.append((event, time, confidence, topk_rows))

        template = self._load_template(template_name)

        format_strs = []
        for i, (event, time, confidence, topk_rows) in enumerate(state.timeline_with_topk_rows):
            entry_str = (
                f"Event {i + 1}: {event}, time: {time:.2f}, confidence: {confidence}\n"
                "Top-10 EHR rows:\n"
                "name | value | time | similarity"
            )
            for td in topk_rows:
                time_str = " ".join([f"{t:.2f}" for t in td[2]])
                entry_str += f"\n{td[0]} | {td[1]} | {time_str} | {td[3]:.3f}"
            format_strs.append(entry_str)
        timeline_with_topk_rows_str = "\n\n".join(format_strs)

        prompt = template.format(
            central_timeline_with_topk_rows=timeline_with_topk_rows_str,
            discharge_summary=state.discharge_summary,
        )

        for attempt in range(n_retries):
            response = self._invoke_llm(prompt)
            parsed = self.extract_bsv(response, self.updated_timeline_parser)
            if parsed and parsed.get("updated_timeline"):
                state.updated_timeline = self._merge_confidence_from_original(
                    state.timeline,
                    parsed["updated_timeline"],
                )
                return state
            print(f"Attempt {attempt + 1}: failed to update timeline, retrying...")

        print(f"Failed to update timeline after {n_retries} attempts")
        return state


def print_outputs(state: State):
    for i, ((event, time, confidence, topk_rows), (event2, time2, confidence2)) in enumerate(
        zip(state.timeline_with_topk_rows, state.updated_timeline)
    ):
        entry_str = (
            f"Event {i + 1}: {event}, time: {time:.2f}, confidence: {confidence}\n"
            "Top-10 EHR rows:\n"
            "name | value | time | similarity"
        )
        for td in topk_rows:
            time_str = " ".join([f"{t:.2f}" for t in td[2]])
            entry_str += f"\n{td[0]} | {td[1]} | {time_str} | {td[3]:.3f}"
        print(entry_str)
        print(f"-> Updated to: {event2}: {time2:.2f} hours (confidence {confidence2})\n")


def is_non_empty_content(content: Any) -> bool:
    return content not in (None, "", [], {})


def load_json_output(log_folder: str, filename: str) -> Optional[Any]:
    file_path = os.path.join(log_folder, filename)
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
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


def save_outputs(
    state: State,
    log_folder: str,
    only_files: Optional[List[str]] = None,
    skip_empty: bool = False,
    print_summary: bool = True,
    overwrite_non_empty_existing: bool = False,
):
    os.makedirs(log_folder, exist_ok=True)

    output_map = {
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

        with open(os.path.join(log_folder, filename), "w") as f:
            f.write(json.dumps(content, indent=2))

    if print_summary:
        print(f"Outputs saved to: {log_folder}")
        print("Reconstructed Timeline:")
        for event, t, confidence in state.updated_timeline:
            print(f"{event}: {t} hours (confidence {confidence})")

        print("\nDetailed Timeline with Top-K EHR Rows:")
        print_outputs(state)


PIPELINE_STEPS = [
    {
        "name": "timeline",
        "primary_output": "timeline.json",
        "outputs": ["timeline.json"],
        "dependencies": [],
        "runner": "extract_events_with_time",
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
    required_files = [
        "timeline.json",
        "timeline_with_topk_rows.json",
        "updated_timeline.json",
    ]
    return all(load_json_output(log_folder, f) is not None for f in required_files)


def run_graph(
    discharge_summary: str,
    hadm_id: int,
    llm: Any,
    n_retries: int = 3,
    log_folder: str = None,
    llm_endpoint: str = None,
    model_basename: Optional[str] = None,
    omit_tags: List[str] = None,
) -> State:
    """Single-step ablation pipeline: extract full timeline -> update timeline."""
    agent = TimelineAgent(
        llm=llm,
        llm_endpoint=llm_endpoint,
        model_basename=model_basename,
        omit_tags=omit_tags,
    )
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
            dep
            for dep in step["dependencies"]
            if load_json_output(log_folder, dep) is None
            and not is_non_empty_content(getattr(state, OUTPUT_FILE_TO_STATE_ATTR[dep]))
        ]
        if missing_dependencies:
            print(f"Cannot run {step['name']} because these dependency files are missing or empty: {missing_dependencies}")
            break

        print(f"Running {step['name']} because {primary_output} is missing or empty")
        state = getattr(agent, step["runner"])(state, n_retries)
        save_outputs(state, log_folder, only_files=step["outputs"], skip_empty=True, print_summary=False)
        load_existing_outputs_into_state(state, log_folder)

    return state


def infer_hadm_id_from_filename(path: str) -> Optional[int]:
    basename = os.path.splitext(os.path.basename(path))[0]
    if basename.endswith(".txt"):
        basename = os.path.splitext(basename)[0]
    matches = re.findall(r"\d+", basename)
    return int(matches[-1]) if matches else None


def main():
    parser = argparse.ArgumentParser(
        description="Ablation 3: extract events+time+confidence directly, then update with structured EHR"
    )

    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to local GGUF (only used if --llm_endpoint is not provided)",
    )
    parser.add_argument(
        "--llm_endpoint",
        type=str,
        help="Endpoint URL for llama-server (e.g., http://127.0.0.1:8001)",
    )
    parser.add_argument(
        "--model_basename",
        type=str,
        help="Required when using --llm_endpoint (used for request body and log naming)",
    )

    parser.add_argument("--summary_file", type=str, help="Path to a single discharge summary text file (.txt or .txt.gz)")
    parser.add_argument("--summary_directory", type=str, help="Directory containing multiple summaries")
    parser.add_argument("--summary_ext", type=str, default=".txt", help="Extension filter when using --summary_directory")

    parser.add_argument("--hadm_id", type=int, help="HADM_ID for structured retrieval. Required for single-file mode unless inferable.")

    parser.add_argument("--log_folder", type=str, help="Folder to save outputs")
    parser.add_argument("--debug", action="store_true", help="Enable timestamped log folder names")
    parser.add_argument(
        "--omit_tags",
        type=str,
        default="think",
        help="Comma-separated list of tags to remove before parsing (default: think)",
    )
    parser.add_argument("--n_retries", type=int, default=3, help="Retries per step")

    args = parser.parse_args()

    if not args.summary_file and not args.summary_directory:
        raise ValueError("Must provide either --summary_file or --summary_directory")
    if args.summary_file and args.summary_directory:
        raise ValueError("Cannot provide both --summary_file and --summary_directory")

    if args.llm_endpoint:
        if not args.model_basename:
            raise ValueError("When using --llm_endpoint, you must provide --model_basename")
        llm = None
        llm_endpoint = args.llm_endpoint
        model_basename = args.model_basename
    else:
        if not args.model_path:
            raise ValueError("Provide --model_path if not using --llm_endpoint")
        llm = LlamaCpp(model_path=args.model_path, n_gpu_layers=9999, n_ctx=16000)
        llm_endpoint = None
        model_basename = os.path.splitext(os.path.basename(args.model_path))[0]

    omit_tags = [tag.strip() for tag in args.omit_tags.split(",") if tag.strip()]

    def read_summary(path: str) -> str:
        if path.endswith(".gz"):
            import gzip
            try:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                with gzip.open(path, "rt", encoding="unicode_escape") as f:
                    return f.read()
        else:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="unicode_escape") as f:
                    return f.read()

    if args.summary_file:
        basename = os.path.basename(args.summary_file)
        if basename.endswith(".gz"):
            basename = os.path.splitext(os.path.splitext(basename)[0])[0]
        else:
            basename = os.path.splitext(basename)[0]

        summary = read_summary(args.summary_file)
        hadm_id = args.hadm_id if args.hadm_id is not None else infer_hadm_id_from_filename(args.summary_file)
        if hadm_id is None:
            raise ValueError("Could not determine hadm_id. Please provide --hadm_id.")

        log_folder = (
            args.log_folder
            if args.log_folder
            else f"logs/{basename}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" if args.debug else f"logs/{basename}_{model_basename}"
        )
        os.makedirs(log_folder, exist_ok=True)

        if should_skip_processing(log_folder):
            print(f"Skipping processing as outputs already exist in {log_folder}")
            return

        state = run_graph(
            discharge_summary=summary,
            hadm_id=hadm_id,
            llm=llm,
            n_retries=args.n_retries,
            log_folder=log_folder,
            llm_endpoint=llm_endpoint,
            model_basename=model_basename,
            omit_tags=omit_tags,
        )
        if state.timeline or state.updated_timeline:
            save_outputs(state, log_folder)
        else:
            print(f"No non-empty outputs produced for {basename}; not saving empty JSON files.")
        return

    if not os.path.isdir(args.summary_directory):
        raise ValueError(f"Directory not found: {args.summary_directory}")

    files = [
        f
        for f in os.listdir(args.summary_directory)
        if f.endswith(args.summary_ext) or (args.summary_ext == ".txt" and f.endswith(".txt.gz"))
    ]
    random.shuffle(files)

    for filename in files:
        filepath = os.path.join(args.summary_directory, filename)
        summary = read_summary(filepath)

        basename = filename
        if basename.endswith(".gz"):
            basename = os.path.splitext(os.path.splitext(basename)[0])[0]
        else:
            basename = os.path.splitext(basename)[0]

        hadm_id = args.hadm_id if args.hadm_id is not None else infer_hadm_id_from_filename(filename)
        if hadm_id is None:
            print(f"Skipping {filename}: could not infer hadm_id and none was provided")
            continue

        log_folder = (
            args.log_folder
            if args.log_folder
            else f"logs/{basename}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" if args.debug else f"logs/{basename}_{model_basename}"
        )
        os.makedirs(log_folder, exist_ok=True)

        if should_skip_processing(log_folder):
            print(f"Skipping {filename} (outputs exist in {log_folder})")
            continue

        print(f"\nProcessing: {filename}")
        state = run_graph(
            discharge_summary=summary,
            hadm_id=hadm_id,
            llm=llm,
            n_retries=args.n_retries,
            log_folder=log_folder,
            llm_endpoint=llm_endpoint,
            model_basename=model_basename,
            omit_tags=omit_tags,
        )
        if state.timeline or state.updated_timeline:
            save_outputs(state, log_folder)
        else:
            print(f"No non-empty outputs produced for {basename}; not saving empty JSON files.")


if __name__ == "__main__":
    main()
