import os
import json
import math
import re
import unicodedata
from difflib import SequenceMatcher


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def unwrap_json(data, key):
    if isinstance(data, dict):
        return data.get(key, [])
    return data


def find_file(folder, candidates):
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def normalize_value(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def times_equal(t1, t2, tol=1e-6):
    t1 = normalize_value(t1)
    t2 = normalize_value(t2)

    if t1 is None and t2 is None:
        return True
    if t1 is None or t2 is None:
        return False

    try:
        return abs(float(t1) - float(t2)) <= tol
    except (TypeError, ValueError):
        return t1 == t2


def normalize_event_text(text):
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    dash_map = {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
    }
    for src, dst in dash_map.items():
        text = text.replace(src, dst)

    text = text.replace("…", "...")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def events_match(event1, event2, fuzzy_threshold=0.88):
    e1 = normalize_event_text(event1)
    e2 = normalize_event_text(event2)

    if e1 == e2:
        return True

    # allow one to be a shortened/truncated version of the other
    if e1 and e2 and (e1 in e2 or e2 in e1):
        return True

    # allow near-match
    return SequenceMatcher(None, e1, e2).ratio() >= fuzzy_threshold


def confidence_to_certain_central(confidence):
    """
    Central timeline rule:
    1-5 -> 0
    6-9 -> 1
    NA  -> 1
    """
    if confidence is None or confidence == "NA":
        return 1
    if 1 <= confidence <= 5:
        return 0
    if 6 <= confidence <= 9:
        return 1
    return 1


def confidence_to_certain_timeline(confidence):
    """
    Full timeline rule:
    1-5 -> 0
    6-9 -> 1
    NA  -> NA
    """
    if confidence is None or confidence == "NA":
        return "NA"
    if 1 <= confidence <= 5:
        return 0
    if 6 <= confidence <= 9:
        return 1
    return "NA"


def percent_flag_is_one(rows, flag_name):
    """
    Ignores rows where flag is NA.
    """
    valid = [row for row in rows if row.get(flag_name) != "NA"]
    if not valid:
        return 0.0
    count_1 = sum(1 for row in valid if row.get(flag_name) == 1)
    return 100.0 * count_1 / len(valid)


def build_central_confidence_map(time_distances):
    conf_map = {}
    for row in time_distances:
        if len(row) < 4:
            continue
        _, event2, _, confidence = row
        if event2 not in conf_map:
            conf_map[event2] = confidence
        else:
            conf_map[event2] = max(conf_map[event2], confidence)
    return conf_map


def build_non_central_confidence_map(non_central_events):
    conf_map = {}
    for row in non_central_events:
        if len(row) < 4:
            continue
        non_central_event, _, _, confidence = row
        if non_central_event not in conf_map:
            conf_map[non_central_event] = confidence
        else:
            conf_map[non_central_event] = max(conf_map[non_central_event], confidence)
    return conf_map


def validate_timeline_alignment(
    original_timeline,
    updated_timeline,
    timeline_name="timeline",
    strict_event_match=False,
):
    if len(original_timeline) != len(updated_timeline):
        raise ValueError(
            f"{timeline_name}: original and updated timelines have different lengths "
            f"({len(original_timeline)} vs {len(updated_timeline)})"
        )

    for i, (orig_row, upd_row) in enumerate(zip(original_timeline, updated_timeline)):
        if len(orig_row) < 2:
            raise ValueError(f"{timeline_name}: original row {i} is malformed: {orig_row}")
        if len(upd_row) < 2:
            raise ValueError(f"{timeline_name}: updated row {i} is malformed: {upd_row}")

        orig_event = orig_row[0]
        upd_event = upd_row[0]

        if not events_match(orig_event, upd_event):
            msg = (
                f"{timeline_name}: event mismatch at row {i}: "
                f"original={orig_event!r} updated={upd_event!r}"
            )
            if strict_event_match:
                raise ValueError(msg)
            print(f"Warning: {msg} -- proceeding by row order")


def build_updated_time_lookup(original_timeline, updated_timeline, timeline_name="timeline"):
    """
    Returns event -> list of (original_time, updated_time, certain_EHR)
    preserving repeated events by occurrence order.
    """
    validate_timeline_alignment(
        original_timeline,
        updated_timeline,
        timeline_name=timeline_name,
    )

    event_map = {}
    for orig_row, upd_row in zip(original_timeline, updated_timeline):
        event = orig_row[0]
        original_time = orig_row[1]
        updated_time = upd_row[1]
        certain_ehr = 0 if times_equal(original_time, updated_time) else 1

        event_map.setdefault(event, []).append({
            "original_time": original_time,
            "updated_time": updated_time,
            "certain_EHR": certain_ehr,
        })

    return event_map


def format_central_events(central_events):
    return [{"central_event": event} for event in central_events]


def format_time_distances(time_distances):
    rows = []
    for row in time_distances:
        if len(row) < 4:
            continue
        event1, event2, e2_minus_e1, confidence = row
        rows.append({
            "event1": event1,
            "event2": event2,
            "e2_minus_e1": e2_minus_e1,
            "confidence": confidence,
        })
    return rows


def format_non_central_events(non_central_events):
    rows = []
    for row in non_central_events:
        if len(row) < 4:
            continue
        non_central_event, central_event, relative_time, confidence = row
        rows.append({
            "non_central_event": non_central_event,
            "central_event": central_event,
            "relative_time": relative_time,
            "confidence": confidence,
        })
    return rows


def build_central_event_timeline(central_timeline, time_distances, updated_central_timeline):
    """
    Output:
    event | original_time | updated_time | confidence | certain | certain_EHR
    """
    central_conf_map = build_central_confidence_map(time_distances)
    time_map = build_updated_time_lookup(
        central_timeline,
        updated_central_timeline,
        timeline_name="central_timeline",
    )

    event_seen_count = {}
    rows = []

    for row in central_timeline:
        if len(row) < 2:
            continue

        event, _ = row
        idx = event_seen_count.get(event, 0)
        event_seen_count[event] = idx + 1

        time_info = time_map[event][idx]
        confidence = central_conf_map.get(event, "NA")
        certain = confidence_to_certain_central(confidence)

        rows.append({
            "event": event,
            "original_time": time_info["original_time"],
            "updated_time": time_info["updated_time"],
            "confidence": confidence,
            "certain": certain,
            "certain_EHR": time_info["certain_EHR"],
        })

    return rows


def build_full_timeline(timeline, non_central_events, time_distances, updated_timeline):
    """
    Output:
    event | original_time | updated_time | confidence | certain | certain_EHR

    Rules:
    - if event is a non-central event and non_central_events has confidence, use it
    - else if event is in central time_distances, use that confidence
    - else confidence = NA

    certain:
    - for timeline only, confidence NA -> certain NA
    """
    non_central_conf_map = build_non_central_confidence_map(non_central_events) if non_central_events else {}
    central_conf_map = build_central_confidence_map(time_distances)
    time_map = build_updated_time_lookup(
        timeline,
        updated_timeline,
        timeline_name="timeline",
    )

    event_seen_count = {}
    rows = []

    for row in timeline:
        if len(row) < 2:
            continue

        event, _ = row
        idx = event_seen_count.get(event, 0)
        event_seen_count[event] = idx + 1

        time_info = time_map[event][idx]

        if event in non_central_conf_map:
            confidence = non_central_conf_map[event]
        elif event in central_conf_map:
            confidence = central_conf_map[event]
        else:
            confidence = "NA"

        certain = confidence_to_certain_timeline(confidence)

        rows.append({
            "event": event,
            "original_time": time_info["original_time"],
            "updated_time": time_info["updated_time"],
            "confidence": confidence,
            "certain": certain,
            "certain_EHR": time_info["certain_EHR"],
        })

    return rows


def print_central_events(rows):
    print("central_event")
    for r in rows:
        print(r["central_event"])


def print_time_distances(rows):
    print("time_distances")
    print("event1 | event2 | e2_minus_e1 | confidence")
    for r in rows:
        print(f"{r['event1']} | {r['event2']} | {r['e2_minus_e1']} | {r['confidence']}")


def print_central_event_timeline(rows):
    print("central_event_timeline")
    print("event | original_time | updated_time | confidence | certain | certain_EHR")
    for r in rows:
        print(
            f"{r['event']} | {r['original_time']} | {r['updated_time']} | "
            f"{r['confidence']} | {r['certain']} | {r['certain_EHR']}"
        )


def print_non_central_events(rows):
    print("non_central_events")
    print("non_central_event | central_event | relative_time | confidence")
    for r in rows:
        print(
            f"{r['non_central_event']} | {r['central_event']} | "
            f"{r['relative_time']} | {r['confidence']}"
        )


def print_full_timeline(rows):
    print("timeline")
    print("event | original_time | updated_time | confidence | certain | certain_EHR")
    for r in rows:
        print(
            f"{r['event']} | {r['original_time']} | {r['updated_time']} | "
            f"{r['confidence']} | {r['certain']} | {r['certain_EHR']}"
        )


def process_case_folder(case_folder):
    central_events_path = find_file(case_folder, [
        "central_events.json",
        "1_central_events.json",
    ])
    time_distances_path = find_file(case_folder, [
        "time_distances.json",
        "2_time_distances.json",
    ])
    central_timeline_path = find_file(case_folder, [
        "central_timeline.json",
        "3_central_timeline.json",
    ])
    non_central_events_path = find_file(case_folder, [
        "non_central_events.json",
        "4_non_central_events.json",
    ])
    updated_central_timeline_path = find_file(case_folder, [
        "updated_central_timeline.json",
        "5_updated_central_timeline.json",
    ])
    timeline_path = find_file(case_folder, [
        "timeline.json",
        "6_timeline.json",
    ])
    updated_timeline_path = find_file(case_folder, [
        "updated_timeline.json",
        "7_updated_timeline.json",
    ])

    required = {
        "central_events": central_events_path,
        "time_distances": time_distances_path,
        "central_timeline": central_timeline_path,
        "updated_central_timeline": updated_central_timeline_path,
        "timeline": timeline_path,
        "updated_timeline": updated_timeline_path,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        print(f"Skipping {os.path.basename(case_folder)}: missing {', '.join(missing)}")
        return None

    central_events = unwrap_json(load_json(central_events_path), "central_events")
    time_distances = unwrap_json(load_json(time_distances_path), "time_distances")
    central_timeline = unwrap_json(load_json(central_timeline_path), "central_timeline")

    non_central_events = []
    if non_central_events_path is not None:
        non_central_events = unwrap_json(load_json(non_central_events_path), "non_central_events")

    updated_central_timeline = unwrap_json(
        load_json(updated_central_timeline_path),
        "updated_central_timeline",
    )
    timeline = unwrap_json(load_json(timeline_path), "timeline")
    updated_timeline = unwrap_json(load_json(updated_timeline_path), "updated_timeline")

    central_event_rows = format_central_events(central_events)
    time_distance_rows = format_time_distances(time_distances)
    non_central_event_rows = format_non_central_events(non_central_events)

    central_event_timeline_rows = build_central_event_timeline(
        central_timeline=central_timeline,
        time_distances=time_distances,
        updated_central_timeline=updated_central_timeline,
    )

    full_timeline_rows = build_full_timeline(
        timeline=timeline,
        non_central_events=non_central_events,
        time_distances=time_distances,
        updated_timeline=updated_timeline,
    )

    result = {
        "central_events": central_event_rows,
        "time_distances": time_distance_rows,
        "central_event_timeline": central_event_timeline_rows,
        "non_central_events": non_central_event_rows,
        "timeline": full_timeline_rows,
        "central_event_timeline_percent_certain_1": percent_flag_is_one(
            central_event_timeline_rows, "certain"
        ),
        "central_event_timeline_percent_certain_EHR_1": percent_flag_is_one(
            central_event_timeline_rows, "certain_EHR"
        ),
        "timeline_percent_certain_1": percent_flag_is_one(
            full_timeline_rows, "certain"
        ),
        "timeline_percent_certain_EHR_1": percent_flag_is_one(
            full_timeline_rows, "certain_EHR"
        ),
    }

    return result


#########################################################
#/Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/multimodal_simple

def main(parent_folder):
#parent_folder = "/Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/LLM_TTS/multistep/DeepSeek_R1/"

    summary = []

    case_folders = [
        os.path.join(parent_folder, name)
        for name in sorted(os.listdir(parent_folder))
        if os.path.isdir(os.path.join(parent_folder, name))
    ]

    for case_folder in case_folders:
        case_name = os.path.basename(case_folder)

        try:
            result = process_case_folder(case_folder)
        except ValueError as e:
            print(f"Skipping {case_name}: validation failed - {e}")
            continue

        if result is None:
            continue

        print("\n" + "=" * 120)
        print(f"FILE: {case_name}\n")

        print_central_events(result["central_events"])
        print()
        print_time_distances(result["time_distances"])
        print()
        print_central_event_timeline(result["central_event_timeline"])
        print(f"\n% certain = 1 (central_event_timeline): {result['central_event_timeline_percent_certain_1']:.2f}%")
        print(f"% certain_EHR = 1 (central_event_timeline): {result['central_event_timeline_percent_certain_EHR_1']:.2f}%")
        print()
        print_non_central_events(result["non_central_events"])
        print()
        print_full_timeline(result["timeline"])
        print(f"\n% certain = 1 (timeline): {result['timeline_percent_certain_1']:.2f}%")
        print(f"% certain_EHR = 1 (timeline): {result['timeline_percent_certain_EHR_1']:.2f}%")

        output_path = os.path.join(case_folder, "structured_timelines_with_certain_and_ehr.json")
        save_json(result, output_path)

        summary.append({
            "file": case_name,
            "central_event_timeline_percent_certain_1": round(result["central_event_timeline_percent_certain_1"], 2),
            "central_event_timeline_percent_certain_EHR_1": round(result["central_event_timeline_percent_certain_EHR_1"], 2),
            "timeline_percent_certain_1": round(result["timeline_percent_certain_1"], 2),
            "timeline_percent_certain_EHR_1": round(result["timeline_percent_certain_EHR_1"], 2),
            "n_central_timeline_events": len(result["central_event_timeline"]),
            "n_non_central_timeline_events": len(result["non_central_events"]),
            "n_timeline_events": len(result["timeline"]),
        })

    summary_path = os.path.join(parent_folder, "certain_summary_with_ehr.json")
    save_json(summary, summary_path)

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("file | central_%certain | central_%certain_EHR | timeline_%certain | timeline_%certain_EHR | n_central | n_noncentral | n_timeline")
    for row in summary:
        print(
            f"{row['file']} | "
            f"{row['central_event_timeline_percent_certain_1']:.2f} | "
            f"{row['central_event_timeline_percent_certain_EHR_1']:.2f} | "
            f"{row['timeline_percent_certain_1']:.2f} | "
            f"{row['timeline_percent_certain_EHR_1']:.2f} | "
            f"{row['n_central_timeline_events']} | "
            f"{row['n_non_central_timeline_events']} | "
            f"{row['n_timeline_events']}"
        )

    print(f"\nSaved summary to: {summary_path}")

if __name__ == "__main__":
    parent_folder = "/Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/LLM_TTS/multistep/glm5/"
    main(parent_folder)
