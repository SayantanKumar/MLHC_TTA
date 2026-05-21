# Templates

This folder contains prompt templates for extracting clinical events from case reports or discharge summaries and assigning each event a timestamp in hours. The templates correspond to the study workflow in *Text Knows What, Tables Know When: Clinical Timeline Reconstruction via Retrieval-Augmented Multimodal Alignment*. They are designed for textual temporal annotation workflows where an LLM produces strict BSV (Bar-Separated Values) outputs that can be parsed by downstream scripts.

All timestamps are relative to time zero:

- Use admission as time `0` when admission is present.
- If admission is not stated, use the case presentation time as time `0`.
- Events before time zero use negative hours.
- Events after time zero use positive hours.
- Duration events should use the start of the interval.

## Template Summary

| Template | Purpose | Inputs | Output |
| --- | --- | --- | --- |
| `extract_central_events.template` | Extract timeline anchor events, also called central or referent events. | `{discharge_summary}` | `event` |
| `compute_time_distances.template` | Estimate pairwise time differences between central events. | `{central_events}`, `{discharge_summary}` | `event1|event2|e2_minus_e1|confidence` |
| `reconstruct_central_timeline.template` | Convert central-event pairwise distances into absolute central-event times. | `{central_events}`, `{time_distances}`, `{discharge_summary}` | `event|time` |
| `extract_non_central_events.template` | Extract all other patient-related events and place them relative to a central event. | `{central_events}`, `{discharge_summary}` | `event|central_event|relative_time|confidence` |
| `reconstruct_timeline.template` | Combine central and non-central events into a complete absolute timeline. | `{discharge_summary}`, `{central_timeline}`, `{non_central_events}` | `event|time` |
| `singlestep_extract_events_with_time.template` | Extract all events and absolute timestamps in one prompt. | `{discharge_summary}` | `event|time|confidence` |
| `update_central_timeline.template` | Calibrate either the central scaffold or the full timeline using top-k retrieved structured EHR rows. | `{discharge_summary}`, `{central_timeline_with_topk_rows}` | `event|time` |

## Recommended Workflows

### Study Default: Multistep Multimodal Pipeline

This is the main workflow described in the paper. It decomposes timeline reconstruction into a central-event scaffold, structured-EHR calibration of that scaffold, full timeline assembly, and final structured-EHR refinement.

1. Run `extract_central_events.template` to identify temporally informative anchor events.
2. Run `compute_time_distances.template` to estimate pairwise relative offsets among central events.
3. Run `reconstruct_central_timeline.template` to convert those offsets into an initial absolute central scaffold.
4. Retrieve the top-k structured EHR rows for each central event, then run `update_central_timeline.template` to produce a calibrated central scaffold.
5. Run `extract_non_central_events.template` to extract remaining events and attach each one to a central event with a relative offset.
6. Run `reconstruct_timeline.template` to combine the calibrated central scaffold with non-central events into an initial complete timeline.
7. Retrieve the top-k structured EHR rows for every event in the complete timeline, then run `update_central_timeline.template` again to produce the final patient trajectory.

Although the file is named `update_central_timeline.template`, the paper uses this same structured-EHR update prompt for both central-scaffold calibration and final full-timeline refinement.

### Text-Only Multistep Baseline

The unimodal text-only multistep baseline follows the same central/non-central decomposition, but skips both structured-EHR update calls:

1. Run central-event extraction, pairwise distance estimation, and central timeline reconstruction.
2. Run non-central event extraction and full timeline reconstruction.
3. Use the resulting `event|time` table directly.

This baseline preserves the graph-style scaffold but does not use retrieved structured rows as temporal anchors.

### Ablation Variants

The paper compares several variants to isolate where structured evidence helps:

| Variant | Workflow difference |
| --- | --- |
| `Update both central and final timeline` | Default multimodal pipeline; run `update_central_timeline.template` after the central scaffold and again after full assembly. |
| `Update central timeline only` | Run the structured update after central reconstruction, then assemble the full timeline without final refinement. |
| `Update final timeline only` | Skip central-scaffold calibration, assemble the full timeline, then run the structured update once on all events. |
| `Single-step` | Remove the central/non-central decomposition and extract all events with timestamps in one prompt. |

### Single-Step Baseline

Use `singlestep_extract_events_with_time.template` for the one-prompt baseline:

1. Fill `{discharge_summary}`.
2. Ask the model for the strict BSV output.
3. Parse the returned `event|time|confidence` table.

This path is simpler, but it does not expose the central-event scaffold or the relative-time reasoning used by the multistep pipeline.

### Structured-EHR Time Calibration Input

Use `update_central_timeline.template` whenever an existing event timeline should be calibrated with structured EHR evidence. In the default study workflow, this happens twice:

- after `reconstruct_central_timeline.template`, using central events only
- after `reconstruct_timeline.template`, using all events in the complete timeline

The input `{central_timeline_with_topk_rows}` should include each event, its initial time, and the top-10 nearest structured rows in BSV form:

```text
Event 1: <event>, time: <initial_time>
Top-10 EHR rows:
name | value | time | similarity
<row name> | <row value> | <row time(s)> | <similarity>
```

The template instructs the model to change an event time only when the structured evidence is clinically relevant.

## Placeholders

The templates use simple brace-delimited placeholders. Replace each placeholder before sending the prompt to a model.

| Placeholder | Meaning |
| --- | --- |
| `{discharge_summary}` | The full case report, discharge summary, or note text. |
| `{central_events}` | A newline-separated list or table of central timeline anchor events. |
| `{time_distances}` | Output from `compute_time_distances.template`. |
| `{central_timeline}` | Output from `reconstruct_central_timeline.template`. |
| `{non_central_events}` | Output from `extract_non_central_events.template`. |
| `{central_timeline_with_topk_rows}` | An existing central or full event timeline with timestamps plus retrieved structured EHR rows for each event. |

## Output Format

All templates ask for raw BSV output:

- Use `|` as the delimiter.
- Include exactly one header line.
- Do not include Markdown code fences.
- Do not include prose explanations.
- Do not include blank lines.
- Use numeric hour values for all time fields.
- Use numeric confidence values when a confidence column is requested.

Some examples in the templates show spaces around `|`, such as `event | time`. For parsing, it is safest to trim whitespace around each field.

## Event Extraction Rules

Across the templates, events are patient-related findings or mentions that can be temporally located. The prompts ask the model to include:

- diagnoses
- symptoms and signs
- labs and imaging findings
- procedures
- medications and interventions
- discontinuation or termination events
- pertinent negatives
- demographic or baseline patient descriptors when temporally anchored

Conjunctive phrases should be split into component events. For example, `fever and rash` should become separate `fever` and `rash` events. Contextual phrases may be copied across split events when needed for the event text to stand alone.

## Confidence Scores

Confidence scores represent certainty in the timestamp:

| Score | Meaning |
| --- | --- |
| `1-3` | Low confidence; based mainly on indirect evidence. |
| `4-6` | Moderate confidence; some direct evidence is available. |
| `7-9` | High confidence; timing is explicitly documented. |

`extract_non_central_events.template` allows `0` as the lowest confidence score. Other templates describe the scale as `1-9`.

## Practical Notes

- Keep model outputs strict. Extra prose, Markdown fences, or blank lines can break BSV parsing.
- Preserve original text spans when possible, with minimal normalization.
- Prefer numeric estimates over missing values; the templates explicitly discourage null or undefined time fields.
- Review low-confidence events manually if they materially affect downstream evaluation.
- If using clinical notes with protected health information, follow the privacy and security requirements of your environment before sending data to any model endpoint.
