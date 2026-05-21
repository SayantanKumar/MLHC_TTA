import os
import sys
import json
import time
import re
import collections
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm

# File paths for MIMIC-III and IV
# m3_dir = '~/workspace/mimic3/csv'
# m4_dir = '~/workspace/mimiciv/physionet.org/files/mimiciv/3.1'
m3_struct_path = '/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/timeline_i2b2_5col_new.csv'
m4_struct_path = '/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/timeline_mimiciv_5col_new.csv'

# df_admin_m3 = pd.read_csv(os.path.join(m3_dir, 'ADMISSIONS.csv'), low_memory=False)
# df_admin_m4 = pd.read_csv(os.path.join(m4_dir, 'hosp', 'admissions.csv'), low_memory=False)

df_admin_m3 = pd.read_csv("/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/MIMICIII_ADMISSIONS.csv", low_memory=False)
df_admin_m4 = pd.read_csv("/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/MIMICIV_ADMISSIONS.csv", low_memory=False)

df_struct_m3 = pd.read_csv(m3_struct_path, low_memory=False)
df_struct_m4 = pd.read_csv(m4_struct_path, low_memory=False)

# MIMIC-III HADM_IDs that DST is observed
m3_dst_hadm_ids = [100831, 101280, 103859, 125310, 133706, 145785]

# MIMIC-IV subject_id to HADM_ID mapping for selected patients
m4_pt_to_hadm = {
    10056200: 28143635,
    12051619: 24799933,
    13528310: 25054328,
    14216696: 28837354,
    16266622: 23933519
}

# Cache for struct data
struct_data = collections.defaultdict(collections.defaultdict)

# Sentence Transformer embedding extraction
model_id = "mohammadkhodadad/MedTE-cl15-step-8000" # MedTE model
tokenizer = None
model = None

def get_sentence_embedding_tokenizer_model():
    global tokenizer, model
    if model is None:
        model = transformers.AutoModel.from_pretrained(model_id)
        model.eval()
        model.cuda()
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return tokenizer, model

def extract_sentence_embeddings(texts, batch_size=256, show_progress=True):
    """
    Extract sentence embeddings for a list of texts (mean-pool over all tokens).
    """
    tokenizer, model = get_sentence_embedding_tokenizer_model()

    embeddings = []
    if show_progress:
        pbar = tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
            
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        masked_embeddings = last_hidden_state * mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = mask.sum(dim=1)
        mean_pooled = sum_embeddings / sum_mask  # shape: [batch_size, hidden_dim]
        embeddings.append(mean_pooled.cpu())
        if show_progress:
            pbar.update(1)
    embeddings = torch.cat(embeddings)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

# Time functions
def convert_row_time(time_str, offset_hours=5, hadm_id=None, to_datetime=False):
    """
    Convert time str in 5-col files into datetime object or (correct) time str.
    MIMIC-III: Subtract offset_hours hours and output without T and Z.
               e.g. 2072-12-29T05:00:00Z	-> 2072-12-29 00:00:00
    MIMIC-IV: Simply detach T and Z and interpret as ET (New York time).
    """
    if hadm_id < 1000000:
        offset_hours = 4 if hadm_id in m3_dst_hadm_ids else 5
        dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone(timedelta(hours=-offset_hours)))
    else:
        dt = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=ZoneInfo('America/New_York'))
    if to_datetime:
        return dt
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def get_admission_time(hadm_id):
    # Get admission time for hadm_id
    if hadm_id < 1000000:  # MIMIC-III
        admit_time_str = df_admin_m3[df_admin_m3['HADM_ID'] == hadm_id].iloc[0]['ADMITTIME']
        admit_time = datetime.strptime(admit_time_str, '%Y-%m-%d %H:%M:%S')
        admit_time = admit_time.replace(tzinfo=timezone(timedelta(hours=-5)))
    else:  # MIMIC-IV
        admit_time_str = df_admin_m4[df_admin_m4['subject_id'] == hadm_id].iloc[0]['admittime']
        admit_time = datetime.strptime(admit_time_str, '%Y-%m-%d %H:%M:%S')
        admit_time = admit_time.replace(tzinfo=ZoneInfo('America/New_York'))
    return admit_time

def get_struct_embeddings(event_names, event_values, batch_size=256):
    # Get embeddings for event names and values
    texts = [f"{event_name},{value}" for event_name, value in zip(event_names, event_values)]
    embeds = extract_sentence_embeddings(texts, batch_size=batch_size, show_progress=False)
    embeds = embeds.cuda()
    return embeds

def _equal(a, b):
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except Exception:
        pass
    try: 
        return a == b
    except Exception:
        return False

def merge_struct_by_event(event_names, event_values, event_times):
    # Merge events with same name and values by concatenating their times
    # Assumes event_names, event_values, event_times are sorted by (name, value, time)
    merged_names = []
    merged_values = []
    merged_times = []

    for name, value, time in zip(event_names, event_values, event_times):
        if merged_names and merged_names[-1] == name and _equal(merged_values[-1], value):
            merged_times[-1].append(time)
        else:
            merged_names.append(name)
            merged_values.append(value)
            merged_times.append([time])

    return merged_names, merged_values, merged_times

def get_struct_data(hadm_id, merge_events=False):
    # Get (event names, values, times, embeddings) for hadm_id
    global struct_data
    if hadm_id in struct_data and merge_events in struct_data[hadm_id]:
        return struct_data[hadm_id][merge_events]

    # Get admission time, struct data
    admit_dt = get_admission_time(hadm_id)
    if hadm_id < 1000000:
        df_struct_patient = df_struct_m3[df_struct_m3['hid'] == hadm_id].copy().reset_index(drop=True)
    else:
        df_struct_patient = df_struct_m4[df_struct_m4['hid'] == m4_pt_to_hadm[hadm_id]].copy().reset_index(drop=True)
    
    if merge_events:
        df_struct_patient.sort_values(by=['event', 'value', 't'], inplace=True)

    event_names, event_values, event_times = [], [], []
    for _, row in df_struct_patient.iterrows():
        row_dt = convert_row_time(row.t, hadm_id=hadm_id, to_datetime=True)
        time_from_admit = (row_dt - admit_dt).total_seconds() / 3600.0  # in hours
        event_names.append(row.event)
        event_values.append(row.value)
        event_times.append(time_from_admit)

    if merge_events:
        event_names, event_values, event_times = merge_struct_by_event(event_names, event_values, event_times)
    event_embeds = get_struct_embeddings(event_names, event_values)

    ret = (event_names, event_values, event_times, event_embeds)
    struct_data[hadm_id][merge_events] = ret
    return ret

def get_topk_struct_events(hadm_id, query, topk=20, max_times=20, value_ndigits=4):
    event_names, event_values, event_times, event_embeds = get_struct_data(hadm_id, merge_events=True)
    query_embed = extract_sentence_embeddings([query], show_progress=False)[0].cuda()
    similarities = (event_embeds @ query_embed).cpu()
    del query_embed

    topk_indices = torch.topk(similarities, k=topk).indices.tolist()
    ret = []
    for idx in topk_indices:
        ret.append((
            event_names[idx],
            round_float_in_str(event_values[idx], ndigits=value_ndigits),
            event_times[idx] if max_times <= 0 else reduce_event_times(event_times[idx], max_times=max_times),
            similarities[idx].item()
        ))
    return ret
    
def reduce_event_times(event_times, max_times=20):
    # If there are too many event times, reduce them by sampling with equal intervals, leaving first and last times
    cnt = len(event_times)
    if cnt <= max_times:
        return event_times
    indices = [cnt * i // (max_times-1) for i in range(max_times-1)] + [cnt-1]
    reduced_times = [event_times[i] for i in indices]
    return reduced_times

def round_float_in_str(x, ndigits=3):
    if pd.isna(x):
        return x
    if isinstance(x, str):
        s = x.strip()
        # find if the string looks like a float
        m = re.fullmatch(r"[+-]?\d*\.\d+", s)
        if m:
            try:
                decimals = len(s.split('.')[-1])
                if decimals <= ndigits:
                    return x  # keep as is
                num = float(s)
                rounded = round(num, ndigits)
                if re.fullmatch(r"[+-]?\.\d+", s):
                    sign = '-' if s.startswith('-') else ('+' if s.startswith('+') else '')
                    s_new = f"{sign}.{str(abs(rounded)).split('.')[-1]}"
                else:
                    s_new = str(rounded)
                return s_new
            except ValueError:
                return x
        else:
            return x
    return x