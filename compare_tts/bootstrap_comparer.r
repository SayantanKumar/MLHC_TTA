#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(jsonlite)
  library(lubridate)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

check_directory_exists <- function(path, description) {
  if (!dir.exists(path)) stop(paste(description, "not found at:", path))
}

# --- Source compare_tts.r without executing its CLI block ---
# We keep using YOUR exact metric + prep functions (prepare_match_data, generate_match_stats, etc.)
source_compare_tts_functions_only <- function(path) {
  if (!file.exists(path)) stop(paste0("Cannot find compare_tts file at: ", path))
  lines <- readLines(path, warn = FALSE)

  # Drop the CLI runner block at the bottom.
  # Your repo version may include either:
  #   (A) a comment marker, OR
  #   (B) a plain `if (!interactive()) { ... }` block.
  # Allow leading whitespace in case the file is indented.
  marker <- grep("^\\s*#\\s*Run the comparison if not in interactive mode", lines)
  if (length(marker) == 0) {
    marker <- grep("^\\s*if\\s*\\(\\s*!interactive\\(\\)\\s*\\)\\s*\\{", lines)
  }
  if (length(marker) > 0) {
    lines <- lines[seq_len(marker[1] - 1)]
  }

  # Evaluate into the global env (so functions are available)
  eval(parse(text = paste(lines, collapse = "\n")), envir = .GlobalEnv)
}

# --- config loader (bootstrap-specific) ---
read_bootstrap_config <- function(config_path) {
  if (!file.exists(config_path)) stop(paste("Config file not found at:", config_path))
  config <- fromJSON(config_path)

  required_fields <- c("man_loc", "ref_locs", "pilot_names", "out", "distance", "bootstrap")
  missing_fields <- setdiff(required_fields, names(config))
  if (length(missing_fields) > 0) {
    stop(paste("Missing required config fields:", paste(missing_fields, collapse = ", ")))
  }

  if (length(config$ref_locs) != length(config$pilot_names)) {
    stop("ref_locs and pilot_names must have the same length")
  }

  check_directory_exists(config$man_loc, "Manual annotation directory")
  sapply(config$ref_locs, function(x) check_directory_exists(x, "Reference directory"))

  # defaults compatible with your pipeline
  config$man_suffix <- config$man_suffix %||% ".csv"
  config$ref_suffix <- config$ref_suffix %||% ".csv"
  config$man_header <- config$man_header %||% FALSE
  config$ref_header <- config$ref_header %||% FALSE
  config$approach   <- config$approach %||% "strcmp"
  config$use_lap    <- config$use_lap %||% TRUE

  # Output locations
  # - folder: base output directory
  # - relfolder: either a relative folder name (e.g., "matches") OR an absolute path
  config$out$folder <- config$out$folder %||% paste0(tempdir(), "/")
  config$out$relfolder <- config$out$relfolder %||% "matches"
  config$out$event_out_locs <- config$out$event_out_locs %||% file.path(config$out$folder, "event_outputs")

  # bootstrap settings
  config$bootstrap$n_boot <- config$bootstrap$n_boot %||% 200
  config$bootstrap$ci_alpha <- config$bootstrap$ci_alpha %||% 0.05
  config$bootstrap$seed <- config$bootstrap$seed %||% 1

  # output
  config$bootstrap$out_csv <- config$bootstrap$out_csv %||% "bootstrap_summary.csv"

  return(config)
}

# Find latest best_matches file for each pilot (under <out.folder>/<out.relfolder>/<pilot>/best_matches*.csv)
# find_latest_match_files <- function(config) {
#   root <- if (startsWith(config$out$relfolder, "/")) {
#     config$out$relfolder
#   } else {
#     file.path(config$out$folder, config$out$relfolder)
#   }

#   files <- map_chr(config$pilot_names, function(pilot) {
#     pilot_dir <- file.path(root, pilot)
#     if (!dir.exists(pilot_dir)) {
#       stop(paste0(
#         "Pilot match dir not found: ", pilot_dir,
#         " (did you run the original pipeline to generate best_matches?)"
#       ))
#     }
#     candidates <- list.files(pilot_dir, pattern = "^best_matches.*\\.csv$", full.names = TRUE)
#     if (length(candidates) == 0) {
#       stop(paste0("No best_matches*.csv found in ", pilot_dir))
#     }
#     candidates[which.max(file.info(candidates)$mtime)]
#   })

#   names(files) <- config$pilot_names
#   return(files)
# }


get_match_files <- function(config) {
  # If user provided explicit match files, use them
  if (!is.null(config$match_files)) {

    # Case 1: named object (preferred) -> config$match_files$PILOT_NAME
    if (is.list(config$match_files) && !is.null(names(config$match_files))) {
      missing_names <- setdiff(config$pilot_names, names(config$match_files))
      if (length(missing_names) > 0) {
        stop(paste0(
          "match_files is missing entries for pilots: ",
          paste(missing_names, collapse = ", ")
        ))
      }
      files <- unlist(config$match_files[config$pilot_names], use.names = TRUE)
    } else {
      # Case 2: plain list/vector aligned with pilot_names
      files <- unlist(config$match_files, use.names = FALSE)
      if (length(files) != length(config$pilot_names)) {
        stop("match_files must have the same length as pilot_names.")
      }
      names(files) <- config$pilot_names
    }

    # Validate files exist
    bad <- files[!file.exists(files)]
    if (length(bad) > 0) {
      stop(paste0(
        "These match_files do not exist:\n",
        paste(paste0(names(bad), ": ", bad), collapse = "\n")
      ))
    }
    return(files)
  }
}

# Compute metrics using YOUR existing functions, assuming match_tbl is already prepared
compute_metrics_prepared <- function(match_tbl_prepared, config) {
  thr <- config$distance$threshold

  match_stats <- generate_match_stats(match_tbl_prepared, thr) %>%
    rename(pilot.names = Version,
           match_rate = `Match Rate`,
           match_count = count) %>%
    select(pilot.names, match_rate, match_count)

  time_parse_rate <- calculate_time_parse_rate(match_tbl_prepared, thr) %>%
    rename(pilot.names = Version)

  concordance_results <- calculate_concordance(match_tbl_prepared, thr, na.times = "remove") %>%
    select(pilot.names, Concordance, C25, C75) %>%
    rename(concordance = Concordance, c25 = C25, c75 = C75)

  time_discrepancy <- calculate_time_discrepancy(match_tbl_prepared, thr, log(60 * 24 * 365.26)) %>%
    select(Version, auc) %>%
    rename(pilot.names = Version)

  out <- match_stats %>%
    full_join(time_parse_rate, by = "pilot.names") %>%
    full_join(concordance_results, by = "pilot.names") %>%
    full_join(time_discrepancy, by = "pilot.names")

  return(out)
}

bootstrap_compare <- function(config_path) {
  config <- read_bootstrap_config(config_path)

  # Ensure output dir exists
  dir.create(config$out$folder, recursive = TRUE, showWarnings = FALSE)

  # Load your function definitions (without running compare_tts CLI)
  script_dir <- tryCatch(
    dirname(normalizePath(sys.frame(1)$ofile)),
    error = function(e) getwd()
  )
  source_compare_tts_functions_only(file.path(script_dir, "compare_tts_2.r"))

  # Fill in event_match_files automatically (latest per pilot)
  #match_files <- find_latest_match_files(config)

  match_files <- get_match_files(config)

  # Create a config object compatible with your original functions
  config2 <- config
  config2$event_match_files <- unname(match_files)
  config2$event_match <- FALSE

  # Build sources and load match table
  sources <- initialize_sources(config2)
  match_tbl_full <- load_saved_matches(sources, config2)

  # Identify case reports by manual basename
  case_ids <- match_tbl_full %>% distinct(common.bns) %>% pull(common.bns)
  n_cases <- length(case_ids)
  if (n_cases == 0) stop("No case reports found after loading saved matches.")
  message(paste0("Found N = ", n_cases, " manual case reports."))

  # --- CACHING: prepare once ---
  # This is the expensive step (reads manual/pilot files, joins times, etc.)
  message("Preparing match table once (cached) ...")
  match_tbl_prepared_full <- prepare_match_data(match_tbl_full, config2)
  message("Cached preparation complete.")

  # Point estimate (no resampling)
  point <- compute_metrics_prepared(match_tbl_prepared_full, config2) %>%
    mutate(stat = "point")

  # Bootstrap
  set.seed(config$bootstrap$seed)
  B <- config$bootstrap$n_boot
  boot_list <- vector("list", B)

  for (b in seq_len(B)) {
    sampled <- sample(case_ids, size = n_cases, replace = TRUE)

    # Build a bootstrap match table by duplicating whole-case rows.
    # We MUST make common.bns unique per draw, otherwise later nesting can collapse duplicates.
    boot_tbl <- map2_dfr(sampled, seq_along(sampled), function(cid, k) {
      match_tbl_prepared_full %>%
        filter(common.bns == cid) %>%
        mutate(common.bns = paste0(cid, "__boot", b, "__draw", k))
    })

    boot_list[[b]] <- compute_metrics_prepared(boot_tbl, config2) %>%
      mutate(stat = "boot", boot_id = b)

    if (b %% 10 == 0) message(paste0("bootstrap ", b, "/", B))
  }

  boot_all <- bind_rows(boot_list)

  alpha <- config$bootstrap$ci_alpha
  lo <- alpha / 2
  hi <- 1 - alpha / 2

  metrics <- c("match_rate", "parse_rate", "concordance", "c25", "c75", "auc")

  ci_tbl <- boot_all %>%
    group_by(pilot.names) %>%
    summarise(
      across(all_of(metrics),
             list(lo = ~ quantile(.x, lo, na.rm = TRUE),
                  hi = ~ quantile(.x, hi, na.rm = TRUE)),
             .names = "{.col}_{.fn}"),
      .groups = "drop"
    )

  # Final combined summary
  final <- point %>%
    select(pilot.names, match_rate, match_count, parse_rate, concordance, c25, c75, auc) %>%
    left_join(ci_tbl, by = "pilot.names") %>%
    arrange(pilot.names)

  out_path <- file.path(config$out$folder, config$bootstrap$out_csv)
  write_csv(final, out_path)
  message(paste0("Wrote bootstrap summary to: ", out_path))

  return(list(point = point, ci = ci_tbl, final = final, match_files = match_files))
}

# CLI
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript bootstrap_comparer.r <bootstrap_config.json>")
bootstrap_compare(args[1])
