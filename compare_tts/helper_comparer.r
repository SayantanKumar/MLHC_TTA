library(tidyverse)
library(stringdist)
library(reticulate)
library(TreeDist)

use_condaenv("/opt/anaconda3/envs/Rtest")
if(!exists("script_dir")) {
    script_dir = tryCatch(
        dirname(normalizePath(sys.frame(1)$ofile)),
        error = function(e) "/Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/compare_tts/"
    )
}
source_python(file.path(script_dir,"distance_helper_v2.py"))
# use_condaenv("~/.miniconda3/envs/tta_unimodal/")
# source_python("/usr0/home/juyongk/workspace/Textual_tabular_alignment/compare_tts/distance_helper_v2.py")

# Linear assignment problem functions with ignoring matching if exceeding threshold (wrapper around LAPJV)
dummy_pad = function(m, value=0.1) {
    rc = ncol(m) + nrow(m)
    dm = matrix(value, nrow=rc, ncol=rc)
    dm[1:nrow(m),1:ncol(m)] = m
    return(dm)
}
threshold_LAPJV = function(m, value=0.1, omit_mismatch_cost=F) {
    if(any(dim(m)<1)) { return(NULL) }
    dm = dummy_pad(m, value=value) %>% signif(8)
    excess_cost = value*(nrow(dm)-nrow(m))
    result = LAPJV(dm)
    matching = ifelse(result$matching <= ncol(m), result$matching, NA)[1:nrow(m)]
    return(
        list(
            score = ifelse(omit_mismatch_cost,
                        result$score - excess_cost - value*sum(is.na(matching)),
                        result$score - excess_cost
            ),
            matching = matching
        )
    )
}


#' Recursively find the best match between rows (v1) and columns (v2) in a distance matrix
#' Returns a tibble of matches with distance, row/column indices, and iteration info
recursive_match <- function(dists, tbl = NULL, rids = NULL, cids = NULL, iter = 0) {
    if (nrow(dists) == 0) {
        return(tbl)
    }
    if (is.null(rids)) {
        rids <- 1:nrow(dists)
    }
    if (is.null(cids)) {
        cids <- 1:ncol(dists)
    }

    best.match <- tibble(
        temp.rid = 1:nrow(dists),
        rowmins = dists %>% apply(1, min),
        wrowmins = dists %>% apply(1, which.min)
    ) %>%
    group_by(rowmins) %>%
    mutate(count=n()) %>%
    ungroup()

    uptodist = best.match %>%
        mutate(uptodist = ifelse(count>1, rowmins,Inf)) %>%
        summarise(uptodist=min(rowmins)) %>% .[[1,1]]
    # num.rowmins = rowSums(dists == best.match$rowmins[row(dists)])
    # # run up to and including the min distance where there are >1 rowmins.
    # uptodist = min(best.match$rowmins[num.rowmins>1],Inf)
    best.match = best.match %>%
    filter(rowmins <= uptodist) %>%
        group_by(wrowmins) %>% 
        arrange(rowmins) %>%
        slice(1) %>% 
        ungroup()

    rids.left <- rids[-best.match$temp.rid]
    cids.left <- cids[-best.match$wrowmins]

    found.tbl <- tibble(
        dist = best.match$rowmins,
        rid = rids[best.match$temp.rid],
        cid = cids[best.match$wrowmins],
        iter = iter
    )

    # browser()
    if (is.null(tbl)) {
        tbl <- list(found.tbl)
    } else {
        tbl <- append(tbl, list(found.tbl))
    }

    if (length(rids.left) == 0) {
        return(tbl)
    }
    if (length(cids.left) == 0) {
        return(append(tbl,
            list(
                tibble(
                    dist = Inf,
                    rid = rids.left,
                    cid = NA,
                    iter = iter + 1
                ) 
            )
        ))
        # return(tbl %>% bind_rows(
        #     tibble(
        #         dist = Inf,
        #         rid = rids.left,
        #         cid = NA,
        #         iter = iter + 1
        #     )
        # ))
    }

    return(recursive_match(
        dists[-best.match$temp.rid, -best.match$wrowmins, drop = FALSE],
        tbl,
        rids.left,
        cids.left,
        iter + 1
    ))
}

compute_dist_matrix = function(events1, events2, method="lv") {
    if (method == "embedding_cosine") {
        write_csv(tibble(event = events1, noskip = 0), paste0(tempdir(), "1.csv"))
        write_csv(tibble(event = events2, noskip = 0), paste0(tempdir(), "2.csv"))
        get_and_write_embeddings(
            paste0(tempdir(), "1.csv"),
            paste0(tempdir(), "2.csv"),
            paste0(tempdir(), "out.csv")
        )
        dists <- read_csv(paste0(tempdir(), "out.csv"), skip_empty_rows = FALSE) %>% as.matrix()
    } else {
        dists <- stringdist::stringdistmatrix(events1, events2, method = method)
    }
    return(dists)
}

get_match_table_from_dists = function(dists, events1, events2, threshold = 0.6, use_lap=T, rowids=F) {
    ### Investigating low event match rates (AULTC is now higher); found the above problem
    ### Investigating why the double join tibble has different num entries than single join table when the key is in theory 1:1 (dups, set diff); added rowids but still issue
    ### Still problem of sequential matching doesn't work when skip an event; possible use textual order (as context) in reconciliation of tie-breaking? # nolint: line_length_linter.
    ### Then rerun for assessments (everywhere!)    
    if(use_lap) {
        mincost.match = dists %>% threshold_LAPJV(value=threshold)
        result = tibble(
            rid = which(!is.na(mincost.match$matching)),
            cid = mincost.match$matching[!is.na(mincost.match$matching)]
            ) %>%
            mutate(
                v1 = events1[rid],
                v2 = events2[cid]
                # v2 = events2[mincost.match$matching][!is.na(mincost.match$matching)]
            ) %>% 
            mutate(
                error.rate = map2_dbl(rid, cid, ~ dists[.x,.y])
            ) %>% select(v1, v2, error.rate, rid, cid) %>%
            # The above omits the no-matches, which we want to pass back (with NA's in many columns)
            bind_rows(
                tibble(
                    rid = which(is.na(mincost.match$matching)),
                ) %>%
                mutate(v1=events1[rid])
                # will fill events2, cid, error.rate with NA
            )
        # browser()
        if(rowids) {
            return(result)
        } else {
            return(results %>% select(-rid, -cid))
        }
    } else {
        tbl <- recursive_match(dists) %>% bind_rows()

        if(rowids) {
            return(tbl %>% 
                mutate(v1 = events1[rid], v2 = events2[cid]) %>%
                select(v1, v2, error.rate = dist, rid, cid) %>%
                mutate(keep = error.rate < threshold))
        } else {
            return(tbl %>% 
                mutate(v1 = events1[rid], v2 = events2[cid]) %>%
                select(v1, v2, error.rate = dist) %>%
                mutate(keep = error.rate < threshold))
        }
    }
    # browser()
    
}

#' Get match table between two event lists using specified distance method
#' @param events1 First list of events
#' @param events2 Second list of events
#' @param threshold Error rate threshold for keeping matches
#' @param method Distance method ("lv" for Levenshtein or "embedding_cosine")
#' @return Tibble with matched events, error rates, and keep flag
get_match_table <- function(events1, events2, threshold = 0.6, method = "lv", use_lap=T, rowids=F) {
    events1 <- ifelse(is.na(events1), "", events1)
    events2 <- ifelse(is.na(events2), "", events2)

    dists = compute_dist_matrix(events1, events2, method)

    return(get_match_table_from_dists(dists=dists,
                                      events1=events1, events2=events2,
                                      threshold=threshold, use_lap=use_lap, rowids=rowids))
}

compute_dist_matrix_featurized = function(sources = c("string_sim"), source_objs = list(NULL), w=c(1)) {
    mats = list()
    weights = list()
    if("string_sim" %in% sources) {
        si = which(sources=="string_sim")[1]
        source_obj = source_objs[[si]]
        mats = mats %>% append(list(m=compute_dist_matrix(source_obj[["events1"]], source_obj[["events2"]], source_obj[["method"]])))
        weights = weights %>% append(w[si])
    }
    if("char_pos" %in% sources) {
        if(w[which(sources=="char_pos")[1]] > 0) { # avoid NA's in char_pos if you have NAs charpos's.
            si = which(sources=="char_pos")[1]
            source_obj = source_objs[[si]]
            
            dist_fn = source_obj[["dist_fn"]]
            dists = tibble(
                pos1 = source_obj[["char_pos1"]],
                pos1_ub = source_obj[["char_pos1_ub"]]
            ) %>% 
            mutate(i1 = 1:n()) %>%
            crossing(
                tibble(
                    pos2 = source_obj[["char_pos2"]],
                    pos2_ub = source_obj[["char_pos2_ub"]]
                ) %>%
                mutate(i2= 1:n())
            ) %>%
            mutate(min.dist = pmin(pmin(abs(pos1 - pos2), abs(pos1_ub-pos2), na.rm=T),pmin(abs(pos1 - pos2_ub), abs(pos1_ub-pos2_ub), na.rm=T), na.rm=T)) %>%
            (function(df) Matrix::sparseMatrix(i=df$i1,j=df$i2,x=df$min.dist))(.) %>% as.matrix()        

            mats = mats %>% append(list(m=dists %>% dist_fn))
            weights = weights %>% append(w[si])
            # browser()
        }
    }

    # browser()
    return(
        map2(mats, weights, ~ .y * .x) %>% Reduce("+",.)
    )
}

get_match_table_wtimes = function(ref.tibble, pilot.tibble, threshold = 0.6, approach="strcmp", approach.w=c(1,0), method = "lv", use_lap=T, rowids=F) {
    # if(any(str_detect(names(ref.tibble),"char_pos")) & any(str_detect(names(pilot.tibble),"char_pos"))) {
    if(approach == "featurized") { #for now, this means use embedding and character position
        feature_obj = list(
            event.comp = list(
                events1=ref.tibble$referent.events, events2=pilot.tibble$pilot.events, method=method
            ),
            pos.comp = list(
                char_pos1 = ref.tibble$char_pos,
                char_pos1_ub = ref.tibble$char_pos_ub,
                char_pos2 = pilot.tibble$char_pos,
                char_pos2_ub = pilot.tibble$char_pos_ub,
                # dist_fn = function(x) { ifelse(is.na(x) | is.infinite(x), 1, ((x > 30) + (x > 100)) / 2)  }
                dist_fn = function(x) { ifelse(is.na(x) | is.infinite(x), 1., pmin(pmax(0,x-10),500)/500.)  }
            )
        )
        # dists = compute_dist_matrix_featurized(sources=c("string_sim","char_pos"), feature_obj, w=c(0.95,0.05))
        dists = compute_dist_matrix_featurized(sources=c("string_sim","char_pos"), feature_obj, w=approach.w)
        # dists = compute_dist_matrix_featurized(sources=c("string_sim","char_pos"), feature_obj, w=c(0.85,0.5))
        # dists = compute_dist_matrix_featurized(sources=c("string_sim","char_pos"), feature_obj, w=c(1,0))

        tbl = get_match_table_from_dists(dists=dists,
                                      events1=ref.tibble$referent.events, events2=pilot.tibble$pilot.events,
                                      threshold=threshold, use_lap=use_lap, rowids=rowids) %>%
                  rename(idx=rid, match.idx=cid)
    } else {
        tbl = get_match_table(ref.tibble$referent.events, pilot.tibble$pilot.events,
                              threshold=threshold, method=method, use_lap=use_lap, rowids=T) %>%
            mutate(
                ref.time = ref.tibble$referent.time[rid],
                pilot.time=as.character(pilot.tibble$pilot.time[cid])
            ) %>% 
            rename(
                idx=rid,
                match.idx = cid
            )
    }
    if(rowids) {
        return(tbl)
    } else {
        return(tbl %>% select(-idx, -match.idx))
    }
}

# Example usage (commented out)
if (FALSE) {
    # # Basic usage
    # matches <- get_match_table(dat2$event, dat1$event, method = "embedding_cosine")
    
    # # Pipeline example
    # ann_matching <- common.files %>%
    #     mutate(
    #         result = map(v1, ~ get_match_table(
    #             read_csv(paste0(v1.loc, .x), skip_empty_rows = FALSE)[["event"]],
    #             read_csv(paste0(v2.loc, .x), skip_empty_rows = FALSE)[["event"]],
    #             method = "embedding_cosine"
    #         ))
    #     ) %>%
    #     rename(file.name = v1) %>%
    #     unnest(everything())
    
    # # Plot error rate distribution
    # ann_matching$error.rate %>% ecdf() %>% plot(xlim = c(0, 0.1), xlab = "Error rate (Cosine)")

    # TEST compute_matrix_featurized/get_match_table_wtimes
    ref.tibble = tibble(
        referent.events = c(
            "hi",
            "how are you",
            "bye"
        ),
        char_pos=c(0, 4, 20),
        char_pos_ub=c(2,9, 23)
    )
    pilot.tibble = tibble(
        pilot.events = c(
            "hello",
            "comment ca va",
            "au revoir",
            "ciao"
        ),
        char_pos=c(0, 8, 30, 36),
        char_pos_ub=c(10,16, 35, NA)
    )
    get_match_table_wtimes(ref.tibble, pilot.tibble, threshold = 8, method = "lv", use_lap=T, rowids=T)

}