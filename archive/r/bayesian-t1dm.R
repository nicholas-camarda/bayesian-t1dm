library(tidyverse)
library(GetoptLong)
library(lubridate)
library(ggpubr)
# bayesian
# library(rstan)
library(brms)
# library(distr)

# for pretty breaks
library(scales)

# for AUC
# library(zoo)

# for progress
library(progressr)

# addtl for time series analysis
# library(fable)

find_project_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  normalizePath(getwd())
}

project_root <- find_project_root()
source(file.path(project_root, "helper_functions.R"))

# these help Stan run faster, for bayesian computation
# rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores() - 1)

# progress details
handlers("progress")
handlers("txtprogressbar", "beepr")
handlers(handler_pbcol(enable_after = 3.0))
handlers(handler_progress(complete = "#"))


# TODO:  READ IN NEW DATA retreived from API calls

##
bolus_data_combined <- bind_rows(temp_df$bolus_data)
cgm_data_combined <- bind_rows(temp_df$cgm_data) %>%
  group_by(DateTime) %>%
  summarize(BG = mean(BG))

library(tictoc)
tic()
expanded_bolus_data_combined <- bolus_data_combined %>%
  dplyr::select(-BG) %>%
  mutate(expanded_bolus_data = map(ActualTotalBolusRequested, .f = get_insulin_levels), .before = 1)
final_expanded_bolus_data_combined <- expanded_bolus_data_combined %>%
  unnest(expanded_bolus_data) %>%
  mutate(CompletionDateTime = CompletionDateTime + minutes(Time)) %>%
  mutate(DateTime = lubridate::round_date(CompletionDateTime, "5 minutes"), .before = 1) 
toc()

complete_dataset <- left_join(final_expanded_bolus_data_combined, cgm_data_combined, by = "DateTime") %>%
dplyr::select(DateTime, Ia, IOB, ​ActualTotalBolusRequested, TargetBG, CorrectionFactor, )


# write_csv(bolus_data_combined, "processed-data/processed-tslim-bolus-data.csv")
# write_csv(cgm_data_combined, "processed-data/processed-cgm-data.csv")


# isTrue <- function(x) is.logical(x) && length(x) == 1 && !is.na(x) && x

# my_auc <- function(df, target_bg = 110){
#   x <- 1:nrow(df)
#   y <- abs(df$BG - target_bg) # abs because any difference from 110 still is error
#   auc <- sum(diff(x)*rollmean(y, k = 2))
#   return(auc)
# }

# ### From ChatGPT ###

# # Load the 'rstan' and 'brms' packages for Bayesian modeling
# # install.packages(c("rstan", "brms", "truncnorm"))
# # library(rstan)
# library(brms)
# # library(truncnorm)


# # Collect the blood sugar, insulin bolus, and carbohydrate intake data
# bolus_data_combined <- bind_rows(temp_df$bolus_data)
# cgm_data_combined <- bind_rows(temp_df$cgm_data) %>%
#   group_by(DateTime) %>%
#   summarize(BG = mean(BG))

# # from chatgpt
# round_datetime_5min <- function(dt) {
#   # Extract the hour, minute, and second from the POSIXct object
#   hour <- as.integer(format(dt, "%H"))
#   minute <- as.integer(format(dt, "%M"))
#   second <- as.integer(format(dt, "%S"))

#   # Calculate the number of seconds from the start of the day
#   seconds <- hour * 3600 + minute * 60 + second

#   # Calculate the number of seconds until the next 5 minute mark
#   rounding <- ((seconds + 2.5 * 60) %/% (5 * 60)) * (5 * 60)

#   # Calculate the number of seconds to add or subtract to round the POSIXct object
#   delta <- rounding - seconds

#   # Round the POSIXct object by adding or subtracting the delta
#   res <- ifelse(delta >= 0, dt + delta, dt - abs(delta))
#   p_res <- as_datetime(res)
#   return(p_res)
# }


# trunc_bolus_data_combined <- bolus_data_combined %>%
#   dplyr::select(CompletionDateTime, BG, ActualTotalBolusRequested,CarbSize,InsulinToCarbRatio,CorrectionFactor) %>%
#   na.omit() %>%
#   mutate(DateTime = round_datetime_5min(CompletionDateTime))

# data_temp <- left_join(cgm_data_combined, trunc_bolus_data_combined) %>%
#   mutate(idx = row_number(),
#          InsulinToCarbRatio = as.numeric(str_split(InsulinToCarbRatio, ":", simplify = T)[,2])) %>%
#   rename(insulin_bolus = ActualTotalBolusRequested,
#          guessed_carbohydrate = CarbSize,
#          blood_sugar = BG,
#          datetime = DateTime) %>%
#   dplyr::select(-CompletionDateTime, -idx, -InsulinToCarbRatio, -CorrectionFactor)

# data <- data_temp %>%
#   mutate(time_point = as.factor(seq(0, nrow(data)*5 - 5, by = 5))) %>%
#   dplyr::select(-datetime)

# install.packages("rstanarm")
# library(rstanarm)



#  #############################################################################
# # result <- read_rds("output/bolus_cgm_combined-analysis_ready.rds")

# # result$ActualTotalBolusRequested[1]
# # result$CGM[1][[1]] %>% filter(Timing == "After Bolus") %>%

# extract_cgm_data_for_bolus <- function(cgm_dat, bolus_dat, num_sec_window = 10000){
#   # DEBUG
#   # cgm_dat <- cgm_data_combined
#   # bolus_dat <- bolus_data_combined
#   # num_sec_window = 10000

#   # create output directory for plots
#   output_dir_top_ <- "output/bolus_cgm_plots"
#   dir.create(output_dir_top_, showWarnings = FALSE, recursive = TRUE)

#   message("Finding CGM data around boluses...")
#   with_progress({
#     sequence_ <- 1:nrow(bolus_dat)
#     p <- progressor(length(sequence_))

#     # this is currnetly by each individual bolus
#     # i think it would be better to do this by each DAY, so as to capture all the boluses in the day and
#     # the effects each one have (if they are close together, especially!!)

#     result <- lapply(sequence_, FUN = function(i){

#       b0 <- bolus_dat[i-1,]
#       b1 <- bolus_dat[i,]
#       b2 <- bolus_dat[i+1,]

#       targetBG <- b1$TargetBG
#       datetime <- b1$CompletionDateTime
#       FoodBolusSize <- b1$FoodBolusSize
#       CorrectionBolusSize <- b1$CorrectionBolusSize
#       ActualTotalBolusRequested <- b1$ActualTotalBolusRequested
#       BolusType <- b1$BolusType
#       CarbSize <- b1$CarbSize

#       CorrectionFactor <- b1$CorrectionFactor
#       InsulinToCarbRatio <- b1$InsulinToCarbRatio

#       window_down <- datetime - seconds(num_sec_window)
#       num_sec_in_5_hrs <- 60*60*5
#       window_up <- datetime + seconds(num_sec_in_5_hrs)

#       window_before <- lubridate::interval(start = window_down, end = datetime)
#       if (isTrue(b0) && b0$CompletionDateTime %within% window_before) {
#         window_down <- b0$CompletionDateTime
#         window_before <- lubridate::interval(start = window_down, end = datetime)
#       }

#       window_after <- lubridate::interval(start = datetime, end = window_up)
#       if (isTrue(b2) && (b2$CompletionDateTime %within% window_after)) {
#         window_up <- b2$CompletionDateTime
#         window_after <- lubridate::interval(start = datetime, end = window_up)
#       }

#       cgm_b1 <- bind_rows(
#         cgm_dat %>%
#           # mutate(intervals_5 = interval(start = EventDateTime, end = lead(EventDateTime))) %>%
#           mutate(dist_m = DateTime - datetime) %>%
#           filter(DateTime %within% window_before) %>%
#           mutate(Timing = "Before Bolus"),
#         cgm_dat %>%
#           # mutate(intervals_5 = interval(start = EventDateTime, end = lead(EventDateTime))) %>%
#           mutate(dist_m = DateTime - datetime) %>%
#           filter(DateTime %within% window_after) %>%
#           mutate(Timing = "After Bolus")) %>%
#         mutate(Timing = factor(Timing, levels = c("Before Bolus", "After Bolus")))


#       # my kinda dumb metric for assessing the effectiveness of the bolus
#       # closer to zero the better
#       error_after_bolus <- my_auc(df = cgm_b1 %>% filter(Timing == "After Bolus"),
#                                   target_bg = targetBG)
#       error_before_bolus <- my_auc(df = cgm_b1 %>% filter(Timing == "Before Bolus"),
#                                    target_bg = targetBG)
#       auc_diff <- error_after_bolus - error_before_bolus

#       # maybe this shouldn't be in this loop but it's just so easy
#       # helper stuff for plotting
#       month_idx <- month(datetime,label = FALSE)
#       month_ <- month(datetime,label = TRUE)
#       day_ <- day(datetime)
#       year_ <- year(datetime)
#       edited_datetime <- str_replace_all(string = datetime, pattern = ":", replacement = ".")
#       final_dir <- qq("@{output_dir_top_}/@{year_}-@{month_idx}@{month_}-@{day_}")
#       dir.create(final_dir, showWarnings = FALSE, recursive = TRUE)
#       trace_fn <- qq("@{final_dir}/Bolus on @{edited_datetime}.png")

#       if (!file.exists(trace_fn)){
#         num_breaks_y <- (max(cgm_b1$BG, na.rm = TRUE) - min(min(cgm_b1$BG, na.rm = TRUE), targetBG)) / 10
#         if (is.infinite(num_breaks_y)) num_breaks_y <- 5

#         trace <- ggline(cgm_b1, x = "DateTime", y = "BG", color = "Timing",
#                         palette = "jco",
#                         ggtheme = theme_bw(),
#                         title = qq("Bolus on @{datetime}")) +
#           geom_vline(xintercept = datetime, color = "red") +
#           geom_hline(yintercept = targetBG, color = "purple") +
#           grids() +
#           scale_y_continuous(breaks = scales::breaks_pretty(n = num_breaks_y)) +
#           annotate(x=datetime,y=+Inf,
#                    label=qq("Bolus: @{ActualTotalBolusRequested}u \nCarb: @{CarbSize}g"),
#                    vjust=2,geom="label") +
#           labs(subtitle = qq("IC: @{InsulinToCarbRatio} || CF: 1:@{CorrectionFactor}"),
#                caption = qq("Insulin duration: 5 hours\nBefore-bolus window size: @{scales::label_comma(accuracy = 1)(num_sec_window/(60*60))} hrs"))
#         suppressMessages(ggsave(filename = trace_fn, plot = trace, dpi = 320))
#       }

#       p()
#       cmd_res <- tibble(BolusDateTime = datetime, FoodBolusSize, CorrectionBolusSize,
#                         ActualTotalBolusRequested, CarbSize = CarbSize,
#                         TargetBG = targetBG, CGM = list(cgm_b1),
#                         BolusType, AUCdiff = auc_diff)
#       return(cmd_res)
#     }) %>%
#       bind_rows()
#   })

#   # How off was I? need a cost function to describe how bad the guess was
#   # How much more insulin should I have given?
#   #
#   # # figure out how to model and forecast within a fable
#   # trends_df <- result %>%
#   #   mutate(trend_model = map(.x = CGM, .f = function(dat) {
#   #     ts_dat <- dat %>%
#   #       dplyr::select(DateTime, BG) %>%
#   #       mutate(DateTime = floor_date(DateTime, unit = "5 minutes")) %>%
#   #       group_by(DateTime) %>%
#   #       summarize(BG = mean(BG)) %>%
#   #       as_tsibble(index = DateTime)
#   #
#   #     TSLM(ts_dat ~ trend())
#   #   }))



#   # expanded_result <- result %>% unnest(c(CGM))
#   # expanded_result
#   # this is wrong...
#   # model1 <- glm(BG ~ ActualTotalBolusRequested + AUCdiff + CarbSize, data = expanded_result)
#   # summary(model1)
#   # plot(model1)
#   # new_data <- tibble(ActualTotalBolusRequested = 15.45,
#   #                    BolusDateTime = as_datetime("2022-09-15 19:57:56"),
#   #                    DateTime = as_datetime("2022-09-15 21:57:56"),
#   #                    AUCdiff = 200,
#   #                    FoodBolusSize = 15.45,
#   #                    CorrectionBolusSize = 0.0)
#   # predict(model1, new_data)
# }


# # optimize non-seasonal pdq
# # optimize_pdq <- function(df){
# #   df <- df %>% column_to_rownames(var = "DateTime") %>%
# #     dplyr::select(1)
# #   print(head(df))
# #   azfinal_aic <- Inf
# #   azfinal_order <- c(0,0,0)
# #   for (p in 1:4) for (d in 0:1) for (q in 1:4) {
# #     azcurrent.aic <- AIC(arima(df, order=c(p, d, q)))
# #     if (azcurrent.aic < azfinal_aic) {
# #       azfinal_aic <- azcurrent.aic
# #       azfinal.order <- c(p, d, q)
# #       azfinal.arima <- arima(df, order=azfinal.order)
# #     }
# #     return(azfinal.order)
# #   }
# # }

# # combined_data <- map_dfr(.x = data_fns, .f = read_csv, skip = 6, show_col_types = FALSE)
