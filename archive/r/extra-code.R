cgm_b1 %>% plot_time_series(DateTime, BG)

auto.arima(cgm_b1)

splits <- initial_time_split(cgm_b1, prop = 0.9) # 

# check out auto.arima from forecast

model_fit_arima <- arima_reg(non_seasonal_ar = 0,
                             non_seasonal_differences = 1,
                             non_seasonal_ma = 1,
                             seasonal_period = 12,
                             seasonal_ar = 0,
                             seasonal_differences = 1,
                             seasonal_ma = 1) %>%
  set_engine(engine = "arima") %>%
  fit(BG ~ DateTime, 
      data = training(splits))

model_fit_arima_bayes<- sarima_reg(non_seasonal_ar = 0,
                                   non_seasonal_differences = 1,
                                   non_seasonal_ma = 1,
                                   seasonal_period = 12,
                                   seasonal_ar = 0,
                                   seasonal_differences = 1,
                                   seasonal_ma = 1, 
                                   pred_seed = 100) %>%
  set_engine(engine = "stan") %>%
  fit(BG ~ DateTime, data = training(splits))

model_fit_naive <- random_walk_reg(seasonal_random_walk = TRUE, seasonal_period = 12) %>%
  set_engine("stan") %>%
  fit(BG ~ DateTime + month(DateTime), data = training(splits))

models_tbl <- modeltime_table(
  model_fit_arima,
  model_fit_arima_bayes,
  model_fit_naive
)
models_tbl

calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))
calibration_tbl

# interactive plot
calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = cgm_b1
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = FALSE
  )

# To make table-creation a bit easier, I’ve included table_modeltime_accuracy() 
# for outputing results in either interactive (reactable) or static (gt) tables.
calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = interactive
  )

refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = cgm_b1)

refit_tbl %>%
  modeltime_forecast(h = "2 hours", actual_data = cgm_b1) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = FALSE
  )
