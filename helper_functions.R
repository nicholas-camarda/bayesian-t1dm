#' @note this function calculates the amount of insulin leftover every
#' five minutes for a bolus of a given size, and an insulin duration of a fixed amount
#' @param dosage numeric, in units of insulin, of insulin bolus
#' @param duration numeric, in hours, of insulin duration
get_insulin_levels <- function(dosage, duration_hours = 5, plot_curve = FALSE) {
    # Convert duration to minutes and initialize time and insulin levels
    td <- duration_hours * 60
    time_interval <- seq(from = 0, to = td, by = 5)

    ## from Dragan Maksimovic https://github.com/LoopKit/Loop/issues/388
    # peak minutes of insulin action, based on generalized data of humalog/novalog
    tp <- 75
    # time constant of exponential decay
    tau <- tp * (1 - tp / td) / (1 - (2 * tp / td))
    # rise time factor
    a <- 2 * tau / td
    # auxilliary scale factor
    S <- 1 / (1 - a + (1 + a) * exp(-td / tau))

    # insulin activity
    Ia_t <- tibble(Time = time_interval, res = sapply(time_interval, FUN = function(t) {
        dosage * (S / tau^2) * t * (1 - t / td) * exp(-t / tau)
    })) %>% mutate(measurement = "Ia")

    # inuslin on board
    IOB_t <- tibble(Time = time_interval, res = sapply(time_interval, FUN = function(t) {
        dosage * (1 - S * (1 - a) * ((t^2 / (tau * td * (1 - a)) - t / tau - 1) * exp(-t / tau) + 1))
    })) %>% mutate(measurement = "IOB")

    Ia_IOB_t <- bind_rows(Ia_t, IOB_t) %>% mutate(measurement = factor(measurement))

    # Return a data frame with the time intervals and insulin levels
    if (plot_curve) {
        p1 <- ggplot(Ia_IOB_t, mapping = aes(x = Time, y = res, color = measurement)) +
            geom_line() +
            geom_point() +
            theme_bw() +
            scale_color_manual(values = c("#E69F00", "#56B4E9")) + # "#999999",
            ggtitle("Insulin Activity (Ia) and Insulin-On-Board (IOB)") +
            facet_wrap(~measurement, scales = "free_y") +
            ylab("Insulin (units)") +
            xlab("Time (minutes)") +
            labs(caption = qq("Insulin dosage = @{dosage}\nInsulin duration = @{duration_hours}"))
        print(p1)
    }

    return(Ia_IOB_t %>% pivot_wider(id_cols = Time, values_from = res, names_from = measurement))
}

# get_insulin_levels(dosage = 6)
