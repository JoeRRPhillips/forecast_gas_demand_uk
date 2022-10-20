# Use R 4.1.2 interpreter
library("lubridate")
library("magrittr")
library("tidyverse")
library("useful")


root_dir = "C:/Users/JPhillips/ldz"
raw_data_dir = "/data/raw"
fr_train = readr::read_csv(file.path(root_dir, raw_data_dir, "train.csv"))
fr_test = readr::read_csv(file.path(root_dir, raw_data_dir, "test.csv"))

processed_data_dir = "data/processed/lagged_temp"
# processed_data_dir = "data/processed"


fr_plt = fr_train %>%
    select(demand, date) %>%
    mutate(date = as.Date(date))

# fr_plt %>%
#     ggplot() +
#     geom_line(aes(x = date, y = demand))


date_to_year_decimal <- function(x) {
    # E.g. 2020-07-02 -> 2020.5
    return (lubridate::year(x) + (lubridate::yday(x)-1) / 366)
}

date_to_day_of_year <- function(date) {
    return (lubridate::yday(date)-1)
}

add_seasonal_and_weekend_features <- function(
    fr_raw,
    format = "%d/%m/%Y",
    base_year = 2009,
    base_year_normaliser = 10.0,
    days_per_year = 365.2425
) {
    fr_features = fr_raw %>%
        mutate(date = as.Date(date, format = format))

    fr_features = fr_features %>%
        mutate(
            is_weekend = ifelse(grepl("S(at|un)", wday(date, label = TRUE)), 1, 0),
            sin_yday = sin(2 * pi * date_to_day_of_year(date) / days_per_year),
            cos_yday = cos(2 * pi * date_to_day_of_year(date) / days_per_year),
            sin_2yday = sin(2 * 2 * pi * date_to_day_of_year(date) / days_per_year),
            cos_2yday = cos(2 * 2 * pi * date_to_day_of_year(date) / days_per_year),
            sin_3yday = sin(3 * 2 * pi * date_to_day_of_year(date) / days_per_year),
            cos_3yday = cos(3 * 2 * pi * date_to_day_of_year(date) / days_per_year),
            year_decimal = (date_to_year_decimal(date) - base_year)/base_year_normaliser
        )
    return (fr_features)
}


# Normalise features in the train and test sets by the values observed in the train set only
normalise_features <- function(fr_list, feature_prefix) {
    fr_train = fr_list$train
    fr_test = fr_list$test

    fr_feature_train = fr_train %>%
        select(starts_with(feature_prefix))

    min_train = min(fr_feature_train)
    max_train = max(fr_feature_train)

    fr_train = fr_train %>%
        mutate_at(vars(starts_with(feature_prefix)), function(x) {(x - min_train) / (max_train - min_train)})

    fr_test = fr_test %>%
        mutate_at(vars(starts_with(feature_prefix)), function(x) {(x - min_train) / (max_train - min_train)})

    return (list(train = fr_train, test = fr_test))
}


lag_demand <- function(fr_list) {
    fr_train = fr_list$train
    fr_test = fr_list$test

    # Provide the previous demand at the previous timestep as a feature for the current timestep.
    fr_train = fr_train %>%
        mutate(demand_lag1 = lag(demand))

    # Replace the NA induced during lagging by repeating the first value.
    fr_train$demand_lag1[[1]] = fr_train$demand_lag1[[2]]

    # Add placeholders for the lagged demand. These will be overwritten by the model, which will should not be allowed to
    # see the placeholder values. Autoregressive bootstrapping.
    # Adds demand_lag1 as the last column.
    fr_test = fr_test %>%
        mutate(demand_lag1 = 0)

    # Move demand column to first column index for convenience during data partitioning.
    # n = length(fr_train)
    # fr_train = fr_train[, c(n, 1:n-1)]

    # Not needed - mangles column order
    # n = length(fr_test)
    # fr_test = fr_test[, c(n, 1:n-1)]

    return (list(train = fr_train, test = fr_test))
}


# Drop duplicated columns
fr_train = fr_train %>% select(-contains("_5"))
fr_test = fr_test %>% select(-contains("_5"))

fr_train = add_seasonal_and_weekend_features(fr_train)
fr_test = add_seasonal_and_weekend_features(fr_test)

fr_list = list(train = fr_train, test = fr_test)
fr_list = normalise_features(fr_list, "ssrd")
fr_list = normalise_features(fr_list, "wind")
fr_list = normalise_features(fr_list, "temp")

# fr_list = lag_demand(fr_list)

# fr_full = fr_train %>%
#     select(-demand) %>%
#     rbind(fr_test)

# fr_full_s = fr_full %>%
#     shift.column(columns = startsWith(colnames(fr_full), prefix = "temp"), newNames = sprintf("%s._lag1", columns), len = 1L, up = FALSE)

fr_train = fr_list$train
fr_test = fr_list$test

# Remove unwanted columns
fr_train = fr_train %>%
    select(-id, -date)

fr_test = fr_test %>%
    select(-id, -date)


readr::write_csv(fr_train, file.path(root_dir, processed_data_dir, "train.csv"))
readr::write_csv(fr_test, file.path(root_dir, processed_data_dir, "test.csv"))
