# Use R 4.1.2 interpreter
library("lubridate")
library("magrittr")
library("tidyverse")


root_dir = "C:/Users/JPhillips/ldz"
raw_data_dir = "data/raw/"
data_dir = file.path(root_dir, raw_data_dir)

fr_train = readr::read_csv(file.path(data_dir, "train.csv"))
fr_test = readr::read_csv(file.path(data_dir, "test.csv"))

format = "%d/%m/%Y"
fr_train = fr_train %>%
    select(demand, date) %>%
    mutate(date = as.Date(date, format = format))

fr_test = fr_test %>%
    mutate(date = as.Date(date, format = format))

# "C:/Users/JPhillips/ldz/predictions/2022-06-07/09-26-22/MLP/test.csv"
# pred_fp = "predictions/2022-06-07/09-26-22/MLP/test.csv"
# pred_fp = "predictions/2022-06-07/15-43-27/MLP/test.csv"
# pred_fp = "predictions/2022-06-07/19-28-06/MLP/test.csv"
# pred_fp = "predictions/2022-06-07/20-44-35/MLP/test.csv"
pred_fp = "predictions/2022-06-22/17-53-33/MLP/test.csv"
fr_pred = readr::read_csv(file.path(root_dir, pred_fp))
fr_pred$date = fr_test$date

fr_pred = fr_pred %>%
    select(-id)

fr_all = fr_train %>%
    bind_rows(fr_pred)

fr_all %>%
    ggplot() +
    geom_line(aes(x = date, y = demand)) +
    geom_vline(aes(xintercept = tail(fr_train, n=1)$date), color = "red")
