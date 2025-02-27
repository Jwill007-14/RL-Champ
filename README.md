# Rocket League Championship Predictions!

![download (4)](https://github.com/user-attachments/assets/a8a1dd06-afa3-4db6-ad9c-7e8c2c698470)

## Problem Statement
Rocket League is a very popular game that mixes soccar with rocket powered vehichels. There is both casual and ranked modes as well as tournaments allowing players to really test their metal. This data set is of Profesional Teams going head to head. Join me as I do some EDA to show you a few of the stats that can affect a teams winning chances. Then we will do some modeling on the data to predict how often a team wins as opposed to losing.

## Data Dictionary
Below is a data dictionary to explain the meaning of each variable or field in the dataset.

| Column Name | Description |
|------------- |-------------|
| color                    | Color of the team, Orange or Blue. (object)           |
| team_name                | Name of the Team. (object)                            |
| team_region              | Region the team is from. (object)                     |
| core_shots               | The amount of Shots made in a match. (int64)          |
| core_goals               | The amount of Goals made in a match. (int64)          |
| core_saves               | The amount of Saves made in a match. (int64)          |
| core_assists             | The amount of Assists made in a match. (int64)        |
| core_score               | The score per match. (int64)                          |
| core_shooting_percentage | Overall shooting accuracy. (Float)                    |
| winner                   | Shows if a game was won or lost by the team. (bool)   |

## Executive Summary
The main factors in this dataset that are being used for modeling are Shots on net,goals made, shot accuracy, saves, and assists. there are afew factors that also affect the overall score like centering the ball and simply being the first one to touch the ball, yet the stats that really matter are tied to making or preventing goals. so in theory the more often a team shoots on the net then the more often the ball will go in and score a point for a team. The team with the most goals by the end usually wins. Lets use these stats to predict some wins and losses(essintially creating ghost matches).
### Data Cleaning Steps
I removed all the stats that would have cuased to much noise with the modeling they are as follows;['ball_possesion_time','ball_time_in_side','boost_bpm','boost_bcpm','boost_avg_amount','boost_amount_collected','boost_count_collected','boost_amount_stolen','boost_amount_collected_big','boost_amount_stolen_big','boost_amount_collected_small','boost_amount_stolen_small','boost_count_collected_big','boost_count_collected_small','boost_count_stolen_big','boost_count_stolen_small','boost_amount_overfill','boost_amount_overfill_stolen','boost_amount_used_while_supersonic','boost_time_zero_boost','boost_time_full_boost','boost_time_boost_0_25','boost_time_boost_25_50','boost_time_boost_50_75','boost_time_boost_75_100','movement_total_distance','movement_time_supersonic_speed','movement_time_boost_speed','movement_time_slow_speed','movement_time_ground','movement_time_low_air','movement_time_high_air','movement_time_powerslide','movement_count_powerslide','positioning_time_defensive_third','positioning_time_neutral_third','positioning_time_offensive_third','positioning_time_defensive_half','positioning_time_offensive_half','positioning_time_behind_ball','positioning_time_in_front_ball','demo_inflicted','demo_taken']

I also removed the variables that where place holders and not nedded for any EDA;
['game_id','team_id','team_slug']

### Key Visualizations

#### Visualization 1: Count of each Genre of Games
This Histogram shows that teams with a lower score will lose more than they win. This is the standard for all Rocket League matches.

![RLC](https://github.com/user-attachments/assets/4dce03b6-f85d-4350-b91a-4c4bd1596ea6)

#### Visualization 2: North American Sales vs Global Sales
The scatterplot below shows that Shots have a significant correlation with Score, with a correlation of 0.72.This means more shots on net equate to a higher Score.

![RLC2](https://github.com/user-attachments/assets/fddebeba-1402-49d7-b185-f13589cae626)

## Model Performance

### Model Selection
I used KNN for this dataset. This was so I could use the classifier to get a more accurtate prediction.
### Evaluation Metrics
Summarize the performance of the model(s) using key evaluation metrics (e.g., RMSE, R²).

| Model             | RMSE     | R²       |
|-------------------|----------|----------|
| KNN               | [80%]    | [71%]    |

## Conclusions/Recommendations

My conclusion is rather simple the more often a team shoots on net the more often they will score or win more often in overtime as well, so simply shoot as often as you can finding new and creative ways to get it past the goalie.
