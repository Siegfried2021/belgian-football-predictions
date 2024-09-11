# Football Match Result Predictor

This project is a football match result prediction system that utilizes historical football match data to predict the result of upcoming matches using machine learning techniques. The system is built using **Python** and **scikit-learn's** `RandomForestClassifier` for classification tasks. The project involves several key stages, including data preprocessing, feature engineering, model training, and prediction of match results.

## Project Structure

### 1. **Preprocessing Football Data**

The `FootballDataPreprocessor` class is used to clean and preprocess the football data. This class is responsible for:
- Handling missing data.
- Converting date formats.
- Labeling match results numerically (e.g., home win, draw, away win).
- Feature engineering, including generating team form statistics, head-to-head results, and home/away ratios based on past matches.

Key methods:
- `create_season`: Adds a `season` column to the dataset based on match dates.
- `window_rows_last_n_games`: Fetches the last `n` games for a team, either home or away.
- `compute_team_form`: Calculates form scores for teams based on their past matches over multiple last `n` games.
- `compute_team_stats`: Computes aggregate statistics (e.g., yellow cards, shots) for teams over multiple last `n` games.
- `add_features`: Adds computed features to the dataset based on historical team performance.

### 2. **Model Experimentation and Selection**

The system finds the best combination of features for model training using the `run_model_experiments` function. It performs multiple combinations of feature sets using historical data to identify which features maximize the model's performance.

### 3. **Training the Model**

Once the optimal feature set is determined, the model is retrained using a `RandomForestClassifier`. The `train_model_with_best_features` function:
- Trains the model with the best feature set found during experimentation.
- Saves the trained model as a `.joblib` file for future use.

### 4. **Predicting Match Results**

The `predict_match_result` function uses the trained model to predict the result of a new match. It takes in the updated dataset, the model, and the best feature set, and returns a readable prediction (home win, draw, or away win).

### 5. **Full Workflow Execution**

The `preprocessing_and_model_selection` function integrates all the steps:
1. Preprocesses the football data.
2. Runs experiments to find the best feature set.
3. Retrains the model on the best feature set and saves the model.
4. Predicts the result of the next football match using the saved model.

### Requirements

- **Python 3.8+**
- **Pandas**: Data manipulation and analysis.
- **Numpy**: Support for mathematical operations.
- **scikit-learn**: Machine learning library for training models, imputing missing data, and evaluating results.
- **Joblib**: Saving and loading models.
