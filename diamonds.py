import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt


# Filter data by carat less than 1.0 and Ideal cut type
def plot_diamonds(
    min_carat=0.75,
    max_carat=1.0,
    cut_type="Ideal",
):
    # Load the dataset
    # https://www.kaggle.com/datasets/enashed/diamond-prices
    diamonds_df = pd.read_csv("./diamonds2022.csv")

    # Filter by carat and cut
    filtered_df = diamonds_df[
        (diamonds_df["carat"] > min_carat)
        & (diamonds_df["carat"] < max_carat)
        & (diamonds_df["cut"] == cut_type)
    ]

    # Compute the mean price grouped by color and clarity
    mean_prices = (
        filtered_df.groupby(["color", "clarity"])["price"].mean().reset_index()
    )

    # Define the order of columns
    clarity_order = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]

    # Pivot the data for heatmap plotting and reorder columns
    pivot_table = mean_prices.pivot("color", "clarity", "price").reindex(
        columns=clarity_order
    )

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(
        f"Mean Price of Diamonds ({min_carat} < Carat < {max_carat}, {cut_type} Cut) by Color and Clarity"
    )
    plt.xlabel("Clarity")
    plt.ylabel("Color")
    plt.show()


def train_diamonds():
    # Load the dataset
    diamonds_df = pd.read_csv("diamonds2022.csv")

    # Select features and target variable
    X = diamonds_df[["carat", "cut", "color", "clarity"]]
    y = diamonds_df["price"]

    # Convert categorical variables into dummy/indicator variables
    X = pd.get_dummies(X, columns=["cut", "color", "clarity"])
    print(X.columns)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create the Extra Trees Regressor model
    extra_trees = ExtraTreesRegressor(n_estimators=100, random_state=42)

    # Fit the model to the training data
    extra_trees.fit(X_train, y_train)

    # Error
    y_pred = extra_trees.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # print("Mean squared error:", mse)
    print(f"Mean absolute error: ${mae:0.2f}")

    # Save the trained model to a file
    joblib.dump(extra_trees, "extra_trees_model.pkl")

    return extra_trees


def pred_price(
    carat: float,
    cut: str,
    color: str,
    clarity: str,
    model_path="extra_trees_model.pkl",
    train_cols=[
        "carat",
        "cut_Fair",
        "cut_Good",
        "cut_Ideal",
        "cut_Premium",
        "cut_Very Good",
        "color_D",
        "color_E",
        "color_F",
        "color_G",
        "color_H",
        "color_I",
        "color_J",
        "clarity_I1",
        "clarity_IF",
        "clarity_SI1",
        "clarity_SI2",
        "clarity_VS1",
        "clarity_VS2",
        "clarity_VVS1",
        "clarity_VVS2",
    ],
):
    assert carat >= 0.2 and carat <= 5.01, f"Carat out of range"
    assert cut in [
        "Ideal",
        "Premium",
        "Good",
        "Very Good",
        "Fair",
    ], f"Invalid cut: {cut}"
    assert color in ["E", "I", "J", "H", "F", "G", "D"], f"Invalid color: {color}"
    assert clarity in [
        "SI2",
        "SI1",
        "VS1",
        "VS2",
        "VVS2",
        "VVS1",
        "I1",
        "IF",
    ], f"Invalid clarity: {clarity}"
    # Load the trained model from the file
    loaded_model = joblib.load(model_path)

    # Example prediction using loaded model
    example_data = {
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
    }
    example_df = pd.DataFrame(example_data)

    # Convert categorical variables into dummy/indicator variables
    example_df = pd.get_dummies(example_df, columns=["cut", "color", "clarity"])

    # Ensure example dataframe has all possible dummy variables
    missing_cols = set(train_cols) - set(example_df.columns)
    for col in missing_cols:
        example_df[col] = 0

    # Reorder columns to match the model's input
    example_df = example_df[train_cols]

    # Predict on the example data using the loaded model
    predicted_price = loaded_model.predict(example_df)
    print("Predicted Price:", predicted_price[0])
    return predicted_price[0]


if __name__ == "__main__":
    # plot_diamonds(min_carat=0.9, max_carat=1.10, cut_type="Premium")
    # train_diamonds()
    pred_price(carat=1.0, cut="Premium", color="J", clarity="VS1")
