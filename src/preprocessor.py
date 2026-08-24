import pandas as pd


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    df["Gender"] = df["Gender"].map({
        "Male": 0,
        "Female": 1
    })

    return df


def get_features_and_target(df):
    features = [
        "Age",
        "Gender",
        "EducationYears",
        "MMSE",
        "MemoryComplaints",
        "BehavioralChanges",
        "DailyActivityScore",
        "FamilyHistory"
    ]

    X = df[features]
    y = df["Risk"]

    return X, y