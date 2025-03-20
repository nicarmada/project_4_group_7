# project_4_group_7
Data Project 4 - Group 7













# Overview of cmagor_code.ipynb:
* Part 1:
 *  In this section we start by importing all necessary dependencies, reading in the file path of the "cleaned_mortality_data.csv", and transforming into a DataFrame
  * For example: file_path = Path("data\cleaned_mortality_data.csv")
cleaned_df = pd.read_csv(file_path)
cleaned_df

* (insert your own file path to csv file.)

* Part 2:
* In this section we apply K-means clustering by selecting the relevant features for clustering, standardize the features, determine the optimal k value using elbow method, and finally plotting the elbow graph
 
* Part 3:
* In this section we apply K-means using the optimal k value (4), applying K-means grouped by year, standardizing the features, creating a for loop to store inertia values, and then concatenating the clustered_df DataFrame.

* Part 4:
* For this next part, we start by grouping and clustering by state, creating the StandardScaler instance, creating another for loop to store inertia values, using the optimal number of clusters(4), and concatenating and displaying the state_clustered_df.

* Part 5: Graphs
* For the graphs, we first make a baar chart for yearly clustering, create a line chart for yearly clustering, create a bar chart of state-wise clustering, and performing the Isolation Forest method and generating a scatter plot & a bar chart for it.

* Part 6:
* Here we identify features and target variables to set up for StandardScaler, apply the StandardScaler, train the model and split data into train and test sets, make predictions and get an accuracy score, we do some hyperparameter tuning for improved results, and apply the RandomForestRegressor using population as the target variable.

* Part 7:
* In this final section, we apply StandardScaler and RandomForestRegressor using crude_rate as the target, we get an accuracy score, and generate a scatter plot based off that RandomForestRegressor using crude_rate as the target.

# We got/referenced the following lines of code from Xpert Learning Assistant/ChatGPT:

* file = r'C:\Users\Michele\Documents\GitHub\project_4_group_7\data\cleaned_mortality_data.csv'

* df = pd.read_csv(file)
  
* features = ["crude_rate", "deaths", "prescriptions_dispensed_by_us_retailers_in_that_year_(millions)"]
  
X = df[features]

X = X.fillna(X.median())

* Train Isolation Forest model

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(X)  # -1 indicates anomaly, 1 is normal
num_anomalies = (df["anomaly"] == -1).sum()

* plt.figure(figsize=(12, 6))
plt.scatter(df["year"], df["crude_rate"], c=df["anomaly"], cmap="coolwarm", edgecolors="k")
plt.xlabel("Year")
plt.ylabel("Crude Rate")
plt.title(f"Anomaly Detection in Mortality Data (Detected {num_anomalies} anomalies)")
plt.colorbar(label="Anomaly (-1: Yes, 1: No)")
plt.show()

* df[df["anomaly"] == -1].head()
fig = px.scatter(
    df, 
    x="year", 
    y="crude_rate", 
    color=df["anomaly"].map({1: "Normal", -1: "Anomaly"}), 
    hover_data=["state", "deaths", "prescriptions_dispensed_by_us_retailers_in_that_year_(millions)"],
    title="Interactive Anomaly Detection in Mortality Data",
    labels={"crude_rate": "Crude Mortality Rate", "year": "Year"},)
fig.show()

* params = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]}

* Run search grid

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5)
grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)

* model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
cleaned_filtered_df["anomoly_score"] = model.fit_predict(cleaned_filtered_df)

* clustered_data = []
for year, group in cleaned_df.groupby("year"):
    features = group[["deaths", "population", "crude_rate", "prescriptions_dispensed_by_us_retailers_in_that_year_(millions)"]]

* clustered_data = []
for state, group in cleaned_df.groupby("state"):
    features = group[["deaths", "population", "crude_rate", "prescriptions_dispensed_by_us_retailers_in_that_year_(millions)"]]

* sns.lineplot(data=state_clustered_df.groupby(["year", "cluster"])["crude_rate"].mean().reset_index(), x="year", y="crude_rate", hue="cluster", palette="tab10", marker="o")
