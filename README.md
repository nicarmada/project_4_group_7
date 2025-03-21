# project_4_group_7
Data Project 4 - Group 7

# Overview Of MayraM-BarChart:LineChart.ipynb

Bar Chart

My portion  of the project begins by importing essential Python libraries:
The project begins by importing essential Python libraries:
* pandas for data manipulation.
* matplotlib.pyplot and seaborn for visualization.
* StandardScaler and KMeans from sklearn for machine learning clustering.

Loading the Dataset
df = pd.read_csv("cleaned_mortality_data.csv")
* The dataset includes data from 816 records containing columns like state, year, deaths, crude_rate, and prescriptions dispensed.
 Initial Data Inspection
df.info(), df.head()
* This step helps understand the structure of the dataset, including column types and previewing the first few rows.

Feature Selection and Preprocessing
features = df[['crude_rate', 'prescriptions_dispensed_by_us_retailers_in_that_year_(millions)']]
* Two key features were selected for clustering:
    * Crude death rate per 100,000 population.
    * Number of prescriptions dispensed (in millions).
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
* These features were standardized using StandardScaler to normalize values and ensure equal weighting in clustering.
 
Applying KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)
* KMeans is applied to group the states into 4 clusters based on crude death rate and prescription volume.
* A new cluster column is added to the dataframe indicating each record’s assigned cluster.

Visualizing the Results
sns.barplot(data=df, x='state', y='deaths', hue='cluster', palette='viridis')
* A bar chart is created to show total opioid-related deaths per state, colored by cluster.
* This makes it visually clear how states differ in terms of death totals and cluster assignments.
 Analyzing Cluster Characteristics
cluster_analysis = df.groupby('cluster')[['crude_rate', 'prescriptions_dispensed_by_us_retailers_in_that_year_(millions)', 'deaths']].mean()
* This step calculates the average crude rate, prescriptions, and deaths per cluster, helping interpret what each cluster represents.

Summary:
This code builds a foundation for data-driven opioid trend analysis using machine learning clustering techniques and effective visual storytelling with bar charts and summaries.

Line Chart 

This code is designed to visualize and compare trends in opioid-related crude death rates (per 100,000 population) across selected U.S. states from 1994 to 2014. It uses line plotting to highlight how mortality patterns have changed over time in specific regions impacted by the opioid crisis.

 Loading the Dataset
df = pd.read_csv('/mnt/data/cleaned_mortality_data.csv')
* Loads a cleaned dataset containing opioid mortality information by state and year.

 Cleaning Column Names
df.columns = [col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for col in df.columns]
* Standardizes column headers by converting to lowercase and replacing spaces/symbols with underscores to ensure code compatibility.
 
Defining Target States
target_states = ['New Jersey', 'Ohio', 'Louisiana', 'California', 'Hawaii', 'Rhode Island', 'Washington', 'Michigan']
* Specifies the states to be included in the visualization for focused trend analysis.

 Filtering the Dataset
df_filtered = df[(df['state'].isin(target_states)) & (df['year'] >= 1994) & (df['year'] <= 2014)]
* Filters the dataset to include only records from the target states and the years 1994 to 2014.

 Creating the Line Plot
for state in target_states:
    state_data = df_filtered[df_filtered['state'] == state]
    plt.plot(state_data['year'], state_data['crude_rate'], marker='o', linestyle='-', label=state)
* Plots each state’s crude death rate over time using a line chart, with time on the x-axis and crude rate on the y-axis.

Chart Customization
plt.title('Crude Death Rate (1994–2014) by State')
plt.xlabel('Year')
plt.ylabel('Crude Death Rate (per 100,000 population)')
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
* Titles, axis labels, grid, and legend enhance readability and clarity of the chart.

Key Insights from the Visualization:
* Each line represents the opioid-related crude death rate for a particular state over time.
* You can identify:
    * Sharp increases in states like Ohio and Michigan, indicating earlier or more severe impacts.
    * Gradual trends in places like California or Washington, suggesting differing opioid spread patterns.
    * Consistently low rates in states like Hawaii, indicating less severe impact or effective public health controls.
* The chart enables quick cross-state comparison and highlights periods of escalation or stability.

Summary:
This line plot effectively captures how opioid-related death rates have evolved across different U.S. states. It supports data-driven decision-making for policy development, public health intervention planning, and further research on regional opioid trends.






# Overview of michele aguilar code
Anomaly detection in mortality data

Using the data chosen I decided to look further into anomalies in the dataset to see if there was a correlation in years and crude rate. I wanted to identify if our dataset could discover trends or unusual spikes regarding death rate in correlation to pandemics, environmental factors, or other unexpected events.

Feature Selection: Focus on crude_rate, deaths, and prescriptions_dispensed_by_us_retailers_in_that_year_(millions).
Machine Learning: Isolation Forest
Normal points get a label 1 (blue in chart).
Anomalies get a label -1 (red in chart).

Code steps:
- imported necessary dependencies
- Imported csv file
- Select relevant features for anomaly detection

features = ["crude_rate", "deaths", "prescriptions_dispensed_by_us_retailers_in_that_year_(millions)"] 
X = df[features]
- Handle missing values by filling with median values

X = X.fillna(X.median())

- Train Isolation Forest model. -1 indicates anomaly, 1 is normal

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42) df["anomaly"] = model.fit_predict(X) 

- Visualize anomalies in crude rate by using matplotlib
- interactive scatter plot with plotly 










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
