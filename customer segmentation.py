# ============================================================
# CUSTOMER SEGMENTATION PROJECT
# Python + Pandas + Scikit-learn + Matplotlib + Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# PAGE CONFIGURATION (PAGE VISUALISATION)
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE OF THE PAGE
# ============================================================

st.title("📊 Customer Segmentation System")

st.write(
    "This application uses K-Means Machine Learning "
    "to divide customers into different groups."
)


# ============================================================
# SIDEBAR OF PROJECT FOR UPLOADING CSV FILE
# ============================================================

st.sidebar.title("Customer Segmentation")

uploaded_file = st.sidebar.file_uploader(
    "Upload customers.csv",
    type=["csv"]
)


# ============================================================
# LOAD DATA FROM THE UPLOADED CSV FILE
# ============================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

else:

    try:

        df = pd.read_csv("customers.csv")

        st.info(
            "customers.csv loaded from project folder."
        )

    except FileNotFoundError:

        st.error(
            "customers.csv not found. "
            "Please upload the CSV file."
        )

        st.stop()


# ============================================================
#  HERE ARE THE PREVIEW OF DATASET
# ============================================================

st.header("1. Datasets")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "Total Columns",
        len(df.columns)
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )


st.subheader("Customers Data")

st.dataframe(
    df,
    use_container_width=True
)


# ============================================================
# DATA CLEANING TO AVOIDE MISTAKES 
# ============================================================

st.header("2. Data Cleaning")

# Remove duplicates

df = df.drop_duplicates()


# Find numeric columns

numeric_columns = df.select_dtypes(
    include=np.number
).columns


# Fill missing numerical values

for column in numeric_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )


st.success(
    "Duplicate records removed and missing numerical "
    "values handled successfully."
)


# ============================================================
#  SHOWINNG MISSING VALUES IF PRESENT
# ============================================================

st.subheader("Missing Values From Given Datasts:")

missing_values = df.isnull().sum()

missing_df = pd.DataFrame({
    "Column": missing_values.index,
    "Missing Values": missing_values.values
})

st.dataframe(
    missing_df,
    use_container_width=True
)


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

st.header("3. Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)


# ============================================================
# FEATURE SELECTION
# ============================================================

st.header("4. Feature Selection")

available_features = list(
    df.select_dtypes(
        include=np.number
    ).columns
)


if len(available_features) < 2:

    st.error(
        "At least two numerical columns are required."
    )

    st.stop()


# Preferred features

preferred_features = [
    "Income",
    "SpendingScore",
    "PurchaseFrequency"
]


default_features = [
    feature
    for feature in preferred_features
    if feature in available_features
]


if len(default_features) < 2:

    default_features = available_features[:2]


selected_features = st.multiselect(
    "Select features for clustering:",
    available_features,
    default=default_features
)


if len(selected_features) < 2:

    st.warning(
        "Please select at least two features."
    )

    st.stop()


# ============================================================
# EDA
# ============================================================

st.header("5. Exploratory Data Analysis")


eda_column = st.selectbox(
    "Select a column for distribution:",
    selected_features
)


fig, ax = plt.subplots()

ax.hist(
    df[eda_column],
    bins=10,
    edgecolor="black"
)

ax.set_xlabel(eda_column)
ax.set_ylabel("Number of Customers")
ax.set_title(
    f"{eda_column} Distribution"
)

st.pyplot(fig)


# ============================================================
# SCATTER PLOT
# ============================================================

st.subheader("Relationship Between Features")

x_feature = st.selectbox(
    "Select X-axis:",
    selected_features,
    index=0
)

y_feature = st.selectbox(
    "Select Y-axis:",
    selected_features,
    index=min(1, len(selected_features) - 1)
)


fig, ax = plt.subplots()

ax.scatter(
    df[x_feature],
    df[y_feature],
    s=70
)

ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.set_title(
    f"{x_feature} vs {y_feature}"
)

st.pyplot(fig)


# ============================================================
# FEATURE SCALING
# ============================================================

X = df[selected_features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# ELBOW METHOD
# ============================================================

st.header("6. Elbow Method")

max_clusters = min(
    10,
    len(df) - 1
)


if max_clusters < 2:

    st.error(
        "Not enough customers for clustering."
    )

    st.stop()


inertia = []


for k in range(2, max_clusters + 1):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(
        model.inertia_
    )


fig, ax = plt.subplots()

ax.plot(
    range(2, max_clusters + 1),
    inertia,
    marker="o"
)

ax.set_xlabel(
    "Number of Clusters"
)

ax.set_ylabel(
    "Inertia"
)

ax.set_title(
    "Elbow Method"
)

st.pyplot(fig)


# ============================================================
# NUMBER OF CLUSTERS
# ============================================================

st.sidebar.header(
    "Machine Learning Settings"
)


number_of_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=max_clusters,
    value=min(4, max_clusters)
)


# ============================================================
# K-MEANS MODEL
# ============================================================

st.header("7. K-Means Customer Segmentation")


kmeans = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=10
)


df["Cluster"] = kmeans.fit_predict(
    X_scaled
)


# ============================================================
# SILHOUETTE SCORE
# ============================================================

silhouette = silhouette_score(
    X_scaled,
    df["Cluster"]
)


st.metric(
    "Silhouette Score",
    round(silhouette, 3)
)


# ============================================================
# CLUSTER PROFILE
# ============================================================

st.header("8. Customer Segment Profile")


profile = df.groupby(
    "Cluster"
)[selected_features].mean()


profile["Customer Count"] = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)


st.dataframe(
    profile.round(2),
    use_container_width=True
)


# ============================================================
# SEGMENT NAMES
# ============================================================

# Calculate a score for each cluster

cluster_scores = profile[
    selected_features
].mean(axis=1)


sorted_clusters = (
    cluster_scores
    .sort_values(ascending=False)
    .index
    .tolist()
)


segment_names = {}


if len(sorted_clusters) >= 1:

    segment_names[
        sorted_clusters[0]
    ] = "High Value Customers"


if len(sorted_clusters) >= 2:

    segment_names[
        sorted_clusters[1]
    ] = "Potential Customers"


if len(sorted_clusters) >= 3:

    segment_names[
        sorted_clusters[-1]
    ] = "Low Value Customers"


for cluster in sorted_clusters:

    if cluster not in segment_names:

        segment_names[
            cluster
        ] = "Regular Customers"


df["Segment"] = df[
    "Cluster"
].map(segment_names)


# ============================================================
# SEGMENTED CUSTOMER TABLE
# ============================================================

st.header("9. Segmented Customers")


display_columns = []

if "CustomerID" in df.columns:

    display_columns.append(
        "CustomerID"
    )


display_columns.extend(
    selected_features
)

display_columns.extend(
    [
        "Cluster",
        "Segment"
    ]
)


st.dataframe(
    df[display_columns],
    use_container_width=True
)


# ============================================================
# CUSTOMER COUNT BY SEGMENT
# ============================================================

st.header("10. Customer Distribution")


segment_count = (
    df["Segment"]
    .value_counts()
)


fig, ax = plt.subplots()

ax.bar(
    segment_count.index,
    segment_count.values
)

ax.set_xlabel(
    "Customer Segment"
)

ax.set_ylabel(
    "Number of Customers"
)

ax.set_title(
    "Customers by Segment"
)

plt.xticks(
    rotation=30
)

st.pyplot(fig)


# ============================================================
# CLUSTER VISUALIZATION
# ============================================================

st.header("11. Customer Segmentation Visualization")


fig, ax = plt.subplots()


for cluster in range(
    number_of_clusters
):

    cluster_data = df[
        df["Cluster"] == cluster
    ]

    ax.scatter(
        cluster_data[x_feature],
        cluster_data[y_feature],
        s=80,
        label=segment_names.get(
            cluster,
            f"Cluster {cluster}"
        )
    )


ax.set_xlabel(
    x_feature
)

ax.set_ylabel(
    y_feature
)

ax.set_title(
    "Customer Segmentation using K-Means"
)

ax.legend()


st.pyplot(fig)


# ============================================================
# CLUSTER CENTERS
# ============================================================

st.header("12. Cluster Centers")


centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)


centers_df = pd.DataFrame(
    centers,
    columns=selected_features
)


centers_df["Cluster"] = range(
    number_of_clusters
)


centers_df["Segment"] = (
    centers_df["Cluster"]
    .map(segment_names)
)


st.dataframe(
    centers_df.round(2),
    use_container_width=True
)


# ============================================================
# MARKETING RECOMMENDATIONS
# ============================================================

st.header("13. Marketing Recommendations")


for cluster in range(
    number_of_clusters
):

    cluster_data = df[
        df["Cluster"] == cluster
    ]


    segment = segment_names.get(
        cluster,
        f"Cluster {cluster}"
    )


    with st.expander(
        f"📌 {segment}"
    ):

        avg_values = (
            cluster_data[
                selected_features
            ].mean()
        )


        st.write(
            "Average Customer Profile:"
        )


        for feature in selected_features:

            st.write(
                f"**{feature}:** "
                f"{avg_values[feature]:.2f}"
            )


        # Marketing strategy

        if segment == "High Value Customers":

            st.success(
                "Strategy: Provide loyalty rewards, "
                "premium products, exclusive offers "
                "and personalized services."
            )


        elif segment == "Potential Customers":

            st.info(
                "Strategy: Use personalized discounts, "
                "product recommendations and targeted "
                "promotional campaigns."
            )


        elif segment == "Low Value Customers":

            st.warning(
                "Strategy: Use re-engagement campaigns, "
                "discounts and low-cost marketing."
            )


        else:

            st.info(
                "Strategy: Use loyalty points, "
                "bundle offers and regular promotions."
            )


# ============================================================
# CUSTOMER SEARCH 
# ============================================================

st.header("14. Search Customer")


if "CustomerID" in df.columns:

    customer_id = st.text_input(
        "Enter Customer ID:"
    )


    if customer_id:

        result = df[
            df["CustomerID"]
            .astype(str)
            .str.contains(
                customer_id,
                case=False,
                na=False
            )
        ]


        if len(result) > 0:

            st.success(
                "Customer found."
            )

            st.dataframe(
                result,
                use_container_width=True
            )

        else:

            st.error(
                "Customer not found."
            )


# ============================================================
# DOWNLOAD SEGMENTED DATA 
# ============================================================

st.header("15. Download Results")


csv_data = df.to_csv(
    index=False
)


st.download_button(
    label="⬇️ Download Segmented Customers CSV",
    data=csv_data,
    file_name="segmented_customers.csv",
    mime="text/csv"
)


# ============================================================
# SAVE DATA LOCALLY
# ============================================================

df.to_csv(
    "segmented_customers.csv",
    index=False
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.write(
    "Customer Segmentation Project | "
    "Python + Pandas + Scikit-learn + Streamlit"
)