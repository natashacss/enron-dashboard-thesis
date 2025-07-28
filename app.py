import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import pickle
import gdown
import os

st.set_page_config(page_title="Enron Anomaly Dashboard", layout="wide")

# Title + description
st.title("Enron Email Anomaly Dashboard")
st.markdown("Visualizing communication anomalies using ML models on the Enron email dataset.")

@st.cache_data
def load_graph():
    with open("data/anomaly_graph.pkl", "rb") as f:
        return pickle.load(f)

# Load data
@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1SayGoQQzHWA0YBoGgNCgk1yBkLxVnXJa"
    output_path = "data/enron_anomalies.csv"

    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

    return pd.read_csv(output_path)

graph = load_graph()
df = load_data()

# --- JSON stats loader (from summary_stats.json) ---
@st.cache_data
def load_stats():
    with open("data/summary_stats.json") as f:
        return json.load(f)

stats = load_stats()

# Fix LOF score values just for plotting
df['lof_score_clipped'] = np.clip(df['lof_score'], a_min=-3, a_max=-0.5)

# --- Sidebar Controls ---
st.sidebar.header("Filters")

# Step 1: Get user filter inputs first
min_combo = st.sidebar.slider("Minimum Anomaly Combo Score", 0, 3, 0)
min_emails = st.sidebar.slider("Minimum Emails Sent", 1, 100, 5)

# Step 2: Pre-filter data based on these inputs
df_filtered = df.copy()
df_filtered = df_filtered[df_filtered['anomaly_combo'] >= min_combo]
df_filtered = df_filtered[df_filtered['from'].notna()]
df_filtered['email_count'] = df_filtered.groupby('from')['from'].transform('count')
df_filtered = df_filtered[df_filtered['email_count'] >= min_emails]

# Step 3: Compute the dynamic max
max_possible_nodes = df_filtered['from'].nunique()

# Step 4: Use that in the max_nodes slider
max_nodes = st.sidebar.slider(
    "Max Nodes Displayed",
    min_value=10,
    max_value=max_possible_nodes if max_possible_nodes > 10 else 10,
    value=min(150, max_possible_nodes),
    step=10
)

# Optional feedback
st.sidebar.caption(f"{max_possible_nodes} nodes available with current filters.")

# Step 5: Apply to graph data sampling
df_graph = df_filtered[['from', 'to']].dropna().sample(
    min(len(df_filtered), max_nodes),
    random_state=42
)

# Apply filters
if st.sidebar.button("Apply Filter"):
    st.session_state.filtered = True
else:
    if "filtered" not in st.session_state:
        st.session_state.filtered = False

# --- Filter Nodes ---
G = graph.copy()
if st.session_state.filtered:
    st.toast("🔄 Graph has been rebuilt!", icon="✨")

if st.session_state.filtered:
    filtered_nodes = [n for n, d in G.nodes(data=True)
                      if d.get("anomaly_combo", 0) >= min_combo and d.get("email_count", 0) >= min_emails]
    G = G.subgraph(filtered_nodes).copy()
    G = G.subgraph(sorted(G.nodes(), key=lambda n: G.nodes[n].get("email_count", 0), reverse=True)[:max_nodes])

# Show data preview
if st.checkbox("Show raw data"):
    st.dataframe(df.head(20))

# Score distributions
st.subheader("Anomaly Score Distributions")

st.markdown("""
Use the dropdown to select which model's anomaly scoring distribution you'd like to inspect.
This chart helps us understand how each model assigns anomaly scores to email behavior.
Scores further to the right typically indicate higher anomaly likelihood (depending on the model).
""")

model = st.selectbox("Select model to view score distribution:", ["Isolation Forest", "LOF", "One-Class SVM"])

score_col = {
    "Isolation Forest": "iso_score",
    "LOF": "lof_score_clipped",
    "One-Class SVM": "svm_score"
}[model]

fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df[score_col], bins=50, kde=True, ax=ax)

# Improve axis labels
ax.set_xlabel("Anomaly Score", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title(f"{model} Anomaly Score Distribution", fontsize=13)

st.pyplot(fig)

st.caption("""
**How to read this:**  
- The x-axis shows how 'anomalous' an email was considered by the selected model.  
- The y-axis shows how many emails received that score.  
- Taller bars = more emails with that anomaly score.  
- Rightmost tails (or dips) may suggest extreme outliers.
""")

st.caption("""
**Model Score Interpretation Guide:**

- **Isolation Forest**: Lower scores (e.g., near 0 or negative) are more anomalous. Normal behavior clusters closer to 0.5–1.0.
- **LOF (Local Outlier Factor)**: Anomalies tend to have more negative scores, often below -1.5. Plot clipped to [-3, -0.5] for clarity.
- **One-Class SVM**: Scores closer to 0 are more normal. Negative scores typically indicate anomalies.

Each model uses a different scoring range and logic. Focus on the **relative tails** of each distribution to identify outliers.
""")

st.subheader("Total Anomalies Detected by Each Model")

st.markdown("""
This chart shows how many emails each model identified as anomalous using its internal detection rules.
Differences in count reflect how sensitive or conservative each model is.
""")

counts = {
    "Isolation Forest": int(df['iso_anomaly'].eq(-1).sum()),
    "LOF": int(df['lof_anomaly'].eq(-1).sum()),
    "One-Class SVM": int(df['svm_anomaly'].eq(-1).sum())
}

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette='Set2', ax=ax)

# Add count labels
for i, v in enumerate(counts.values()):
    ax.text(i, v + 50, str(v), ha='center', fontsize=10)

ax.set_ylabel("Number of Anomalies")
ax.set_xlabel("Detection Model")
ax.set_title("Anomaly Counts Across Models")

st.pyplot(fig)

st.caption("""
**How to read this:**  
- Each bar shows how many emails were flagged as outliers by the model.  
- A higher bar means the model found more suspicious emails.  
- Use this to compare detection aggressiveness between models.
""")

st.subheader("Anomaly Agreement Across Models")

st.markdown("""
This pie chart shows how often the anomaly detection models agree on which emails are suspicious.
Cases where multiple models agree are usually more trustworthy signals of risk.
""")

labels = ['None flagged', 'Flagged by 1 model', 'Flagged by 2 models', 'Flagged by ALL 3']
sizes = [stats['overlap'].get(str(i), 0) for i in range(4)]
colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756']
explode = [0.03, 0.05, 0.08, 0.1]

# Calculate percentages
total = sum(sizes)
percentages = [(s / total) * 100 for s in sizes]

# Use count + percentage in legend
legend_labels = [
    f"{labels[i]}: {sizes[i]} email(s) ({percentages[i]:.4f}%)"
    for i in range(4)
]

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(sizes, colors=colors, explode=explode, startangle=140)

ax.legend(
    legend_labels,
    title="Flagged By",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=9  # Smaller font to fit long labels
)

ax.set_title("Model Agreement on Anomalies", fontsize=14)
ax.axis('equal')
st.pyplot(fig)

st.caption("""
**How to read this:**  
- Each slice shows how many emails were flagged by 0, 1, 2, or all 3 models.  
- Emails flagged by **2 or 3 models** are likely higher-risk.  
- Even if a slice is tiny, it's still meaningful — those are your rare but critical events.
""")

st.subheader("Enron Communication Network Graph")

# Get number of unique sender nodes
max_possible_nodes = df_filtered['from'].nunique()

# Build the directed graph
G = nx.DiGraph()
for _, row in df_graph.iterrows():
    sender = row['from'].strip()
    recipients = [r.strip() for r in str(row['to']).split(',')]
    for r in recipients:
        G.add_edge(sender, r)

# Create PyVis network
net = Network(height="600px", width="100%", directed=True, notebook=False, bgcolor="#ffffff", font_color="black")

# Define color mapping based on anomaly_combo score
def get_color(anomaly_combo):
    if anomaly_combo == 3:
        return "#e45756"  # dark red
    elif anomaly_combo == 2:
        return "#f58518"  # orange
    elif anomaly_combo == 1:
        return "#ffdd57"  # yellow
    else:
        return "#8ecae6"  # light blue

# Add nodes with correct coloring and metadata
all_nodes = list(G.nodes())
for node in all_nodes:
    user_data = df[df['from'] == node]
    if user_data.empty:
        anomaly_combo = 0
        total_sent = 0
        iso_score = 0.0
        sample_subject = "No subject"
    else:
        anomaly_combo = int(user_data['anomaly_combo'].max())
        total_sent = len(user_data)
        iso_score = user_data['iso_score'].mean()
        subject_series = user_data['subject'].dropna()
        sample_subject = subject_series.sample(1).values[0] if not subject_series.empty else "No subject"
        if len(sample_subject) > 100:
            sample_subject = sample_subject[:100] + "..."

    hover_text = (
        f"{node}\n"
        f"📬 Total Sent: {total_sent}\n"
        f"📊 Anomaly Combo Score: {anomaly_combo}\n"
        f"🌡️ Avg ISO Score: {iso_score:.4f}\n"
        f"📝 Sample Subject:\n{sample_subject}"
    )

    node_size = 10 + total_sent // 10
    color = get_color(anomaly_combo)

    net.add_node(node, label=node, title=hover_text, color=color, size=node_size)

# Add edges
for edge in G.edges():
    net.add_edge(edge[0], edge[1])

# Save and render
net.set_edge_smooth('dynamic')
net.save_graph("graph.html")
st.download_button("📥 Download Graph HTML", data=open("graph.html").read(), file_name="anomaly_graph.html")
components.html(open("graph.html", "r", encoding='utf-8').read(), height=650)

import streamlit as st
import pandas as pd
import numpy as np

st.subheader("High-Risk Entities")

# Create a button to trigger refresh
if st.button("🔄 Refresh Table"):
    st.session_state.refresh_ews = True

# Create session state to store randomized high-risk table
if "refresh_ews" not in st.session_state or st.session_state.refresh_ews:
    high_risk_candidates = df[df['anomaly_combo'] >= 2].copy()
    high_risk_candidates = high_risk_candidates.dropna(subset=['from'])
    
    # Get one row per 'from' only
    high_risk_unique = (
        high_risk_candidates
        .sort_values(by='iso_score', ascending=False)
        .drop_duplicates(subset='from', keep='first')
        .sample(10, random_state=np.random.randint(0, 99999))  # randomize sample
    )
    
    # Clean column names
    high_risk_display = high_risk_unique[['from', 'anomaly_combo', 'iso_score', 'subject']].copy()
    high_risk_display.columns = ['Sender Email', 'Anomaly Score', 'ISO Score', 'Sample Subject']
    
    st.session_state.ews_table = high_risk_display
    st.session_state.refresh_ews = False

# Display the current EWS table
st.table(st.session_state.ews_table)

st.subheader("Search Email or Sender")

# User input
user_input = st.text_input("Enter email address or keyword to search:")

if user_input:
    # Filter by 'from' field
    user_records = df[df['from'].str.contains(user_input, case=False, na=False)].copy()
    
    if not user_records.empty:
        st.success(f"Found {len(user_records)} email(s) from '{user_input}'.")

        # Prettify table
        display_df = user_records[['date', 'to', 'subject', 'anomaly_combo', 'iso_score']].copy()
        display_df.columns = ['📅 Date', '📥 Recipient(s)', '📝 Subject', '⚠️ Anomaly Score', '🌡 ISO Score']
        
        # Reset index for selectbox reference
        display_df = display_df.reset_index(drop=True)
        
        # Show table
        st.dataframe(display_df.head(10), use_container_width=True)
        
        # Subject selector
        selected_idx = st.selectbox(
            "🧾 Click to view full message from selected subject:",
            options=display_df.index,
            format_func=lambda i: display_df.loc[i, '📝 Subject']
        )
        
        # Show full message below
        full_msg = user_records.iloc[selected_idx]['message']
        st.markdown("### Full Message")
        st.info(full_msg if isinstance(full_msg, str) and full_msg.strip() else "No message content available.")
    
    else:
        st.warning("No matching sender found.")
