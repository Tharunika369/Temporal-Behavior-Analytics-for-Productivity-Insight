import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import textwrap
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Behavioral Data Explorer", layout="wide")

# apply a simple dark theme via CSS
dark_css = """
<style>
/* set dark background for main app and sidebar */
.stApp, .css-1d391kg, .css-1lcbmhc {background-color: #0e1117 !important; color: #fafafa !important;}
.stSidebar .css-1d391kg, .stSidebar .css-1lcbmhc {background-color: #0e1117 !important; color: #fafafa !important;}
a, .stMarkdown > p {color: #fafafa !important;}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)


# Apply a seaborn style for Matplotlib plots
sns.set_theme(style="whitegrid")

# Load custom CSS if available
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path, 'r', encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Behavioral Data Explorer</h1>", unsafe_allow_html=True)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

# File uploader and default path
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"] )
default_path = Path(__file__).parent / "final_behavior_500.csv"
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded uploaded CSV — {df.shape[0]} rows")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
elif default_path.exists():
    try:
        df = load_csv(default_path)
        st.sidebar.success(f"Loaded dataset: {default_path.name} — {df.shape[0]} rows")
    except Exception as e:
        st.sidebar.error(f"Failed to load {default_path.name}: {e}")
else:
    st.sidebar.info("No CSV found. Upload a CSV to get started.")

if df is None:
    st.info("Upload `final_behavior_500.csv` or another CSV to begin exploring your data.")
    st.stop()

# Normalize selection: show normalized columns option
norm_cols = [c for c in df.columns if c.endswith("_norm")]
use_norm = st.sidebar.checkbox("Prefer normalized metrics when available (columns ending with _norm)", value=True if norm_cols else False)

# Determine categorical and numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Categorical filters per user's request
# try loading available label encoders to show human labels instead of codes
import joblib
encoders = {}
en_path = Path(__file__).parent / "label_encoders.pkl"
if en_path.exists():
    try:
        encoders = joblib.load(en_path)
    except Exception:
        encoders = {}

cat_filters = {}
for c in ['gender','region','income_level','education_level','daily_role','device_type']:
    if c in df.columns:
        vals = df[c].dropna().unique()
        # if we have an encoder for this column, map numeric codes to labels for display
        if c in encoders:
            le = encoders[c]
            try:
                label_map = {code: label for code, label in enumerate(le.classes_)}
                display_vals = [label_map.get(v, v) for v in vals]
            except Exception:
                display_vals = vals.tolist()
        else:
            display_vals = vals.tolist()
        options = ['All'] + sorted(display_vals)
        sel = st.sidebar.multiselect(f"Filter {c}", options=options, default=['All'])
        cat_filters[c] = sel

# Generic filters
if 'age' in df.columns:
    min_age, max_age = int(df['age'].min()), int(df['age'].max())
    age_range = st.sidebar.slider('Age range', min_value=min_age, max_value=max_age, value=(min_age, max_age))
else:
    age_range = None

st.sidebar.markdown('---')
plot_type = st.sidebar.selectbox('Plot type', options=['Scatter','Line','Histogram','Boxplot','Bar','Correlation Heatmap'])

# Column selectors
st.sidebar.markdown('Select columns for plotting')
if plot_type in ['Scatter','Line','Boxplot','Bar']:
    x_col = st.sidebar.selectbox('X axis', options=numeric_cols + cat_cols, index=0 if numeric_cols else 0)
    y_col = st.sidebar.selectbox('Y axis', options=numeric_cols + cat_cols, index=1 if len(numeric_cols)>1 else 0)
    color_col = st.sidebar.selectbox('Color by (categorical)', options=['None'] + cat_cols, index=0)
elif plot_type == 'Histogram':
    hist_col = st.sidebar.selectbox('Histogram column', options=numeric_cols, index=0 if numeric_cols else 0)
elif plot_type == 'Correlation Heatmap':
    corr_cols = st.sidebar.multiselect('Numeric columns for correlation', options=numeric_cols, default=numeric_cols[:10])

st.sidebar.markdown('---')
show_summary = st.sidebar.checkbox('Show summary statistics', value=True)
use_seaborn = st.sidebar.checkbox('Render plots with Seaborn/Matplotlib when available', value=False)

def apply_filters(df):
    d = df.copy()
    # categorical filters (decode selections back to codes if necessary)
    for c, sel in cat_filters.items():
        if sel and 'All' not in sel:
            if c in encoders:
                le = encoders[c]
                # try to transform selected labels to codes
                try:
                    codes = le.transform(sel)
                    d = d[d[c].isin(codes)]
                except Exception:
                    d = d[d[c].isin(sel)]
            else:
                d = d[d[c].isin(sel)]
    # age
    if age_range is not None:
        d = d[(d['age'] >= age_range[0]) & (d['age'] <= age_range[1])]
    return d

dff = apply_filters(df)

st.markdown(f"### Data preview — {dff.shape[0]} rows × {dff.shape[1]} columns")
st.dataframe(dff.head(10))

if show_summary:
    st.subheader('Summary statistics')
    num = dff.describe().T
    st.dataframe(num)

st.markdown('---')

def plot_scatter(df, x, y, color=None):
    title = f"Scatter: {y} vs {x}"
    if use_seaborn:
        fig, ax = plt.subplots(figsize=(8,5))
        if color and color in df.columns:
            sns.scatterplot(data=df, x=x, y=y, hue=color, ax=ax, alpha=0.8)
        else:
            sns.scatterplot(data=df, x=x, y=y, ax=ax, alpha=0.7)
        ax.set_title(title)
        st.pyplot(fig)
    else:
        fig = px.scatter(df, x=x, y=y, color=color if color!='None' else None, hover_data=df.columns)
        fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
        st.plotly_chart(fig, width='stretch')

def plot_line(df, x, y, color=None):
    title = f"Line: {y} over {x}"
    if use_seaborn:
        fig, ax = plt.subplots(figsize=(8,5))
        if color and color in df.columns:
            sns.lineplot(data=df, x=x, y=y, hue=color, ax=ax)
        else:
            sns.lineplot(data=df, x=x, y=y, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)
    else:
        fig = px.line(df, x=x, y=y, color=color if color!='None' else None)
        fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
        st.plotly_chart(fig, width='stretch')

def plot_hist(df, col):
    title = f"Histogram: {col}"
    if use_seaborn:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(data=df, x=col, kde=True, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)
    else:
        fig = px.histogram(df, x=col, nbins=40)
        fig.update_layout(title=title, xaxis_title=col)
        st.plotly_chart(fig, width='stretch')

def plot_box(df, x, y):
    title = f"Boxplot: {y} by {x}"
    if use_seaborn:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(data=df, x=x, y=y, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)
    else:
        fig = px.box(df, x=x, y=y, points='outliers')
        fig.update_layout(title=title)
        st.plotly_chart(fig, width='stretch')

def plot_bar(df, x, y):
    title = f"Bar: {y} by {x}"
    if use_seaborn:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(data=df, x=x, y=y, ax=ax)
        ax.set_title(title)
        st.pyplot(fig)
    else:
        fig = px.bar(df, x=x, y=y)
        fig.update_layout(title=title)
        st.plotly_chart(fig, width='stretch')

def plot_corr(df, cols):
    title = "Correlation heatmap"
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu', center=0, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# If normalized toggle is set, prefer norm columns for certain metrics
if use_norm:
    # Attempt to replace columns with normalized equivalents where available
    def get_preferred(col):
        norm = f"{col}_norm"
        if norm in dff.columns:
            return norm
        return col
else:
    def get_preferred(col):
        return col

# Main plotting logic
if plot_type == 'Scatter':
    x = get_preferred(x_col)
    y = get_preferred(y_col)
    plot_scatter(dff, x, y, color=(None if color_col=='None' else color_col))
elif plot_type == 'Line':
    x = get_preferred(x_col)
    y = get_preferred(y_col)
    plot_line(dff, x, y, color=(None if color_col=='None' else color_col))
elif plot_type == 'Histogram':
    col = get_preferred(hist_col)
    plot_hist(dff, col)
elif plot_type == 'Boxplot':
    plot_box(dff, x_col, y_col)
elif plot_type == 'Bar':
    plot_bar(dff, x_col, y_col)
elif plot_type == 'Correlation Heatmap':
    if corr_cols:
        plot_corr(dff, corr_cols)
    else:
        st.info('Pick numeric columns for correlation heatmap')

# Key charts focusing on stress, sleep, social media, focus, happiness
st.markdown('---')
st.header('Key charts')
key_features = ['stress_level','sleep_hours','social_media_mins','focus_score','happiness_score']
present = [c for c in key_features if c in dff.columns]

if len(present) >= 2:
    # 1) Scatter: stress_level vs sleep_hours
    if 'stress_level' in dff.columns and 'sleep_hours' in dff.columns:
        st.subheader('Stress vs Sleep')
        if use_seaborn:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.scatterplot(data=dff, x='sleep_hours', y='stress_level', hue=( 'device_type' if 'device_type' in dff.columns else None), alpha=0.7, ax=ax)
            ax.set_xlabel('Sleep hours')
            ax.set_ylabel('Stress level')
            ax.set_title('Stress level vs Sleep hours')
            st.pyplot(fig)
        else:
            fig = px.scatter(dff, x='sleep_hours', y='stress_level', color=( 'device_type' if 'device_type' in dff.columns else None), hover_data=cat_cols)
            fig.update_layout(title='Stress level vs Sleep hours', xaxis_title='Sleep hours', yaxis_title='Stress level')
            st.plotly_chart(fig, width='stretch')

    # 2) Scatter: Stress Level vs Anxiety Score (identify high-risk individuals)
    if 'stress_level' in dff.columns and 'anxiety_score' in dff.columns:
        st.subheader('Stress Level vs Anxiety Score — identify high-risk individuals')
        # Color by risk_score if present, else by device_type if available
        color_by = 'risk_score' if 'risk_score' in dff.columns else ('device_type' if 'device_type' in dff.columns else None)
        if use_seaborn:
            fig, ax = plt.subplots(figsize=(8,5))
            if color_by == 'risk_score':
                sc = ax.scatter(dff['anxiety_score'], dff['stress_level'], c=dff['risk_score'], cmap='Reds', alpha=0.8)
                fig.colorbar(sc, ax=ax, label='risk_score')
            elif color_by:
                sns.scatterplot(data=dff, x='anxiety_score', y='stress_level', hue=color_by, ax=ax, alpha=0.8)
            else:
                sns.scatterplot(data=dff, x='anxiety_score', y='stress_level', ax=ax, alpha=0.8)
            ax.set_xlabel('Anxiety score')
            ax.set_ylabel('Stress level')
            ax.set_title('Stress vs Anxiety')
            st.pyplot(fig)
        else:
            fig = px.scatter(dff, x='anxiety_score', y='stress_level', color=color_by if color_by else None, hover_data=cat_cols)
            fig.update_layout(title='Stress vs Anxiety', xaxis_title='Anxiety score', yaxis_title='Stress level')
            st.plotly_chart(fig, width='stretch')

        # Optionally highlight top-risk individuals if risk_score exists
        if 'risk_score' in dff.columns:
            thresh = st.slider('Highlight top-risk percentile (risk_score)', min_value=90, max_value=100, value=95)
            top_thresh = np.percentile(dff['risk_score'].dropna(), thresh)
            top = dff[dff['risk_score'] >= top_thresh]
            if not top.empty:
                st.markdown(f"Top {100-thresh}% risk threshold: >= {top_thresh:.2f} — {len(top)} rows")
                if use_seaborn:
                    fig, ax = plt.subplots(figsize=(8,4))
                    sns.scatterplot(data=top, x='anxiety_score', y='stress_level', color='red', s=40, ax=ax)
                    ax.set_title('Highlighted high-risk individuals')
                    st.pyplot(fig)
                else:
                    fig = px.scatter(top, x='anxiety_score', y='stress_level', color_discrete_sequence=['red'], hover_data=cat_cols)
                    fig.update_layout(title='Highlighted high-risk individuals')
                    st.plotly_chart(fig, width='stretch')

    # 3) Boxplot: Focus Score by Daily Role or Department
    if 'focus_score' in dff.columns and ('daily_role' in dff.columns or 'department' in dff.columns):
        group_col = 'daily_role' if 'daily_role' in dff.columns else 'department'
        st.subheader(f'Focus score by {group_col} — roles/departments with attention challenges')
        agg_count = dff.groupby(group_col)['focus_score'].count().sort_values(ascending=False)
        # Show top groups for readability
        top_n = st.number_input(f'Number of {group_col} groups to show', min_value=3, max_value=50, value=10)
        top_groups = agg_count.head(top_n).index.tolist()
        plot_df = dff[dff[group_col].isin(top_groups)]
        if use_seaborn:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.boxplot(data=plot_df, x=group_col, y='focus_score', ax=ax)
            ax.set_xlabel(group_col)
            ax.set_ylabel('Focus score')
            ax.set_title(f'Focus score by {group_col}')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            fig = px.box(plot_df, x=group_col, y='focus_score', points='outliers')
            fig.update_layout(title=f'Focus score by {group_col}', xaxis_title=group_col, yaxis_title='Focus score')
            st.plotly_chart(fig, width='stretch')

    # 4) Histogram of focus_score (kept as optional)
    elif 'focus_score' in dff.columns:
        st.subheader('Focus score distribution')
        if use_seaborn:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.histplot(dff['focus_score'].dropna(), kde=True, ax=ax)
            ax.set_title('Distribution of Focus Score')
            ax.set_xlabel('Focus score')
            st.pyplot(fig)
        else:
            fig = px.histogram(dff, x='focus_score', nbins=40)
            fig.update_layout(title='Distribution of Focus Score', xaxis_title='Focus score')
            st.plotly_chart(fig, width='stretch')

    # 4) Bar chart: happiness_score across income_level
    if 'happiness_score' in dff.columns and 'income_level' in dff.columns:
        st.subheader('Average happiness by income level')
        agg = dff.groupby('income_level')['happiness_score'].mean().reset_index().sort_values('happiness_score', ascending=False)
        if use_seaborn:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(data=agg, x='happiness_score', y='income_level', palette='viridis', ax=ax)
            ax.set_xlabel('Mean happiness score')
            ax.set_ylabel('Income level')
            ax.set_title('Average happiness by income level')
            st.pyplot(fig)
        else:
            fig = px.bar(agg, x='happiness_score', y='income_level', orientation='h')
            fig.update_layout(title='Average happiness by income level', xaxis_title='Mean happiness score', yaxis_title='Income level')
            st.plotly_chart(fig, width='stretch')

    # 5) Correlation heatmap of the five features plus risk_score
    corr_features = [c for c in (key_features + ['risk_score']) if c in dff.columns]
    if len(corr_features) >= 2:
        st.subheader('Correlation heatmap (key features + risk_score)')
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(dff[corr_features].corr(), annot=True, fmt='.2f', cmap='RdBu', center=0, ax=ax)
        ax.set_title('Correlation matrix')
        st.pyplot(fig)
else:
    st.info('Not enough of the key metrics are present to show the key charts.')

st.markdown('---')

st.markdown('---')

st.write('Export filtered data:')
st.download_button('Download CSV', data=dff.to_csv(index=False), file_name='filtered_behavioral.csv', mime='text/csv')

st.markdown('### Tips')
st.markdown('- Use the filters on the left to focus on subgroups (gender, region, income, etc.).')
st.markdown('- Toggle Seaborn rendering for publication-quality plots or use Plotly for interactivity.')

