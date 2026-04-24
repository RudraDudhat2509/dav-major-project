import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─── page config ───
st.set_page_config(
    page_title="AI Tool Usage — IIT Survey",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    letter-spacing: -0.5px;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #e8c97e33;
    border-radius: 12px;
    padding: 20px 24px;
    color: #f0ead6;
    text-align: center;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #e8c97e;
    font-family: 'DM Serif Display', serif;
}
.metric-label {
    font-size: 0.82rem;
    color: #a0a8b8;
    margin-top: 4px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.section-header {
    border-left: 4px solid #e8c97e;
    padding-left: 12px;
    margin: 24px 0 16px 0;
    color: #1a1a2e;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f3460 100%);
}
[data-testid="stSidebar"] * {
    color: #f0ead6 !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════
# DATA LOAD + CLEAN (cached)
# ════════════════════════════════════════
@st.cache_data
def load_and_clean():
    df = pd.read_csv("survey.csv")
    df.columns = [
        'timestamp','year','field','gender','institution',
        'exp','first_tool','primary_tool','tools_used','use_cases',
        'freq','hours_saved','trust','subscription','submitted_ai',
        'skill_concern','nps'
    ]

    # joke rows hatao
    joke_kw = ['boyfriend','virgin','siri']
    bad = df['field'].str.lower().apply(lambda x: any(k in str(x) for k in joke_kw))
    bad |= df['year'].isna()
    bad |= df['first_tool'].str.lower().apply(lambda x: any(k in str(x) for k in joke_kw))
    df = df[~bad].reset_index(drop=True)

    # ordinal encoding
    df['freq_n']   = df['freq'].map({'Rarely':1,'A few times a week':2,'Once a day':3,'Multiple times a day':4})
    df['hours_n']  = df['hours_saved'].map({'0 hours':0,'1–2 hours':1.5,'3–5 hours':4,'6–10 hours':8,'More than 10 hours':12})
    df['exp_n']    = df['exp'].map({'Never used':0,'3–6 months':0.5,'6–12 months':0.75,'1–2 years':1.5,'2+ years':2.5})
    df['sub_n']    = df['subscription'].map({'No, I only use free tiers/models':0,
                                              'Yes, I pay for a subscription myself':2,
                                              'Yes, my institution/scholarship pays for my subscription':1})
    df['ethics_n'] = df['submitted_ai'].map({'Never':0,'Rarely':1,'Sometimes':2,'Often':3,'Prefer not to say':np.nan})
    df['trust']    = pd.to_numeric(df['trust'], errors='coerce')
    df['concern']  = pd.to_numeric(df['skill_concern'], errors='coerce')
    df['nps']      = pd.to_numeric(df['nps'], errors='coerce')

    # multi-select explode
    ALL_TOOLS = ['ChatGPT','Claude','Gemini','Microsoft Copilot','GitHub Copilot',
                 'Perplexity','Grok','DeepSeek','Cursor','Notion AI','Grammarly AI',
                 'DALL·E','Midjourney','Stable Diffusion','Suno','ElevenLabs']
    ALL_CASES = [
        'Coding & debugging',
        'Assignments & homework (conceptual help)',
        'Studying concepts/explaining difficult topics',
        'Writing & essays (proofreading, idea generation)',
        'Research & summarization',
        'Image/video generation (creative projects)',
        'Career prep (e.g., mock interviews, resume review)',
        'Productivity (e.g., scheduling, email drafts)',
        'Entertainment'
    ]
    for t in ALL_TOOLS:
        df[f't_{t}'] = df['tools_used'].str.contains(t, na=False).astype(int)
    uc_cols = []
    for c in ALL_CASES:
        col = 'uc_' + c.split('(')[0].strip().replace(' & ','_').replace(' ','_').replace('/','_')
        df[col] = df['use_cases'].str.contains(c, na=False).astype(int)
        uc_cols.append(col)

    df['tool_count']    = df[[f't_{t}' for t in ALL_TOOLS]].sum(axis=1)
    df['usecase_count'] = df[uc_cols].sum(axis=1)

    def map_primary(t):
        t = str(t)
        if 'Claude' in t: return 'Claude'
        if 'ChatGPT' in t: return 'ChatGPT'
        return 'Other'

    def clean_tool(t):
        t = str(t)
        for kw, label in [('ChatGPT','ChatGPT'),('Claude','Claude'),
                           ('Gemini','Gemini'),('Bard','Gemini'),
                           ('Grammarly','Grammarly'),('Copilot','Copilot'),
                           ('Perplexity','Perplexity')]:
            if kw in t: return label
        if "Haven't" in t or 'Never' in t: return 'None'
        return 'Other'

    df['tool_label']    = df['primary_tool'].apply(map_primary)
    df['first_clean']   = df['first_tool'].apply(clean_tool)
    df['current_clean'] = df['primary_tool'].apply(clean_tool)

    return df, ALL_TOOLS, ALL_CASES, uc_cols

df, ALL_TOOLS, ALL_CASES, uc_cols = load_and_clean()

UC_LABELS = ['Coding','Assignments','Study/Explain','Writing',
             'Research','Image Gen','Career Prep','Productivity','Entertainment']

CLUSTER_NAMES = {0:'Pragmatist', 1:'Power User', 2:'Skeptic/Casual'}
COLORS = px.colors.qualitative.Set2


# ════════════════════════════════════════
# SIDEBAR NAV
# ════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🤖 AI Survey Dashboard")
    st.markdown("**IIT Students · April 2026**")
    st.divider()
    page = st.radio("Navigate", [
        "📊 Overview",
        "🔥 Tool × Use Case",
        "🔀 Tool Migration",
        "👥 Clustering",
        "📈 Regression",
        "🌳 Classification"
    ])
    st.divider()
    st.markdown(f"**N = {len(df)} valid responses**")
    st.markdown("*After removing joke/invalid rows*")


# ════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════
if page == "📊 Overview":
    st.title("Right Tool, Right Job?")
    st.markdown("#### Decoding AI Adoption Patterns Among IIT Students")
    st.divider()

    # summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (str(len(df)), "Respondents"),
        (str(df['tool_count'].median().astype(int)), "Median Tools Used"),
        (f"{df['hours_n'].median():.0f}h", "Median Hours Saved/Week"),
        (f"{df['trust'].mean():.1f}/5", "Avg Trust Score"),
        (f"{df['nps'].mean():.1f}/10", "Avg NPS"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4,c5], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Respondents by Year")
        yr = df['year'].value_counts().reset_index()
        yr.columns = ['Year','Count']
        fig = px.bar(yr, x='Year', y='Count', color='Count',
                     color_continuous_scale='Blues', text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("##### Primary Tool Distribution")
        pt = df['tool_label'].value_counts().reset_index()
        pt.columns = ['Tool','Count']
        fig = px.pie(pt, values='Count', names='Tool', hole=0.45,
                     color_discrete_sequence=COLORS)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("##### Field of Study Breakdown")
        fd = df['field'].value_counts().reset_index()
        fd.columns = ['Field','Count']
        fig = px.bar(fd, x='Count', y='Field', orientation='h',
                     color='Count', color_continuous_scale='Oranges', text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown("##### Ethics — Submitted AI Content as Own Work")
        eth = df['submitted_ai'].value_counts().reset_index()
        eth.columns = ['Response','Count']
        fig = px.bar(eth, x='Response', y='Count', color='Response',
                     color_discrete_sequence=px.colors.qualitative.Pastel, text='Count')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Trust & Skill Atrophy Concern Distribution")
    col_e, col_f = st.columns(2)
    with col_e:
        fig = px.histogram(df, x='trust', nbins=5, color_discrete_sequence=['#5B8DB8'],
                           labels={'trust':'Trust Score (1–5)'}, title='Trust in AI Outputs')
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)
    with col_f:
        fig = px.histogram(df, x='concern', nbins=5, color_discrete_sequence=['#E07B54'],
                           labels={'concern':'Concern Level (1–5)'}, title='Skill Atrophy Concern')
        fig.update_layout(height=280)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════
# PAGE: TOOL × USE CASE
# ════════════════════════════════════════
elif page == "🔥 Tool × Use Case":
    st.title("Tool × Use Case Heatmap")
    st.markdown("Which tools are being used for which jobs — across all respondents.")
    st.divider()

    focus_tools = st.multiselect("Select tools to display",
        ['ChatGPT','Claude','Gemini','GitHub Copilot','Perplexity',
         'DeepSeek','Cursor','Grammarly AI','Microsoft Copilot','Notion AI'],
        default=['ChatGPT','Claude','Gemini','GitHub Copilot','Perplexity','DeepSeek'])

    hm_data = []
    for tool in focus_tools:
        col_name = f't_{tool}'
        if col_name not in df.columns: continue
        tool_users = df[df[col_name] == 1]
        row = [tool_users[c].sum() for c in uc_cols]
        hm_data.append(row)

    if hm_data:
        hm_df = pd.DataFrame(hm_data, index=focus_tools, columns=UC_LABELS)
        fig = px.imshow(hm_df, text_auto=True, color_continuous_scale='YlOrRd',
                        aspect='auto', title='Tool × Use Case Matrix')
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Use Case Breakdown per Tool (Stacked Bar)")
        hm_melt = hm_df.reset_index().melt(id_vars='index', var_name='UseCase', value_name='Users')
        hm_melt.columns = ['Tool','UseCase','Users']
        fig2 = px.bar(hm_melt, x='Tool', y='Users', color='UseCase',
                      color_discrete_sequence=px.colors.qualitative.Alphabet,
                      title='Stacked Use Cases per Tool')
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Hours Saved — Violin by Primary Tool")
    vdf = df[df['tool_label'].isin(['Claude','ChatGPT','Other']) & df['hours_n'].notna()]
    fig3 = px.violin(vdf, y='hours_n', x='tool_label', box=True, points='all',
                     color='tool_label', color_discrete_sequence=COLORS,
                     labels={'hours_n':'Hours Saved/Week','tool_label':'Primary Tool'})
    fig3.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("##### Use Cases by Field of Study")
    main_fields = ['Data Science & AI','CS/IT','Engineering (Non-CS/IT)']
    field_df = df[df['field'].isin(main_fields)]
    uc_by_field = field_df.groupby('field')[uc_cols].sum()
    uc_by_field.columns = UC_LABELS
    uc_melt = uc_by_field.reset_index().melt(id_vars='field', var_name='UseCase', value_name='Count')
    fig4 = px.bar(uc_melt, x='UseCase', y='Count', color='field', barmode='group',
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig4.update_layout(height=400, xaxis_tickangle=-30)
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════
# PAGE: TOOL MIGRATION (SANKEY)
# ════════════════════════════════════════
elif page == "🔀 Tool Migration":
    st.title("First Tool → Current Tool")
    st.markdown("How students migrated from their first-ever AI tool to their current primary one.")
    st.divider()

    flow = df.groupby(['first_clean','current_clean']).size().reset_index(name='count')
    flow = flow[flow['count'] >= 1]

    all_nodes = list(pd.unique(flow[['first_clean','current_clean']].values.ravel()))
    node_idx  = {n: i for i, n in enumerate(all_nodes)}

    fig = go.Figure(go.Sankey(
        node=dict(label=all_nodes, pad=20, thickness=24,
                  color='rgba(99,110,250,0.75)',
                  line=dict(color='white', width=0.5)),
        link=dict(
            source=[node_idx[r] for r in flow['first_clean']],
            target=[node_idx[r] for r in flow['current_clean']],
            value=flow['count'].tolist(),
            color='rgba(200,200,255,0.3)'
        )
    ))
    fig.update_layout(title='Tool Migration Sankey', font_size=13, height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Loyalty Rate — % who stayed on their first tool")
    stayed = df[df['first_clean'] == df['current_clean']]
    left   = df[df['first_clean'] != df['current_clean']]
    col1, col2 = st.columns(2)
    col1.metric("Stayed on First Tool", f"{len(stayed)} / {len(df)}", f"{100*len(stayed)/len(df):.0f}%")
    col2.metric("Switched Tools", f"{len(left)} / {len(df)}", f"{100*len(left)/len(df):.0f}%")

    st.markdown("##### Where did ChatGPT starters end up?")
    gpt_starters = df[df['first_clean'] == 'ChatGPT']['current_clean'].value_counts().reset_index()
    gpt_starters.columns = ['Current Tool','Count']
    fig2 = px.bar(gpt_starters, x='Current Tool', y='Count', color='Current Tool',
                  color_discrete_sequence=COLORS, text='Count')
    fig2.update_traces(textposition='outside')
    fig2.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════
# PAGE: CLUSTERING
# ════════════════════════════════════════
elif page == "👥 Clustering":
    st.title("User Archetypes — K-Means Clustering")
    st.markdown("Segmenting students by behavioral patterns, not demographics.")
    st.divider()

    cluster_cols_list = ['freq_n','hours_n','tool_count','usecase_count','trust','sub_n']
    cluster_df = df[cluster_cols_list].dropna()
    cluster_idx = cluster_df.index

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)

    # elbow + silhouette
    inertias, silhouettes = [], []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        from sklearn.metrics import silhouette_score
        silhouettes.append(silhouette_score(X_scaled, km.labels_))

    col_e, col_s = st.columns(2)
    with col_e:
        fig = px.line(x=list(range(2,8)), y=inertias, markers=True,
                      labels={'x':'k','y':'Inertia'}, title='Elbow Method')
        st.plotly_chart(fig, use_container_width=True)
    with col_s:
        fig = px.line(x=list(range(2,8)), y=silhouettes, markers=True,
                      labels={'x':'k','y':'Silhouette Score'}, title='Silhouette Score')
        st.plotly_chart(fig, use_container_width=True)

    k = st.slider("Choose k", 2, 6, 3)
    km_final = KMeans(n_clusters=k, random_state=42, n_init=10)
    df.loc[cluster_idx, 'cluster'] = km_final.fit_predict(X_scaled)
    df['cluster'] = df['cluster'].fillna(-1).astype(int)

    # profile table
    profile = df[df['cluster']>=0].groupby('cluster')[cluster_cols_list].mean().round(2)
    st.markdown("##### Cluster Profiles")
    st.dataframe(profile.style.background_gradient(cmap='Blues'), use_container_width=True)

    # PCA scatter
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        'PC1': X_pca[:,0], 'PC2': X_pca[:,1],
        'Cluster': df.loc[cluster_idx,'cluster'].astype(str)
    })
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                     title='PCA — Cluster Separation (2D)',
                     color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    # radar
    st.markdown("##### Radar — Behavioral Fingerprint per Cluster")
    col_labels = ['Frequency','Hours Saved','Tool Count','UseCase Count','Trust','Subscription']
    fig = go.Figure()
    for cid in range(k):
        if cid not in profile.index: continue
        vals = profile.loc[cid].tolist()
        norm = [(v - profile[c].min()) / (profile[c].max() - profile[c].min() + 1e-6)
                for v, c in zip(vals, cluster_cols_list)]
        norm += [norm[0]]
        cats = col_labels + [col_labels[0]]
        fig.add_trace(go.Scatterpolar(r=norm, theta=cats, fill='toself', name=f'Cluster {cid}'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=450)
    st.plotly_chart(fig, use_container_width=True)

    # cross tab
    st.markdown("##### Primary Tool by Cluster")
    if k == 3:
        name_map = CLUSTER_NAMES
    else:
        name_map = {i: f'Cluster {i}' for i in range(k)}
    cross = pd.crosstab(df[df['cluster']>=0]['cluster'].map(name_map),
                        df[df['cluster']>=0]['tool_label'])
    fig = px.bar(cross.reset_index().melt(id_vars='cluster'),
                 x='cluster', y='value', color='tool_label', barmode='group',
                 labels={'cluster':'Archetype','value':'Count','tool_label':'Primary Tool'},
                 color_discrete_sequence=COLORS)
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════
# PAGE: REGRESSION
# ════════════════════════════════════════
elif page == "📈 Regression":
    st.title("Ridge Regression — What Drives Hours Saved?")
    st.markdown("Predicting how many hours/week a student saves using AI, from their behavioral profile.")
    st.divider()

    from sklearn.model_selection import cross_val_score
    reg_features = ['freq_n','tool_count','usecase_count','trust','exp_n','sub_n','concern']
    feat_labels  = ['Frequency','Tool Count','UseCase Count','Trust','Experience','Subscription','Skill Concern']
    reg_df = df[reg_features + ['hours_n']].dropna()

    X_reg = StandardScaler().fit_transform(reg_df[reg_features].values)
    y_reg = reg_df['hours_n'].values

    alpha = st.slider("Ridge alpha (regularization strength)", 0.01, 10.0, 1.0, 0.1)
    ridge = Ridge(alpha=alpha)
    cv_r2 = cross_val_score(ridge, X_reg, y_reg, cv=5, scoring='r2')
    ridge.fit(X_reg, y_reg)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean CV R²", f"{cv_r2.mean():.3f}")
    col2.metric("Std CV R²", f"± {cv_r2.std():.3f}")
    col3.metric("Training Samples", len(reg_df))

    # coefficient plot
    coef_df = pd.DataFrame({'Feature': feat_labels, 'Coefficient': ridge.coef_})
    coef_df = coef_df.sort_values('Coefficient')
    coef_df['Color'] = coef_df['Coefficient'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

    fig = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
                 color='Color', color_discrete_map={'Positive':'steelblue','Negative':'tomato'},
                 title='Ridge Coefficients — Effect on Hours Saved')
    fig.add_vline(x=0, line_dash='dash', line_color='black', line_width=1)
    fig.update_layout(height=420, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # actual vs predicted
    y_pred = ridge.predict(X_reg)
    scatter_df = pd.DataFrame({'Actual': y_reg, 'Predicted': y_pred})
    fig2 = px.scatter(scatter_df, x='Actual', y='Predicted',
                      trendline='ols', title='Actual vs Predicted Hours Saved',
                      color_discrete_sequence=['mediumorchid'])
    fig2.add_shape(type='line', x0=0, y0=0, x1=12, y1=12,
                   line=dict(dash='dot', color='grey', width=1))
    fig2.update_layout(height=380)
    st.plotly_chart(fig2, use_container_width=True)

    st.info("💡 **Interpretation**: A low R² here is expected and informative — it means hours saved is idiosyncratic and not fully determined by the behavioral features we measured. The coefficient chart is the real finding.", icon="📌")


# ════════════════════════════════════════
# PAGE: CLASSIFICATION
# ════════════════════════════════════════
elif page == "🌳 Classification":
    st.title("Decision Tree — Predicting Primary Tool")
    st.markdown("Can we predict whether a student uses Claude, ChatGPT, or Other — from what they do with AI?")
    st.divider()

    uc_bool_cols = [c for c in df.columns if c.startswith('uc_')]
    clf_features = uc_bool_cols + ['freq_n','trust','exp_n','tool_count']
    feat_labels  = UC_LABELS + ['Frequency','Trust','Experience','ToolCount']

    le  = LabelEncoder()
    clf_df = df[clf_features + ['tool_label']].dropna()
    X_clf  = clf_df[clf_features].values
    y_clf  = le.fit_transform(clf_df['tool_label'])

    from sklearn.model_selection import cross_val_score
    max_depth = st.slider("Max tree depth", 2, 6, 4)
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42, min_samples_leaf=3)
    cv_acc = cross_val_score(dt, X_clf, y_clf, cv=5, scoring='accuracy')
    dt.fit(X_clf, y_clf)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean CV Accuracy", f"{cv_acc.mean():.3f}")
    col2.metric("Std CV Accuracy", f"± {cv_acc.std():.3f}")
    col3.metric("Classes", str(list(le.classes_)))

    # classification report as df
    st.markdown("##### Classification Report")
    report = classification_report(y_clf, dt.predict(X_clf),
                                   target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df.style.background_gradient(cmap='Greens', subset=['precision','recall','f1-score']),
                 use_container_width=True)

    # confusion matrix
    cm = confusion_matrix(y_clf, dt.predict(X_clf))
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    x=list(le.classes_), y=list(le.classes_),
                    labels={'x':'Predicted','y':'Actual'},
                    title='Confusion Matrix')
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    # feature importance
    fi_df = pd.DataFrame({'Feature': feat_labels[:len(clf_features)],
                           'Importance': dt.feature_importances_})
    fi_df = fi_df[fi_df['Importance'] > 0].sort_values('Importance')
    fig2 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale='Oranges',
                  title='Feature Importance — Decision Tree')
    fig2.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.info("💡 **Interpretation**: The top splits in the tree reveal which behaviors most strongly predict tool preference — e.g. if 'Writing' use case is the first split, it means Claude users disproportionately use AI for writing.", icon="🌳")
