import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("C:\VSC University\Semester 2\ML Project\spotify.csv.csv")

features = df[['energy', 'valence', 'danceability', 'tempo']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# ---------------- CLUSTER ANALYSIS ---------------- #
cluster_avg = df.groupby('cluster')[['valence','energy']].mean()

# Sort clusters by valence (happiness)
sorted_clusters = cluster_avg.sort_values(by='valence')

mood_map = {
    sorted_clusters.index[0]: 'Sad',
    sorted_clusters.index[1]: 'Chill',
    sorted_clusters.index[2]: 'Happy'
}

df['mood'] = df['cluster'].map(mood_map)

# ---------------- UI ---------------- #
st.title("🎧 AI-Based Music Mood Recommendation System")

st.markdown("""
Select your mood parameters and let the ML model predict your mood 
and recommend songs accordingly 🎶
""")

st.write("Adjust your mood parameters:")

energy = st.slider("Energy", 0.0, 100.0, 1.0)
valence = st.slider("Happiness (Valence)", 0.0, 100.0, 1.0)
danceability = st.slider("Danceability", 0.0, 100.0, 1.0)
tempo = st.slider("Tempo", 50.0, 200.0, 100.0)

st.write("### Your Input:")
st.write(f"Energy: {energy}, Valence: {valence}, Danceability: {danceability}, Tempo: {tempo}")

# ---------------- PREDICT ---------------- #
if st.button("Predict Mood & Recommend Songs"):

    user_input = [[energy, valence, danceability, tempo]]
    user_scaled = scaler.transform(user_input)

    cluster = kmeans.predict(user_scaled)[0]
    mood = mood_map[cluster]

    st.subheader(f"Predicted Mood: {mood}")

    if mood == 'Happy':
        st.success("😄 You seem energetic and happy!")
    elif mood == 'Sad':
        st.warning("😢 You seem a bit low. Here are some songs for you.")
    else:
        st.info("😌 You are in a calm and relaxed mood.")
    # Recommendation
    result = df[df['mood'] == mood]

    if mood == 'Happy':
        result = result.sort_values(by=['energy','valence'], ascending=False)
    elif mood == 'Sad':
        result = result.sort_values(by=['valence','energy'])
    else:
        result = result.sort_values(by=['energy'])

    result = result.head(6)

    st.write("### Recommended Songs 🎶")

    for i, row in result.iterrows():
        st.write(f"🎵 {row['track_name']} by {row['artists']}")
        
st.markdown("---")
st.markdown("Developed using Machine Learning (K-Means Clustering) and Streamlit")
st.markdown("""
<style>
.stButton>button {
    background-color: #1DB954;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


#cd "Semester 2\ML Project"
#python -m streamlit run app.py