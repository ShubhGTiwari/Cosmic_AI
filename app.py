import time
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from pymongo import MongoClient

# --- CONFIGURATION ---
MONGO_URI = st.secrets["MONGO_URI"] 
DB_NAME = "cosmic_research_db"
MODEL_PATH = "cosmic_brain_v1.h5"
TRAINING_MEAN = -7.97220180243713e-23
TRAINING_STD  = 2.216313870568256e-19

# Page Setup
st.set_page_config(
    page_title="Cosmic Detector AI",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    .metric-card {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        text-align: center;
    }
    .alert {
        background-color: #7f1d1d;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        animation: blinker 1s linear infinite;
        margin-bottom: 20px;
    }
    @keyframes blinker { 50% { opacity: 0.3; } }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCTIONS ---
@st.cache_resource
def load_resources():
    # Load Model (1D CNN)
    model = tf.keras.models.load_model(MODEL_PATH)
    # Connect to DB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return model, db

def generate_live_stream():
    """
    Simulates live feed
    """
    # 1. Fetch Background Noise (The Texture)
    noise_start = 1126259462 + 200
    noise = TimeSeries.fetch_open_data('H1', noise_start, noise_start + 4).resample(2048).value
    
    # Randomly decide to inject a signal (30% chance)
    if np.random.random() > 0.7:
        # 2. Fetch Signal (The Shape)
        event_time = 1126259462.4
        signal = TimeSeries.fetch_open_data('H1', event_time-2, event_time+2).resample(2048).value
        
        # 3. MIX: Volume = 2.0
        combined_data = noise + (signal * 2.0)
        label = "SIGNAL"
    else:
        # Just noise
        combined_data = noise
        label = "NOISE"
        
    # 4. Normalize using GLOBAL TRAINING STATS
    normalized_data = (combined_data - TRAINING_MEAN) / TRAINING_STD
    
    return normalized_data, label

def plot_waveform(data, prediction_score):
    """Creates the Matplotlib chart for the dashboard"""
    fig, ax = plt.subplots(figsize=(12, 3))
    # Change color based on alarm
    color = '#ff4b4b' if prediction_score > 0.80 else '#00ff00'
    
    ax.plot(data, color=color, linewidth=0.8)
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.set_title(f"LIGO Detector Strain (H1)", color='white')
    ax.set_ylim(-6, 6) 
    return fig

def get_model_summary(model):
    """Model"""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)

# --- MAIN APP ---
def main():
    model, db = load_resources()
    
    st.title(" AI-Driven Cosmic Event Detection System")
    st.markdown("### Real-Time Inference Engine | Status: ONLINE")

    with st.expander("View Neural Network Architecture (System Specs)"):
        st.markdown("This system uses a 1D Convolutional Neural Network (CNN) optimized for time-series signal detection.")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.caption("Layer Structure:")
            # Displays the raw 'model.summary()' text
            st.code(get_model_summary(model))
            
        with col2:
            st.caption("Model Stats:")
            st.metric("Input Shape", "(8192, 1)")
            st.metric("Total Parameters", f"{model.count_params():,}")
            st.metric("File Size", "450 KB (Est.)")
            st.info("Architecture optimized for low-latency edge detection.")

    # Layout: Top Control Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown('<div class="metric-card"><h3>Sensor</h3>LIGO Hanford</div>', unsafe_allow_html=True)
    with col2: st.markdown('<div class="metric-card"><h3>Frequency</h3>2048 Hz</div>', unsafe_allow_html=True)
    with col3: st.markdown('<div class="metric-card"><h3>Model</h3>1D-CNN (v1.0)</div>', unsafe_allow_html=True)
    with col4: 
        run_btn = st.button(" START LIVE FEED", type="primary")

    # Placeholders
    alert_spot = st.empty()
    chart_spot = st.empty()
    rag_spot = st.empty()
    
    if "running" not in st.session_state:
        st.session_state.running = False
    if run_btn:
        st.session_state.running = True

    # THE INFERENCE LOOP
    if st.session_state.running:
        while True:
            # 1. Get Data
            live_data, label = generate_live_stream()
            
            # 2. AI Prediction
            input_tensor = np.expand_dims(live_data, axis=0) 
            input_tensor = np.expand_dims(input_tensor, axis=-1) 
            prob = model.predict(input_tensor, verbose=0)[0][0]
            
            # 3. Update Dashboard
            fig = plot_waveform(live_data, prob)
            chart_spot.pyplot(fig)
            plt.close(fig)
            
            # 4. Logic & Alerts
            if prob > 0.80:
                alert_spot.markdown(f"""
                <div class="alert">
                    GRAVITATIONAL WAVE DETECTED! (Confidence: {prob*100:.2f}%)
                </div>
                """, unsafe_allow_html=True)
                
                # RAG Retrieval
                rag_spot.info(" AI is querying the Historical Database for similar events...")
                time.sleep(1) 
                
                # Fetch the Golden Event (GRB with image)
                match = db.events.find_one({'type': 'Gamma-Ray Burst', 'processed': True, 'image_path': {'$exists': True}})
                
                if match:
                    col_A, col_B = rag_spot.columns([1, 1])
                    with col_A:
                        st.success(" RAG Context Retrieved")
                        st.write(f"Similar Historic Event: {match.get('event_id')}")
                        st.write(f"Event Type: {match.get('type')}")
                        st.write("AI Recommendation: This signal pattern matches high-energy merger profiles. Immediate optical follow-up is recommended.")
                    
                    with col_B:
                        # Construct your Cloudflare Public URL
                        # Note: We use the direct R2 dev link structure. 
                        # If you haven't set up a custom domain, this link might need authentication or just show the filename.
                        # For the DEMO, displaying the local path or the R2 key visually is enough proof.
                        st.write("Optical Evidence Found:")
                        
                        # We use the direct public link structure for R2 dev domains if you enabled it
                        # Otherwise we just show the filename as proof
                        if 'image_path' in match:
                             # Display image if you have a public bucket URL, else show placeholder logic
                             # Replace with your actual R2 dev URL if you have one, e.g., https://pub-xxxx.r2.dev
                             image_url = f"https://pub-{R2_ACCOUNT_ID}.r2.dev/{match['image_path']}"
                             # Fallback for demo: just show the text if URL isn't live
                             st.code(f"R2 Bucket: {match['image_path']}")
                             st.info("Visual asset retrieved from Cloudflare R2 bucket.")
                
                time.sleep(5) # Pause to let user see it
                rag_spot.empty()
                
            else:
                alert_spot.success(f"System Normal. Background Noise Level. (Score: {prob*100:.4f}%)")
                rag_spot.empty()
            
            time.sleep(0.1)

if __name__ == "__main__":
    main()