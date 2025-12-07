import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tempfile
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import streamlit as st
st.write("App started - debugging mode")

# ==============================================================
# UTILITY: TRAIN SYNTHETIC ML MODEL (ENVIRONMENTAL DATA)
# ==============================================================

@st.cache_resource
def train_ml_model():
    np.random.seed(42)
    data_size = 5000

    # Synthetic environmental features
    slope = np.random.uniform(0, 60, data_size)
    elevation = np.random.uniform(50, 2500, data_size)
    rainfall = np.random.uniform(10, 350, data_size)
    soil_moisture = np.random.uniform(0.1, 1.0, data_size)
    ndvi = np.random.uniform(-0.2, 0.9, data_size)
    distance_to_road = np.random.uniform(0, 5, data_size)
    terrain_roughness = np.random.uniform(0, 1000, data_size)
    soil_type = np.random.randint(1, 5, data_size)
    groundwater_level = np.random.uniform(0, 50, data_size)
    precipitation_intensity = np.random.uniform(0, 100, data_size)
    slope_aspect = np.random.uniform(0, 360, data_size)
    seismic_activity = np.random.uniform(0, 7, data_size)

    risk_score = (
        (slope / 100) * 0.20 + (rainfall / 400) * 0.20 +
        soil_moisture * 0.15 + (1 - ndvi) * 0.10 +
        (terrain_roughness / 1000) * 0.12 + ((5 - soil_type) / 4) * 0.10 +
        (groundwater_level / 50) * 0.08 + (precipitation_intensity / 100) * 0.05
    )

    landslide = (risk_score > 0.45).astype(int)

    df_ml = pd.DataFrame({
        "slope": slope,
        "elevation": elevation,
        "rainfall": rainfall,
        "soil_moisture": soil_moisture,
        "ndvi": ndvi,
        "distance_to_road": distance_to_road,
        "terrain_roughness": terrain_roughness,
        "soil_type": soil_type,
        "groundwater_level": groundwater_level,
        "precipitation_intensity": precipitation_intensity,
        "slope_aspect": slope_aspect,
        "seismic_activity": seismic_activity,
        "landslide": landslide
    })

    X_ml = df_ml.drop("landslide", axis=1)
    y_ml = df_ml["landslide"]
    X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
        X_ml, y_ml, test_size=0.3, random_state=42
    )

    model_ml = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=15
    )
    model_ml.fit(X_train_ml, y_train_ml)

    return model_ml


# ==============================================================
# VIDEO ANALYZER FOR GROUND / CCTV
# ==============================================================

class VideoAnalyzer:
    """Analyzes ground video frames for movement, cracks, deformation."""

    def __init__(self):
        self.prev_gray = None

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crack approximation: edge density
        edges = cv2.Canny(gray, 100, 200)
        crack_percentage = edges.mean() / 255.0  # 0–1

        # Movement detection: frame difference
        if self.prev_gray is None:
            movement_pixels = 0.0
        else:
            diff = cv2.absdiff(self.prev_gray, gray)
            movement_pixels = float(np.clip(diff.mean() * 0.5, 0, 100))

        self.prev_gray = gray

        deformation = np.clip(movement_pixels / 100.0, 0, 1)

        crack_score = min(crack_percentage * 2, 1.0)
        movement_score = min(movement_pixels / 100.0, 1.0)
        deformation_score = deformation

        video_risk = (
            crack_score * 0.4 +
            movement_score * 0.35 +
            deformation_score * 0.25
        )

        return {
            "crack_percentage": crack_percentage,
            "movement_pixels": movement_pixels,
            "deformation": deformation,
            "video_risk": float(video_risk)
        }


def analyze_ground_video(uploaded_file, max_frames=150):
    """
    Takes a Streamlit uploaded_file object, analyzes first N frames,
    returns average video_risk and metric summary.
    """
    if uploaded_file is None:
        return None, None

    try:
        # Rewind file pointer to beginning
        uploaded_file.seek(0)
        
        # Save to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.flush()
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        
        # Check if video opened successfully
        if not cap.isOpened():
            st.error("❌ Could not open ground video. Please check the file format.")
            return None, None
        
        analyzer = VideoAnalyzer()

        frame_count = 0
        risks = []
        cracks = []
        moves = []
        deforms = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count > max_frames:
                break

            metrics = analyzer.process_frame(frame)
            risks.append(metrics["video_risk"])
            cracks.append(metrics["crack_percentage"])
            moves.append(metrics["movement_pixels"])
            deforms.append(metrics["deformation"])

        cap.release()

        if len(risks) == 0:
            st.error("❌ No frames were extracted from the ground video.")
            return None, None

        summary = {
            "avg_video_risk": float(np.mean(risks)),
            "avg_crack_percentage": float(np.mean(cracks)),
            "avg_movement_pixels": float(np.mean(moves)),
            "avg_deformation": float(np.mean(deforms)),
            "frames_analyzed": frame_count
        }
        return summary["avg_video_risk"], summary

    except Exception as e:
        st.error(f"❌ Ground video analysis error: {str(e)}")
        return None, None
    
    finally:
        # Clean up temporary file
        try:
            if 'tfile' in locals():
                os.unlink(tfile.name)
        except Exception as cleanup_error:
            pass


# ==============================================================
# ANIMAL BEHAVIOR FROM VIDEO
# ==============================================================

class AnimalBehaviorAnalyzer:
    """Monitors animal behavior patterns for stress/panic indicators"""

    def __init__(self):
        self.stress_indicators = {
            'cattle': {'panic_threshold': 0.7, 'weight': 0.4},
            'dogs': {'panic_threshold': 0.6, 'weight': 0.35},
            'birds': {'panic_threshold': 0.65, 'weight': 0.25}
        }

    def analyze_animal_movement(self, animal_type,
                                movement_intensity,
                                activity_level,
                                heart_rate_elevation,
                                vocalizations):

        if animal_type not in self.stress_indicators:
            return 0, False, 0

        stress_score = (
            movement_intensity * 0.35 +
            activity_level * 0.3 +
            heart_rate_elevation * 0.2 +
            vocalizations * 0.15
        )

        threshold = self.stress_indicators[animal_type]['panic_threshold']
        weight = self.stress_indicators[animal_type]['weight']
        is_panicked = stress_score > threshold

        return stress_score, is_panicked, weight

    def analyze_herd_behavior(self, herd_data):
        if len(herd_data) == 0:
            return 0.0, 0.0, "NO DATA"

        herd_panic_ratio = float(np.mean(herd_data))
        collective_stress = float(np.std(herd_data))

        if herd_panic_ratio > 0.6:
            alert = "🔴 CRITICAL animal panic"
        elif herd_panic_ratio > 0.4:
            alert = "🟠 WARNING multiple stressed"
        elif herd_panic_ratio > 0.2:
            alert = "🟡 CAUTION some stress"
        else:
            alert = "🟢 NORMAL animal behavior"

        return herd_panic_ratio, collective_stress, alert

    def get_behavior_risk_score(self, herd_panic_ratio):
        return float(min(herd_panic_ratio * 1.5, 1.0))


class AnimalVideoAnalyzer:
    """Uses simple motion from animal camera video to estimate stress"""

    def __init__(self, base_analyzer: AnimalBehaviorAnalyzer, animal_type='cattle'):
        self.base_analyzer = base_analyzer
        self.animal_type = animal_type
        self.prev_gray = None

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            movement_pixels = 0.0
        else:
            diff = cv2.absdiff(self.prev_gray, gray)
            movement_pixels = float(np.clip(diff.mean() * 0.8, 0, 100))

        self.prev_gray = gray

        movement_intensity = np.clip(movement_pixels / 100.0, 0, 1)
        activity_level = movement_intensity
        heart_rate_elevation = movement_intensity * 0.8
        vocalizations = movement_intensity * 0.5

        herd_scores = []
        for _ in range(3):
            # Add small variation to simulate multiple animals
            noise = np.random.uniform(-0.05, 0.05)
            m = float(np.clip(movement_intensity + noise, 0, 1))
            stress, _, _ = self.base_analyzer.analyze_animal_movement(
                self.animal_type,
                m,
                activity_level,
                heart_rate_elevation,
                vocalizations
            )
            herd_scores.append(stress)

        herd_panic, collective_stress, alert = self.base_analyzer.analyze_herd_behavior(herd_scores)
        animal_risk = self.base_analyzer.get_behavior_risk_score(herd_panic)

        return float(animal_risk), alert


def analyze_animal_video(uploaded_file, animal_type='cattle', max_frames=150):
    """Analyzes animal video for stress indicators"""
    if uploaded_file is None:
        return None, "NO ANIMAL VIDEO", None

    try:
        # Rewind file pointer to beginning
        uploaded_file.seek(0)
        
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.flush()
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        
        # Check if video opened successfully
        if not cap.isOpened():
            st.error("❌ Could not open animal video. Please check the file format.")
            return None, "FAILED TO OPEN", None
        
        base_analyzer = AnimalBehaviorAnalyzer()
        analyzer = AnimalVideoAnalyzer(base_analyzer, animal_type=animal_type)

        frame_count = 0
        risks = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count > max_frames:
                break

            risk, alert = analyzer.process_frame(frame)
            risks.append(risk)

        cap.release()

        if len(risks) == 0:
            st.error("❌ No frames were extracted from the animal video.")
            return None, "NO FRAMES READ", None

        avg_risk = float(np.mean(risks))
        # Recompute alert from avg risk
        herd_ratio = min(avg_risk / 1.5, 1.0)
        _, _, final_alert = base_analyzer.analyze_herd_behavior([herd_ratio])

        summary = {
            "avg_animal_risk": avg_risk,
            "frames_analyzed": frame_count
        }

        return avg_risk, final_alert, summary

    except Exception as e:
        st.error(f"❌ Animal video analysis error: {str(e)}")
        return None, "ERROR", None
    
    finally:
        # Clean up temporary file
        try:
            if 'tfile' in locals():
                os.unlink(tfile.name)
        except Exception as cleanup_error:
            pass


# ==============================================================
# HYBRID SYSTEM (ML + OPTIONAL VIDEO + OPTIONAL ANIMAL)
# ==============================================================

class HybridLandslideDetectionSystem:
    def __init__(self, ml_model):
        self.ml_model = ml_model

    def predict_combined(self, environmental_data,
                         video_risk=None,
                         animal_risk=None,
                         behavior_alert="NO ANIMAL DATA"):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        env_df = pd.DataFrame([environmental_data])
        ml_prediction = self.ml_model.predict_proba(env_df)[0][1]
        ml_risk = float(ml_prediction)

        weights = {
            'ml': 0.80,
            'video': 0.10,
            'animal': 0.10
        }

        risks = {'ml': ml_risk}
        if video_risk is not None:
            risks['video'] = float(video_risk)
        if animal_risk is not None:
            risks['animal'] = float(animal_risk)

        active_keys = list(risks.keys())
        total_weight = sum(weights[k] for k in active_keys)

        combined_risk = 0.0
        for k in active_keys:
            combined_risk += risks[k] * (weights[k] / total_weight)

        if combined_risk > 0.75:
            alert_level = "🔴 CRITICAL ALERT"
            recommendation = "EVACUATE IMMEDIATELY"
        elif combined_risk > 0.60:
            alert_level = "🟠 HIGH ALERT"
            recommendation = "PREPARE FOR EVACUATION"
        elif combined_risk > 0.40:
            alert_level = "🟡 MODERATE ALERT"
            recommendation = "HEIGHTENED MONITORING"
        else:
            alert_level = "🟢 LOW RISK"
            recommendation = "CONTINUE MONITORING"

        confidence = max([r for r in [ml_risk, video_risk, animal_risk] if r is not None])

        result = {
            "timestamp": timestamp,
            "ml_risk": ml_risk,
            "video_risk": video_risk,
            "animal_risk": animal_risk,
            "combined_risk": combined_risk,
            "alert_level": alert_level,
            "recommendation": recommendation,
            "behavior_alert": behavior_alert,
            "confidence": confidence
        }

        return result


# ==============================================================
# STREAMLIT UI
# ==============================================================

def main():
    st.set_page_config(
        page_title="Hybrid Landslide Detection System",
        layout="wide"
    )

    st.title("🌋 Hybrid Landslide Detection System")
    st.markdown("""
This app combines:
- **ML Model (Environmental parameters)**  
- **Ground Video Analysis** (cracks, movement, deformation)  
- **Animal Behavior Analysis** from video  

👉 If you **don't upload any videos**, the system will fall back to **ML-only prediction**.
    """)

    # Train / load ML model
    with st.spinner("Loading ML model..."):
        model_ml = train_ml_model()
    hybrid_system = HybridLandslideDetectionSystem(model_ml)

    # ----------------------------------------------------------
    # Sidebar: Environmental Inputs
    # ----------------------------------------------------------
    st.sidebar.header("🌱 Environmental Parameters")

    slope = st.sidebar.slider("Slope (degrees)", 0.0, 60.0, 35.0)
    elevation = st.sidebar.slider("Elevation (m)", 50.0, 2500.0, 1200.0)
    rainfall = st.sidebar.slider("Rainfall (mm)", 10.0, 350.0, 180.0)
    soil_moisture = st.sidebar.slider("Soil Moisture (0–1)", 0.1, 1.0, 0.6)
    ndvi = st.sidebar.slider("NDVI (-0.2–0.9)", -0.2, 0.9, 0.3)
    distance_to_road = st.sidebar.slider("Distance to Road (km)", 0.0, 5.0, 1.5)
    terrain_roughness = st.sidebar.slider("Terrain Roughness", 0.0, 1000.0, 500.0)
    soil_type = st.sidebar.selectbox("Soil Type (1–4)", [1, 2, 3, 4], index=1)
    groundwater_level = st.sidebar.slider("Groundwater Level (m)", 0.0, 50.0, 20.0)
    precipitation_intensity = st.sidebar.slider("Precipitation Intensity", 0.0, 100.0, 50.0)
    slope_aspect = st.sidebar.slider("Slope Aspect (degrees)", 0.0, 360.0, 180.0)
    seismic_activity = st.sidebar.slider("Seismic Activity (0–7)", 0.0, 7.0, 3.0)

    environmental_data = {
        'slope': slope,
        'elevation': elevation,
        'rainfall': rainfall,
        'soil_moisture': soil_moisture,
        'ndvi': ndvi,
        'distance_to_road': distance_to_road,
        'terrain_roughness': terrain_roughness,
        'soil_type': soil_type,
        'groundwater_level': groundwater_level,
        'precipitation_intensity': precipitation_intensity,
        'slope_aspect': slope_aspect,
        'seismic_activity': seismic_activity
    }

    # ----------------------------------------------------------
    # Video Upload Section
    # ----------------------------------------------------------
    st.subheader("📹 Video Inputs (Optional)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Ground / Slope / Road CCTV Video**")
        ground_video = st.file_uploader(
            "Upload ground video (mp4, avi, mov)", type=["mp4", "avi", "mov"],
            key="ground"
        )
        if ground_video is not None:
            st.video(ground_video)

    with col2:
        st.markdown("**Animal Area Video**")
        animal_video = st.file_uploader(
            "Upload animal video (mp4, avi, mov)", type=["mp4", "avi", "mov"],
            key="animal"
        )
        animal_type = st.selectbox("Animal Type", ["cattle", "dogs", "birds"])
        if animal_video is not None:
            st.video(animal_video)

    # ----------------------------------------------------------
    # Run Analysis Button
    # ----------------------------------------------------------
    if st.button("🚀 Run Landslide Risk Analysis"):
        with st.spinner("Analyzing environmental data and videos..."):

            # 1. Ground video analysis (optional)
            video_risk = None
            video_summary = None
            if ground_video is not None:
                video_risk, video_summary = analyze_ground_video(ground_video)
            
            # 2. Animal video analysis (optional)
            animal_risk = None
            animal_summary = None
            behavior_alert = "NO ANIMAL DATA"
            if animal_video is not None:
                animal_risk, behavior_alert, animal_summary = analyze_animal_video(
                    animal_video,
                    animal_type=animal_type
                )

            # 3. Hybrid prediction
            result = hybrid_system.predict_combined(
                environmental_data,
                video_risk=video_risk,
                animal_risk=animal_risk,
                behavior_alert=behavior_alert
            )

        # ------------------------------------------------------
        # Display Results
        # ------------------------------------------------------
        st.success("✅ Analysis complete!")

        st.subheader("🧠 Combined Risk Assessment")
        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("ML Environmental Risk", f"{result['ml_risk']:.2f}")
            st.metric("Video Risk", "N/A" if result['video_risk'] is None else f"{result['video_risk']:.2f}")
            st.metric("Animal Risk", "N/A" if result['animal_risk'] is None else f"{result['animal_risk']:.2f}")

        with col_b:
            st.metric("Combined Risk Score", f"{result['combined_risk']:.2f}")
            st.metric("Alert Level", result['alert_level'])
            st.metric("System Confidence", f"{result['confidence']:.2f}")

        st.markdown(f"**Recommendation:** {result['recommendation']}")
        st.markdown(f"**Behavior Alert:** {result['behavior_alert']}")
        st.markdown(f"**Timestamp:** {result['timestamp']}")

        # Ground video summary
        if video_summary is not None:
            st.subheader("📊 Ground Video Analysis Summary")
            st.json(video_summary)
        else:
            st.info("ℹ️ No ground video provided – using only ML + animal (if available).")

        # Animal video summary
        if animal_summary is not None:
            st.subheader("🐄 Animal Behavior Analysis Summary")
            st.json(animal_summary)
        else:
            st.info("ℹ️ No animal video provided – using only ML + ground video (if available).")

    else:
        st.info("ℹ️ Set environmental parameters, optionally upload videos, then click **Run Landslide Risk Analysis**.")


if __name__ == "__main__":
    main()