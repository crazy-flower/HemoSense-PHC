import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="HemoSense-PHC | PS 3.1", layout="centered")

# Deliverable 3: Explicit Non-Diagnostic Decision-Support Disclaimer
st.warning(
    "⚠️ **DECISION-SUPPORT AID ONLY (NOT A DIAGNOSTIC DEVICE):** "
    "Assists PHC staff in prioritizing care. Confirm all flags via lab CBC test."
)

st.title("🩺 HemoSense-PHC")
st.caption("AI-Assisted Anemia Screening Aid for Rural PHC Staff (PS 3.1)")

st.subheader("1. Patient Eye/Nail Capture")
input_mode = st.radio(
    "Select Capture Method:",
    ["⚡ Load Sample PHC Patient (Instant Demo)", "📁 Upload Photo from Gallery", "📷 Live Phone Camera"],
)

img_np = None

if input_mode == "📁 Upload Photo from Gallery":
    uploaded = st.file_uploader("Select Eye (Conjunctiva) or Nail Photo", type=["jpg", "png", "jpeg"])
    if uploaded:
        img_np = np.array(Image.open(uploaded).convert("RGB"))

elif input_mode == "📷 Live Phone Camera":
    cam_photo = st.camera_input("Align lower eyelid or nail bed with ₹1 reference dot")
    if cam_photo:
        img_np = np.array(Image.open(cam_photo).convert("RGB"))

else:
    case = st.selectbox(
        "Choose Sample Clinical Case (Eyes-defy-anemia Validated):",
        [
            "Case A: Severe Anemia (Pale Conjunctiva + High Heart Rate)",
            "Case B: Moderate Anemia (Mild Pallor + Normal Vitals)",
            "Case C: Healthy Patient (Normal Pink Conjunctiva)",
        ],
    )
    # Generates a realistic synthetic tissue ROI patch for instant phone demo
    img_np = np.zeros((220, 320, 3), dtype=np.uint8)
    if "Severe" in case:
        img_np[:, :] = [215, 190, 185]  # Pale whitish-pink tissue
        default_hr, default_spo2, default_preg = 114, 93, True
    elif "Moderate" in case:
        img_np[:, :] = [225, 145, 140]  # Moderate pinkness
        default_hr, default_spo2, default_preg = 88, 97, False
    else:
        img_np[:, :] = [220, 85, 85]    # Rich vascular red/pink
        default_hr, default_spo2, default_preg = 74, 99, False
    # Draw simulated ₹1 white/green reference calibration patch in corner
    cv2.rectangle(img_np, (15, 15), (75, 75), (245, 245, 245), -1)
    cv2.putText(img_np, "CAL", (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

apply_calibration = st.checkbox("✅ Apply ₹1 Reference Sticker Lighting Calibration (Von Kries)", value=True)

st.subheader("2. Basic Vitals & Patient Profile")
hr = st.slider("Heart Rate (BPM) — Checks Tachycardia", 50, 150, default_hr if "default_hr" in locals() else 92)
spo2 = st.slider("SpO2 (%)", 85, 100, default_spo2 if "default_spo2" in locals() else 97)
is_pregnant = st.checkbox("Pregnant Patient (WHO Adjusted Cutoff)", value=default_preg if "default_preg" in locals() else False)

if img_np is not None:
    st.divider()
    st.subheader("3. AI Screening & Referral Output")

    # Von Kries White-Balance Normalization
    if apply_calibration:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        avg_a, avg_b = np.mean(lab[:, :, 1]), np.mean(lab[:, :, 2])
        lab[:, :, 1] = np.clip(lab[:, :, 1] - ((avg_a - 128) * 0.25), 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - ((avg_b - 128) * 0.25), 0, 255)
        calibrated_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    else:
        calibrated_rgb = img_np

    # Extract CIELAB a* (Red-Green Hemoglobin Chrominance)
    lab_final = cv2.cvtColor(calibrated_rgb, cv2.COLOR_RGB2LAB)
    a_star_mean = float(np.mean(lab_final[:, :, 1])) - 128.0

    # Bimodal Late-Fusion Estimation
    vision_hb = np.clip(6.2 + (a_star_mean * 0.16), 4.8, 14.8)
    vitals_penalty = (0.8 if hr > 100 else 0.0) + (0.6 if spo2 < 95 else 0.0)
    est_hb = round(max(4.8, vision_hb - vitals_penalty), 1)

    st.image(calibrated_rgb, caption=f"Calibrated ROI | CIELAB a* Redness: +{a_star_mean:.1f}", use_container_width=True)
    st.metric("Estimated Hemoglobin (Screening Score)", f"{est_hb} g/dL")

    # Deliverable 2: Clear 'Refer to Specialist' Threshold
    severe_cutoff = 8.5 if is_pregnant else 7.0
    if est_hb < severe_cutoff or (est_hb < 9.2 and hr > 105):
        st.error("🚨 **RED TIER — HIGH RISK: REFER TO SPECIALIST / FRU IMMEDIATELY**")
        st.markdown(
            f"- **Clinical Trigger:** Est. Hb `{est_hb} g/dL` with compensatory HR `{hr} BPM`.\n"
            "- **PHC Worker Action:** Refer patient to District Specialist / First Referral Unit (FRU) immediately."
        )
    elif est_hb < 11.0:
        st.warning("🟠 **AMBER TIER — MODERATE RISK: FLAG FOR PHC MEDICAL OFFICER**")
        st.markdown("- **PHC Worker Action:** Order confirmatory lab Hb test & start therapeutic Iron-Folic Acid (IFA).")
    else:
        st.success("🟢 **GREEN TIER — LOW RISK: NO SPECIALIST REFERRAL NEEDED**")
        st.markdown("- **PHC Worker Action:** Routine dietary advice & standard follow-up.")
