import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Content Safety Analyzer",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

st.title("Content Safety Analyzer")

# Sidebar navigation
option = st.sidebar.selectbox(
    "Select Analysis Type",
    ("Website Analysis", "Image Analysis", "Text Analysis", "Video Analysis", "Prediction History")
)


def display_probabilities_and_verdict(prob_dict, verdict):
    """
    Helper function to display probabilities in a table,
    optionally show predicted_label if present, and show the final verdict.
    """
    # Attempt to pop out "predicted_label" from the probabilities
    predicted_label = prob_dict.pop("predicted_label", None)

    # Convert dictionary to a list of (class, probability) pairs
    items = list(prob_dict.items())

    # Create a DataFrame and sort by Probability descending
    df = pd.DataFrame(items, columns=["Class", "Probability"])
    df = df.sort_values(by="Probability", ascending=False)

    st.subheader("Verdict")
    st.write(verdict)

    st.subheader("Prediction Probabilities (Highest First)")
    st.table(df)

    if predicted_label:
        st.subheader("Predicted Label")
        st.write(predicted_label)


    
# --------------------------
# 1) Website Analysis
# --------------------------
if option == "Website Analysis":
    st.header("Website Analysis")
    website_url = st.text_input("Enter website URL")
    if st.button("Analyze Website"):
        if website_url:
            try:
                response = requests.post(f"{API_URL}/predict/website", json={"url": website_url})
                if response.ok:
                    result = response.json()
                    # For websites, the probabilities are typically in result["overall"]
                    prob_dict = result.get("overall", {})
                    verdict = result.get("verdict", "N/A")
                    display_probabilities_and_verdict(prob_dict, verdict)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
        else:
            st.error("Please enter a valid URL.")

# --------------------------
# 2) Image Analysis
# --------------------------
elif option == "Image Analysis":
    st.header("Image Analysis")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if st.button("Analyze Image"):
        if image_file is not None:
            try:
                files = {"file": image_file}
                response = requests.post(f"{API_URL}/predict/image", files=files)
                if response.ok:
                    result = response.json()
                    # For images, the probabilities are typically in result["prediction"]
                    prob_dict = result.get("prediction", {})
                    verdict = result.get("verdict", "N/A")
                    display_probabilities_and_verdict(prob_dict, verdict)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
        else:
            st.error("Please upload an image file.")

# --------------------------
# 3) Text Analysis
# --------------------------
elif option == "Text Analysis":
    st.header("Text Analysis")
    text_input = st.text_area("Enter text for analysis")
    if st.button("Analyze Text"):
        if text_input:
            try:
                data = {"text": text_input}
                response = requests.post(f"{API_URL}/predict/text", data=data)
                if response.ok:
                    result = response.json()
                    # For text, the probabilities are typically in result["prediction"]
                    prob_dict = result.get("prediction", {})
                    verdict = result.get("verdict", "N/A")
                    display_probabilities_and_verdict(prob_dict, verdict)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
        else:
            st.error("Please enter some text.")

# --------------------------
# 4) Video Analysis
# --------------------------
elif option == "Video Analysis":
    st.header("Video Analysis")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if st.button("Analyze Video"):
        if video_file is not None:
            try:
                files = {"file": video_file}
                response = requests.post(f"{API_URL}/predict/video", files=files)
                if response.ok:
                    result = response.json()
                    # For videos, the probabilities are typically in result["prediction"]
                    prob_dict = result.get("prediction", {})
                    verdict = result.get("verdict", "N/A")
                    display_probabilities_and_verdict(prob_dict, verdict)
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
        else:
            st.error("Please upload a video file.")

# --------------------------
# 5) Prediction History
# --------------------------
elif option == "Prediction History":
    st.header("Prediction History")
    try:
        response = requests.get(f"{API_URL}/history")
        if response.ok:
            history = response.json()
            if history:
                log_entries = []
                for entry in history:
                    if "url" in entry:
                        media_type = "Website"
                        source = entry["url"]
                        verdict = entry["result"]["verdict"]
                        probabilities = entry["result"].get("overall", {})
                    elif "file" in entry:
                        media_type = "Image"
                        source = entry["file"]
                        verdict = entry["result"]["verdict"]
                        probabilities = entry["result"].get("prediction", {})
                    elif "text" in entry:
                        media_type = "Text"
                        source = entry["text"]
                        verdict = entry["result"]["verdict"]
                        probabilities = entry["result"].get("prediction", {})
                    elif "video" in entry:
                        media_type = "Video"
                        source = entry["video"]
                        verdict = entry["result"]["verdict"]
                        probabilities = entry["result"].get("prediction", {})
                    else:
                        media_type = "Unknown"
                        source = ""
                        verdict = ""
                        probabilities = {}

                    log_entries.append({
                        "Media Type": media_type,
                        "Source": source,
                        "Verdict": verdict,
                        "Probabilities": probabilities
                    })

                df = pd.DataFrame(log_entries)
                st.dataframe(df)
            else:
                st.info("No predictions have been made yet.")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")

