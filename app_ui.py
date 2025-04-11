import streamlit as st
import requests

st.set_page_config(page_title="QA Assistant", layout="wide")

st.title("Question Answering Assistant")
st.markdown("Ask any question based on your context!")

context = st.text_area("Enter Context", height=200)
question = st.text_input("Enter your Question")

if st.button("Get Answer"):
    if not context or not question:
        st.warning("Please provide both context and question.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",  # Update if running elsewhere
                    json={"context": context, "question": question}
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success("Answer Retrieved!")
                    st.markdown(f"**Answer:** {result['answer']}")
                    st.markdown(f"**Score:** {result['score']:.4f}")
                    st.markdown(f"**Latency:** {result['latency_ms']:.2f} ms")
                else:
                    st.error(f"Error: {response.status_code} - {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Exception: {str(e)}")
