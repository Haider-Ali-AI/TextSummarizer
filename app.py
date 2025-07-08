import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5_summarizer_model")
    model = T5ForConditionalGeneration.from_pretrained("t5_summarizer_model")
    return tokenizer, model

tokenizer, model = load_model()

def summarize(text, max_length=50):
    input_text = "summarize: " + text.strip()
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=10, do_sample=False)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("Text Summarizer Model")
user_input = st.text_area("Enter text to summarize:", height=250)

if st.button("Summarize"):
    if user_input.strip():
        summary = summarize(user_input)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text first.")
