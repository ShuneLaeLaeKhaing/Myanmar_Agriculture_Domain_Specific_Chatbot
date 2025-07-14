import streamlit as st

st.set_page_config(page_title="မြန်မာစကားပြော Chatbot")

from utils.response_generator import ResponseGenerator

# Initialize
response_generator = ResponseGenerator()

# Myanmar UI Setup

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Padauk&display=swap');
* {
    font-family: 'Padauk', sans-serif !important;
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("မေးခွန်းမေးပါ..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    with st.spinner("ဖြေကြားနေပါသည်..."):
        response = response_generator.generate_response(query)
        
        # Format output
        if response["source"] == "Language Error":
            output = f"❌ {response['response']}"

        elif response["source"] == "💰 Market Price":
            output = f"""
            💰 **ဈေးနှုန်းအဖြေ**
            {response['response']}
            """

        elif response["source"] == "📚 FAQ":
            output = f"""
            📌 **FAQ အဖြေ**
            {response['response']}
            """
        elif response["source"] == "🌐 Web":
            output = f"""
            🌐 **Web အဖြေ**
            {response['response']}
            """    

        elif response["source"] == "Out of scope":
            output = f"❌ {response['response']}"
            
        else:
            output = f" {response.get('response', 'မသိသောအဖြေ')}"
        
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.chat_message("assistant").write(output)