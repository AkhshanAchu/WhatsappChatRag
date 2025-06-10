import streamlit as st
from some import WhatsAppChatRAG
import tempfile
import os
import pandas as pd
import time
import random
import plotly.express as px
import plotly.graph_objects as go


# 🔧 Optional: Uncomment if using OpenAI directly
# from openai import OpenAI
# client = OpenAI()

st.set_page_config(page_title="📱 WhatsApp Chat RAG", layout="wide")
st.title("📱 WhatsApp Chat Q&A with RAG")

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = WhatsAppChatRAG()
if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False
if 'history' not in st.session_state:
    st.session_state.history = []

rag = st.session_state.rag

# Upload WhatsApp chat
uploaded_file = st.file_uploader("📄 Upload your WhatsApp chat `.txt` file", type=["txt"])
if uploaded_file and not st.session_state.chat_started:
    # Funny messages to display while loading
    funny_messages = [
        "🔍 Stealing emojis from your messages...",
        "🕵️‍♂️ Counting your 'Yeah yeah's...",
        "🤖 Trying to decode your late night texts...",
        "💬 Finding out who's the most active spammer...",
        "😂 Translating 'Sus' to emotional intelligence...",
        "⌛ Searching for unread 'Good Morning' messages...",
        "📊 Calculating your meme-to-message ratio...",
        "👀 Judging your grammar silently...",
        "📆 Checking who texts more: weekdays or weekends..."
    ]


    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    messages = rag.parse_whatsapp_chat(temp_path)
    os.unlink(temp_path)

    if messages:
        with st.spinner(random.choice(funny_messages)):
            rag.create_embeddings()
            stats = rag.get_chat_statistics()
            insights = rag.get_advanced_analytics()
            st.success(f"✅ Parsed {len(messages)} messages and embeddings created.")
            st.session_state.chat_started = True

            # 🧠 Funny LLM Summary
            funny_prompt = f"""
            Here are some statistics from a WhatsApp chat. Please give a funny roast, sarcastic summary, or meme-style commentary:

            - Total messages: {stats['total_messages']}
            - Top senders: {stats['top_senders']}
            - Message types: {dict(stats['message_types'])}
            - Total emojis used: {stats['total_emojis']}
            - Date range: {stats['date_range']['first']} to {stats['date_range']['last']}
            - Avg messages/day: {insights['avg_messages_per_day']:.2f}
            - Avg message length: {insights['avg_message_length']:.2f}
            - Participant stats: {insights['participant_msg_count']}
            - Hourly activity: {insights['hourly_activity'].to_dict()}

            Be short and funny. Think like a sarcastic group admin summarizing this chaos.
            """

            try:
                funny_prompt = f"""
                    Here are some statistics from a WhatsApp chat. Please give a funny roast, sarcastic summary, or meme-style commentary:

                    - Total messages: {stats['total_messages']}
                    - Top senders: {stats['top_senders']}
                    - Message types: {dict(stats['message_types'])}
                    - Total emojis used: {stats['total_emojis']}
                    - Date range: {stats['date_range']['first']} to {stats['date_range']['last']}
                    - Avg messages/day: {insights['avg_messages_per_day']:.2f}
                    - Avg message length: {insights['avg_message_length']:.2f}
                    - Participant stats: {insights['participant_msg_count']}
                    - Hourly activity: {insights['hourly_activity'].to_dict()}

                    Be short and funny. Think like a sarcastic group admin summarizing this chaos.
                    """
                funny_summary = rag.generate_response(funny_prompt)
                # Optional: use OpenAI directly
                # response = client.chat.completions.create(
                #     model="gpt-4",
                #     messages=[{"role": "user", "content": funny_prompt}]
                # )
                # funny_summary = response.choices[0].message.content.strip()
            except:
                funny_summary = "🤖 AI was too stunned to roast your chat properly."

        with st.expander("📊 Chat Statistics", expanded=True):
            st.write(f"**Total messages:** {stats['total_messages']}")
            st.write(f"**Top Senders:** {[f'{name} ({count})' for name, count in stats['top_senders']]}")
            st.write(f"**Message Types:** {dict(stats['message_types'])}")
            st.write(f"**Total Emojis Used:** {stats['total_emojis']}")
            st.write(f"**Date Range:** {stats['date_range']['first']} ➡ {stats['date_range']['last']}")

            st.markdown("### 📈 Advanced Analytics")
            st.write(f"**Average Messages per Day:** {insights['avg_messages_per_day']:.2f}")
            st.write(f"**Average Message Length:** {insights['avg_message_length']:.2f} characters")

            st.markdown("**👥 Per-Participant Stats**")
            st.dataframe(pd.DataFrame({
                'Total Messages': insights['participant_msg_count'],
                'Avg Length': insights['participant_avg_length'],
                'Avg Messages/Day': insights['participant_avg_per_day']
            }))

            st.markdown("**📌 Message Type Distribution**")
            st.dataframe(insights['message_type_distribution'])

            st.markdown("**📆 Messages by Weekday**")
            st.bar_chart(insights['weekday_activity'])

            st.markdown("**⏰ Hourly Activity**")
            st.line_chart(insights['hourly_activity'])

            st.markdown("**📆 Message Distribution Over Time**")
            st.line_chart(insights['message_trend_over_time'])

            st.markdown("### ⏱ Hourly Activity per Sender")
            hourly_melted = insights['hourly_by_sender'].T.reset_index().melt(id_vars='hour', var_name='Sender', value_name='Message Count')
            hourly_melted = hourly_melted.sort_values(by='hour')

            # Plotly bar chart (interactive)
            fig = px.bar(
                hourly_melted,
                x='hour',
                y='Message Count',
                color='Sender',
                barmode='group',
                title='⏱ Hourly Message Distribution per Sender',
                labels={'hour': 'Hour of Day'},
            )

            fig.update_layout(
                xaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    tickvals=list(range(24)),
                    ticktext=[f"{i:02d}:00" for i in range(24)]
                ),
                xaxis_title="Hour of Day",
                yaxis_title="Message Count",
                bargap=0.1,
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### 🔥 Heatmap of Daily Activity by Participant")
            heatmap_data = insights['participant_daily_activity'].T  # sender as rows
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns.astype(str),
                y=heatmap_data.index,
                colorscale='Viridis'
            ))
            fig_heatmap.update_layout(
                xaxis_title='Date',
                yaxis_title='Sender',
                title='Heatmap of Messages Sent per Day by Each Participant',
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            st.markdown("### 👤 Daily Activity by Participant")

            daily_activity_df = insights['participant_daily_activity'].reset_index().melt(id_vars='date', var_name='Sender', value_name='Message Count')
            fig = px.bar(
                daily_activity_df,
                x='date',
                y='Message Count',
                color='Sender',
                title='👤 Daily Message Activity per Participant',
                labels={'date': 'Date'},
                barmode='group'
            )

            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Number of Messages',
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)




        # Display Funny LLM Roast
        with st.expander("🤣 AI's Roast of Your Group", expanded=True):
            st.markdown(f"_{funny_summary}_")

    else:
        st.error("❌ Failed to parse the uploaded file. Make sure it's a valid WhatsApp export.")

# Chat interface
if st.session_state.chat_started:
    st.markdown("---")
    st.subheader("💬 Ask Questions from Your Chat")

    user_input = st.chat_input("Type your question here...")
    if user_input:
        with st.spinner("🔍 Thinking..."):
            answer = rag.answer_question(user_input)
            st.session_state.history.append(("user", user_input))
            st.session_state.history.append(("ai", answer))

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    st.markdown("---")
    st.subheader("🎯 Contextual Search")

    with st.form("contextual_search_form"):
        query = st.text_input("🔍 Search term", value="")
        question = st.text_input("What Do you want me to infer ? 😏 (Optional)", value="")
        sender_filter = st.text_input("👤 Filter by sender (optional)")
        date_filter = st.text_input("📅 Filter by date (optional, e.g., 12/05/2023)")
        message_type = st.selectbox("💬 Message type", options=["", "text", "emoji", "media"])
        submitted = st.form_submit_button("Search Contextually")


    if submitted:
        with st.spinner("Searching contextually..."):
            results = rag.contextual_search(
                query=query or " ",
                sender_filter=sender_filter or None,
                date_filter=date_filter or None,
                message_type=message_type or None,
                User_query = question or None
            )
            if results:
                st.success(f"Found {len(results['results'])} relevant message(s). Showing top 5:")
                for msg in results['results'][:5]:
                    st.markdown(f"""
                    <div style='padding:10px;border:1px solid #333;margin-bottom:10px;border-radius:10px;background:#2f2f2f;color:#fff'>
                    <strong style='color:#4CAF50'>{msg.sender}</strong> 
                    <span style='color:#999;font-size:12px'>[{msg.timestamp}]</span><br/>
                    <span style='color:#fff'>{msg.original_content}</span><br/>
                    <code style='background:#444;padding:2px 4px;border-radius:2px;color:#FF9800'>Type: {msg.message_type}</code>
                    </br>
                    <span style='color:#999;font-size:12px'> | {results['Report']} </span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No relevant messages found for the filters.")
