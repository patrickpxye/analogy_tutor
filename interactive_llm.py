import gradio as gr
import os
from openai import OpenAI

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Backend function for handling chat
def chat_with_model(user_message, history):
    if not isinstance(history, list):
        history = []  # Ensure history is a list

    # Append user message to history
    history.append(("user", user_message))

    # Format history for OpenAI API
    formatted_history = [{"role": "system", "content": "You are a helpful assistant."}]
    for sender, message in history:
        role = "user" if sender == "user" else "assistant"
        formatted_history.append({"role": role, "content": message})

    # Generate model response
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=formatted_history,
            temperature=0.7,
            max_tokens=150,
        )
        model_response = response.choices[0].message.content.strip()
    except Exception as e:
        model_response = f"Error: {e}"

    # Append model response to history
    history.append(("model", model_response))
    return history, ""  # Return updated history and clear the user input

# Function to show evaluation form
def show_evaluation():
    return gr.update(visible=True)

# Function to submit evaluation
def submit_evaluation(interactivity, efficiency, helpfulness, feedback, multiple_choice_answer):
    return (
        f"Thank you for your feedback!\n"
        f"Interactivity: {interactivity}/5\n"
        f"Efficiency: {efficiency}/5\n"
        f"Helpfulness: {helpfulness}/5\n"
        f"Quiz Answer: {multiple_choice_answer}\n"
        f"Feedback: {feedback}"
    )

# Custom CSS for styling
custom_css = """
#chatbot {
    height: 500px;
    overflow-y: auto;
    background-color: #f9f9f9;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #ddd;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    white-space: pre-wrap;
}
.message {
    display: flex;
    margin-bottom: 10px;
    align-items: flex-start;
}
.message.user {
    justify-content: flex-end;
    flex-direction: row;
    width: 100%;
}
.message.model {
    justify-content: flex-start;
    flex-direction: row;
    width: 100%;
}
.message .content {
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 60%;
    font-size: 14px;
    line-height: 1.4;
    word-wrap: break-word;
    white-space: pre-wrap;
}
.message.user .content {
    background-color: #007BFF;
    color: white;
    border-radius: 15px 15px 0 15px;
    text-align: right;
    margin-left: auto;
}
.message.model .content {
    background-color: #F1F0F0;
    color: black;
    border-radius: 15px 15px 15px 0;
    text-align: left;
    margin-right: auto;
}
button {
    padding: 5px 10px;
    font-size: 14px;
    width: auto;
}
.criteria {
    font-size: 12px;
    color: #555;
    margin-top: -10px;
    margin-bottom: 15px;
    text-align: center;
}
#evaluation-section {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 20px;
    margin-top: 20px;
}
.slider-column {
    width: 100%;
    margin-bottom: 20px;
}
.feedback-column {
    width: 100%;
}
"""

# Custom chatbot rendering with auto-scroll
def render_chat(history):
    if not isinstance(history, list):
        history = []

    html = ""
    for sender, message in history:
        alignment = "user" if sender == "user" else "model"
        html += f"""
        <div class="message {alignment}">
            <div class="content">{message}</div>
        </div>
        """
    html += """
    <script>
        var chatbox = document.getElementById("chatbot");
        if (chatbox) {
            chatbox.scrollTop = chatbox.scrollHeight;
        }
    </script>
    """
    return html

# Define the Gradio interface
with gr.Blocks(css=custom_css, title="Chat Interface with Evaluation Form") as demo:
    gr.Markdown("""
    # 💬 Chat Interface with AI Model
    Messages from the user appear on the rightmost side, while model responses are on the leftmost side. You can end the chat anytime and provide feedback below.
    """)

    history = gr.State([])  # Initialize chat history
    chatbot = gr.HTML(label="Chat History", elem_id="chatbot")
    user_input = gr.Textbox(placeholder="Type your message...", label="", lines=1)
    with gr.Row():
        send_button = gr.Button("Send", elem_id="send-btn")
        clear_button = gr.Button("Clear Chat", elem_id="clear-btn")
        end_chat_button = gr.Button("End Chat & Evaluate", variant="secondary", elem_id="end-btn")

    evaluation_form = gr.Row(visible=False, elem_id="evaluation-section")
    with evaluation_form:
        gr.Markdown("## 📝 Evaluation")
        with gr.Column(elem_classes="slider-column"):
            interactivity = gr.Slider(1, 5, step=1, label="The assistant is very engaging and interactive")
            gr.HTML("<p class='criteria'>1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree</p>")
            efficiency = gr.Slider(1, 5, step=1, label="The assistant is efficient in problem solving")
            gr.HTML("<p class='criteria'>1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree</p>")
            helpfulness = gr.Slider(1, 5, step=1, label="The assistant is helpful and gets the job done")
            gr.HTML("<p class='criteria'>1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree</p>")
        with gr.Column(elem_classes="feedback-column"):
            multiple_choice_answer = gr.Radio(
                ["Choice1Choice1Choice1Choice1Choice1", "Choice2Choice2Choice2Choice2Choice2", "Choice3Choice3Choice3Choice3Choice3", "Choice4Choice4Choice4Choice4Choice4"],
                label="Quiz",
            )
            feedback = gr.Textbox(lines=3, placeholder="Write your feedback here...", label="Additional Feedback")
            submit_button = gr.Button("Submit Evaluation")
        evaluation_output = gr.Textbox(visible=True, interactive=False, label="Feedback Summary")

    send_button.click(chat_with_model, inputs=[user_input, history], outputs=[history, user_input]).success(render_chat, inputs=[history], outputs=[chatbot])
    clear_button.click(lambda: ([], ""), inputs=None, outputs=[history, chatbot])
    end_chat_button.click(show_evaluation, outputs=[evaluation_form])
    submit_button.click(submit_evaluation, inputs=[interactivity, efficiency, helpfulness, feedback, multiple_choice_answer], outputs=[evaluation_output])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4, max_size=4)
    demo.launch(max_threads=50)
