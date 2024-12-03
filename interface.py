import gradio as gr
import os
import json
import random
from datetime import datetime
from hashlib import sha256
from openai import OpenAI
from analogy_tutor.ds.sciq import SciQ
from analogy_tutor.modules import Extractor
from analogy_tutor.utils.llm_lib.get_llm_outputs import get_llm_output

assistant_generation_kwargs = {
    'model': 'gpt-4o-mini',
    'max_tokens': 2048
}
dataset = SciQ()
extractor = Extractor()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load system prompt template
with open("analogy_tutor/prompts/llm_assistant/few-shot-analogy.txt", "r") as f:
    SYSTEM_PROMPT = f.read()


with open("analogy_tutor/prompts/llm_assistant/few-shot-non-analogy.txt", "r") as f:
    NON_ANALOGY_PROMPT = f.read()


# Backend function for handling chat
def chat_with_model(user_message, history, user_profile, concepts, use_analogy):
    if not isinstance(history, list):
        history = []  # Ensure history is a list

    # Dynamically construct system prompt
    if use_analogy:
        system_prompt = SYSTEM_PROMPT.format(
            user_profile=user_profile,
            target_concepts=", ".join(concepts),
        )
    else:
        system_prompt = NON_ANALOGY_PROMPT.format(
            target_concepts=", ".join(concepts)
        )

    if not user_message == '':
        # Append user message to history
        history.append(("user", user_message))

    # Format history for OpenAI API
    formatted_history = [{"role": "user", "content": system_prompt}]
    for sender, message in history:
        role = "user" if sender == "user" else "assistant"
        formatted_history.append({"role": role, "content": message})

    # Generate model response
    try:
        response_with_cot = get_llm_output(formatted_history, **assistant_generation_kwargs)
        print("response_with_cot: ", response_with_cot)
        model_response = response_with_cot.split("Explanation:", 1)[-1].strip()
    except Exception as e:
        model_response = f"Error: {e}"

    # Append model response to history
    history.append(("model", model_response))
    return history, ""  # Return updated history and clear the user input


# Function to show evaluation form
def show_evaluation():
    return gr.update(visible=True)


# Function to submit evaluation
def submit_evaluation(idx, entry, use_analogy, helpfulness, efficiency, impressiveness, feedback, multiple_choice_answer, history):
    # Save evaluation and chat history
    log_data = {
        "idx": idx,
        "use_analogy": use_analogy, 
        "entry": entry,
        "timestamp": datetime.now().isoformat(),
        "history": history,
        "evaluation": {
            "helpfulness": helpfulness,
            "efficiency": efficiency,
            "impressiveness": impressiveness, 
            "quiz_answer": multiple_choice_answer,
            "feedback": feedback,
        },
    }

    # Generate a unique hash for the file name
    hash_key = sha256(str(log_data).encode()).hexdigest()[:8]
    file_name = f"logs/{idx}_{use_analogy}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

    os.makedirs("logs", exist_ok=True)
    with open(file_name, "w") as f:
        json.dump(log_data, f, indent=4)

    return (
        f"Thank you for your feedback!\n"
        f"Helpfulness: {helpfulness}/5\n"
        f"Efficiency: {efficiency}/5\n"
        f"Impressiveness: {impressiveness}/5\n"
        f"Quiz Answer: {multiple_choice_answer}\n"
        f"Feedback: {feedback}\n"
        f"Your evaluation and chat history have been saved."
        f"\n\n The correct answer: {multiple_choice_answer}"
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

def get_random_example():
    """Fetch a random example from the dataset."""
    idx = random.randint(len(dataset) - 50, len(dataset) - 1)  # Random index from the last 50 entries
    use_analogy = random.randint(0, 1)  # Randomly choose analogy or non-analogy
    entry = dataset[idx]
    concepts = extractor(entry)
    print(concepts)
    quiz_options = [f"{key}. {item}" for key, item in entry["choices"].items()]
    quiz_question = entry["question"]
    return idx, entry, concepts, use_analogy, quiz_options, quiz_question

# Initialize example function
def initialize_example():
    """Initialize a new random example for the session."""
    return get_random_example()

# Define the Gradio interface
with gr.Blocks(css=custom_css, title="Chat Interface with Evaluation Form") as demo:
    gr.Markdown("""
    # 💬 Chat Interface with AI Model
    Instructions:
    Thank you for attending the study! In this chat interface, you will chat with an AI model to understand some concepts which will be used to answer a quiz question.

    Steps:
    1. Click "Start Session" to initialize the chat session.
    2. Enter your background information such as hobbies in the "User Profile" box and then hit "Send" to start the conversation.
    3. You can ask the AI model questions related to the concepts provided.
    4. When you are ready, hit "End Chat & Evaluate" to provide feedback and take the quiz.
    5. After filling out the evaluation form, hit "Submit Evaluation" to complete the process. 

    You are encouraged to interact with the assistant for 2-6 rounds before ending the chat.
    """)

    # Quiz and user profile input
    files = os.listdir("logs")

    idx_state = gr.State()      # Store `idx` as a state
    entry_state = gr.State()    # Store `entry` as a state
    concept_state = gr.State()  # Store concepts as a state
    analogy_state = gr.State()  # Store analogy choice as a state
    quiz_state = gr.State()     # Store quiz options as a state
    question_state = gr.State()  # Store quiz question as a state

    # Start session button to initialize random example
    start_session_button = gr.Button("Start Session")
    start_session_button.click(
        initialize_example,
        inputs=[],
        outputs=[idx_state, entry_state, concept_state, analogy_state, quiz_state, question_state],
    )

    user_profile = gr.Textbox(
        placeholder="Describe your background (e.g., profession, age)...",
        label="User Profile",
        lines=2,
    )

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
        with gr.Column():
            helpfulness = gr.Slider(1, 5, step=1, label="The assistant is very helpful at teaching the concepts.")
            gr.HTML("<p class='criteria'>1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree</p>")
            efficiency = gr.Slider(1, 5, step=1, label="The assistant helps me learn these concepts efficiently.")
            gr.HTML("<p class='criteria'>1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree</p>")
            impressiveness = gr.Slider(1, 5, step=1, label="The assistant's explanations leave me with an impressive understanding about the concepts")
            gr.HTML("<p class='criteria'>1: Strongly Disagree, 2: Disagree, 3: Neutral, 4: Agree, 5: Strongly Agree</p>")
        with gr.Column():
            gr.Markdown("""
                ### Instructions:
                - **Do not refer back to the teaching content**: You are not encouraged to revisit the teaching materials while answering the quiz.
                - **Avoid external sources**: Please refrain from using tools like Google or GPT to find answers.
                - **Provide honest feedback**: To best contribute to our experiment, evaluate the AI model truthfully and thoughtfully.
                """)
            multiple_choice_answer = gr.Radio(
                choices=[],  # Initialize with an empty list
                label="",  # Initialize with an empty label
                interactive=True
            )

            feedback = gr.Textbox(lines=3, placeholder="Write your feedback here...", label="Additional Feedback")
            submit_button = gr.Button("Submit Evaluation")
        evaluation_output = gr.Textbox(visible=True, interactive=False, label="Feedback Summary")

    # Dynamically update quiz options and question when the session starts
    send_button.click(
        chat_with_model,
        inputs=[user_input, history, user_profile, concept_state, analogy_state],
        outputs=[history, user_input],
    ).success(render_chat, inputs=[history], outputs=[chatbot])

    clear_button.click(lambda: ([], ""), inputs=None, outputs=[history, chatbot])
    end_chat_button.click(
        show_evaluation, outputs=[evaluation_form]
    ).then(
        lambda quiz_options, quiz_question: gr.update(choices=quiz_options, label=quiz_question),
        inputs=[quiz_state, question_state],
        outputs=[multiple_choice_answer],
    )
    submit_button.click(
        submit_evaluation,
        inputs=[idx_state, entry_state, analogy_state, helpfulness, efficiency, impressiveness, feedback, multiple_choice_answer, history],
        outputs=[evaluation_output],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=50, max_size=50)
    demo.launch(max_threads=50, share=True)
