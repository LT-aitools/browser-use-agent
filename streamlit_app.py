import streamlit as st
import base64
from pathlib import Path
import json
from PIL import Image
import io
from browser_use import Agent
from datetime import datetime
import asyncio
from browser_use.browser.views import BrowserStateHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from browser_use import Browser, BrowserConfig, Controller
from pydantic import SecretStr

# Initialize session state for task management
if 'task_running' not in st.session_state:
    st.session_state.task_running = False
if 'task_future' not in st.session_state:
    st.session_state.task_future = None
if 'agent' not in st.session_state:
    st.session_state.agent = None

def load_and_display_results():
    results_dir = Path("saved_trajectories")
    if not results_dir.exists():
        st.warning("No results found. Run a test first!")
        return

    # Get all task directories
    task_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    if not task_dirs:
        st.warning("No tasks found in results directory")
        return

    # Let user select which task to view
    selected_task = st.selectbox(
        "Select Task to View",
        options=task_dirs,
        format_func=lambda x: x.name
    )

    if selected_task:
        trajectory_dir = selected_task / "trajectory"
        
        # Display screenshots
        st.header("Screenshots")
        screenshots = sorted(trajectory_dir.glob("*.png"))
        
        # Create tabs for each screenshot
        if screenshots:
            tabs = st.tabs([f"Step {i+1}" for i in range(len(screenshots))])
            for i, (tab, screenshot) in enumerate(zip(tabs, screenshots)):
                with tab:
                    st.image(str(screenshot), caption=f"Step {i+1}")
        
        # Display extracted information
        st.header("Extracted Information")
        info_file = selected_task / "info.json"
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
                
            if "urls" in info:
                st.subheader("URLs Visited")
                st.json(info["urls"])
            
            if "actions" in info:
                st.subheader("Actions Taken")
                st.json(info["actions"])
            
            if "extracted_content" in info:
                st.subheader("Extracted Content")
                st.json(info["extracted_content"])

def serialize_action(action):
    """Convert action to JSON serializable format"""
    if isinstance(action, dict):
        return {k: serialize_action(v) for k, v in action.items() if not k.startswith('_')}
    elif hasattr(action, 'model_dump'):
        return action.model_dump()
    elif hasattr(action, '__dict__'):
        return {k: serialize_action(v) for k, v in action.__dict__.items() if not k.startswith('_')}
    elif isinstance(action, (list, tuple)):
        return [serialize_action(item) for item in action]
    elif isinstance(action, (str, int, float, bool, type(None))):
        return action
    else:
        return str(action)

async def run_test():
    """Run the browser test"""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY environment variable is not set")
            return
            
        st.session_state.task_running = True
        
        # Get the task from session state
        task = st.session_state.task
        if not task:
            st.error("Please enter a task")
            return
            
        # Initialize the agent if not already done
        if not st.session_state.agent:
            # Initialize the LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                api_key=SecretStr(api_key),
                convert_system_message_to_human=True,
                temperature=0.7
            )
            
            # Initialize browser and controller
            browser = Browser(config=BrowserConfig())
            controller = Controller()
            
            # Initialize the agent with all required components
            st.session_state.agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
                controller=controller,
                use_vision=True,
                max_actions_per_step=1
            )
        
        # Create results directory
        results_dir = Path("saved_trajectories")
        results_dir.mkdir(exist_ok=True)
        
        # Create timestamp-based directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = results_dir / timestamp
        run_dir.mkdir(exist_ok=True)
        
        # Create trajectory directory
        trajectory_dir = run_dir / "trajectory"
        trajectory_dir.mkdir(exist_ok=True)
        
        # Run the agent and get history
        history = await st.session_state.agent.run()
        
        # Check if task was killed
        if not st.session_state.task_running:
            st.warning("Task was interrupted")
            return
        
        # Save screenshots
        for i, state in enumerate(history.history):
            if isinstance(state.state, BrowserStateHistory) and state.state.screenshot:
                screenshot_path = trajectory_dir / f"step_{i}.png"
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(state.state.screenshot))
        
        # Save other information
        info = {
            "urls": history.urls(),
            "actions": [serialize_action(action) for action in history.model_actions()],
            "extracted_content": history.extracted_content()
        }
        
        with open(run_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
            
        st.success("Test completed!")
            
    except Exception as e:
        st.error(f"Error during test: {str(e)}")
        st.session_state.task_running = False
        st.session_state.task_future = None

# Streamlit UI
st.title("Browser-Use Test Results Viewer")

# Add a task input
st.text_area("Enter task:", key="task", height=100)

# Run button
if st.button("Run Test"):
    if not st.session_state.task_running:
        st.session_state.task_future = asyncio.run(run_test())
    else:
        st.warning("A task is already running. Please wait for it to complete.")

# Display task status
if st.session_state.task_running:
    st.info("Task is running...")

# Display results
load_and_display_results() 