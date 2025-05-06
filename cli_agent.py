import openai
import click
import os
import re
import subprocess
import sys

# --- Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-4o-mini" # Or "openai/gpt-4o" for the full model if preferred
# To refer to your app, you can set a custom site URL or name.
# This helps OpenRouter identify your application.
# Read more: https://openrouter.ai/docs#headers
YOUR_SITE_URL = "http://localhost:3000" # Optional, replace with your actual site or app name
YOUR_APP_NAME = "My CLI AI Agent" # Optional


# --- OpenAI Client Setup for OpenRouter ---
if not OPENROUTER_API_KEY:
    click.echo(click.style("Error: OPENROUTER_API_KEY environment variable not set.", fg="red"))
    sys.exit(1)

client = openai.OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    default_headers={ # Optional headers
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_APP_NAME,
    }
)

# --- Conversation History ---
conversation_history = [
    {"role": "system", "content": "You are a helpful AI assistant integrated into a CLI. When asked to create code, provide the code in a single, clearly marked block (e.g., ```python ... ```). You can also explain the code outside the block."},
]

# --- Helper Functions ---
def extract_code_blocks(text):
    """
    Extracts code blocks from the AI's response.
    Handles blocks with or without language specifiers.
    Returns a list of (language, code_block) tuples.
    """
    code_blocks = []
    # Regex to find ```optional_lang\ncode\n```
    pattern = re.compile(r"```(\w+)?\s*\n(.*?)\n```", re.DOTALL)
    matches = pattern.finditer(text)
    for match in matches:
        language = match.group(1) if match.group(1) else "unknown"
        code = match.group(2).strip()
        code_blocks.append((language, code))
    return code_blocks

def execute_code(language, code_block):
    """
    Executes the given code block.
    WARNING: This is a simplified execution and is NOT SECURE.
    For production, use proper sandboxing (Docker, VMs, etc.).
    """
    click.echo(click.style("\n--- Code to Execute ---", fg="yellow"))
    click.echo(f"Language: {language}")
    click.echo(code_block)
    click.echo(click.style("--- End Code ---", fg="yellow"))

    if not click.confirm(click.style("Do you want to execute this code?", fg="magenta"), default=False):
        click.echo("Code execution aborted by user.")
        return "Code execution aborted by user."

    try:
        if language.lower() == "python":
            # For Python, we can try to execute it more directly, but still risky
            # A safer approach for Python would be to write to a temp file and run with subprocess
            # For simplicity here, we'll use subprocess for a generic approach
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(code_block)
                temp_file_name = tmp_file.name
            
            click.echo(f"Executing Python script: {temp_file_name}")
            process = subprocess.run([sys.executable, temp_file_name], capture_output=True, text=True, timeout=30)
            os.remove(temp_file_name) # Clean up the temporary file

        elif language.lower() in ["bash", "shell", "sh"]:
            click.echo(f"Executing {language} script...")
            # For shell scripts, ensure commands are safe. This example is still very risky.
            # Consider what commands an LLM might generate.
            process = subprocess.run(code_block, shell=True, capture_output=True, text=True, timeout=30, check=False)
        
        # Add more language handlers as needed (e.g., javascript with node)
        # elif language.lower() == "javascript":
        #     with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as tmp_file:
        #         tmp_file.write(code_block)
        #         temp_file_name = tmp_file.name
        #     click.echo(f"Executing JavaScript (Node.js) script: {temp_file_name}")
        #     process = subprocess.run(["node", temp_file_name], capture_output=True, text=True, timeout=30)
        #     os.remove(temp_file_name)

        else:
            return f"Execution for language '{language}' is not supported by this agent."

        output = "--- Execution Output ---\n"
        if process.stdout:
            output += f"STDOUT:\n{process.stdout}\n"
        if process.stderr:
            output += f"STDERR:\n{process.stderr}\n"
        if not process.stdout and not process.stderr:
            output += "No output produced.\n"
        output += "--- End Execution Output ---"
        return output

    except subprocess.TimeoutExpired:
        return "Code execution timed out."
    except Exception as e:
        return f"An error occurred during code execution: {e}"

# --- CLI Command ---
@click.command()
@click.option('--prompt', '-p', help="Single prompt to send to the AI. Exits after.")
def chat(prompt):
    """
    CLI AI Agent that communicates with gpt-4o-mini via OpenRouter.
    Can create and (with confirmation) run code.
    """
    global conversation_history

    if prompt:
        # Single shot mode
        _process_prompt(prompt)
        return

    # Interactive mode
    click.echo(click.style("Starting interactive chat with AI Agent.", fg="cyan"))
    click.echo("Type 'exit' or 'quit' to end the session.")
    click.echo(f"Using model: {MODEL_NAME} via OpenRouter.")

    while True:
        try:
            user_input = click.prompt("You")
            if user_input.lower() in ["exit", "quit"]:
                click.echo(click.style("Exiting agent.", fg="yellow"))
                break
            
            _process_prompt(user_input)

        except click.exceptions.Abort: # Handle Ctrl+C
            click.echo(click.style("\nExiting agent (Ctrl+C).", fg="yellow"))
            break
        except Exception as e:
            click.echo(click.style(f"An unexpected error occurred: {e}", fg="red"))


def _process_prompt(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})

    try:
        with click.progressbar(length=100, label=click.style("AI is thinking...", fg="blue"), show_eta=False, show_percent=False, bar_template='%(label)s [%(bar)s]', fill_char='.', empty_char=' ') as bar:
            # This progress bar is indeterminate; actual progress depends on API response time
            # For a real progress bar, you might need to estimate or use streaming if available
            
            # Simulate some progress for visual feedback
            for _ in range(30):
                bar.update(1)

            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation_history,
                stream=True # Enable streaming
            )
            
            bar.update(40) # Update after request sent

            assistant_response_content = ""
            click.echo(click.style(f"\nAI ({MODEL_NAME}): ", fg="green"), nl=False)
            for chunk in stream:
                if chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    click.echo(content_piece, nl=False)
                    sys.stdout.flush() # Ensure immediate printing
                    assistant_response_content += content_piece
            click.echo() # Newline after streaming is complete

            bar.update(30) # Final update

        if assistant_response_content:
            conversation_history.append({"role": "assistant", "content": assistant_response_content})

            # Check for code blocks
            code_blocks = extract_code_blocks(assistant_response_content)
            if code_blocks:
                click.echo(click.style("\nCode block(s) detected in response:", fg="yellow"))
                for i, (lang, code) in enumerate(code_blocks):
                    click.echo(click.style(f"--- Code Block {i+1} ({lang}) ---", fg="blue"))
                    click.echo(code)
                    click.echo(click.style(f"--- End Code Block {i+1} ---", fg="blue"))

                    # Ask to execute each block (or you could choose to execute only the first, or specific languages)
                    if click.confirm(click.style(f"Do you want to attempt to execute Code Block {i+1} ({lang})?", fg="magenta"), default=False):
                        # WARNING: THIS IS THE DANGEROUS PART
                        click.echo(click.style("WARNING: Executing AI-generated code can be dangerous!", bold=True, fg="red"))
                        execution_result = execute_code(lang, code) # execute_code already has its own confirmation
                        click.echo(click.style(execution_result, fg="cyan"))
                        # Optionally add execution result back to conversation history for context
                        conversation_history.append({
                            "role": "user", # Or a custom role like "tool_output" or "execution_result"
                            "content": f"Output of executing the {lang} code:\n{execution_result}"
                        })
            else:
                click.echo(click.style("No executable code blocks detected in the AI's latest response.", fg="blue"))


    except openai.APIConnectionError as e:
        click.echo(click.style(f"OpenRouter API Connection Error: {e}", fg="red"))
    except openai.RateLimitError as e:
        click.echo(click.style(f"OpenRouter Rate Limit Exceeded: {e}", fg="red"))
    except openai.APIStatusError as e:
        click.echo(click.style(f"OpenRouter API Error (Status {e.status_code}): {e.response}", fg="red"))
    except Exception as e:
        click.echo(click.style(f"An error occurred: {e}", fg="red"))


if __name__ == '__main__':
    # Temporary file import for execute_code (Python)
    import tempfile 
    chat()
