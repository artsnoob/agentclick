import openai
import click
import os
import re
import subprocess
import sys
import json
import tempfile
import requests # For making HTTP requests to Brave API

# Attempt to import BeautifulSoup for stripping HTML, optional
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# --- Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY") # Brave Search API Key
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "openai/gpt-4o-mini"
YOUR_SITE_URL = "http://localhost:3000"
YOUR_APP_NAME = "My CLI AI Agent (with Brave Search)"

# Brave Search API Endpoint
BRAVE_SEARCH_API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# --- OpenAI Client Setup for OpenRouter ---
if not OPENROUTER_API_KEY:
    click.echo(click.style("Error: OPENROUTER_API_KEY environment variable not set.", fg="red"))
    sys.exit(1)

if not BRAVE_API_KEY:
    click.echo(click.style("Warning: BRAVE_API_KEY environment variable not set. Web search will not function.", fg="yellow"))
    # click.echo(click.style("Please get a key from Brave Search API and set the environment variable.", fg="yellow")) # Redundant with perform_web_search

client = openai.OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": YOUR_APP_NAME,
    }
)

# --- Conversation History ---
INITIAL_SYSTEM_PROMPT = (
    "You are a helpful AI assistant integrated into a CLI. "
    "When asked to create code, provide the code in a single, clearly marked block (e.g., ```python ... ```). "
    "You can also explain the code outside the block. "
    "If you need current information or information beyond your training data to answer a user's query, "
    "you can request a web search. To do this, respond *only* with a JSON object in the following format: "
    "`{\"action\": \"web_search\", \"query\": \"your search query string\"}`. "
    "Do not include any other text or explanation before or after this JSON object if you are requesting a search. "
    "I will then perform the search and provide you with the results in a follow-up message. "
    "You should then use these results to answer the original user query. "
    "If a web search fails, I will inform you with a message explaining the failure; acknowledge the failure and try to answer without search if possible, or inform the user you could not complete the request due to the search failure."
)
conversation_history = [
    {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
]

# --- Helper Functions ---

def strip_html(html_string):
    if BS4_AVAILABLE:
        soup = BeautifulSoup(html_string, "html.parser")
        return soup.get_text()
    else:
        return re.sub(r'<[^>]+>', '', html_string)

def extract_code_blocks(text):
    code_blocks = []
    pattern = re.compile(r"```(\w+)?\s*\n(.*?)\n```", re.DOTALL)
    matches = pattern.finditer(text)
    for match in matches:
        language = match.group(1) if match.group(1) else "unknown"
        code = match.group(2).strip()
        code_blocks.append((language, code))
    return code_blocks

def execute_code(language, code_block):
    click.echo(click.style("\n--- Code to Execute ---", fg="yellow"))
    click.echo(f"Language: {language}")
    click.echo(code_block)
    click.echo(click.style("--- End Code ---", fg="yellow"))
    if not click.confirm(click.style("Confirm execution of the above code block?", fg="magenta"), default=False):
        click.echo("Code execution aborted by user.")
        return "Code execution aborted by user."
    try:
        if language.lower() == "python":
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(code_block)
                temp_file_name = tmp_file.name
            click.echo(f"Executing Python script: {temp_file_name}")
            process = subprocess.run([sys.executable, temp_file_name], capture_output=True, text=True, timeout=30)
            os.remove(temp_file_name)
        elif language.lower() in ["bash", "shell", "sh"]:
            click.echo(f"Executing {language} script...")
            process = subprocess.run(code_block, shell=True, capture_output=True, text=True, timeout=30, check=False)
        else: return f"Execution for language '{language}' is not supported by this agent."
        output = "--- Execution Output ---\n"
        if process.stdout: output += f"STDOUT:\n{process.stdout}\n"
        if process.stderr: output += f"STDERR:\n{process.stderr}\n"
        if not process.stdout and not process.stderr: output += "No output produced.\n"
        output += "--- End Execution Output ---"
        return output
    except subprocess.TimeoutExpired: return "Code execution timed out."
    except Exception as e: return f"An error occurred during code execution: {e}"


def perform_web_search(query, num_results=3):
    if not BRAVE_API_KEY:
        click.echo(click.style("Brave API Key not configured. Cannot perform web search.", fg="red"))
        return "Web search failed: Brave API Key not configured. I cannot perform the search."
    click.echo(click.style(f"Performing Brave web search for: '{query}'...", fg="blue"))
    headers = {"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY, "Accept-Encoding": "gzip"}
    params = {"q": query, "count": num_results}
    response_obj = None 
    try:
        response_obj = requests.get(BRAVE_SEARCH_API_ENDPOINT, headers=headers, params=params, timeout=10)
        response_obj.raise_for_status()
        search_data = response_obj.json()
        results = search_data.get("web", {}).get("results", [])
        if not results:
            click.echo(click.style(f"Brave Search successful but returned no results for '{query}'.", fg="yellow"))
            return f"No results found for '{query}' using Brave Search."
        
        # Simplified format to avoid potential issues
        formatted_results = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results[:num_results]):
            title = strip_html(result.get("title", "N/A"))
            snippet = strip_html(result.get("description", "N/A"))
            url = result.get("url", "N/A")
            # Keep it simple and concise
            formatted_results += f"Result {i+1}: {title} - {url}\n{snippet}\n\n"
        
        final_output = formatted_results.strip()
        if not BS4_AVAILABLE: click.echo(click.style("DEBUG perform_web_search: BeautifulSoup4 not available, used basic regex for HTML stripping.", fg="magenta"))
        click.echo(click.style(f"DEBUG perform_web_search: HTML stripped results (len {len(final_output)}). Preview: {final_output[:100]}...", fg="magenta"))
        return final_output
    except requests.exceptions.HTTPError as http_err:
        error_message_for_cli = f"Brave API HTTP error: {http_err}"
        error_message_for_ai = f"Web search failed due to an API error with status code: {http_err.response.status_code if http_err.response else 'Unknown'}."
        if http_err.response is not None:
            try:
                response_json = http_err.response.json()
                api_error_detail = response_json.get("error", {}).get("detail", "")
                api_error_code = response_json.get("error", {}).get("code", "")
                error_message_for_cli += f" - API Response Code: {api_error_code}, Detail: {api_error_detail}"
                if http_err.response.status_code == 422 and "SUBSCRIPTION_TOKEN_INVALID" in str(api_error_code).upper():
                    critical_cli_message = "\n" + click.style("CRITICAL: Brave Search API key is invalid/expired. Check BRAVE_API_KEY.", fg="red", bold=True)
                    error_message_for_cli += critical_cli_message
                    error_message_for_ai = "Web search failed: The Brave Search API key is invalid or expired. I cannot perform the search."
                elif api_error_detail: error_message_for_ai += f" API message: {api_error_detail}"
            except json.JSONDecodeError: error_message_for_cli += f" - Non-JSON API Error Response: {http_err.response.text}"
        else: error_message_for_cli += " - No response object available from the HTTPError."
        click.echo(click.style(error_message_for_cli, fg="red"))
        return error_message_for_ai
    except requests.exceptions.RequestException as req_err:
        click.echo(click.style(f"Brave API Request failed (e.g., network issue): {req_err}", fg="red"))
        return f"Web search failed due to a network or request error: {str(req_err)}"
    except json.JSONDecodeError:
        click.echo(click.style(f"Failed to parse Brave API response as JSON. Response: {response_obj.text if response_obj else 'No response object'}", fg="red"))
        return "Web search failed: Could not parse the response from the search API."
    except Exception as e:
        click.echo(click.style(f"An unexpected error in perform_web_search: {e}", fg="red"))
        return f"An unexpected error occurred during web search: {str(e)}"

# --- CLI Command ---
@click.command()
@click.option('--prompt', '-p', help="Single prompt to send to the AI. Exits after.")
def chat(prompt):
    global conversation_history
    if prompt: _process_prompt(prompt); return
    click.echo(click.style("Starting interactive chat with AI Agent.", fg="cyan"))
    click.echo("Type 'exit' or 'quit' to end. Using model: " + click.style(MODEL_NAME, fg="blue") + " via OpenRouter.")
    if BRAVE_API_KEY: click.echo("Web search via " + click.style("Brave Search API", fg="blue") + " is enabled.")
    else: click.echo(click.style("Web search is DISABLED (BRAVE_API_KEY not set).", fg="yellow"))
    if not BS4_AVAILABLE: click.echo(click.style("Note: BeautifulSoup4 not found. HTML stripping in search results will be basic.", fg="yellow"))
    while True:
        try:
            user_input = click.prompt(click.style("You", fg="bright_blue"))
            if user_input.lower() in ["exit", "quit"]: break
            _process_prompt(user_input)
        except click.exceptions.Abort: break
        except Exception as e: click.echo(click.style(f"An unexpected error in chat loop: {e}", fg="red"))
    click.echo(click.style("Exiting agent.", fg="yellow"))


def _process_prompt(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})

    MAX_TOOL_CYCLES = 2
    for cycle in range(MAX_TOOL_CYCLES + 1):
        assistant_response_content = ""
        click.echo(click.style(f"\nDEBUG: Cycle {cycle + 1}/{MAX_TOOL_CYCLES + 1}", fg="magenta"))
        
        messages_for_this_api_call = conversation_history 

        if cycle == 1 and len(conversation_history) >= 3:
            click.echo(click.style("DEBUG: Attempting to use simplified history for this API call.", fg="magenta"))
            simplified_history_for_tool_follow_up = [
                conversation_history[0],  
                conversation_history[-3], 
                conversation_history[-2], 
                conversation_history[-1]  
            ]
            messages_for_this_api_call = simplified_history_for_tool_follow_up
            click.echo(click.style("DEBUG: Using simplified history:", fg="yellow"))
            for i, msg in enumerate(messages_for_this_api_call):
                role = msg.get('role'); name_part = f", name: {msg.get('name')}" if msg.get('name') else ""
                content_preview = str(msg.get('content'))[:150] + "..." if msg.get('content') and len(str(msg.get('content'))) > 150 else msg.get('content')
                click.echo(click.style(f"  {i}. Role: {role}{name_part}, Content Preview: '{content_preview}'", fg="yellow"))
        elif cycle > 0 : 
            click.echo(click.style("DEBUG: Conversation history before current AI call (showing full content for tool message):", fg="yellow"))
            for i, msg in enumerate(messages_for_this_api_call):
                role = msg.get('role'); name_part = f", name: {msg.get('name')}" if msg.get('name') else ""
                if role == "tool" and name_part == ", name: web_search":
                    content_full = str(msg.get('content')); click.echo(click.style(f"  {i}. Role: {role}{name_part}, Content:\n-----\n{content_full}\n-----", fg="yellow"))
                else:
                    content_preview = str(msg.get('content'))[:150] + "..." if msg.get('content') and len(str(msg.get('content'))) > 150 else msg.get('content')
                    click.echo(click.style(f"  {i}. Role: {role}{name_part}, Content Preview: '{content_preview}'", fg="yellow"))
        else: 
            click.echo(click.style("DEBUG: Conversation history before current AI call:", fg="yellow"))
            for i, msg in enumerate(messages_for_this_api_call):
                content_preview = str(msg.get('content'))[:150] + "..." if msg.get('content') and len(str(msg.get('content'))) > 150 else msg.get('content')
                click.echo(click.style(f"  {i}. Role: {msg.get('role')}, Content Preview: '{content_preview}'", fg="yellow"))
        try:
            with click.progressbar(length=100, label=click.style("AI is thinking...", fg="blue"), bar_template='%(label)s [%(bar)s]', fill_char='.', empty_char=' ') as bar:
                bar.update(30)
                stream = client.chat.completions.create(model=MODEL_NAME, messages=messages_for_this_api_call, stream=True)
                bar.update(20)
                click.echo(click.style(f"\nAI ({MODEL_NAME}): ", fg="green"), nl=False)
                for chunk in stream:
                    if chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content_piece = chunk.choices[0].delta.content
                        click.echo(content_piece, nl=False); sys.stdout.flush()
                        assistant_response_content += content_piece
                click.echo()
                bar.update(50)
        except openai.APIStatusError as e: 
            click.echo(click.style(f"\n--- OpenRouter openai.APIStatusError ---", fg="red", bold=True))
            click.echo(click.style(f"Status Code: {e.status_code}", fg="red"))
            if e.response:
                click.echo(click.style(f"Raw Response Body:\n{e.response.text}", fg="red"))
                try: error_details_json = e.response.json(); click.echo(click.style(f"Parsed JSON Error Body:\n{json.dumps(error_details_json, indent=2)}", fg="red"))
                except json.JSONDecodeError: click.echo(click.style("Could not parse error response body as JSON.", fg="red"))
            else: click.echo(click.style("No response object in APIStatusError.", fg="red"))
            if hasattr(e, 'request') and e.request and hasattr(e.request, 'content'): click.echo(click.style(f"Request that failed (body preview):\n{str(e.request.content)[:500]}...", fg="red"))
            if assistant_response_content: click.echo(click.style(f"AI's last (possibly incomplete) message before error: {assistant_response_content}", fg="yellow"))
            return
        except openai.APIError as e: 
            click.echo(click.style(f"\n--- OpenRouter openai.APIError ---", fg="red", bold=True))
            click.echo(click.style(f"Error Type: {type(e).__name__}", fg="red"))
            click.echo(click.style(f"Error Message: {str(e)}", fg="red"))
            click.echo(click.style(f"Error Args: {e.args}", fg="red"))
            if hasattr(e, 'http_body') and e.http_body:
                click.echo(click.style(f"HTTP Body:\n{e.http_body}", fg="red"))
                try: error_json = json.loads(e.http_body); click.echo(click.style(f"Parsed HTTP Body JSON:\n{json.dumps(error_json, indent=2)}", fg="red"))
                except json.JSONDecodeError: click.echo(click.style("Could not parse HTTP Body as JSON.", fg="red"))
            if hasattr(e, 'http_status') and e.http_status: click.echo(click.style(f"HTTP Status: {e.http_status}", fg="red"))
            if hasattr(e, 'json_body') and e.json_body: click.echo(click.style(f"JSON Body from Error:\n{json.dumps(e.json_body, indent=2)}", fg="red"))
            if hasattr(e, 'headers') and e.headers: click.echo(click.style(f"Headers from Error: {e.headers}", fg="red"))
            if assistant_response_content: click.echo(click.style(f"AI's last (possibly incomplete) message before error: {assistant_response_content}", fg="yellow"))
            return
        except openai.APIConnectionError as e: click.echo(click.style(f"OpenRouter API Connection Error: {e}", fg="red")); return
        except openai.RateLimitError as e: click.echo(click.style(f"OpenRouter Rate Limit Exceeded: {e}", fg="red")); return
        except Exception as e: 
            click.echo(click.style(f"\n--- An unexpected generic error occurred during AI communication ---", fg="red", bold=True))
            click.echo(click.style(f"Error Type: {type(e).__name__}", fg="red"))
            click.echo(click.style(f"Error Args: {e.args}", fg="red"))
            if hasattr(e, 'response') and e.response is not None: click.echo(click.style(f"Error Response Text (from generic exception): {e.response.text}", fg="red"))
            return

        is_tool_request = False
        if assistant_response_content:
            try:
                clean_response = assistant_response_content.strip()
                if not (clean_response.startswith("{") and clean_response.endswith("}")):
                    raise json.JSONDecodeError("Not a plain JSON object.", clean_response, 0)
                tool_request_data = json.loads(clean_response)
                if isinstance(tool_request_data, dict) and \
                   tool_request_data.get("action") == "web_search" and \
                   isinstance(tool_request_data.get("query"), str):
                    if cycle < MAX_TOOL_CYCLES:
                        if not BRAVE_API_KEY:
                            click.echo(click.style("AI requested web search, but BRAVE_API_KEY is not set. Informing AI.", fg="red"))
                            conversation_history.append({"role": "assistant", "content": clean_response})
                            conversation_history.append({"role": "user", "content": "Web search could not be performed: API key not configured. Please try to answer my question without search results."})
                            continue
                        is_tool_request = True
                        search_query = tool_request_data["query"] 
                        click.echo(click.style(f"AI requested Brave web search for: '{search_query}'.", fg="magenta"))
                        conversation_history.append({"role": "assistant", "content": clean_response})
                        search_results_content = perform_web_search(search_query)
                        click.echo(click.style(f"DEBUG _process_prompt: Using search_results_content (len {len(search_results_content)}). Preview: '{search_results_content[:100]}...'", fg="yellow"))
                        # Instead of using tool mechanism, add search results as a user message
                        conversation_history.append({"role": "user", "content": f"Here are the search results for '{search_query}':\n\n{search_results_content}\n\nPlease use these results to answer my original question."})
                        continue 
                    else:
                        click.echo(click.style(f"\nMax web search cycles ({MAX_TOOL_CYCLES}) reached. AI search request ignored.", fg="yellow"))
            except json.JSONDecodeError: pass 
            except Exception as e: click.echo(click.style(f"Error processing potential tool call: {type(e).__name__} - {e}", fg="red")); pass 
        
        if not is_tool_request:
            if assistant_response_content:
                conversation_history.append({"role": "assistant", "content": assistant_response_content})
                code_blocks = extract_code_blocks(assistant_response_content)
                if code_blocks:
                    click.echo(click.style("\nCode block(s) detected:", fg="yellow"))
                    for i, (lang, code) in enumerate(code_blocks):
                        click.echo(click.style(f"--- Code Block {i+1} ({lang}) ---", fg="blue"))
                        click.echo(code)
                        click.echo(click.style(f"--- End Code Block {i+1} ---", fg="blue"))
                        if click.confirm(click.style(f"Execute Code Block {i+1} ({lang})?", fg="magenta"), default=False):
                            click.echo(click.style("WARNING: Executing AI code can be DANGEROUS!", bold=True, fg="red"))
                            execution_result = execute_code(lang, code)
                            click.echo(click.style(execution_result, fg="cyan"))
                            conversation_history.append({"role": "user", "content": f"Output of executing {lang} code (Block {i+1}):\n{execution_result}"})
                elif assistant_response_content.strip():
                     click.echo(click.style("No executable code blocks in AI's latest response.", fg="blue"))
            else: click.echo(click.style("AI produced an empty response.", fg="yellow"))
            return 
    click.echo(click.style("Exited processing loop unexpectedly.", fg="red"))

if __name__ == '__main__':
    if not BS4_AVAILABLE:
        print("Consider installing BeautifulSoup4 for better HTML stripping from search results: pip install beautifulsoup4")
    chat()
