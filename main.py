import os
import openai
import subprocess
from pathlib import Path
import sys
from dotenv import load_dotenv 

load_dotenv()

class CodingAssistant:
    def __init__(self):
        self.client = openai.OpenAI()
        self.root_directory = self.set_root_directory()
        self.assistant = self.create_assistant()
        self.thread = self.client.beta.threads.create()

    def set_root_directory(self):
        print("Enter the root directory path (press Enter to use current directory):")
        path = input().strip()
        
        if path and os.path.exists(path):
            return Path(path)
        else:
            if path:
                print("Directory not found. Using current directory instead.")
            return Path.cwd()

    def create_assistant(self):
        return self.client.beta.assistants.create(
            name="Coding Assistant",
            instructions="""You are a coding assistant that helps users with programming tasks. 
            When working with files and commands:
            1. Always use the provided file system functions (read_file, write_file) instead of the code interpreter
            2. Use the execute_command function for system commands
            3. All paths should be relative to the user's root directory
            4. Do not create files in the sandbox environment
            5. When modifying or creating files, use the write_file function
            6. When reading files, use the read_file function
            7. Provide clear, concise responses about what actions were taken
            
            Important: Never use the sandbox environment for file operations - always use the provided functions.""",
            model="gpt-4o-mini",
            tools=[
                {"type": "function", 
                "function": {
                    "name": "execute_command",
                    "description": "Execute a system command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }},
                {"type": "function", 
                "function": {
                    "name": "read_file",
                    "description": "Read content of a file relative to root directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file relative to root directory"
                            }
                        },
                        "required": ["file_path"]
                    }
                }},
                {"type": "function", 
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file relative to root directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file relative to root directory"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["file_path", "content"]
                    }
                }}
            ]
        )

    def execute_command(self, command):
        try:
            print(f"🔄 Running: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.stderr and not result.stdout:
                print(f"⚠️ Warning: {result.stderr}")
            return f"Output: {result.stdout}\nErrors: {result.stderr}"
        except Exception as e:
            print(f"❌ Command failed: {str(e)}")
            return f"Error executing command: {str(e)}"

    def read_file(self, file_path):
        try:
            full_path = self.root_directory / file_path
            with open(full_path, 'r') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, file_path, content):
        try:
            full_path = self.root_directory / file_path
            print(f"📝 Updating file: {file_path}")
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as file:
                file.write(content)
            return "File written successfully"
        except Exception as e:
            print(f"❌ File operation failed: {str(e)}")
            return f"Error writing file: {str(e)}"


    def handle_tool_calls(self, tool_calls):
        tool_outputs = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = eval(tool_call.function.arguments)
            
            if function_name == "execute_command":
                output = self.execute_command(function_args["command"])
            elif function_name == "read_file":
                output = self.read_file(function_args["file_path"])
            elif function_name == "write_file":
                output = self.write_file(function_args["file_path"], function_args["content"])
            
            tool_outputs.append({"tool_call_id": tool_call.id, "output": output})
        
        return tool_outputs

    def process_query(self, query):
        print("\n🤖 Understanding your request...")
        attempt = 1
        max_retries = 3
        last_error = None
        
        while attempt <= max_retries:
            if attempt > 1:
                print(f"\n🔄 Attempt {attempt}/{max_retries} - Retrying due to: {last_error}")
                # Add context about the previous failure to the query
                query = f"""Previous attempt failed with error: {last_error}
                Please try again with a different approach.
                Original query: {query}"""
            
            try:
                self.client.beta.threads.messages.create(
                    thread_id=self.thread.id,
                    role="user",
                    content=query
                )

                run = self.client.beta.threads.runs.create(
                    thread_id=self.thread.id,
                    assistant_id=self.assistant.id
                )

                while True:
                    run = self.client.beta.threads.runs.retrieve(
                        thread_id=self.thread.id,
                        run_id=run.id
                    )

                    if run.status == "requires_action":
                        print("⚙️ Executing commands...")
                        tool_outputs = self.handle_tool_calls(
                            run.required_action.submit_tool_outputs.tool_calls
                        )
                        run = self.client.beta.threads.runs.submit_tool_outputs(
                            thread_id=self.thread.id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )
                    elif run.status == "completed":
                        messages = self.client.beta.threads.messages.list(
                            thread_id=self.thread.id
                        )
                        print("✅ Done!")
                        return messages.data[0].content[0].text.value
                    elif run.status in ["failed", "cancelled", "expired"]:
                        last_error = f"Run {run.status}"
                        raise Exception(last_error)

            except Exception as e:
                last_error = str(e)
                if attempt == max_retries:
                    print(f"❌ Failed after {max_retries} attempts. Last error: {last_error}")
                    return f"Error: Operation failed after {max_retries} attempts. Last error: {last_error}"
                attempt += 1
                continue

    def start(self):
        print(f"\n📂 Root directory set to: {self.root_directory}")
        print("\n💡 Enter your commands in natural language (type 'exit' to quit):")
        
        while True:
            query = input("\n🔷 Command: ").strip()
            if query.lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            
            try:
                response = self.process_query(query)
                print(f"\n🤖 {response}")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    assistant = CodingAssistant()
    assistant.start()