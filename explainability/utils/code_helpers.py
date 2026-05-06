import os
import sys
import io
import traceback
from typing import Dict, Any

def execute_python_code(code: str, working_dir: str = "./output") -> Dict[str, Any]:
    """
    Executes Python code in a controlled environment and returns the output and errors.
    It temporarily changes the current working directory so the script saves files in `working_dir`.
    """
    # Create the target directory if it doesn't exist
    target_path = os.path.abspath(working_dir)
    os.makedirs(target_path, exist_ok=True)
        
    # Capture standard output and errors
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_output
    
    # Track original png files in the directory
    initial_files = set(os.listdir(target_path))
    
    # Prepare execution environment
    exec_globals = {"__name__": "__main__"}
    
    original_cwd = os.getcwd()
    os.chdir(target_path)
    
    error = None
    try:
        exec(code, exec_globals)
    except Exception as e:
        error = traceback.format_exc()
    finally:
        # Restore cwd and stdout/stderr
        os.chdir(original_cwd)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    output_str = redirected_output.getvalue()
    
    if error:
        output_str += "\nException traceback:\n" + error
        
    # Find new plots
    current_files = set(os.listdir(target_path))
    new_files = current_files - initial_files
    new_pngs = [os.path.join(target_path, f) for f in new_files if f.endswith('.png')]
    
    return {
        "output": output_str,
        "error": error if error else "",
        "new_plots": new_pngs
    }
