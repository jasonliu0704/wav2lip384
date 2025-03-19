#!/usr/bin/env python3
"""
Smart Command Executor (smart_cmd.py)

This utility executes a sequence of commands from a file with advanced features:
- Robust error handling with customizable retry logic
- Progress tracking and reporting
- Command validation before execution
- Variable substitution in commands
- Conditional command execution
- Execution timeout protection

Usage:
    python smart_cmd.py [options] command_file.txt

Options:
    --dry-run             Show commands without executing them
    --continue-on-error   Continue execution even if a command fails
    --log-file FILE       Log output to specified file
    --env-file FILE       Load environment variables from file
    --timeout SECONDS     Set default command timeout (default: 300s)
    --quiet               Suppress non-error output
    --vars KEY=VALUE      Define variables for substitution

Command File Syntax:
    # Comments start with '#'
    
    # Simple commands
    pip install -r requirements.txt
    
    # Command with timeout override (in seconds)
    [timeout:600] python train.py --epochs 100
    
    # Conditional command (only runs if previous command succeeded)
    [if-success] echo "Previous command succeeded"
    
    # Conditional command (only runs if previous command failed)
    [if-failure] echo "Previous command failed"
    
    # Commands with variable substitution
    [var:model_name=wav2lip]
    python infer.py --model ${model_name}
    
    # Command with retry logic
    [retry:3] wget https://example.com/large-file.zip
"""

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union


class CommandTimeoutError(Exception):
    """Exception raised when a command exceeds its timeout."""
    pass


class CommandFailedError(Exception):
    """Exception raised when a command fails to execute."""
    pass


class SmartCommand:
    """Represents a command with its options and modifiers."""
    
    def __init__(self, line: str, line_number: int, variables: Dict[str, str]):
        self.original_line = line
        self.line_number = line_number
        self.command = line
        self.timeout = None
        self.retries = 0
        self.condition = None
        self.variables = variables
        self.parse()
        
    def parse(self):
        """Parse command and extract modifiers."""
        # Extract modifiers within square brackets
        pattern = r'^\[([^]]+)\]\s*(.*)$'
        match = re.match(pattern, self.command)
        
        if match:
            modifier, command = match.groups()
            self.command = command.strip()
            
            # Parse timeout modifier
            timeout_match = re.match(r'timeout:(\d+)', modifier)
            if timeout_match:
                self.timeout = int(timeout_match.group(1))
                
            # Parse retry modifier
            retry_match = re.match(r'retry:(\d+)', modifier)
            if retry_match:
                self.retries = int(retry_match.group(1))
                
            # Parse condition modifier
            if modifier == 'if-success':
                self.condition = 'success'
            elif modifier == 'if-failure':
                self.condition = 'failure'
                
            # Parse variable definition
            var_match = re.match(r'var:([^=]+)=(.*)', modifier)
            if var_match:
                var_name, var_value = var_match.groups()
                self.variables[var_name.strip()] = var_value.strip()
                self.command = ""  # No command to execute for variable definition
        
        # Apply variable substitution
        self.command = self.substitute_variables(self.command)
    
    def substitute_variables(self, text: str) -> str:
        """Substitute variables in the command string."""
        # Replace ${var} or $var with their values
        for var_name, var_value in self.variables.items():
            text = text.replace(f"${{{var_name}}}", var_value)
            text = text.replace(f"${var_name}", var_value)
        return text
    
    def should_execute(self, previous_success: bool) -> bool:
        """Determine if this command should be executed based on conditions."""
        if not self.command:
            return False  # Skip empty commands or variable definitions
            
        if self.condition == 'success':
            return previous_success
        elif self.condition == 'failure':
            return not previous_success
        return True
    
    def __str__(self) -> str:
        """Return string representation of the command for display."""
        modifiers = []
        if self.timeout:
            modifiers.append(f"timeout:{self.timeout}s")
        if self.retries:
            modifiers.append(f"retry:{self.retries}")
        if self.condition:
            modifiers.append(self.condition)
            
        if modifiers:
            return f"[{', '.join(modifiers)}] {self.command}"
        return self.command


class SmartCommandExecutor:
    """Executes a sequence of commands with advanced features."""
    
    def __init__(self, 
                 command_file: str, 
                 dry_run: bool = False,
                 continue_on_error: bool = False,
                 log_file: Optional[str] = None,
                 timeout: int = 300,
                 quiet: bool = False,
                 variables: Dict[str, str] = None):
        
        self.command_file = command_file
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.log_file = log_file
        self.default_timeout = timeout
        self.quiet = quiet
        self.variables = variables or {}
        self.log_handle = None
        
        self.commands = []
        self.previous_success = True
        
        # Environment variables
        self.env = os.environ.copy()
        for key, value in self.variables.items():
            self.env[key] = value
    
    def __enter__(self):
        """Context manager entry."""
        if self.log_file:
            self.log_handle = open(self.log_file, 'a')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.log_handle:
            self.log_handle.close()
    
    def load_commands(self) -> int:
        """Load commands from file and return number of commands."""
        with open(self.command_file, 'r') as f:
            lines = f.readlines()
        
        self.commands = []
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            self.commands.append(SmartCommand(line, i + 1, self.variables))
        
        return len(self.commands)
    
    def log(self, message: str, error: bool = False):
        """Log a message to console and log file if configured."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_msg = f"[{timestamp}] {message}"
        
        if error:
            print(formatted_msg, file=sys.stderr)
        elif not self.quiet:
            print(formatted_msg)
        
        if self.log_handle:
            self.log_handle.write(formatted_msg + '\n')
            self.log_handle.flush()
    
    def execute_command(self, cmd: SmartCommand, infinite_retries=False) -> bool:
        """Execute a single command and return True if successful."""
        if not cmd.command:
            return True  # Variable definition is always successful
            
        if not cmd.should_execute(self.previous_success):
            self.log(f"Skipping command {cmd} due to condition")
            return True  # Skipped commands don't affect success status
        
        if self.dry_run:
            self.log(f"Would execute: {cmd}")
            return True
        
        timeout = cmd.timeout or self.default_timeout
        success = False
        attempts = 0
        max_attempts = max(1, cmd.retries + 1)
        
        while (attempts < max_attempts) or infinite_retries:
            attempts += 1
            attempt_str = f"(attempt {attempts}/{max_attempts})" if max_attempts > 1 else ""
            
            try:
                self.log(f"Executing: {cmd} {attempt_str}")
                
                # Start time for tracking execution duration
                start_time = time.time()
                
                # Run the command with timeout and capture output
                try:
                    process = subprocess.Popen(
                        shlex.split(cmd.command),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        env=self.env
                    )
                    
                    # Stream output in real-time
                    stdout_data, stderr_data = "", ""
                    stdout_lines, stderr_lines = [], []
                    
                    # Set up polling
                    while process.poll() is None:
                        # Check if process has timed out
                        if time.time() - start_time > timeout:
                            process.terminate()
                            try:
                                process.wait(timeout=5)  # Give it a chance to terminate gracefully
                            except subprocess.TimeoutExpired:
                                process.kill()  # Force kill if it doesn't terminate
                            
                            raise CommandTimeoutError(f"Command timed out after {timeout} seconds")
                        
                        # Process stdout
                        if process.stdout:
                            line = process.stdout.readline()
                            if line:
                                if not self.quiet:
                                    print(line, end='')
                                stdout_lines.append(line)
                                stdout_data += line
                        
                        # Process stderr
                        if process.stderr:
                            line = process.stderr.readline()
                            if line:
                                print(line, end='', file=sys.stderr)
                                stderr_lines.append(line)
                                stderr_data += line
                                
                        time.sleep(0.1)  # Small sleep to avoid CPU hogging
                    
                    # Get any remaining output
                    if process.stdout:
                        for line in process.stdout:
                            if not self.quiet:
                                print(line, end='')
                            stdout_lines.append(line)
                            stdout_data += line
                    
                    if process.stderr:
                        for line in process.stderr:
                            print(line, end='', file=sys.stderr)
                            stderr_lines.append(line)
                            stderr_data += line
                    
                    # Check return code
                    if process.returncode != 0:
                        raise CommandFailedError(f"Command failed with return code {process.returncode}")
                    
                    # Command succeeded
                    execution_time = time.time() - start_time
                    self.log(f"Command completed successfully in {execution_time:.2f}s")
                    success = True
                    break
                    
                except CommandFailedError as e:
                    execution_time = time.time() - start_time
                    self.log(f"Command failed after {execution_time:.2f}s: {str(e)}", error=True)
                    if attempts < max_attempts:
                        retry_delay = min(5 * attempts, 30)  # Incremental backoff with cap
                        self.log(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        self.log(f"All {max_attempts} attempts failed", error=True)
                
            except CommandTimeoutError as e:
                self.log(f"Command timed out: {str(e)}", error=True)
                if attempts < max_attempts:
                    self.log(f"Retrying command after timeout...")
                else:
                    self.log(f"All {max_attempts} attempts timed out", error=True)
                    
            except Exception as e:
                self.log(f"Error executing command: {str(e)}", error=True)
                break
        
        return success
    
    def execute_all(self, infinite_retries=False) -> bool:
        """Execute all commands and return True if all succeeded."""
        self.log(f"Starting execution of {len(self.commands)} commands from {self.command_file}")
        overall_success = True
        
        for i, cmd in enumerate(self.commands):
            self.log(f"Command {i+1}/{len(self.commands)}")
            
            success = self.execute_command(cmd, infinite_retries)
            self.previous_success = success
            
            if not success:
                overall_success = False
                if not self.continue_on_error:
                    self.log("Stopping execution due to command failure", error=True)
                    break
        
        status = "successfully" if overall_success else "with some failures"
        self.log(f"Execution completed {status}")
        return overall_success


def load_env_file(filename: str) -> Dict[str, str]:
    """Load environment variables from a file."""
    env_vars = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error loading environment file: {str(e)}", file=sys.stderr)
    
    return env_vars


def parse_vars(var_list: List[str]) -> Dict[str, str]:
    """Parse variables from command line arguments."""
    variables = {}
    for var in var_list:
        if '=' in var:
            key, value = var.split('=', 1)
            variables[key.strip()] = value.strip()
    return variables


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Execute a sequence of commands with advanced features.')
    parser.add_argument('command_file', help='File containing commands to execute')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without executing')
    parser.add_argument('--continue-on-error', action='store_true', help='Continue execution if a command fails')
    parser.add_argument('--log-file', help='Log output to specified file')
    parser.add_argument('--env-file', help='Load environment variables from file')
    parser.add_argument('--timeout', type=int, default=300, help='Default command timeout in seconds')
    parser.add_argument('--quiet', action='store_true', help='Suppress non-error output')
    parser.add_argument('--vars', action='append', default=[], help='Define variables for substitution (KEY=VALUE)')
    parser.add_argument('--infinite-retries', action='store_true', help='Keep retrying a failing command until it succeeds')
    
    args = parser.parse_args()
    
    # Load variables from environment file and command line
    variables = {}
    if args.env_file:
        variables.update(load_env_file(args.env_file))
    variables.update(parse_vars(args.vars))
    
    try:
        with SmartCommandExecutor(
            args.command_file,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
            log_file=args.log_file,
            timeout=args.timeout,
            quiet=args.quiet,
            variables=variables
        ) as executor:
            
            num_commands = executor.load_commands()
            if num_commands == 0:
                executor.log("No commands found in file", error=True)
                return 1
                
            success = executor.execute_all(infinite_retries=args.infinite_retries)
            return 0 if success else 1
            
    except FileNotFoundError:
        print(f"Error: Command file '{args.command_file}' not found", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
