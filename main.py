import json
import os
import shlex
import time
import unicodedata
import csv
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from pygments.lexers.sql import SqlLexer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint
import re

@dataclass
class QueryStats:
    start_time: float
    end_time: float
    rows_processed: int
    rows_returned: int
    cache_hit: bool = False

class Index:
    def __init__(self, key: str):
        self.key = key
        self.index: Dict[str, List[int]] = defaultdict(list)
    
    def build(self, data: List[dict]):
        self.index.clear()
        for i, item in enumerate(data):
            value = str(self.get_nested_value(item, self.key))
            self.index[value].append(i)
    
    def get_nested_value(self, item: dict, key_path: str) -> Any:
        """Enhanced get_nested_value to properly handle arrays and nested structures"""
        if not key_path or not item:
            return None

        def process_array_value(array_data, remaining_path):
            """Helper function to process array values"""
            if not remaining_path:
                return array_data
            # If we have more path to process, apply it to each element
            results = []
            for element in array_data:
                if isinstance(element, (dict, list)):
                    result = self.get_nested_value(element, remaining_path)
                    if result is not None:
                        results.append(result)
            return results if results else None

        parts = []
        current_part = ""
        in_quotes = False
        
        # Split path handling array notation
        for char in key_path:
            if char == '"':
                in_quotes = not in_quotes
            elif char == '.' and not in_quotes:
                if current_part:
                    parts.append(current_part)
                current_part = ""
            else:
                current_part += char
        
        if current_part:
            parts.append(current_part)

        current = item
        for i, part in enumerate(parts):
            if current is None:
                return None

            # Check for array index notation
            array_match = re.match(r'(.+?)\[(\d+)\]$', part)
            if array_match:
                key, index = array_match.groups()
                index = int(index)
                
                # Get the array first
                if isinstance(current, dict):
                    current = current.get(key)
                
                # Then access the index
                if isinstance(current, list) and 0 <= index < len(current):
                    current = current[index]
                    continue
                return None

            # Handle regular nested access
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                # If we're accessing a property of array elements
                remaining_path = '.'.join(parts[i:])
                return process_array_value(current, remaining_path)
            else:
                return None

        return current

    def find(self, value: str) -> List[int]:
        return self.index.get(str(value), [])

class Cache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[List[dict], datetime]] = {}
    
    def get(self, query: str) -> Optional[List[dict]]:
        if query in self.cache:
            results, timestamp = self.cache[query]
            # Cache invalidation after 5 minutes
            if (datetime.now() - timestamp).total_seconds() < 300:
                return results
            else:
                del self.cache[query]
        return None
    
    def set(self, query: str, results: List[dict]):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest]
        self.cache[query] = (results, datetime.now())


class JsonAnalyzer:
    def __init__(self):
        self.loaded_files: Dict[str, Any] = {}
        self.key_stats: Dict[str, Dict] = {}
        self.indexes: Dict[str, Dict[str, Index]] = {}
        self.cache = Cache()
        self.console = Console()
        
        # Setup command history
        history_file = os.path.expanduser('~/.json_query_history')
        self.history = FileHistory(history_file)
        
        # Setup SQL completer
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'IN', 'LIKE', 
                       'LIMIT', 'OFFSET', 'ORDER', 'BY', 'ASC', 'DESC']
        self.completer = WordCompleter(sql_keywords, ignore_case=True)
        
        # Setup prompt session with styling
        style = Style.from_dict({
            'completion-menu.completion': 'bg:#008888 #ffffff',
            'completion-menu.completion.current': 'bg:#00aaaa #000000',
            'scrollbar.background': 'bg:#88aaaa',
            'scrollbar.button': 'bg:#222222',
        })
        
        self.session = PromptSession(
            history=self.history,
            completer=self.completer,
            lexer=PygmentsLexer(SqlLexer),
            style=style
        )

    def create_index(self, table: str, key: str):
        """Create an index for faster querying"""
        if table not in self.indexes:
            self.indexes[table] = {}
        
        index = Index(key)
        data = self.loaded_files[table]
        if isinstance(data, list):
            with Progress() as progress:
                task = progress.add_task(f"[cyan]Building index for {key}...", total=len(data))
                index.build(data)
                progress.update(task, advance=len(data))
        
        self.indexes[table][key] = index
        self.console.print(f"[green]Index created for {key} on {table}[/green]")

    def load_json_file(self, filepath: str, lazy: bool = True) -> Optional[dict]:
        try:
            filename = Path(filepath).stem
            if lazy:
                # Store file path for lazy loading
                self.loaded_files[filename] = filepath
                self.console.print(f"[cyan]File {filepath} registered for lazy loading[/cyan]")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.loaded_files[filename] = data
                self._analyze_data(filename, data)
                return data
                
        except Exception as e:
            self.console.print(f"[red]Error loading {filepath}: {str(e)}[/red]")
            return None
        
    def analyze_keys(self, data: Any, prefix: str = "", stats: Dict = None) -> Dict:
        """Analyze the keys in JSON data and count their occurrences"""
        if stats is None:
            stats = {}

        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if new_prefix not in stats:
                    stats[new_prefix] = {"count": 1}
                else:
                    stats[new_prefix]["count"] += 1
                self.analyze_keys(value, new_prefix, stats)
        elif isinstance(data, list):
            for item in data:
                self.analyze_keys(item, prefix, stats)

        return stats
    
    def get_nested_value(self, item: dict, key_path: str) -> Any:
        """Enhanced get_nested_value to properly handle arrays and nested structures"""
        if not key_path or not item:
            return None

        def process_array_value(array_data, remaining_path):
            """Helper function to process array values"""
            if not remaining_path:
                return array_data
            # If we have more path to process, apply it to each element
            results = []
            for element in array_data:
                if isinstance(element, (dict, list)):
                    result = self.get_nested_value(element, remaining_path)
                    if result is not None:
                        results.append(result)
            return results if results else None

        parts = []
        current_part = ""
        in_quotes = False
        
        # Split path handling array notation
        for char in key_path:
            if char == '"':
                in_quotes = not in_quotes
            elif char == '.' and not in_quotes:
                if current_part:
                    parts.append(current_part)
                current_part = ""
            else:
                current_part += char
        
        if current_part:
            parts.append(current_part)

        current = item
        for i, part in enumerate(parts):
            if current is None:
                return None

            # Check for array index notation
            array_match = re.match(r'(.+?)\[(\d+)\]$', part)
            if array_match:
                key, index = array_match.groups()
                index = int(index)
                
                # Get the array first
                if isinstance(current, dict):
                    current = current.get(key)
                
                # Then access the index
                if isinstance(current, list) and 0 <= index < len(current):
                    current = current[index]
                    continue
                return None

            # Handle regular nested access
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return None
            elif isinstance(current, list):
                # If we're accessing a property of array elements
                if i < len(parts) - 1:
                    # We're not at the last part, keep processing
                    new_results = []
                    for element in current:
                        if isinstance(element, dict):
                            value = element.get(part)
                            if value is not None:
                                new_results.append(value)
                    current = new_results if new_results else None
                else:
                    # We're at the last part, collect all values
                    values = []
                    for element in current:
                        if isinstance(element, dict):
                            value = element.get(part)
                            if value is not None:
                                values.append(value)
                    return values if values else None
            else:
                return None

        return current

    def display_key_analysis(self):
        """Display analysis of keys and their occurrences"""
        # Clear existing key stats
        self.key_stats.clear()
        
        # Analyze all loaded files
        for filename, data in self.loaded_files.items():
            # Ensure data is loaded if lazy loading is enabled
            if isinstance(data, str):
                self._ensure_data_loaded(filename)
                data = self.loaded_files[filename]
            
            if isinstance(data, list):
                # For list of objects, analyze first item and then update counts
                if data:
                    self.key_stats.update(self.analyze_keys(data[0]))
                    for item in data[1:]:
                        self.analyze_keys(item, stats=self.key_stats)
            else:
                self.key_stats.update(self.analyze_keys(data))

        # Display results in a table
        table = Table(show_header=True, header_style="bold magenta", title="JSON Key Analysis")
        table.add_column("Key Name", style="cyan")
        table.add_column("Occurrence Count", justify="right", style="green")

        # Sort keys alphabetically
        for key, info in sorted(self.key_stats.items()):
            table.add_row(key, str(info["count"]))

        self.console.print(table)

    def load_directory(self, directory: str, lazy: bool = True):
        try:
            for file in os.listdir(directory):
                if file.endswith('.json'):
                    filepath = os.path.join(directory, file)
                    self.load_json_file(filepath, lazy)
        except Exception as e:
            self.console.print(f"[red]Error loading directory {directory}: {str(e)}[/red]")

    def _analyze_data(self, filename: str, data: Any):
        """Analyze data structure and create initial indexes"""
        if isinstance(data, list) and data:
            # Create indexes for commonly queried fields
            common_fields = ['id', 'name', 'type', 'status']
            for field in common_fields:
                if any(field in item for item in data):
                    self.create_index(filename, field)

    def _ensure_data_loaded(self, table: str) -> bool:
        """Ensure data is loaded for lazy-loaded files"""
        if isinstance(self.loaded_files[table], str):
            filepath = self.loaded_files[table]
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.loaded_files[table] = data
                self._analyze_data(table, data)
            return True
        return False

    def parse_complex_condition(self, condition: str) -> List[tuple]:
        """Parse complex WHERE conditions with AND/OR/IN operators"""
        conditions = []
        current_condition = ""
        in_parentheses = 0
        in_quotes = False
        
        for char in condition:
            if char == '"':
                in_quotes = not in_quotes
            elif char == '(' and not in_quotes:
                in_parentheses += 1
            elif char == ')' and not in_quotes:
                in_parentheses -= 1
            
            if char.upper() in ('A', 'O') and not in_quotes and not in_parentheses:
                if condition[len(current_condition):].upper().startswith('AND'):
                    conditions.append((current_condition.strip(), 'AND'))
                    current_condition = ""
                    continue
                elif condition[len(current_condition):].upper().startswith('OR'):
                    conditions.append((current_condition.strip(), 'OR'))
                    current_condition = ""
                    continue
            
            current_condition += char
        
        conditions.append((current_condition.strip(), None))
        return conditions
    def evaluate_simple_condition(self, item: dict, condition: str) -> bool:
        """
        Evaluates a simple condition against an item, with enhanced array support
        while maintaining original functionality
        """
        operators = ["=", "!=", ">", "<", ">=", "<=", "LIKE"]
        operator = None
        
        # Find the operator while avoiding false matches
        for op in operators:
            # Add spaces around operator to avoid partial matches
            spaced_op = f" {op} "
            if spaced_op in condition:
                operator = op
                break
            # Also check for operators at the start of condition
            elif condition.startswith(op + " "):
                operator = op
                break
            # And at the end of condition
            elif condition.endswith(" " + op):
                operator = op
                break
            # If none of above, check original way as fallback
            elif op in condition:
                operator = op
                break

        if not operator:
            return False

        # Split with maxsplit=1 to handle cases where operator might appear multiple times
        left, right = condition.split(operator, maxsplit=1)
        left = left.strip()
        right = right.strip().strip('"\'')
        
        # Get the value using enhanced get_nested_value
        value = self.get_nested_value(item, left)

        # Handle array values
        if isinstance(value, list):
            # For array values, check if any element satisfies the condition
            return any(self._evaluate_single_value(v, right, operator) for v in value)
        else:
            return self._evaluate_single_value(value, right, operator)

    def _evaluate_single_value(self, value, right: str, operator: str) -> bool:
        """Helper method to evaluate a single value against the condition"""
        # Handle None values
        if value is None:
            str_value = ""
        else:
            str_value = str(value).strip()

        # Normalize unicode and prepare comparison values
        str_value = unicodedata.normalize('NFKC', str_value)
        str_right = unicodedata.normalize('NFKC', right.strip())

        if operator == "=":
            return str_value == str_right
                
        elif operator == "!=":
            return str_value != str_right
                
        elif operator in [">", "<", ">=", "<="]:
            try:
                # Handle numeric comparisons
                value_num = float(value) if value is not None else 0
                right_num = float(right)
                
                if operator == ">":
                    return value_num > right_num
                elif operator == "<":
                    return value_num < right_num
                elif operator == ">=":
                    return value_num >= right_num
                else:  # <=
                    return value_num <= right_num
            except (TypeError, ValueError):
                # If numeric conversion fails, return False
                return False
                    
        elif operator.upper() == "LIKE":
            try:
                # Prepare strings for LIKE comparison
                if value is None:
                    str_value = ""
                else:
                    str_value = str(value)
                pattern = str(right)
                
                # Normalize unicode and strip whitespace
                str_value = unicodedata.normalize('NFKC', str_value.strip())
                pattern = unicodedata.normalize('NFKC', pattern.strip())
                
                # Case insensitive comparison
                str_value = str_value.upper()
                pattern = pattern.upper()
                
                # Handle different LIKE pattern cases
                if pattern.startswith('%') and pattern.endswith('%'):
                    search_text = pattern[1:-1]  # Remove % from both ends
                    return search_text in str_value
                elif pattern.startswith('%'):
                    search_text = pattern[1:]  # Remove starting %
                    return str_value.endswith(search_text)
                elif pattern.endswith('%'):
                    search_text = pattern[:-1]  # Remove ending %
                    return str_value.startswith(search_text)
                else:
                    return str_value == pattern
                    
            except (TypeError, ValueError, AttributeError) as e:
                print(f"Error in LIKE comparison: {e}")
                return False

        return False
    
    def evaluate_complex_condition(self, item: dict, condition: str) -> bool:
        """Evaluate complex WHERE conditions"""
        if not condition:
            return True

        # Parse conditions (for AND/OR operations)
        conditions = self.parse_complex_condition(condition)
        result = True
        last_operator = None
        
        # If only one condition, evaluate it directly
        if len(conditions) == 1:
            single_condition = conditions[0][0]
            return self.evaluate_simple_condition(item, single_condition)
        
        # For multiple conditions connected with AND/OR
        for cond, operator in conditions:
            if 'IN' in cond.upper():
                key, values = cond.split(' IN ')
                key = key.strip()
                values = [v.strip(' ()"\'"') for v in values[1:-1].split(',')]
                current_result = str(self.get_nested_value(item, key)) in values
            else:
                current_result = self.evaluate_simple_condition(item, cond)
            
            if last_operator == 'AND':
                result = result and current_result
            elif last_operator == 'OR':
                result = result or current_result
            else:
                result = current_result
            
            last_operator = operator
            
        return result
    

    def execute_query(self, query: str) -> Optional[QueryStats]:
        """Enhanced execute_query with better array handling and error management"""
        start_time = time.time()
        stats = QueryStats(start_time=start_time, end_time=0, rows_processed=0, rows_returned=0)
        
        try:
            # Parse query - updated regex to capture DISTINCT keyword
            match = re.match(
                r'SELECT\s+(DISTINCT\s+)?(.*?)\s+FROM\s+(\w+)((?:\s+WHERE\s+.*?)?)((?:\s+LIMIT\s+\d+)?)((?:\s+OFFSET\s+\d+)?)\s*;?$', 
                query, 
                re.IGNORECASE
            )
            
            if not match:
                raise ValueError("Invalid query format")
                
            distinct_part, select_part, table, where_part, limit_part, offset_part = match.groups()
            is_distinct = bool(distinct_part)
            
            # Check cache after parsing the query
            cached_results = self.cache.get(query)
            if cached_results is not None:
                stats.cache_hit = True
                stats.rows_returned = len(cached_results)
                stats.end_time = time.time()
                # Store the last query results and columns
                self.last_results = cached_results
                self.last_columns = select_part
                self.display_results(cached_results, select_part, stats)
                return stats
            
            # Handle LIMIT and OFFSET
            limit = int(limit_part.split()[-1]) if limit_part else None
            offset = int(offset_part.split()[-1]) if offset_part else 0
            
            # Ensure data is loaded
            self._ensure_data_loaded(table)
            
            # Get and process data
            data = self.loaded_files[table]
            if not isinstance(data, list):
                data = [data]
                
            # Apply WHERE conditions
            where_condition = where_part.replace('WHERE', '', 1).strip() if where_part else ""
            results = []
            for item in data:
                if where_condition:
                    if self.evaluate_complex_condition(item, where_condition):
                        results.append(item)
                else:
                    results.append(item)
                
            stats.rows_processed = len(data)
            
            # Apply DISTINCT if needed
            if is_distinct:
                # Get the columns we're selecting
                if select_part.strip() == "*":
                    columns = list(results[0].keys()) if results else []
                else:
                    columns = [col.strip().strip('"') for col in select_part.split(',')]
                
                # Create a set of tuples containing only the selected columns' values
                unique_results = {
                    tuple(str(self.get_nested_value(row, col)).strip() for col in columns)
                    for row in results
                }
                
                # Convert back to list of dicts
                results = [
                    dict(zip(columns, row_tuple))
                    for row_tuple in unique_results
                ]
            
            # Apply OFFSET and LIMIT
            if offset:
                results = results[offset:]
            if limit is not None:
                results = results[:limit]
                
            stats.rows_returned = len(results)
            stats.end_time = time.time()
            
            # Store results and display
            self.last_results = results
            self.last_columns = select_part
            self.display_results(results, select_part, stats)
            self.cache.set(query, results)
            
            return stats
            
        except Exception as e:
            self.console.print(f"[red]Error executing query: {str(e)}[/red]")
            return None


    def display_results(self, results: List[dict], select_part: str, stats: QueryStats = None):
        """Enhanced display_results to handle array results better"""
        if not results:
            self.console.print("[yellow]No results found[/yellow]")
            return

        # Store results for potential export
        self.last_results = results

        table = Table(show_header=True, header_style="bold magenta")
        
        # Add the column header
        columns = []
        if select_part.strip() == "*":
            columns = list(results[0].keys()) if results else []
        else:
            # Split columns handling quoted names
            current = ""
            in_quotes = False
            for char in select_part:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    if current.strip():
                        columns.append(current.strip())
                    current = ""
                else:
                    current += char
            if current.strip():
                columns.append(current.strip())
            columns = [col.strip().strip('"') for col in columns]

        # Add columns to table
        for col in columns:
            table.add_column(col)

        # Process and display each row
        for result in results:
            row = []
            has_content = False
            for col in columns:
                value = self.get_nested_value(result, col)
                if isinstance(value, list):
                    # For array of simple values (strings, numbers)
                    if value and not isinstance(value[0], (dict, list)):
                        value = ", ".join(str(v) for v in value)
                    else:
                        # For array of objects or nested arrays
                        value = json.dumps(value, indent=2)
                elif isinstance(value, dict):
                    value = json.dumps(value, indent=2)
                
                str_value = str(value) if value is not None else ""
                row.append(str_value)
                if str_value:
                    has_content = True
                    
            if has_content:
                table.add_row(*row)

        self.console.print(table)


    def export_results(self, filename: str):
        try:
            if not hasattr(self, 'last_results') or not self.last_results:
                self.console.print("[yellow]No results to export. Run a query first.[/yellow]")
                return
                
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Get column names from the last query
            if self.last_columns.strip() == "*":
                fieldnames = list(self.last_results[0].keys())
            else:
                fieldnames = [col.strip().strip('"') for col in self.last_columns.split(',')]
                
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Only write the fields that were in the query
                for result in self.last_results:
                    row = {field: result.get(field, '') for field in fieldnames}
                    writer.writerow(row)
                    
            self.console.print(f"[green]Results exported to {filename}[/green]")
                
        except Exception as e:
            self.console.print(f"[red]Error exporting results: {str(e)}[/red]")

    def display_query_stats(self, stats: QueryStats):
        """Display query execution statistics"""
        execution_time = stats.end_time - stats.start_time
        
        table = Table(title="Query Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Execution Time", f"{execution_time:.4f} seconds")
        table.add_row("Rows Processed", str(stats.rows_processed))
        table.add_row("Rows Returned", str(stats.rows_returned))
        table.add_row("Cache Hit", "Yes" if stats.cache_hit else "No")
        
        self.console.print(table)

    def run_interactive(self):
        while True:
            try:
                command = self.session.prompt("json-query> ").strip()

                if not command:
                    continue

                if command.lower() == "exit":
                    break
                elif command.lower() == "clear":
                    os.system('cls' if os.name == 'nt' else 'clear')
                elif command.lower() == "help":
                    self.show_help()
                elif command.lower().startswith("load "):
                    args = shlex.split(command)
                    path = args[1]
                    lazy = "--lazy" in args
                    if os.path.isdir(path):
                        self.load_directory(path, lazy)
                    else:
                        self.load_json_file(path, lazy)
                elif command.lower() == "show tables":
                    self.show_loaded_files()
                elif command.lower() == "analyze":
                    self.display_key_analysis()
                elif command.lower().startswith("create index"):
                    _, _, table, key = command.split()
                    self.create_index(table, key)
                elif command.lower().startswith("export "):
                    _, filename = command.split(maxsplit=1)
                    self.export_results(filename)
                elif command.lower().startswith("select"):
                    stats = self.execute_query(command)
                    if stats:
                        self.display_query_stats(stats)
                else:
                    self.console.print("[red]Unknown command. Type 'help' for available commands.[/red]")

            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"[red]Error: {str(e)}[/red]")

    def show_help(self):
        help_text = """
        Available Commands:
        ------------------
        load <path> [--lazy] : Load JSON file(s) from path (file or directory)
        create index <table> <key> : Create an index for faster querying
        analyze             : Show analysis of JSON keys and their occurrences
        show tables        : Display loaded JSON files
        export <filename>  : Export last query results to CSV file
        select            : Query data (SQL-like syntax)
        clear             : Clear the screen
        help              : Show this help message
        exit              : Exit the program

        Query Syntax:
        ------------
        - Basic query: 
          SELECT * FROM table_name
        
        - Select specific columns:
          SELECT col1, col2 FROM table_name
        
        - With conditions: 
          SELECT * FROM table_name WHERE column = "value"
        
        - Complex conditions:
          SELECT * FROM table_name WHERE col1 = "value1" AND col2 = "value2"
          SELECT * FROM table_name WHERE col1 IN ("value1", "value2")
        
        - With LIMIT and OFFSET:
          SELECT * FROM table_name LIMIT 10 OFFSET 20
        
        Operators:
        ---------
        - Comparison: =, !=, >, <, >=, <=
        - Logical: AND, OR
        - Pattern matching: LIKE (with % as wildcard)
        - List membership: IN
        
        Notes:
        ------
        - All queries are case-sensitive
        - Keys with spaces must be wrapped in double quotes
        - String values in conditions must be wrapped in double quotes
        - Use --lazy flag with load to enable lazy loading
        - Create indexes on frequently queried columns for better performance
        """
        self.console.print(help_text)

    def show_loaded_files(self):
        table = Table(show_header=True, header_style="bold magenta", title="Loaded JSON Files")
        table.add_column("File Name", style="cyan")
        table.add_column("Entry Count", style="green")
        table.add_column("Indexed Fields", style="yellow")
        table.add_column("Status", style="blue")

        for filename, data in self.loaded_files.items():
            if isinstance(data, str):
                status = "Lazy (Not Loaded)"
                count = "N/A"
            else:
                status = "Loaded"
                count = str(len(data) if isinstance(data, list) else 1)
            
            indexed_fields = ", ".join(self.indexes.get(filename, {}).keys())
            
            table.add_row(filename, count, indexed_fields or "None", status)

        self.console.print(table)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive JSON Parser and Analyzer')
    parser.add_argument('files', nargs='*', help='JSON files or directories to load')
    parser.add_argument('-a', '--analyze', action='store_true', help='Show key analysis and exit')
    parser.add_argument('-q', '--query', help='Execute SQL-like query and exit')
    parser.add_argument('--lazy', action='store_true', help='Enable lazy loading')
    parser.add_argument('--no-interactive', action='store_true', help='Do not start interactive mode')
    
    args = parser.parse_args()
    
    analyzer = JsonAnalyzer()
    
    # Load files if provided
    for path in args.files:
        if os.path.isdir(path):
            analyzer.load_directory(path, args.lazy)
        else:
            analyzer.load_json_file(path, args.lazy)
    
    # Handle non-interactive commands if specified
    if args.analyze:
        analyzer.display_key_analysis()
        if not args.no_interactive:
            analyzer.run_interactive()
        return
    
    if args.query:
        stats = analyzer.execute_query(args.query)
        if stats and not args.no_interactive:
            analyzer.display_query_stats(stats)
            analyzer.run_interactive()
        return
    
    # By default, always run interactive mode
    analyzer.run_interactive()

if __name__ == "__main__":
    main()
