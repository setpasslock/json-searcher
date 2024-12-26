# json-searcher
A tool that allows you to query JSON files using SQL-like syntax.

## Features

- SQL-like query syntax
- WHERE conditions with AND/OR operators
- LIKE operator with % wildcard support
- Case-insensitive search
- Unicode character support
- Query result caching
- CSV export
- Nested JSON structure support
- DISTINCT keyword for unique results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/json-query-tool.git
cd json-query-tool
```

2. Install dependencies:
```bash
pip install prompt_toolkit rich pygments
```

## Usage

### Basic Usage

```bash
python json_parser.py data.json
```

### Command Line Options

```bash
python json_parser.py [--lazy] [--analyze] [--query "SQL QUERY"] [--no-interactive] file1.json [file2.json ...]
```

- `--lazy`: Enable lazy loading for large files
- `--analyze`: Analyze JSON structure
- `--query`: Run a single query
- `--no-interactive`: Don't enter interactive mode

### Commands

- `load <path> [--lazy]`: Load JSON file(s)
- `show tables`: Display loaded JSON files
- `analyze`: Analyze JSON keys
- `create index <table> <key>`: Create an index for faster queries
- `export <filename>`: Export last query results to CSV
- `help`: Show help message
- `exit`: Exit program

### Query Examples

```sql
-- Basic query
SELECT * FROM table_name

-- Select specific columns
SELECT col1, col2 FROM table_name

-- With conditions
SELECT * FROM table_name WHERE column = "value"

-- Multiple conditions
SELECT * FROM table_name WHERE col1 = "value1" AND col2 = "value2"

-- Text search
SELECT * FROM table_name WHERE column LIKE "%search%"

-- Unique results
SELECT DISTINCT column FROM table_name

-- Pagination
SELECT * FROM table_name LIMIT 10 OFFSET 20
```

### Supported Operators

- Comparison: `=`, `!=`, `>`, `<`, `>=`, `<=`
- Logical: `AND`, `OR`
- Text search: `LIKE` (wildcard: `%`)
- List search: `IN`

## Usage Examples

```sql
-- Load JSON file
load data.json

-- Analyze structure
analyze

-- Create index
create index users id

-- Query with conditions
SELECT name, age FROM users WHERE age > 25 AND city LIKE "%York%"

-- Export results
export results.csv
```

## Notes

- Written in Python 3
- Supports Unicode characters
- Lazy loading option available for large files
- Query results are cached for 5 minutes
- Case-insensitive column name lookup
- Column names with spaces should be wrapped in quotes

## Contributing

Feel free to open issues for bug reports or suggestions.
