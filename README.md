# json-searcher
A tool that allows you to query JSON files using SQL-like syntax, with support for nested JSON structures and arrays.

## Features
- SQL-like query syntax
- WHERE conditions with AND/OR operators
- LIKE operator with % wildcard support
- Case-insensitive search
- Unicode character support
- Query result caching
- CSV export
- Advanced nested JSON structure support
  - Dot notation for nested objects
  - Array indexing
  - Array property access
  - Nested array handling
- DISTINCT keyword for unique results

## Installation
1. Clone the repository:
```bash
git clone https://github.com/setpasslock/json-searcher.git
cd json-searcher
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

### Query Examples for Nested JSON
#### Basic Nested Object Access
```sql
-- Access nested object properties using dot notation
SELECT user.profile.name FROM users
SELECT user.address.city FROM users

-- Multiple nested fields
SELECT user.profile.name, user.address.city FROM users
```

#### Array Access
```sql
-- Access specific array element
SELECT items[0].name FROM inventory
SELECT users.friends[1].email FROM users

-- Access all array elements' properties
SELECT users.friends.name FROM users
```

#### Filtering with Nested Fields
```sql
-- Filter using nested object properties
SELECT * FROM users WHERE user.profile.age > 25
SELECT * FROM orders WHERE order.items.price > 100

-- Filter using array properties
SELECT * FROM users WHERE user.friends.name = "John"
```

#### Complex Nested Structures
```sql
-- Example JSON structure:
{
  "user": {
    "profile": {
      "name": "John",
      "contacts": [
        {"type": "email", "value": "john@example.com"},
        {"type": "phone", "value": "123-456-7890"}
      ]
    },
    "orders": [
      {
        "id": 1,
        "items": [
          {"name": "Book", "price": 29.99},
          {"name": "Pen", "price": 4.99}
        ]
      }
    ]
  }
}

-- Query examples for above structure:
SELECT user.profile.contacts[0].value FROM users
SELECT user.orders[0].items.name FROM users
SELECT * FROM users WHERE user.profile.contacts.type = "email"
```

#### Working with Arrays
```sql
-- Get all item names from all orders
SELECT user.orders.items.name FROM users

-- Filter based on array contents
SELECT * FROM users WHERE user.orders.items.price > 20

-- Access nested array elements
SELECT user.orders[0].items[1].name FROM users
```

### Supported Operators
- Comparison: `=`, `!=`, `>`, `<`, `>=`, `<=`
- Logical: `AND`, `OR`
- Text search: `LIKE` (wildcard: `%`)
- List search: `IN`

## Complete Example with Sample Data
```sql
-- Sample JSON data:
{
  "store": {
    "departments": [
      {
        "name": "Electronics",
        "products": [
          {"id": 1, "name": "Laptop", "price": 999.99},
          {"id": 2, "name": "Phone", "price": 499.99}
        ]
      },
      {
        "name": "Books",
        "products": [
          {"id": 3, "name": "Python Guide", "price": 29.99},
          {"id": 4, "name": "SQL Manual", "price": 39.99}
        ]
      }
    ]
  }
}

-- Query Examples:
-- Get all department names
SELECT store.departments.name FROM inventory

-- Get products from specific department
SELECT store.departments[0].products.name FROM inventory

-- Find expensive products across all departments
SELECT store.departments.products.name 
FROM inventory 
WHERE store.departments.products.price > 100

-- Search for specific product names
SELECT * FROM inventory 
WHERE store.departments.products.name LIKE "%Python%"
```

## Notes
- Written in Python 3
- Supports Unicode characters
- Lazy loading option available for large files
- Query results are cached for 5 minutes
- Case-insensitive column name lookup
- Column names with spaces should be wrapped in quotes
- Dot notation used for accessing nested properties
- Array indices start at 0
- Can access properties across all array elements

## Contributing
Feel free to open issues for bug reports or suggestions. We especially welcome examples of complex nested JSON queries that could be added to the documentation.
