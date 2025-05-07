# dummy_data_generator
This generator poroduces dummy data based on DDL SQL scripts provided by the user. 
# SQL DDL Dummy Data Generator

A Python tool for generating realistic dummy data based on SQL DDL (CREATE TABLE) statements.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-line Options](#command-line-options)
  - [Environment Variables](#environment-variables)
- [Output](#output)
- [Examples](#examples)
- [Testing](#testing)
- [Supported SQL Dialects](#supported-sql-dialects)
- [Data Generation Details](#data-generation-details)
- [Known Limitations](#known-limitations)

## Features

- Parses SQL DDL files to extract table structures
- Generates intelligent data based on column names and data types
- Maintains referential integrity for foreign key relationships
- Supports multiple SQL dialects (Standard SQL, BigQuery, Snowflake, Redshift)
- Outputs data in both CSV and SQL INSERT formats
- Handles schema-qualified table names
- Configurable NULL value generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dummy_data_generator.git
cd dummy_data_generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python dummy_data_generator.py --ddl path/to/schema.sql --rows 100 --outdir ./output
```

### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ddl` | Path to SQL DDL file | Required |
| `--rows` | Number of rows to generate per table | 10 |
| `--outdir` | Directory to write output files | Current directory |
| `--allow-nulls` | Allow NULL values in nullable fields | True |
| `--no-allow-nulls` | Do not allow NULL values in any fields | - |

### Environment Variables

Create a `.env` file in the project directory:

```
DDL_PATH=path/to/schema.sql
NUM_ROWS=100
OUT_DIR=./output
ALLOW_NULLS=TRUE
```

## Output

The generator creates two output directories:

- `<outdir>/csv/`: CSV files for each table
- `<outdir>/sql/`: SQL INSERT statements for each table

Output directories are automatically created if they don't exist.

## Examples

### Standard SQL Schema

```bash
python dummy_data_generator.py --ddl sample_schema.sql --rows 50 --outdir ./output
```

### BigQuery Schema

```bash
python dummy_data_generator.py --ddl sample_bigquery_schema.sql --rows 25 --outdir ./output
```

### Preventing NULL Values

```bash
python dummy_data_generator.py --ddl sample_schema.sql --rows 100 --no-allow-nulls --outdir ./output
```

## Testing

The repository includes a pytest-based test script to validate generated data:

```bash
python test_dummy_data_generator.py --ddl sample_schema.sql --outdir ./output
```

Tests validate:
- All tables from DDL have corresponding output files
- Column structures match between DDL and output
- SQL files reference correct table names
- NULL value compliance
- Referential integrity

## Supported SQL Dialects

- **Standard SQL** (PostgreSQL, MySQL, SQLite)
- **BigQuery** (project.dataset.table naming, STRING, INT64 types)
- **Snowflake** (schema-qualified naming)
- **Redshift**

### Supported Data Types

- **Numeric**: INT, BIGINT, SMALLINT, DECIMAL, FLOAT/REAL, INT64, NUMERIC
- **String**: VARCHAR, CHAR, TEXT, STRING
- **Date/Time**: DATE, TIMESTAMP, TIMESTAMP_TZ
- **Boolean**: BOOLEAN, BOOL

## Data Generation Details

The tool generates intelligent data based on column names:

- **ID fields**: Appropriate unique identifiers (automatically tracked for FK relationships)
- **Name fields**: Realistic first/last names
- **Email addresses**: Valid-format email addresses
- **Phone numbers**: Formatted phone numbers
- **Addresses**: Street addresses, cities, states, postal codes
- **Dates**: Realistic dates in appropriate ranges

Foreign key relationships are maintained by reusing values across tables.

## Known Limitations

- Complex composite primary/foreign keys may not be fully supported
- Custom data types require mapping to standard types
- Very complex DDL parsing may miss some column definitions
- Default values in DDL are not used in data generation