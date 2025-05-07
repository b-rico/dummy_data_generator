#!/usr/bin/env python3
"""
SQL DDL Dummy Data Generator

This script generates realistic dummy data based on SQL DDL (CREATE TABLE statements).
It parses the DDL file, extracts table structures, and generates appropriate random data
for each column based on its data type.

Dependencies:
    - sqlparse: For parsing SQL DDL files
    - faker: For generating realistic data
    - pandas: For data handling and CSV export
    - python-dotenv: For loading environment variables

Installation:
    pip install sqlparse faker pandas python-dotenv

Usage:
    python dummy_data_generator.py --ddl path/to/ddl.sql --rows 100 --format sql --outdir ./output

Arguments:
    --ddl:     Path to the SQL DDL file containing CREATE TABLE statements
    --rows:    Number of rows to generate per table (default: 10)
    --format:  Output format: 'sql' or 'csv' (default: 'sql')
    --outdir:  Directory to write output files (default: current directory)

Environment Variables (.env file):
    DDL_PATH:  Path to the SQL DDL file
    NUM_ROWS:  Number of rows to generate per table
    FORMAT:    Output format ('sql' or 'csv')
    OUT_DIR:   Directory to write output files
"""

import os
import re
import random
import argparse
import datetime
import sqlparse
import pandas as pd
import uuid
from faker import Faker
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import logging
import sys
import traceback

# Initialize Faker
fake = Faker()

# Global ID tracking for relational integrity
entity_id_mapping = {}  # Maps entity type and name to its UUID
generated_id_values = {}  # Maps table.column to a list of generated IDs

def generate_or_reuse_uuid(entity_type, entity_name=None):
    """Generate a new UUID or reuse an existing one for a given entity."""
    if entity_name:
        key = f"{entity_type}:{entity_name}"
        if key not in entity_id_mapping:
            entity_id_mapping[key] = str(uuid.uuid4())
        return entity_id_mapping[key]
    return str(uuid.uuid4())

def track_generated_id(table_name, column_name, value):
    """Track generated ID values for reuse across tables."""
    key = f"{table_name}.{column_name}"
    if key not in generated_id_values:
        generated_id_values[key] = []
    generated_id_values[key].append(value)
    return value

def get_existing_id(table_name, column_name):
    """Get a previously generated ID if available."""
    for source_key in generated_id_values:
        # If it's the same column name (even in a different table)
        if source_key.endswith(f".{column_name}") and generated_id_values[source_key]:
            return random.choice(generated_id_values[source_key])
    # If no existing ID found, return None
    return None

def parse_arguments():
    """Parse command line arguments or load from .env file."""
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Generate dummy data from SQL DDL files.')
    parser.add_argument('--ddl', help='Path to the SQL DDL file', default=os.getenv('DDL_PATH'))
    parser.add_argument('--rows', type=int, help='Number of rows to generate per table', 
                        default=int(os.getenv('NUM_ROWS', 10)))
    parser.add_argument('--outdir', help='Directory to write output files', 
                        default=os.getenv('OUT_DIR', '.'))
    
    # Add mutually exclusive group for allow-nulls
    null_group = parser.add_mutually_exclusive_group()
    null_group.add_argument('--allow-nulls', dest='allow_nulls', action='store_true',
                        help='Allow NULL values in nullable fields')
    null_group.add_argument('--no-allow-nulls', dest='allow_nulls', action='store_false',
                        help='Do not allow NULL values in any fields')
    
    # Set the default from environment variable
    parser.set_defaults(allow_nulls=os.getenv('ALLOW_NULLS', 'TRUE').upper() == 'TRUE')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.ddl:
        parser.error("No DDL file specified. Use --ddl argument or set DDL_PATH in .env file.")
    
    # Check if DDL path is valid
    if not os.path.exists(args.ddl):
        parser.error(f"The specified DDL file does not exist: {args.ddl}")
    
    if os.path.isdir(args.ddl):
        parser.error(f"The specified DDL path is a directory, not a file: {args.ddl}")
    
    # Ensure outdir exists
    os.makedirs(args.outdir, exist_ok=True)
    # Create subdirectories for CSV and SQL output
    os.makedirs(os.path.join(args.outdir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "sql"), exist_ok=True)
    
    return args

def parse_ddl(ddl_path):
    """
    Parse the SQL DDL file and extract table schemas.
    
    Returns a list of dictionaries, each containing:
    - table_name: Name of the table (simple name)
    - full_name: Fully qualified name (project.dataset.table)
    - columns: List of column definitions (name, data_type, size, precision, scale, nullable)
    - foreign_keys: List of foreign key references (column_name, ref_table, ref_column)
    - primary_key: Primary key column name(s)
    """
    try:
        with open(ddl_path, 'r') as f:
            sql = f.read()
        
        if not sql.strip():
            print(f"Warning: The DDL file '{ddl_path}' is empty.")
            return []
        
        # Print the first 200 characters for debugging
        print(f"Parsing SQL file: {ddl_path}")
        print(f"First part of SQL content: {sql[:min(200, len(sql))]}...")
        
        # Split the SQL into statements
        statements = sqlparse.parse(sql)
        print(f"Found {len(statements)} SQL statements")
        
        tables = []
        
        # Special handling for jiva_members table specifically
        jiva_members_pattern = re.compile(r'create\s+table\s+([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\.jiva_members\s*\(\s*(.*?)\s*\)\s*using\s+delta', re.DOTALL | re.IGNORECASE)
        jiva_members_match = jiva_members_pattern.search(sql)
        
        if jiva_members_match:
            print("Found jiva_members table definition - using specialized parsing")
            full_name = jiva_members_match.group(1) + ".jiva_members"
            jiva_members_columns_text = jiva_members_match.group(2)
            
            # Extract column definitions line by line
            jiva_members_columns = []
            for line in jiva_members_columns_text.split(','):
                line = line.strip()
                if not line:
                    continue
                    
                # Skip constraints, keys, etc.
                if re.search(r'^\s*(?:CONSTRAINT|PRIMARY\s+KEY|FOREIGN\s+KEY|KEY|INDEX|UNIQUE)', line, re.IGNORECASE):
                    continue
                
                # Try to extract column name and type
                col_match = re.match(r'\s*([`"\[]?)([a-zA-Z0-9_]+)([`"\]]?)\s+(.*?)(?:\s+COMMENT\s+\'.*\')?$', line, re.IGNORECASE)
                if col_match:
                    quote1, col_name, quote2, data_type_part = col_match.groups()
                    
                    # For BigQuery/Snowflake, clean up the data type
                    data_type_part = data_type_part.strip()
                    data_type = re.match(r'([a-zA-Z0-9_]+)(?:\(([^)]+)\))?', data_type_part)
                    
                    if data_type:
                        data_type_name = data_type.group(1).upper()
                        size_str = data_type.group(2) if data_type.group(2) else None
                        
                        # Map non-standard types to SQL standard types
                        if data_type_name == 'STRING':
                            data_type_name = 'VARCHAR'
                            print(f"Mapped non-standard data type STRING to VARCHAR")
                        
                        # Process size/precision/scale
                        size = None
                        precision = None
                        scale = None
                        
                        if size_str:
                            if ',' in size_str:
                                # Handle decimal precision and scale
                                prec_scale = size_str.split(',')
                                precision = int(prec_scale[0].strip())
                                scale = int(prec_scale[1].strip())
                            else:
                                # Regular size parameter
                                size = int(size_str.strip())
                        
                        # Determine if nullable
                        nullable = not re.search(r'\bNOT\s+NULL\b', line, re.IGNORECASE)
                        
                        # Add column to the list
                        jiva_members_columns.append({
                            'name': col_name,
                            'data_type': data_type_name,
                            'size': size,
                            'precision': precision,
                            'scale': scale,
                            'nullable': nullable
                        })
                        print(f"Found column: {col_name} ({data_type_name})")
            
            # Add the table definition
            tables.append({
                'table_name': 'jiva_members',
                'full_name': full_name,
                'columns': jiva_members_columns,
                'foreign_keys': [],
                'primary_key': None
            })
            # Note: We don't return here anymore, we continue processing other tables
        
        statement_count = 0
        for statement in statements:
            statement_str = str(statement).strip()
            statement_type = None
            
            # Check if this is a CREATE TABLE statement
            if re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE', statement_str, re.IGNORECASE):
                statement_type = "CREATE"
            else:
                # Skip non-CREATE statements
                continue
            
            statement_count += 1
            print(f"Statement {statement_count} type: {statement_type}")
            
            # Try to extract the table name
            # For BigQuery: project.dataset.table format
            full_name = None
            simple_name = None
            
            # First, try to match the comment at the start which has the full table name 
            # Pattern like "-- 1) edp_dev.healthplanoptimasensitive.odw_member_kaisereligibility"
            comment_match = re.search(r'--\s*\d+\)\s+([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)', statement_str, re.IGNORECASE)
            if comment_match:
                full_name = comment_match.group(1)
                simple_name = full_name.split('.')[-1]
                print(f"Extracted qualified name from comment: {full_name}")
                print(f"Simple table name: {simple_name}")
            else:
                # Try standard format: CREATE TABLE project.dataset.table
                bigquery_match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)', statement_str, re.IGNORECASE)
                
                if bigquery_match:
                    full_name = bigquery_match.group(1)
                    simple_name = full_name.split('.')[-1]  # Get the table part
                    print(f"BigQuery match - Project: {full_name.split('.')[0]}, Dataset: {full_name.split('.')[1]},")
                    print(f"Table: {simple_name}")
                else:
                    # For standard SQL: just the table name
                    table_match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+(?:`|\[|")?([^(`\[\s".]+)(?:`|\]|")?', statement_str, re.IGNORECASE)
                    
                    if table_match:
                        simple_name = table_match.group(1)
                        full_name = simple_name  # For non-qualified tables, full_name = simple_name
                        print(f"Standard match - Table: {simple_name}")
                    else:
                        # If we can't extract the table name, skip this statement
                        print(f"Warning: Could not extract table name from statement: {statement_str[:100]}...")
                        continue
            
            # Process the jiva_members table with the special parser
            if simple_name and simple_name.lower() == 'jiva_members' and any(t['table_name'].lower() == 'jiva_members' for t in tables):
                # Skip if already processed by the specialized parser
                continue
            
            # Extract column definitions
            columns = []
            foreign_keys = []
            primary_key = []
            
            # Look for the parenthesized content
            paren_match = re.search(r'\(\s*(.*?)\s*\)(?:\s*using\s+\w+\s*)?(?:\s*(?:with|partitioned\s+by|clustered\s+by|;)\s*)', statement_str, re.DOTALL | re.IGNORECASE)
            
            if paren_match:
                columns_text = paren_match.group(1)
                print(f"Processing table: {simple_name}")
                
                # Split columns and constraints in the CREATE TABLE statement
                # First parse by commas while respecting parentheses
                parts = []
                current_part = ""
                paren_level = 0
                
                for char in columns_text:
                    if char == '(':
                        paren_level += 1
                    elif char == ')':
                        paren_level -= 1
                    
                    if char == ',' and paren_level == 0:
                        parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                
                if current_part.strip():
                    parts.append(current_part.strip())
                
                print(f"Found {len(parts)} column/constraint definitions.")
                
                for part in parts:
                    part = part.strip()
                    print(f"Processing part: {part[:30]}{'...' if len(part) > 30 else ''}")
                    
                    # 1. Check if it's a PRIMARY KEY constraint
                    primary_key_match = re.search(r'^\s*PRIMARY\s+KEY\s*\(\s*([^)]+)\s*\)', part, re.IGNORECASE)
                    if primary_key_match:
                        pk_columns = [col.strip(' `"[]') for col in primary_key_match.group(1).split(',')]
                        primary_key = pk_columns
                        print(f"Found PRIMARY KEY constraint: {pk_columns}")
                        continue
                    
                    # 2. Check if it's a FOREIGN KEY constraint
                    foreign_key_match = re.search(r'^\s*FOREIGN\s+KEY\s*\(\s*([^)]+)\s*\)(?:\s+REFERENCES\s+([a-zA-Z0-9_\.]+)(?:\s*\(\s*([^)]+)\s*\))?)?', part, re.IGNORECASE)
                    if foreign_key_match:
                        fk_columns = [col.strip(' `"[]') for col in foreign_key_match.group(1).split(',')]
                        ref_table = foreign_key_match.group(2).strip(' `"[]') if foreign_key_match.group(2) else None
                        ref_columns = [col.strip(' `"[]') for col in foreign_key_match.group(3).split(',')] if foreign_key_match.group(3) else None
                        
                        # If no reference columns are specified, use the same column names
                        if not ref_columns:
                            ref_columns = fk_columns
                        
                        # If no reference table is specified, try to extract it from a later line
                        if not ref_table and 'REFERENCES' in part:
                            ref_table_match = re.search(r'REFERENCES\s+([a-zA-Z0-9_\.]+)', part, re.IGNORECASE)
                            if ref_table_match:
                                ref_table = ref_table_match.group(1).strip(' `"[]')
                        
                        if ref_table:
                            # Extract simple table name from qualified reference
                            ref_table_simple = ref_table.split('.')[-1] if '.' in ref_table else ref_table
                            
                            # Add foreign key constraint for each column
                            for i, fk_col in enumerate(fk_columns):
                                ref_col = ref_columns[i] if i < len(ref_columns) else ref_columns[-1]
                                foreign_keys.append({
                                    'column_name': fk_col,
                                    'ref_table': ref_table_simple,
                                    'ref_column': ref_col
                                })
                                print(f"Found FOREIGN KEY: {fk_col} -> {ref_table_simple}.{ref_col}")
                        else:
                            print(f"Warning: FOREIGN KEY found but no reference table specified: {part}")
                        continue
                    
                    # Skip other constraint definitions
                    if re.search(r'^\s*(?:CONSTRAINT|UNIQUE|KEY|INDEX|CHECK|ENFORCED|NOT\s+ENFORCED)', part, re.IGNORECASE):
                        continue
                    
                    # 3. Parse column definition - Handle names with underscores
                    # Convert escaped column names (like client\_name) to their actual names (client_name)
                    # This is needed because the DDL may use backslashes to escape underscores
                    unescaped_part = part.replace("\\_", "_")
                    
                    # Look for column definitions - Special case for BigQuery with underscores
                    col_match = None
                    col_name = None
                    data_type_part = None
                    
                    # Try several patterns to capture columns with underscores
                    
                    # Pattern 1: Simplified column pattern for underscored names or escaped names like client\_name
                    column_pattern1 = r'^([a-zA-Z0-9_]+)(\s+|\s*)(STRING|INT|TIMESTAMP|DATE|DECIMAL\(\d+(?:,\d+)?\)|[A-Z0-9_]+(?:\(\d+(?:,\d+)?\))?)(.*)$'
                    match1 = re.match(column_pattern1, unescaped_part, re.IGNORECASE)
                    if match1:
                        col_name = match1.group(1).strip()
                        data_type_part = match1.group(3).strip()
                        col_match = True
                    
                    # Pattern 2: Escaped identifiers
                    if not col_match:
                        column_pattern2 = r'^(?:`|\[|")?([a-zA-Z0-9_]+)(?:`|\]|")?\s+(.*?)$'
                        match2 = re.match(column_pattern2, unescaped_part, re.IGNORECASE)
                        if match2:
                            col_name = match2.group(1).strip()
                            data_type_part = match2.group(2).strip()
                            col_match = True
                    
                    # If we found a column, process it
                    if col_match and col_name and data_type_part:
                        # Extract data type and size/precision
                        data_type = re.match(r'([a-zA-Z0-9_]+)(?:\(([^)]+)\))?', data_type_part)
                        
                        if data_type:
                            data_type_name = data_type.group(1).upper()
                            size_str = data_type.group(2) if data_type.group(2) else None
                            
                            # Map non-standard types to SQL standard types
                            if data_type_name == 'STRING':
                                data_type_name = 'VARCHAR'
                                print(f"Mapped non-standard data type STRING to VARCHAR")
                            
                            # Process size/precision/scale
                            size = None
                            precision = None
                            scale = None
                            
                            if size_str:
                                if ',' in size_str:
                                    # Handle decimal precision and scale
                                    prec_scale = size_str.split(',')
                                    precision = int(prec_scale[0].strip())
                                    scale = int(prec_scale[1].strip())
                                else:
                                    # Regular size parameter
                                    try:
                                        size = int(size_str.strip())
                                    except ValueError:
                                        # Handle cases where size isn't a simple number
                                        print(f"Warning: Could not parse size: {size_str}")
                                        size = 255  # Default size
                            
                            # Check for incomplete decimal definition
                            if data_type_name == 'DECIMAL' and precision is not None and scale is None:
                                print(f"Found incomplete decimal column: {col_name} (precision: {precision})")
                                scale = 4  # Default scale
                                print(f"Completed decimal column: {col_name} (DECIMAL({precision},{scale}))")
                            
                            # Determine if nullable
                            nullable = not re.search(r'\bNOT\s+NULL\b', part, re.IGNORECASE)
                            
                            # Check if this column has an inline PRIMARY KEY constraint
                            if re.search(r'\bPRIMARY\s+KEY\b', part, re.IGNORECASE):
                                primary_key.append(col_name)
                                print(f"Found inline PRIMARY KEY: {col_name}")
                            
                            # Check for inline REFERENCES (foreign key)
                            fk_ref_match = re.search(r'\bREFERENCES\s+([a-zA-Z0-9_\.]+)(?:\s*\(\s*([^)]+)\s*\))?', part, re.IGNORECASE)
                            if fk_ref_match:
                                ref_table = fk_ref_match.group(1).strip(' `"[]')
                                ref_column = fk_ref_match.group(2).strip(' `"[]') if fk_ref_match.group(2) else col_name
                                
                                # Extract simple table name from qualified reference
                                ref_table_simple = ref_table.split('.')[-1] if '.' in ref_table else ref_table
                                
                                foreign_keys.append({
                                    'column_name': col_name,
                                    'ref_table': ref_table_simple,
                                    'ref_column': ref_column
                                })
                                print(f"Found inline FOREIGN KEY: {col_name} -> {ref_table_simple}.{ref_column}")
                            
                            # Add column to the list
                            columns.append({
                                'name': col_name,
                                'data_type': data_type_name,
                                'size': size,
                                'precision': precision,
                                'scale': scale,
                                'nullable': nullable
                            })
                            print(f"Found column: {col_name} ({data_type_name})")
                        else:
                            print(f"Warning: Could not parse data type for column: {part}")
                    else:
                        print(f"Warning: Could not parse column definition: {part}")
            else:
                print(f"Warning: Could not extract column definitions for table {simple_name}")
            
            # Add the table definition with the columns we were able to extract
            if full_name and simple_name:
                if not any(t['table_name'].lower() == simple_name.lower() and t['full_name'].lower() == full_name.lower() for t in tables):
                    tables.append({
                        'table_name': simple_name,
                        'full_name': full_name,
                        'columns': columns,
                        'foreign_keys': foreign_keys,
                        'primary_key': primary_key
                    })
        
        if not tables:
            print("Error: No tables found in DDL file.")
        else:
            print(f"Successfully parsed {len(tables)} tables from DDL file.")
            
        return tables
    
    except Exception as e:
        print(f"Error parsing DDL file: {e}")
        traceback.print_exc()
        return []

def parse_column_definition(column_def):
    """Parse a single column definition and extract its properties."""
    # Fix for incomplete decimal types (e.g., when "decimal(38" is split from "0)")
    if column_def.strip().isdigit():
        # This is likely a scale value from a split decimal type
        # We'll return None and handle the merge elsewhere
        return None
        
    # Handle cases where a decimal type definition is incomplete (missing scale)
    if 'decimal(' in column_def.lower() and not column_def.lower().endswith(')'):
        # This is likely an incomplete decimal type, we'll try to infer scale 0
        decimal_match = re.search(r'(\w+)\s+decimal\((\d+)', column_def, re.IGNORECASE)
        if decimal_match:
            name = decimal_match.group(1).strip()
            precision = int(decimal_match.group(2))
            scale = 0  # Default scale to 0 if not specified
            nullable = 'NOT NULL' not in column_def.upper()
            
            return {
                'name': name,
                'data_type': 'DECIMAL',
                'size': None,
                'precision': precision,
                'scale': scale,
                'nullable': nullable,
                'rest': column_def  # Keep the rest for additional parsing if needed
            }
    
    # Special case for decimal types
    decimal_match = re.search(r'(\w+)\s+decimal\((\d+),(\d+)\)', column_def, re.IGNORECASE)
    if decimal_match:
        name = decimal_match.group(1).strip()
        precision = int(decimal_match.group(2))
        scale = int(decimal_match.group(3))
        nullable = 'NOT NULL' not in column_def.upper()
        
        return {
            'name': name,
            'data_type': 'DECIMAL',
            'size': None,
            'precision': precision,
            'scale': scale,
            'nullable': nullable,
            'rest': column_def  # Keep the rest for additional parsing if needed
        }
        
    # Basic pattern to match column name and data type
    match = re.match(r'[\s]*([`"\[]?)([\w\d_]+)([`"\]]?)[\s]+([\w\d]+)(?:\((.*?)\))?[\s]*(.*)', column_def, re.IGNORECASE)
    
    if not match:
        print(f"Warning: Could not parse column definition: {column_def}")
        return None
    
    quote1, name, quote2, data_type, size_part, rest = match.groups()
    name = name.strip()
    data_type = data_type.upper()
    
    # Parse size/precision/scale
    size = None
    precision = None
    scale = None
    
    if size_part:
        if ',' in size_part:
            # Handle DECIMAL(10,2) format
            parts = size_part.split(',')
            precision = int(parts[0].strip())
            scale = int(parts[1].strip()) if parts[1].strip().isdigit() else 0
        else:
            # Handle VARCHAR(255) format
            try:
                size = int(size_part.strip())
            except ValueError:
                # Handle non-numeric size parameters (e.g., Snowflake VARCHAR(MAX))
                size = 4000  # Default to a large size
    
    # Check nullable
    nullable = 'NOT NULL' not in rest.upper()
    
    # Map non-standard data types to standard ones
    data_type_map = {
        # BigQuery types
        'STRING': 'VARCHAR',
        'FLOAT64': 'FLOAT',
        'INT64': 'INT',
        'BOOL': 'BOOLEAN',
        'TIMESTAMP': 'TIMESTAMP',
        'DATE': 'DATE',
        
        # Snowflake types
        'TEXT': 'VARCHAR',
        'NUMBER': 'DECIMAL',
        'FLOAT': 'FLOAT',
        'TIMESTAMP_TZ': 'TIMESTAMP',
        'TIMESTAMP_NTZ': 'TIMESTAMP',
        'TIMESTAMP_LTZ': 'TIMESTAMP',
        'VARIANT': 'VARCHAR',
        'OBJECT': 'VARCHAR',
        'ARRAY': 'VARCHAR',
        
        # Redshift types
        'SMALLINT': 'INT',
        'BIGINT': 'INT',
        'NUMERIC': 'DECIMAL'
    }
    
    # Use the mapped standard type if available
    standard_type = data_type_map.get(data_type, data_type)
    
    # For debugging
    if data_type != standard_type:
        print(f"Mapped non-standard data type {data_type} to {standard_type}")
    
    return {
        'name': name,
        'data_type': standard_type,
        'size': size,
        'precision': precision,
        'scale': scale,
        'nullable': nullable,
        'rest': rest  # Keep the rest for additional parsing if needed
    }

def build_dependency_graph(tables):
    """
    Build a dependency graph of tables based on foreign key relationships.
    Returns:
    - graph: Dictionary mapping table full names to sets of tables they depend on
    - ordered_tables: List of full table names in order of generation (parents first)
    """
    # Build dependency graph
    graph = defaultdict(set)
    
    # Create a map of simple table names to their full names
    table_name_to_full_name = {}
    for table in tables:
        table_name = table['table_name']
        full_name = table.get('full_name', table_name)
        if table_name not in table_name_to_full_name:
            table_name_to_full_name[table_name] = []
        table_name_to_full_name[table_name].append(full_name)
    
    # Identify dependencies based on foreign keys
    for table in tables:
        table_name = table['table_name']
        full_name = table.get('full_name', table_name)
        
        for fk in table['foreign_keys']:
            ref_table_name = fk['ref_table']
            
            # Find matching full names for the referenced table
            if ref_table_name in table_name_to_full_name:
                for ref_full_name in table_name_to_full_name[ref_table_name]:
                    # This table depends on the referenced table
                    graph[full_name].add(ref_full_name)
    
    # Topological sort to determine generation order
    visited = set()
    temp_visited = set()
    ordered_tables = []
    
    def visit(table_full_name):
        if table_full_name in temp_visited:
            raise ValueError(f"Circular dependency detected involving table {table_full_name}")
        if table_full_name not in visited:
            temp_visited.add(table_full_name)
            for dependency in graph.get(table_full_name, set()):
                visit(dependency)
            temp_visited.remove(table_full_name)
            visited.add(table_full_name)
            # Store the full_name in ordered_tables
            ordered_tables.append(table_full_name)
    
    # Visit each table
    for table in tables:
        full_name = table.get('full_name', table['table_name'])
        if full_name not in visited:
            visit(full_name)
    
    # Reverse to get parents first
    ordered_tables.reverse()
    
    return graph, ordered_tables

def generate_data_for_tables(tables, num_rows, allow_nulls=True):
    """
    Generate data for all tables respecting foreign key relationships.
    
    Args:
        tables: List of table definitions from parse_ddl
        num_rows: Number of rows to generate per table
        allow_nulls: Whether to allow NULL values in nullable fields
    
    Returns:
        Dictionary mapping table names to their generated data
    """
    # Build dependency graph
    graph, ordered_full_names = build_dependency_graph(tables)
    
    # Map full names to their table definitions
    full_name_to_table = {}
    for table in tables:
        full_name = table.get('full_name', table['table_name'])
        full_name_to_table[full_name] = table
    
    # Store generated data by full name
    generated_data = {}
    
    # Create a unified map to store values for all columns, not just primary keys
    # This ensures we have shared values for foreign key relationships
    column_values = {}
    
    # First pass: initialize all primary and foreign key column values 
    # to ensure consistent data across relationships
    for full_name in ordered_full_names:
        if full_name not in full_name_to_table:
            continue
        
        table = full_name_to_table[full_name]
        
        # For each column that might be referenced by a foreign key
        for column in table['columns']:
            col_name = column['name']
            col_key = f"{full_name}.{col_name}"
            
            # Only pre-generate values for columns that might be referenced
            # by foreign keys (usually IDs, codes, etc.)
            if ('id' in col_name.lower() or 'code' in col_name.lower() or 
                'source' in col_name.lower() or 'key' in col_name.lower()):
                column_values[col_key] = []
                
                # Generate distinct values appropriate for the column type
                seen_values = set()
                for _ in range(num_rows * 2):  # Extra values for variety
                    value = generate_value_for_column(column, table['table_name'], {}, 0, False)
                    if value not in seen_values:
                        seen_values.add(value)
                        column_values[col_key].append(value)
    
    # Process tables in order determined by topological sort (parents first)
    for full_name in ordered_full_names:
        if full_name not in full_name_to_table:
            continue
        
        table = full_name_to_table[full_name]
        print(f"Generating data for table: {full_name}")
        
        # Initialize data structure to hold generated values
        table_data = {
            'table_name': table['table_name'],
            'full_name': full_name,
            'columns': table['columns'],
            'data': {}
        }
        
        for column in table['columns']:
            table_data['data'][column['name']] = []
        
        # Identify foreign key columns 
        fk_columns = {}
        for fk in table.get('foreign_keys', []):
            column_name = fk['column_name']
            ref_table = fk['ref_table']
            ref_column = fk['ref_column']
            
            # Find the full name of the referenced table
            ref_full_name = None
            for t in tables:
                if t['table_name'] == ref_table:
                    ref_full_name = t.get('full_name', ref_table)
                    break
            
            if not ref_full_name:
                ref_full_name = ref_table
                
            fk_columns[column_name] = {
                'ref_table': ref_full_name,
                'ref_column': ref_column
            }
        
        # Generate data for each row
        for row_idx in range(num_rows):
            row_entities = {}
            
            # First generate values for non-foreign key columns
            for column in table['columns']:
                col_name = column['name']
                col_key = f"{full_name}.{col_name}"
                
                # Skip foreign keys for now
                if col_name in fk_columns:
                    continue
                
                # If this column has pre-generated values, use them
                if col_key in column_values and column_values[col_key]:
                    # Use one of the pre-generated values, and remove it to avoid duplicates
                    if len(column_values[col_key]) > 1:
                        value = column_values[col_key].pop(0)
                    else:
                        # Keep at least one value for reference
                        value = column_values[col_key][0]
                else:
                    # Generate a new value
                    value = generate_value_for_column(column, table['table_name'], row_entities, row_idx, allow_nulls)
                    
                    # If this is a potential reference column, store it for future foreign keys
                    if ('id' in col_name.lower() or 'code' in col_name.lower() or 
                        'source' in col_name.lower()) and value is not None:
                        if col_key not in column_values:
                            column_values[col_key] = []
                        column_values[col_key].append(value)
                
                table_data['data'][col_name].append(value)
            
            # Now generate values for foreign key columns
            for column in table['columns']:
                col_name = column['name']
                
                # Skip non-foreign key columns
                if col_name not in fk_columns:
                    continue
                
                fk_info = fk_columns[col_name]
                ref_table = fk_info['ref_table']
                ref_column = fk_info['ref_column']
                ref_col_key = f"{ref_table}.{ref_column}"
                
                # Determine if this column should allow NULLs
                nullable = column.get('nullable', True)
                
                # Use values from the referenced column if available
                if ref_col_key in column_values and column_values[ref_col_key]:
                    if not nullable or not allow_nulls or random.random() > 0.15:
                        # Use an existing value
                        value = random.choice(column_values[ref_col_key])
                    else:
                        # Allow NULL if permitted
                        value = None
                elif nullable and allow_nulls:
                    # Use NULL if allowed and no reference values available
                    value = None
                else:
                    # Generate a compatible value and add it to reference values
                    ref_column_def = None
                    if ref_table in full_name_to_table:
                        for col in full_name_to_table[ref_table]['columns']:
                            if col['name'] == ref_column:
                                ref_column_def = col
                                break
                    
                    if ref_column_def:
                        # Generate using reference column definition
                        value = generate_value_for_column(ref_column_def, ref_table, row_entities, row_idx, False)
                    else:
                        # Fallback to common formats based on column name
                        if 'memid' in ref_column.lower():
                            value = f"MEM{random.randint(10000000, 99999999)}"
                        elif 'groupid' in ref_column.lower() or 'groupbaseid' in ref_column.lower() or 'groupsubid' in ref_column.lower():
                            letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
                            digits = ''.join(random.choices('0123456789', k=6))
                            value = f"{letters}{digits}"
                        elif 'datasource' in ref_column.lower():
                            value = random.choice(['SYSTEM', 'MANUAL', 'BATCH', 'API', 'IMPORT', 'USER'])
                        elif 'lob' in ref_column.lower():
                            letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
                            digits = ''.join(random.choices('0123456789', k=4))
                            value = f"{letters}{digits}"
                        elif 'provnpi' in ref_column.lower():
                            value = ''.join(random.choices('0123456789', k=10))
                        else:
                            # Generic ID
                            value = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
                    
                    # Store for future reference
                    if ref_col_key not in column_values:
                        column_values[ref_col_key] = []
                    column_values[ref_col_key].append(value)
                
                # Add the value to our data
                table_data['data'][col_name].append(value)
        
        # Store the generated data for this table
        generated_data[full_name] = table_data
    
    return generated_data

def generate_data_for_table(table, num_rows, entities, allow_nulls=True):
    """
    Generate random data for a single table.
    
    Args:
        table: Table definition from parse_ddl
        num_rows: Number of rows to generate
        entities: Dictionary of previously generated entities to maintain referential integrity
        allow_nulls: Whether to allow NULL values in nullable fields
    
    Returns:
        Dictionary with table name, columns, and generated data
    """
    # Get column definitions from table
    table_name = table['table_name']
    full_name = table.get('full_name', table_name)
    columns = table['columns']
    foreign_keys = table.get('foreign_keys', [])
    primary_keys = table.get('primary_key', [])
    
    # Initialize data structure to hold generated values
    data = {}
    for column in columns:
        data[column['name']] = []
    
    # Create a map of column names to their foreign key info
    fk_map = {fk['column_name']: fk for fk in foreign_keys}
    
    # Generate data for each row
    for row_idx in range(num_rows):
        # Keep track of entities created in this row
        row_entities = {}
        
        # First pass: Generate values for non-foreign key columns
        for column in columns:
            col_name = column['name']
            
            # Skip foreign keys in the first pass
            if col_name in fk_map:
                continue
            
            # Generate value
            value = generate_value_for_column(column, table_name, row_entities, row_idx, allow_nulls)
            data[col_name].append(value)
            
            # Store primary key values for referential integrity
            if primary_keys and col_name in primary_keys:
                entity_type = f"{full_name}.{col_name}"
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(value)
                
                # Also store with generic type for classic lookup
                generic_type = col_name.replace('id', '').strip('_')
                if not generic_type:
                    generic_type = table_name
                if generic_type not in entities:
                    entities[generic_type] = []
                entities[generic_type].append(value)
        
        # Second pass: Generate values for foreign key columns
        for column in columns:
            col_name = column['name']
            
            # Only process foreign keys in second pass
            if col_name not in fk_map:
                continue
            
            fk = fk_map[col_name]
            ref_table = fk['ref_table']
            ref_column = fk['ref_column']
            
            # Look for qualified reference key
            entity_type = f"{ref_table}.{ref_column}"
            ref_values = entities.get(entity_type, [])
            
            # If none found, try table-only reference for backward compatibility 
            if not ref_values:
                generic_type = ref_table.split('.')[-1]  # Handle qualified names
                ref_values = entities.get(generic_type, [])
            
            # Determine if this column should allow NULLs
            nullable = column.get('nullable', False)
            
            if ref_values and (not nullable or not allow_nulls or random.random() > 0.15):
                # Pick a random value from available reference values
                value = random.choice(ref_values)
            elif nullable and allow_nulls:
                # Use NULL if allowed
                value = None
            else:
                # If no reference values available and NULL not allowed, generate a compatible value
                if 'id' in ref_column.lower():
                    if 'mem' in ref_column.lower():
                        value = f"MEM{random.randint(10000000, 99999999)}"
                    elif 'group' in ref_column.lower():
                        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
                        digits = ''.join(random.choices('0123456789', k=6))
                        value = f"{letters}{digits}"
                    else:
                        value = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
                else:
                    value = 'DEFAULT_VALUE'
                
                # Add to reference values for future use
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(value)
            
            data[col_name].append(value)
    
    return {
        'table_name': table_name,
        'full_name': full_name,
        'columns': columns,
        'data': data
    }

def generate_value_for_column(column, table_name, row_entities=None, row_idx=None, allow_nulls=True):
    """Generate a random value for a column based on its data type and name."""
    name = column['name'].lower()
    data_type = column['data_type']
    nullable = column.get('nullable', False)  # Default to not nullable if not specified
    row_entities = row_entities or {}
    
    # If nullable and nulls are allowed, sometimes return None (NULL)
    # Use a consistent approach across all data types
    if nullable and allow_nulls and random.random() < 0.15:  # 15% chance of NULL
        return None
    
    # Generate value based on column type and name
    result = None
    
    # Check if we should use a special ID format for this field
    if 'id' in name and not any(x in name for x in ['void', 'grid', 'valid', 'solid', 'fluid']):
        # Get the entity type from the column name
        entity_type = name.replace('id', '').strip('_')
        if not entity_type:
            entity_type = table_name  # Use table name if no specific type found
            
        # Check for existing related values
        if entity_type+'_first' in row_entities and entity_type+'_last' in row_entities:
            # Use consistent UUID for existing entity
            entity_name = f"{row_entities[entity_type+'_first']} {row_entities[entity_type+'_last']}"
            uuid_value = generate_or_reuse_uuid(entity_type, entity_name)
            return track_generated_id(table_name, name, uuid_value)
        
        # Generate specialized ID formats for different types of IDs
        if name in ['benefitplanid', 'groupbaseid', 'groupsubid', 'grouptierid']:
            # Plan/group IDs often have prefixes and numeric components
            prefix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
            numeric = ''.join(random.choices('0123456789', k=6))
            return f"{prefix}{numeric}"
            
        elif name in ['memid', 'memsid', 'personid', 'memberid']:
            # Member IDs often have standard formats
            prefix = 'MEM'
            numeric = ''.join(random.choices('0123456789', k=8))
            return f"{prefix}{numeric}"
            
        elif name in ['subscriberbadaddressflag', 'subscriberId', 'subscriberid']:
            # Subscriber IDs often start with 'S'
            prefix = 'S'
            numeric = ''.join(random.choices('0123456789', k=9))
            return f"{prefix}{numeric}"
            
        elif 'medicare' in name or 'medicaid' in name:
            # Medicare/Medicaid IDs have specific formats
            one_letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            nine_digits = ''.join(random.choices('0123456789', k=9))
            letter_or_number = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            return f"{one_letter}{nine_digits}{letter_or_number}"
            
        elif 'ssn' in name:
            # Social Security Numbers have a standard format
            area = random.randint(100, 999)
            group = random.randint(10, 99)
            serial = random.randint(1000, 9999)
            return f"{area}-{group}-{serial}"
            
        elif name in ['qnxtenrollid', 'qnxtcarrierid']:
            # Insurance enrollment IDs
            prefix = 'QNX'
            numeric = ''.join(random.choices('0123456789', k=7))
            return f"{prefix}{numeric}"
            
        elif 'unique' in name:
            # For unique identifiers, use UUID format but more readable
            letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
            digits = ''.join(random.choices('0123456789', k=6))
            return f"{letters}-{digits}-{random.randint(1000, 9999)}"
            
        # For other IDs, try to reuse if possible, otherwise create a new one
        existing_id = get_existing_id(table_name, name)
        if existing_id:
            return existing_id
        
        # Generate a mixture of characters and numbers for generic IDs
        if 'code' in name:
            # Code fields often have letter prefixes and digits
            letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
            digits = ''.join(random.choices('0123456789', k=3))
            return f"{letters}{digits}"
        
        # For a generic ID, create an alphanumeric string
        chars = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
        return chars
    
    # Generate values based on column name for common fields
    if 'name' in name:
        # Use consistent names for row entities
        if 'first' in name or 'firstname' in name:
            if 'mem_first' in row_entities:
                return row_entities['mem_first']
            value = fake.first_name()
            entity_type = name.replace('first_name', '').replace('firstname', '')
            if not entity_type:
                entity_type = 'mem'
            row_entities[f"{entity_type}_first"] = value
            return value
        elif 'last' in name or 'lastname' in name:
            if 'mem_last' in row_entities:
                return row_entities['mem_last']
            value = fake.last_name()
            entity_type = name.replace('last_name', '').replace('lastname', '')
            if not entity_type:
                entity_type = 'mem'
            row_entities[f"{entity_type}_last"] = value
            return value
        elif 'middle' in name:
            # Middle initial or name
            if len(name) < 10 or 'initial' in name:  # Likely just an initial
                return random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            return fake.first_name()  # Full middle name
        elif 'full' in name or 'user' in name:
            if 'mem_first' in row_entities and 'mem_last' in row_entities:
                return f"{row_entities['mem_first']} {row_entities['mem_last']}"
            return fake.name()
        elif 'product' in name:
            return fake.catch_phrase()
        return fake.name()
    
    # Generate values based on data type
    if data_type in ('INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT'):
        # If column name suggests an identifier, use positive integers
        if 'id' in name or 'num' in name or 'code' in name:
            result = random.randint(1, 100000)
        else:
            # Otherwise, use the full range including negative numbers
            result = random.randint(-1000, 10000)
    
    elif data_type == 'DECIMAL' or data_type == 'NUMERIC':
        precision = column['precision'] or 10
        scale = column['scale'] or 2
        max_value = min(10 ** (precision - scale) - 1, 1000000)  # Cap at 1 million for realism
        # If column name suggests an identifier or amount, use positive values
        if 'id' in name or 'num' in name or 'code' in name or 'amount' in name:
            value = random.uniform(0, max_value)
        else:
            value = random.uniform(-max_value, max_value)
        result = round(value, scale)
    
    elif data_type in ('FLOAT', 'REAL', 'DOUBLE'):
        # If column name suggests an identifier or amount, use positive values
        if 'id' in name or 'num' in name or 'code' in name or 'amount' in name:
            result = round(random.uniform(0, 10000.0), 4)
        else:
            result = round(random.uniform(-1000.0, 10000.0), 4)
    
    elif data_type in ('VARCHAR', 'CHAR', 'STRING'):
        # Get size with a sensible default if not specified
        size = column.get('size')
        if size is None:
            # Use column name to infer appropriate size
            if any(word in name for word in ['address', 'comment', 'description', 'note']):
                size = 200
            elif any(word in name for word in ['name', 'title', 'email']):
                size = 100
            elif 'phone' in name:
                size = 15
            elif 'code' in name:
                size = 10
            else:
                size = 50  # Default size for most fields
        
        # Special case for very small strings
        if size < 5:
            # For very small strings, use random letters
            result = ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(size))
        # Generate appropriate content based on field name
        elif 'phone' in name:
            result = fake.phone_number()[:size]
        elif 'email' in name:
            result = fake.email()[:size]
        elif 'address' in name:
            if 'line' in name or 'street' in name:
                result = fake.street_address()[:size]
            elif 'city' in name:
                result = fake.city()[:size]
            elif 'state' in name:
                result = fake.state_abbr() if size <= 2 else fake.state()[:size]
            elif 'zip' in name or 'postal' in name:
                result = fake.zipcode()[:size]
            elif 'country' in name:
                result = fake.country_code() if size <= 2 else fake.country()[:size]
            else:
                result = fake.address().replace('\n', ', ')[:size]
        elif 'code' in name:
            # Code fields are usually short alphanumeric
            result = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=min(size, 8)))
        elif any(word in name for word in ['comment', 'note', 'description']):
            # Comments and notes are typically longer text
            result = fake.text(max_nb_chars=min(size, 200))
        elif 'company' in name or 'business' in name or 'vendor' in name:
            result = fake.company()[:size]
        elif 'job' in name or 'title' in name or 'position' in name:
            result = fake.job()[:size]
        elif 'url' in name or 'website' in name:
            result = fake.url()[:size]
        elif 'username' in name or 'login' in name:
            result = fake.user_name()[:size]
        elif 'password' in name:
            result = fake.password(length=min(size, 12))
        elif 'status' in name:
            statuses = ['ACTIVE', 'INACTIVE', 'PENDING', 'APPROVED', 'REJECTED', 'COMPLETED']
            result = random.choice(statuses)[:size]
        elif 'type' in name:
            types = ['PRIMARY', 'SECONDARY', 'BASIC', 'PREMIUM', 'STANDARD', 'CUSTOM']
            result = random.choice(types)[:size]
        elif 'source' in name:
            sources = ['SYSTEM', 'USER', 'IMPORT', 'MANUAL', 'API', 'BATCH']
            result = random.choice(sources)[:size]
        else:
            # Default to a short text string
            length = random.randint(5, min(size, 20))
            result = fake.text(max_nb_chars=length)
    
    elif data_type == 'TEXT':
        result = fake.paragraph()
    
    elif data_type == 'DATE':
        start_date = datetime.date(1970, 1, 1)
        end_date = datetime.date.today()
        days_between = (end_date - start_date).days
        random_days = random.randint(0, days_between)
        result = start_date + datetime.timedelta(days=random_days)
    
    elif data_type == 'TIMESTAMP' or data_type == 'DATETIME':
        # Ensure we always return a valid timestamp even for NULL columns when allow_nulls is False
        try:
            years_ago = random.randint(0, 10)
            value = fake.date_time_between(start_date=f"-{years_ago}y", end_date="now")
            # Check if the value is None or NaT (pandas' version of None for timestamps)
            if pd.isna(value) and not allow_nulls:
                # Fallback to current timestamp if NULL was generated
                value = datetime.datetime.now()
            return value
        except Exception as e:
            # If any exception occurs, return current time when NULL not allowed
            if not allow_nulls:
                return datetime.datetime.now()
            # Otherwise, let the exception propagate
            raise e
    
    elif data_type == 'BOOLEAN' or data_type == 'BOOL':
        return random.choice([True, False])
    
    # Default for unsupported types
    else:
        # Generate a sensible fallback based on column name patterns instead of showing error
        if 'id' in name or 'code' in name:
            result = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
        elif any(word in name for word in ['name', 'title']):
            result = fake.word().capitalize()
        elif 'date' in name:
            start_date = datetime.date(1970, 1, 1)
            end_date = datetime.date.today()
            days_between = (end_date - start_date).days
            random_days = random.randint(0, days_between)
            result = start_date + datetime.timedelta(days=random_days)
        elif 'flag' in name:
            result = random.choice(['Y', 'N'])
        elif 'status' in name:
            result = random.choice(['ACTIVE', 'INACTIVE', 'PENDING'])
        else:
            # Default to a short string
            result = fake.word()[:20]
    
    # Final safety check: never return None or NaN when allow_nulls is False
    if not allow_nulls and (result is None or pd.isna(result)):
        # Generate fallback values based on data type
        if data_type in ('INT', 'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT'):
            return 0
        elif data_type in ('DECIMAL', 'NUMERIC', 'FLOAT', 'REAL', 'DOUBLE'):
            return 0.0
        elif data_type in ('VARCHAR', 'CHAR', 'TEXT', 'STRING'):
            # Generate a sensible default based on column name
            if 'name' in name:
                return 'Unknown'
            elif 'email' in name:
                return 'no-email@example.com'
            elif 'phone' in name:
                return '000-000-0000'
            elif 'address' in name:
                return 'N/A'
            elif 'code' in name:
                return 'DEFAULT'
            else:
                return 'N/A'
        elif data_type == 'DATE':
            return datetime.date.today()
        elif data_type in ('TIMESTAMP', 'DATETIME'):
            return datetime.datetime.now()
        elif data_type in ('BOOLEAN', 'BOOL'):
            return False
        else:
            # For any other type, provide a default string rather than error message
            return 'N/A'
    
    return result

def format_value_for_sql(value):
    """Format a value for SQL INSERT statement."""
    if value is None:
        return 'NULL'
    
    if isinstance(value, (int, float)):
        return str(value)
    
    if isinstance(value, bool):
        return '1' if value else '0'
    
    if isinstance(value, datetime.datetime):
        return f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
    
    if isinstance(value, datetime.date):
        return f"'{value.strftime('%Y-%m-%d')}'"
    
    # Escape single quotes for string values
    escaped_value = str(value).replace("'", "''")
    return f"'{escaped_value}'"

def write_csv(table_data, outdir):
    """Write generated data as CSV files."""
    table_name = table_data['table_name']
    full_name = table_data.get('full_name', table_name)
    data = table_data['data']
    
    df = pd.DataFrame(data)
    
    # Use the full qualified name for the output file, replacing periods with underscores
    output_filename = full_name.replace('.', '_') + ".csv"
    output_path = Path(outdir) / "csv" / output_filename
    
    df.to_csv(output_path, index=False)
    
    return output_path

def write_sql_from_csv(csv_path, outdir):
    """Generate SQL INSERT statements from a CSV file."""
    # Extract table name from the CSV filename
    filename = os.path.basename(csv_path)
    base_name = os.path.splitext(filename)[0]
    
    # Properly convert underscores back to dots for schema-qualified names
    # Format is typically edp_dev_schema_table.csv -> edp_dev.schema.table
    parts = base_name.split('_')
    if len(parts) >= 3:
        # First part is project_dataset, rest joins with dots
        project_dataset = parts[0] + '_' + parts[1]  # Keep edp_dev as is
        schema = parts[2]  # healthplanoptima
        table = '_'.join(parts[3:])  # Join rest for table name
        table_name = f"{project_dataset}.{schema}.{table}"
    else:
        # Fall back to simple table name if not enough parts
        table_name = base_name
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create SQL file path
    sql_filename = base_name + "_insert.sql"
    sql_path = Path(outdir) / "sql" / sql_filename
    
    with open(sql_path, 'w') as f:
        # Write header comment
        f.write(f"-- Generated INSERT statements for table {table_name}\n\n")
        
        # Get column names
        column_names = df.columns.tolist()
        
        # Generate INSERT statements for each row
        for _, row in df.iterrows():
            values = []
            for col in column_names:
                val = row[col]
                # Format value based on its type
                if pd.isna(val):
                    values.append('NULL')
                elif isinstance(val, (int, float)):
                    values.append(str(val))
                elif isinstance(val, (datetime.datetime, datetime.date)):
                    values.append(f"'{val}'")
                else:
                    # Escape single quotes for string values
                    escaped_val = str(val).replace("'", "''")
                    values.append(f"'{escaped_val}'")
            
            # Write INSERT statement
            insert_stmt = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(values)});\n"
            f.write(insert_stmt)
    
    return sql_path

def setup_logging(outdir):
    """Set up logging to a file in the output directory."""
    log_path = Path(outdir) / "dummy_data_generator.log"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler to also output logs to stdout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"Logging initialized to {log_path}")

def main():
    """Main function to parse arguments and generate dummy data."""
    args = parse_arguments()
    
    # Validate input/output paths
    if not args.ddl:
        print("Error: No DDL file specified.")
        return 1
    
    if not os.path.isfile(args.ddl):
        print(f"Error: DDL file '{args.ddl}' not found.")
        return 1
    
    if not args.outdir:
        print("Error: No output directory specified.")
        return 1
    
    # Set up logging
    setup_logging(args.outdir)
    
    # Log settings
    logging.info(f"Using DDL file: {args.ddl}")
    logging.info(f"Generating {args.rows} rows per table")
    logging.info(f"Output directory: {args.outdir}")
    logging.info(f"Allow NULL values: {args.allow_nulls}")
    
    # Parse DDL file
    tables = parse_ddl(args.ddl)
    
    if not tables:
        print("Error: No tables found in DDL file.")
        return 1
    
    print(f"Found {len(tables)} tables in DDL file.")
    
    # Generate data for all tables
    # Convert allow_nulls to boolean if it's a string
    allow_nulls = True
    if isinstance(args.allow_nulls, str):
        allow_nulls = args.allow_nulls.upper() == 'TRUE'
    elif isinstance(args.allow_nulls, bool):
        allow_nulls = args.allow_nulls
    
    table_data = generate_data_for_tables(tables, args.rows, allow_nulls)
    
    # Step 1: Write CSV files
    csv_files = []
    print("\nStep 1: Generating CSV files...")
    for full_name, data in table_data.items():
        try:
            csv_path = write_csv(data, args.outdir)
            csv_files.append(csv_path)
            print(f"CSV data written to {csv_path}")
        except Exception as e:
            print(f"Error writing CSV for table {full_name}: {e}")
    
    # Step 2: Generate SQL INSERT statements from CSV files
    print("\nStep 2: Generating SQL files from CSV...")
    sql_count = 0
    for csv_file in csv_files:
        try:
            sql_path = write_sql_from_csv(csv_file, args.outdir)
            print(f"SQL INSERT statements written to {sql_path}")
            sql_count += 1
        except Exception as e:
            print(f"Error generating SQL from CSV {csv_file}: {e}")
    
    print(f"\nSuccessfully generated relational data for {len(csv_files)} tables:")
    print(f"- {len(csv_files)} CSV files in {os.path.join(args.outdir, 'csv')}")
    print(f"- {sql_count} SQL files in {os.path.join(args.outdir, 'sql')}")
    return 0

if __name__ == "__main__":
    main() 