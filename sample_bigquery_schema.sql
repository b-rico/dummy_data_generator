-- Sample BigQuery SQL schema for dummy_data_generator.py
-- This demonstrates compatibility with cloud data warehouse syntaxes

-- Users table with BigQuery data types
CREATE OR REPLACE TABLE project_id.dataset.users (
    id INT64 NOT NULL,
    first_name STRING,
    last_name STRING,
    email STRING NOT NULL,
    phone_number STRING,
    birth_date DATE,
    created_at TIMESTAMP,
    is_active BOOL DEFAULT TRUE,
    user_type STRING,
    credit_score FLOAT64,
    PRIMARY KEY (id) NOT ENFORCED
);

-- Products table with BigQuery data types
CREATE TABLE project_id.dataset.products (
    product_id INT64 NOT NULL,
    product_name STRING NOT NULL,
    description STRING,
    price NUMERIC(10,2) NOT NULL,
    inventory_count INT64,
    category STRING,
    is_featured BOOL,
    created_date DATE,
    last_updated TIMESTAMP,
    PRIMARY KEY (product_id) NOT ENFORCED
);

-- Address table with BigQuery syntax
CREATE TABLE project_id.dataset.addresses (
    address_id INT64 NOT NULL,
    user_id INT64,
    address_line1 STRING NOT NULL,
    address_line2 STRING,
    city STRING NOT NULL,
    state STRING,
    postal_code STRING NOT NULL,
    country STRING NOT NULL,
    is_primary BOOL,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES project_id.dataset.users(id)
);

-- Orders table with BigQuery syntax
CREATE TABLE project_id.dataset.orders (
    order_id INT64 NOT NULL,
    user_id INT64 NOT NULL,
    order_date TIMESTAMP NOT NULL,
    total_amount NUMERIC(12,2) NOT NULL,
    shipping_address_id INT64,
    status STRING NOT NULL,
    payment_method STRING,
    tracking_number STRING,
    PRIMARY KEY (order_id) NOT ENFORCED,
    FOREIGN KEY (user_id) REFERENCES project_id.dataset.users(id),
    FOREIGN KEY (shipping_address_id) REFERENCES project_id.dataset.addresses(address_id)
); 