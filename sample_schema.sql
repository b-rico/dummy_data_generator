-- Sample SQL DDL file for dummy_data_generator.py

-- Users table with various column types
CREATE TABLE users (
    id INT NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100) NOT NULL,
    phone_number VARCHAR(20),
    birth_date DATE,
    created_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    user_type VARCHAR(10),
    credit_score DECIMAL(6,2),
    PRIMARY KEY (id)
);

-- Products table with different data types
CREATE TABLE products (
    product_id BIGINT NOT NULL,
    product_name VARCHAR(100) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    inventory_count INT,
    category VARCHAR(50),
    is_featured BOOLEAN,
    created_date DATE,
    last_updated TIMESTAMP,
    PRIMARY KEY (product_id)
);

-- Address table to demonstrate relationship and address-specific columns
CREATE TABLE addresses (
    address_id INT NOT NULL,
    user_id INT,
    address_line1 VARCHAR(100) NOT NULL,
    address_line2 VARCHAR(100),
    city VARCHAR(50) NOT NULL,
    state VARCHAR(25),
    postal_code VARCHAR(15) NOT NULL,
    country VARCHAR(50) NOT NULL,
    is_primary BOOLEAN,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Orders table with relationships and various data types
CREATE TABLE orders (
    order_id INT NOT NULL,
    user_id INT NOT NULL,
    order_date TIMESTAMP NOT NULL,
    total_amount DECIMAL(12,2) NOT NULL,
    shipping_address_id INT,
    status VARCHAR(20) NOT NULL,
    payment_method VARCHAR(20),
    tracking_number VARCHAR(50),
    PRIMARY KEY (order_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (shipping_address_id) REFERENCES addresses(address_id)
); 