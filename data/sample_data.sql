-- Sample data for testing the Autonomous Insights Engine
-- This creates a minimal test dataset for the primary_sales table

-- Create table (if not exists)
CREATE TABLE IF NOT EXISTS primary_sales (
    id SERIAL PRIMARY KEY,
    billing_date DATE,
    billing_code VARCHAR(10),
    customer_name VARCHAR(100),
    customer_code VARCHAR(50),
    customer_segment VARCHAR(50),
    region VARCHAR(50),
    product_name VARCHAR(200),
    product_code VARCHAR(50),
    business_unit VARCHAR(20),
    quantity NUMERIC,
    gsv NUMERIC,
    nsv1 NUMERIC,
    nsv2 NUMERIC,
    cpd NUMERIC,
    ctd NUMERIC,
    ebd NUMERIC,
    contribution1 NUMERIC,
    batch_number VARCHAR(50),
    plant VARCHAR(50)
);

-- Insert sample data
INSERT INTO primary_sales (
    billing_date, billing_code, customer_name, customer_code, customer_segment,
    region, product_name, product_code, business_unit, quantity,
    gsv, nsv1, nsv2, cpd, ctd, ebd, contribution1, batch_number, plant
) VALUES
-- Recent sales - Active customers
('2025-11-01', 'SA', 'ABC Auto Workshop', 'CUST001', 'AA-CP', 'North', 'Engine Oil 5W30', 'PROD001', 'ENTI', 100, 50000, 48000, 47000, 500, 1000, 500, 25000, 'BATCH001', 'Plant-A'),
('2025-11-05', 'SA', 'XYZ Motors', 'CUST002', 'AA-DD', 'South', 'Engine Oil 10W40', 'PROD002', 'ENTI', 200, 90000, 87000, 85000, 1000, 1500, 500, 45000, 'BATCH002', 'Plant-A'),
('2025-11-10', 'SA', 'DEF Industries', 'CUST003', 'OEM-FF', 'East', 'Hydraulic Oil', 'PROD003', 'TWOC', 500, 250000, 240000, 235000, 2000, 5000, 3000, 120000, 'BATCH003', 'Plant-B'),

-- Older sales - Same customers (showing regular ordering pattern)
('2025-10-15', 'SA', 'ABC Auto Workshop', 'CUST001', 'AA-CP', 'North', 'Engine Oil 5W30', 'PROD001', 'ENTI', 95, 47500, 45600, 44650, 475, 950, 475, 23750, 'BATCH004', 'Plant-A'),
('2025-10-20', 'SA', 'XYZ Motors', 'CUST002', 'AA-DD', 'South', 'Engine Oil 10W40', 'PROD002', 'ENTI', 190, 85500, 82650, 80750, 950, 1425, 475, 42500, 'BATCH005', 'Plant-A'),
('2025-10-25', 'SA', 'DEF Industries', 'CUST003', 'OEM-FF', 'East', 'Hydraulic Oil', 'PROD003', 'TWOC', 480, 240000, 230400, 225600, 1920, 4800, 2880, 115200, 'BATCH006', 'Plant-B'),

-- Churn signal - Customer that stopped ordering
('2025-08-01', 'SA', 'GHI Service Center', 'CUST004', 'AA-CP', 'North', 'Engine Oil 5W30', 'PROD001', 'ENTI', 120, 60000, 57600, 56400, 600, 1200, 600, 30000, 'BATCH007', 'Plant-A'),
('2025-07-15', 'SA', 'GHI Service Center', 'CUST004', 'AA-CP', 'North', 'Engine Oil 5W30', 'PROD001', 'ENTI', 115, 57500, 55200, 54050, 575, 1150, 575, 28750, 'BATCH008', 'Plant-A'),
-- (No recent orders from CUST004 - churn signal!)

-- Returns - Quality issue signal
('2025-11-12', 'RE', 'JKL Manufacturing', 'CUST005', 'OEM-FW', 'West', 'Gear Oil', 'PROD004', 'TWOC', -50, -25000, -24000, -23500, -250, -500, -250, -12500, 'BATCH009', 'Plant-C'),
('2025-11-01', 'SA', 'JKL Manufacturing', 'CUST005', 'OEM-FW', 'West', 'Gear Oil', 'PROD004', 'TWOC', 300, 150000, 144000, 141000, 1500, 3000, 1500, 75000, 'BATCH009', 'Plant-C'),

-- High discount example - margin erosion signal
('2025-11-15', 'SA', 'MNO Wholesale', 'CUST006', 'INST-DC', 'South', 'Industrial Lubricant', 'PROD005', 'ENTI', 1000, 400000, 360000, 340000, 10000, 20000, 10000, 150000, 'BATCH010', 'Plant-A'),
-- (GSV to NSV2 drop of 15% indicates heavy discounting)

-- Export sales
('2025-11-18', 'SA', 'International Buyer', 'CUST007', 'Export', 'Export', 'Premium Oil Mix', 'PROD006', 'TWOC', 2000, 1000000, 980000, 960000, 5000, 10000, 5000, 500000, 'BATCH011', 'Plant-B');

-- Summary statistics you can run to verify:
-- Total rows: ~12 records
-- Date range: Jul 2025 - Nov 2025
-- Customers: 7
-- Products: 6
-- Signals to find:
--   1. CUST004 churn (no orders since August)
--   2. BATCH009 quality issue (returns)
--   3. CUST006 high discount (margin erosion)
--   4. North region has 2 customers (1 active, 1 churned)

COMMIT;
