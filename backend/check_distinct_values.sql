-- Check distinct value counts for Value Normalization
-- Run this to see what will be embedded and calculate costs

SELECT 'State names' AS category,
       COUNT(DISTINCT state) AS distinct_values,
       'customer_master_veedol.state' AS column_name
FROM customer_master_veedol
WHERE state IS NOT NULL

UNION ALL

SELECT 'Sales channels',
       COUNT(DISTINCT customer_group_report),
       'customer_master_veedol.customer_group_report'
FROM customer_master_veedol
WHERE customer_group_report IS NOT NULL

UNION ALL

SELECT 'Customer names',
       COUNT(DISTINCT customer_name),
       'customer_master_veedol.customer_name'
FROM customer_master_veedol
WHERE customer_name IS NOT NULL

UNION ALL

SELECT 'Product segments',
       COUNT(DISTINCT segment_order),
       'material_master_veedol.segment_order'
FROM material_master_veedol
WHERE segment_order IS NOT NULL

UNION ALL

SELECT 'Product categories',
       COUNT(DISTINCT product_category),
       'material_master_veedol.product_category'
FROM material_master_veedol
WHERE product_category IS NOT NULL

UNION ALL

SELECT 'Product verticals',
       COUNT(DISTINCT product_vertical),
       'material_master_veedol.product_vertical'
FROM material_master_veedol
WHERE product_vertical IS NOT NULL

UNION ALL

SELECT 'Brand names',
       COUNT(DISTINCT brand),
       'material_master_veedol.brand'
FROM material_master_veedol
WHERE brand IS NOT NULL

ORDER BY distinct_values DESC;

-- Get samples of each
SELECT '--- STATE SAMPLES ---' AS info;
SELECT DISTINCT state FROM customer_master_veedol WHERE state IS NOT NULL ORDER BY state LIMIT 10;

SELECT '--- CHANNEL SAMPLES ---' AS info;
SELECT DISTINCT customer_group_report FROM customer_master_veedol WHERE customer_group_report IS NOT NULL ORDER BY customer_group_report;

SELECT '--- SEGMENT SAMPLES ---' AS info;
SELECT DISTINCT segment_order FROM material_master_veedol WHERE segment_order IS NOT NULL ORDER BY segment_order;

SELECT '--- CATEGORY SAMPLES ---' AS info;
SELECT DISTINCT product_category FROM material_master_veedol WHERE product_category IS NOT NULL ORDER BY product_category;

SELECT '--- VERTICAL SAMPLES ---' AS info;
SELECT DISTINCT product_vertical FROM material_master_veedol WHERE product_vertical IS NOT NULL ORDER BY product_vertical;

SELECT '--- BRAND SAMPLES ---' AS info;
SELECT DISTINCT brand FROM material_master_veedol WHERE brand IS NOT NULL ORDER BY brand LIMIT 10;
