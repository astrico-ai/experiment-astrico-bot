-- Comprehensive check of ALL categorical columns in the database
-- This finds columns with relatively few distinct values (good candidates for normalization)

-- Helper CTE to get distinct counts for all text columns
WITH column_stats AS (
    -- Sales Invoices columns
    SELECT 'sales_invoices_veedol' AS table_name,
           'region' AS column_name,
           COUNT(DISTINCT region) AS distinct_count,
           'Region classification' AS description
    FROM sales_invoices_veedol WHERE region IS NOT NULL

    UNION ALL
    SELECT 'sales_invoices_veedol', 'distribution_channel',
           COUNT(DISTINCT distribution_channel), 'Channel code (AM, DE, EC, etc.)'
    FROM sales_invoices_veedol WHERE distribution_channel IS NOT NULL

    UNION ALL
    SELECT 'sales_invoices_veedol', 'billing_type',
           COUNT(DISTINCT billing_type), 'Invoice type (Invoice, Credit Memo, etc.)'
    FROM sales_invoices_veedol WHERE billing_type IS NOT NULL

    UNION ALL
    SELECT 'sales_invoices_veedol', 'billing_code',
           COUNT(DISTINCT billing_code), 'Billing code (F2, RE, ZAMZ, etc.)'
    FROM sales_invoices_veedol WHERE billing_code IS NOT NULL

    UNION ALL
    SELECT 'sales_invoices_veedol', 'item_category',
           COUNT(DISTINCT item_category), 'Item category (G2N, L2N, REN, etc.)'
    FROM sales_invoices_veedol WHERE item_category IS NOT NULL

    UNION ALL
    SELECT 'sales_invoices_veedol', 'customer_group_text',
           COUNT(DISTINCT customer_group_text), 'Customer classification text'
    FROM sales_invoices_veedol WHERE customer_group_text IS NOT NULL

    UNION ALL
    SELECT 'sales_invoices_veedol', 'plant',
           COUNT(DISTINCT plant), 'Plant code'
    FROM sales_invoices_veedol WHERE plant IS NOT NULL

    -- Customer Master columns
    UNION ALL
    SELECT 'customer_master_veedol', 'state',
           COUNT(DISTINCT state), 'State names'
    FROM customer_master_veedol WHERE state IS NOT NULL

    UNION ALL
    SELECT 'customer_master_veedol', 'region',
           COUNT(DISTINCT region), 'Region classification'
    FROM customer_master_veedol WHERE region IS NOT NULL

    UNION ALL
    SELECT 'customer_master_veedol', 'customer_group_report',
           COUNT(DISTINCT customer_group_report), 'Sales channel for reporting'
    FROM customer_master_veedol WHERE customer_group_report IS NOT NULL

    UNION ALL
    SELECT 'customer_master_veedol', 'distribution_channel',
           COUNT(DISTINCT distribution_channel), 'Distribution channel code'
    FROM customer_master_veedol WHERE distribution_channel IS NOT NULL

    UNION ALL
    SELECT 'customer_master_veedol', 'customer_classification',
           COUNT(DISTINCT customer_classification), 'Customer classification'
    FROM customer_master_veedol WHERE customer_classification IS NOT NULL

    -- Material Master columns
    UNION ALL
    SELECT 'material_master_veedol', 'segment_order',
           COUNT(DISTINCT segment_order), 'Product segment (PCMO, MCO, etc.)'
    FROM material_master_veedol WHERE segment_order IS NOT NULL

    UNION ALL
    SELECT 'material_master_veedol', 'segment',
           COUNT(DISTINCT segment), 'Market segment'
    FROM material_master_veedol WHERE segment IS NOT NULL

    UNION ALL
    SELECT 'material_master_veedol', 'main_segment',
           COUNT(DISTINCT main_segment), 'Primary market segment'
    FROM material_master_veedol WHERE main_segment IS NOT NULL

    UNION ALL
    SELECT 'material_master_veedol', 'product_category',
           COUNT(DISTINCT product_category), 'Product category'
    FROM material_master_veedol WHERE product_category IS NOT NULL

    UNION ALL
    SELECT 'material_master_veedol', 'product_vertical',
           COUNT(DISTINCT product_vertical), 'Product vertical'
    FROM material_master_veedol WHERE product_vertical IS NOT NULL

    UNION ALL
    SELECT 'material_master_veedol', 'brand',
           COUNT(DISTINCT brand), 'Brand names'
    FROM material_master_veedol WHERE brand IS NOT NULL

    UNION ALL
    SELECT 'material_master_veedol', 'brand_variant',
           COUNT(DISTINCT brand_variant), 'Brand variant'
    FROM material_master_veedol WHERE brand_variant IS NOT NULL

    UNION ALL
    SELECT 'material_master_veedol', 'pack_type',
           COUNT(DISTINCT pack_type), 'Pack type (SMALL, BULK)'
    FROM material_master_veedol WHERE pack_type IS NOT NULL

    UNION ALL
    SELECT 'material_master_veedol', 'base_unit_of_measure',
           COUNT(DISTINCT base_unit_of_measure), 'Unit of measure (EA, KG, L)'
    FROM material_master_veedol WHERE base_unit_of_measure IS NOT NULL

    UNION ALL
    SELECT 'material_master_veedol', 'company_group',
           COUNT(DISTINCT company_group), 'Company group (ENTI, TWOC)'
    FROM material_master_veedol WHERE company_group IS NOT NULL

    -- Budget Data columns
    UNION ALL
    SELECT 'budget_data_veedol', 'region',
           COUNT(DISTINCT region), 'Region for budget'
    FROM budget_data_veedol WHERE region IS NOT NULL

    UNION ALL
    SELECT 'budget_data_veedol', 'channel',
           COUNT(DISTINCT channel), 'Sales channel for budget'
    FROM budget_data_veedol WHERE channel IS NOT NULL
)

-- Show results ordered by distinct count (candidates with fewer values are better)
SELECT
    table_name,
    column_name,
    distinct_count,
    description,
    CASE
        WHEN distinct_count <= 5 THEN '⭐⭐⭐ CRITICAL'
        WHEN distinct_count <= 15 THEN '⭐⭐ HIGH'
        WHEN distinct_count <= 50 THEN '⭐ MEDIUM'
        WHEN distinct_count <= 100 THEN '△ LOW'
        ELSE '✗ SKIP'
    END AS priority
FROM column_stats
ORDER BY distinct_count, table_name, column_name;

-- Show samples for top priority columns (≤15 distinct values)
SELECT '
═══════════════════════════════════════════════════════════════
SAMPLE VALUES FOR HIGH-PRIORITY COLUMNS (≤15 distinct values)
═══════════════════════════════════════════════════════════════
' AS info;

-- Distribution channels
SELECT '--- distribution_channel (sales_invoices_veedol) ---' AS category;
SELECT DISTINCT distribution_channel, COUNT(*) as usage_count
FROM sales_invoices_veedol
WHERE distribution_channel IS NOT NULL
GROUP BY distribution_channel
ORDER BY distribution_channel;

-- Item categories
SELECT '--- item_category (sales_invoices_veedol) ---' AS category;
SELECT DISTINCT item_category, COUNT(*) as usage_count
FROM sales_invoices_veedol
WHERE item_category IS NOT NULL
GROUP BY item_category
ORDER BY item_category;

-- Billing codes
SELECT '--- billing_code (sales_invoices_veedol) ---' AS category;
SELECT DISTINCT billing_code, COUNT(*) as usage_count
FROM sales_invoices_veedol
WHERE billing_code IS NOT NULL
GROUP BY billing_code
ORDER BY billing_code;

-- Customer group report
SELECT '--- customer_group_report (customer_master_veedol) ---' AS category;
SELECT DISTINCT customer_group_report, COUNT(*) as usage_count
FROM customer_master_veedol
WHERE customer_group_report IS NOT NULL
GROUP BY customer_group_report
ORDER BY customer_group_report;

-- Product segments
SELECT '--- segment_order (material_master_veedol) ---' AS category;
SELECT DISTINCT segment_order, COUNT(*) as usage_count
FROM material_master_veedol
WHERE segment_order IS NOT NULL
GROUP BY segment_order
ORDER BY segment_order;

-- Product categories
SELECT '--- product_category (material_master_veedol) ---' AS category;
SELECT DISTINCT product_category
FROM material_master_veedol
WHERE product_category IS NOT NULL
ORDER BY product_category;

-- Pack types
SELECT '--- pack_type (material_master_veedol) ---' AS category;
SELECT DISTINCT pack_type, COUNT(*) as usage_count
FROM material_master_veedol
WHERE pack_type IS NOT NULL
GROUP BY pack_type
ORDER BY pack_type;

-- Unit of measure
SELECT '--- base_unit_of_measure (material_master_veedol) ---' AS category;
SELECT DISTINCT base_unit_of_measure, COUNT(*) as usage_count
FROM material_master_veedol
WHERE base_unit_of_measure IS NOT NULL
GROUP BY base_unit_of_measure
ORDER BY base_unit_of_measure;

-- Company group
SELECT '--- company_group (material_master_veedol) ---' AS category;
SELECT DISTINCT company_group, COUNT(*) as usage_count
FROM material_master_veedol
WHERE company_group IS NOT NULL
GROUP BY company_group
ORDER BY company_group;

-- Budget channels
SELECT '--- channel (budget_data_veedol) ---' AS category;
SELECT DISTINCT channel, COUNT(*) as usage_count
FROM budget_data_veedol
WHERE channel IS NOT NULL
GROUP BY channel
ORDER BY channel;
