{{
  config(
    materialized = 'view',
    )
}}


WITH
customer AS (
    SELECT
        *
    FROM {{ ref('stg_tpch_sf1__customer') }}
),

nation AS (
    SELECT
        *
    FROM {{ ref('stg_tpch_sf1__nation') }}
)

SELECT
    customer.c_custkey,
    customer.c_name,
    customer.c_address,
    nation.n_name,
    customer.c_phone
FROM customer
LEFT JOIN nation
    ON (customer.c_nationkey = nation.n_nationkey)