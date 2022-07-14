WITH
source AS (
    SELECT
        *
    FROM {{ source('src_tpch_sf1', 'nation') }}
)

SELECT 
    *
FROM source