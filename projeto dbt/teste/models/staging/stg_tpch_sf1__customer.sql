WITH
source AS (
    SELECT
        *
    FROM {{ source('src_tpch_sf1', 'customer') }}
)

SELECT
    *
FROM source