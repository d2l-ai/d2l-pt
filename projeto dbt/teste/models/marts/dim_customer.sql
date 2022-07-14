WITH
source AS (
    SELECT
        *
    FROM {{ ref('int_customer_nation__joined') }}
)

SELECT 
    *
FROM source