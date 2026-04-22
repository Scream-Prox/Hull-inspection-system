CREATE DATABASE IF NOT EXISTS diploma;

CREATE EXTERNAL TABLE IF NOT EXISTS diploma.processed_file_index (
  path STRING,
  length BIGINT
)
STORED AS PARQUET
LOCATION 'hdfs://namenode:8020/data/processed/file_index';

SELECT * FROM diploma.processed_file_index;
