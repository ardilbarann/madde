-- Regulations and their article-level chunks.
-- Embedding width 1024 matches bge-m3; changing models means a new migration.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE regulations (
    id           text PRIMARY KEY,            -- e.g. 'tbdy-2018'
    title        text NOT NULL,
    source_url   text NOT NULL,
    published_on date
);

CREATE TABLE chunks (
    id            bigserial PRIMARY KEY,
    regulation_id text NOT NULL REFERENCES regulations(id) ON DELETE CASCADE,
    article       text NOT NULL,              -- article/section label as printed, e.g. '3.2.1'
    heading       text,
    body          text NOT NULL,
    page          int,
    embedding     vector(1024),
    tsv           tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(heading, '') || ' ' || body)) STORED,
    UNIQUE (regulation_id, article)
);

CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
