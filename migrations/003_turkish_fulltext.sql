-- Full-text search with the Turkish stemmer, so "kaçış yollarının" finds "kaçış yolu".
-- The translate() maps dotless/dotted capitals first: the default lowercasing turns 'I' into 'i',
-- which would split "YANGIN" (heading) from "yangın" (body). Queries apply the same mapping.
DROP INDEX chunks_tsv_idx;
ALTER TABLE chunks DROP COLUMN tsv;
ALTER TABLE chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('turkish', translate(coalesce(heading, '') || ' ' || body, 'Iİ', 'ıi'))
) STORED;
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
