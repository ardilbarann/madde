-- Record which model produced each vector, so a model change is visible and re-embeddable.
ALTER TABLE chunks ADD COLUMN embedding_model text;
