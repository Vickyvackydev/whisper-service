-- 002_add_translation_fields.sql
-- Add enable_translation and target_language columns to transcription_jobs table

ALTER TABLE transcription_jobs 
ADD COLUMN IF NOT EXISTS enable_translation BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS target_language VARCHAR(16);
