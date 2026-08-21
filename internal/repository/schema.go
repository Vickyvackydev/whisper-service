package repository

import (
	_ "embed"
)

//go:embed schema.sql
var InitialSchemaSQL string
