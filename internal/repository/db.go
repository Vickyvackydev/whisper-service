package repository

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

type DB struct {
	Pool *pgxpool.Pool
}

func NewDB(ctx context.Context, databaseURL string) (*DB, error) {
	// First ensure the target database exists (creates it if missing)
	if err := ensureDatabaseExists(ctx, databaseURL); err != nil {
		log.Printf("Warning: Database auto-creation check returned: %v\n", err)
	}

	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("unable to parse database config: %w", err)
	}

	config.MaxConns = 30
	config.MinConns = 5
	config.MaxConnLifetime = 1 * time.Hour
	config.MaxConnIdleTime = 15 * time.Minute
	config.HealthCheckPeriod = 1 * time.Minute

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("unable to create connection pool: %w", err)
	}

	// Ping database
	pingCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := pool.Ping(pingCtx); err != nil {
		return nil, fmt.Errorf("unable to ping database: %w", err)
	}

	return &DB{Pool: pool}, nil
}

func ensureDatabaseExists(ctx context.Context, databaseURL string) error {
	cfg, err := pgx.ParseConfig(databaseURL)
	if err != nil {
		return err
	}

	targetDB := cfg.Database
	if targetDB == "" || targetDB == "postgres" {
		return nil
	}

	// Connect to default maintenance database "postgres"
	defaultCfg := cfg.Copy()
	defaultCfg.Database = "postgres"

	conn, err := pgx.ConnectConfig(ctx, defaultCfg)
	if err != nil {
		// Fallback to template1 if postgres db is not present
		defaultCfg.Database = "template1"
		conn, err = pgx.ConnectConfig(ctx, defaultCfg)
		if err != nil {
			return fmt.Errorf("unable to connect to maintenance database: %w", err)
		}
	}
	defer conn.Close(ctx)

	// Check if database exists
	var exists bool
	err = conn.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1)", targetDB).Scan(&exists)
	if err != nil {
		return fmt.Errorf("failed to check database existence: %w", err)
	}

	if !exists {
		log.Printf("Database '%s' does not exist. Creating automatically...\n", targetDB)
		// Sanitized identifier
		safeDBName := strings.ReplaceAll(targetDB, "\"", "")
		_, err = conn.Exec(ctx, fmt.Sprintf(`CREATE DATABASE "%s";`, safeDBName))
		if err != nil {
			var pgErr *pgconn.PgError
			if errors.As(err, &pgErr) && pgErr.Code == "42P04" { // duplicate_database race condition
				return nil
			}
			return fmt.Errorf("failed to create database '%s': %w", targetDB, err)
		}
		log.Printf("Database '%s' created successfully!\n", targetDB)
	}

	return nil
}

func (db *DB) Close() {
	if db.Pool != nil {
		db.Pool.Close()
	}
}

func (db *DB) RunMigrations(ctx context.Context, schemaSQL string) error {
	_, err := db.Pool.Exec(ctx, schemaSQL)
	if err != nil {
		return fmt.Errorf("migration execution failed: %w", err)
	}
	return nil
}
