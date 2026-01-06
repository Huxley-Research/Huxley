"""
Additional database schemas for autonomous research system.

Extends base schema with exploration-specific tables.
"""

# Exploration sessions table
EXPLORATION_SESSION_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS huxley_exploration_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT UNIQUE NOT NULL,
    domain TEXT NOT NULL,
    objective TEXT NOT NULL,
    curiosity_policy TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    iterations INTEGER DEFAULT 0,
    confidence_delta JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_exploration_sessions_id ON huxley_exploration_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_exploration_sessions_domain ON huxley_exploration_sessions(domain);
"""

EXPLORATION_SESSION_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS huxley_exploration_sessions (
    id TEXT PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    domain TEXT NOT NULL,
    objective TEXT NOT NULL,
    curiosity_policy TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    iterations INTEGER DEFAULT 0,
    confidence_delta TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_exploration_sessions_id ON huxley_exploration_sessions(session_id);
"""

# Hypothesis ledger table
HYPOTHESIS_LEDGER_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS huxley_hypothesis_ledger (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_links JSONB DEFAULT '[]',
    speculative_flag BOOLEAN DEFAULT TRUE,
    revision_history JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES huxley_exploration_sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_hypothesis_ledger_id ON huxley_hypothesis_ledger(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_hypothesis_ledger_session ON huxley_hypothesis_ledger(session_id);
CREATE INDEX IF NOT EXISTS idx_hypothesis_ledger_confidence ON huxley_hypothesis_ledger(confidence);
"""

HYPOTHESIS_LEDGER_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS huxley_hypothesis_ledger (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT UNIQUE NOT NULL,
    session_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_links TEXT DEFAULT '[]',
    speculative_flag INTEGER DEFAULT 1,
    revision_history TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hypothesis_ledger_id ON huxley_hypothesis_ledger(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_hypothesis_ledger_session ON huxley_hypothesis_ledger(session_id);
"""

# Skill registry table
SKILL_REGISTRY_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS huxley_skill_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_name TEXT UNIQUE NOT NULL,
    task_pattern TEXT NOT NULL,
    success_rate REAL DEFAULT 0.0,
    applicability_domain TEXT[],
    usage_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_skill_registry_name ON huxley_skill_registry(skill_name);
CREATE INDEX IF NOT EXISTS idx_skill_registry_success ON huxley_skill_registry(success_rate);
"""

SKILL_REGISTRY_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS huxley_skill_registry (
    id TEXT PRIMARY KEY,
    skill_name TEXT UNIQUE NOT NULL,
    task_pattern TEXT NOT NULL,
    success_rate REAL DEFAULT 0.0,
    applicability_domain TEXT,
    usage_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_skill_registry_name ON huxley_skill_registry(skill_name);
"""

# Risk annotations table  
RISK_ANNOTATIONS_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS huxley_risk_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    safety_relevance TEXT,
    uncertainty_level REAL,
    ethical_sensitivity TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_annotations_entity ON huxley_risk_annotations(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_risk_annotations_uncertainty ON huxley_risk_annotations(uncertainty_level);
"""

RISK_ANNOTATIONS_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS huxley_risk_annotations (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    safety_relevance TEXT,
    uncertainty_level REAL,
    ethical_sensitivity TEXT,
    metadata TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_annotations_entity ON huxley_risk_annotations(entity_type, entity_id);
"""


async def setup_exploration_tables(db_url: str):
    """Setup exploration-specific tables."""
    if db_url.startswith("postgresql") or db_url.startswith("postgres"):
        import asyncpg
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(EXPLORATION_SESSION_SCHEMA_POSTGRES)
            await conn.execute(HYPOTHESIS_LEDGER_SCHEMA_POSTGRES)
            await conn.execute(SKILL_REGISTRY_SCHEMA_POSTGRES)
            await conn.execute(RISK_ANNOTATIONS_SCHEMA_POSTGRES)
        finally:
            await conn.close()
    elif db_url.startswith("sqlite"):
        import aiosqlite
        db_path = db_url.replace("sqlite:///", "").replace("sqlite://", "")
        conn = await aiosqlite.connect(db_path)
        try:
            await conn.execute(EXPLORATION_SESSION_SCHEMA_SQLITE)
            await conn.execute(HYPOTHESIS_LEDGER_SCHEMA_SQLITE)
            await conn.execute(SKILL_REGISTRY_SCHEMA_SQLITE)
            await conn.execute(RISK_ANNOTATIONS_SCHEMA_SQLITE)
            await conn.commit()
        finally:
            await conn.close()
