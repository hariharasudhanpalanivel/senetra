CREATE SCHEMA IF NOT EXISTS senetra;
SET search_path TO senetra;

-- ==========================================
-- ENUMS
-- ==========================================


CREATE TYPE recommendation_status AS ENUM (
    'PENDING',
    'APPROVED',
    'REJECTED'
);

CREATE TYPE transfer_status AS ENUM (
    'PENDING',
    'IN_TRANSIT',
    'DELIVERED'
);

CREATE TYPE alert_status AS ENUM (
    'OPEN',
    'IN_PROGRESS',
    'RESOLVED'
);

-- ==========================================
-- MASTER TABLES
-- ==========================================

CREATE TABLE countries (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    code CHAR(2) NOT NULL UNIQUE
);

CREATE TABLE states (
    id BIGSERIAL PRIMARY KEY,
    country_id BIGINT NOT NULL,
    name VARCHAR(150) NOT NULL,

    CONSTRAINT fk_states_country
        FOREIGN KEY (country_id)
        REFERENCES countries(id)
        ON DELETE CASCADE,

    CONSTRAINT uq_state_country
        UNIQUE (country_id, name)
);

CREATE TABLE districts (
    id BIGSERIAL PRIMARY KEY,
    state_id BIGINT NOT NULL,
    name VARCHAR(150) NOT NULL,
    latitude NUMERIC(10,7),
    longitude NUMERIC(10,7),

    CONSTRAINT fk_district_state
        FOREIGN KEY (state_id)
        REFERENCES states(id)
        ON DELETE CASCADE,

    CONSTRAINT uq_district_state
        UNIQUE (state_id, name)
);

CREATE TABLE phcs (
    id BIGSERIAL PRIMARY KEY,
    district_id BIGINT NOT NULL,
    name VARCHAR(200) NOT NULL,
    latitude NUMERIC(10,7),
    longitude NUMERIC(10,7),
    total_beds INTEGER NOT NULL DEFAULT 0,
    total_doctors INTEGER NOT NULL DEFAULT 0,
    total_nurses INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT fk_phc_district
        FOREIGN KEY (district_id)
        REFERENCES districts(id)
        ON DELETE CASCADE
);

CREATE TABLE medicines (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(100),
    unit VARCHAR(50)
);

-- ==========================================
-- INVENTORY
-- ==========================================

CREATE TABLE inventory (
    id BIGSERIAL PRIMARY KEY,
    phc_id BIGINT NOT NULL,
    medicine_id BIGINT NOT NULL,

    current_stock NUMERIC(15,2) NOT NULL DEFAULT 0,
    daily_consumption NUMERIC(15,2) NOT NULL DEFAULT 0,

    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_inventory_phc
        FOREIGN KEY (phc_id)
        REFERENCES phcs(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_inventory_medicine
        FOREIGN KEY (medicine_id)
        REFERENCES medicines(id)
        ON DELETE CASCADE,

    CONSTRAINT uq_inventory
        UNIQUE (phc_id, medicine_id)
);

-- ==========================================
-- DAILY METRICS
-- ==========================================

CREATE TABLE daily_metrics (
    id BIGSERIAL PRIMARY KEY,

    phc_id BIGINT NOT NULL,
    date DATE NOT NULL,

    patient_footfall INTEGER DEFAULT 0,
    bed_occupancy NUMERIC(5,2),
    staff_availability NUMERIC(5,2),

    outbreak_flag BOOLEAN DEFAULT FALSE,
    outbreak_type VARCHAR(100),

    CONSTRAINT fk_daily_metrics_phc
        FOREIGN KEY (phc_id)
        REFERENCES phcs(id)
        ON DELETE CASCADE,

    CONSTRAINT uq_daily_metric
        UNIQUE (phc_id, date)
);

-- ==========================================
-- MEDICINE CONSUMPTION HISTORY
-- ==========================================

CREATE TABLE medicine_consumption (
    id BIGSERIAL PRIMARY KEY,

    phc_id BIGINT NOT NULL,
    medicine_id BIGINT NOT NULL,

    date DATE NOT NULL,

    opening_stock NUMERIC(15,2) NOT NULL,
    consumption NUMERIC(15,2) NOT NULL,
    closing_stock NUMERIC(15,2) NOT NULL,

    CONSTRAINT fk_med_consumption_phc
        FOREIGN KEY (phc_id)
        REFERENCES phcs(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_med_consumption_medicine
        FOREIGN KEY (medicine_id)
        REFERENCES medicines(id)
        ON DELETE CASCADE,

    CONSTRAINT uq_med_consumption
        UNIQUE (phc_id, medicine_id, date)
);

-- ==========================================
-- PREDICTIONS
-- ==========================================

CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,

    phc_id BIGINT NOT NULL,
    medicine_id BIGINT NOT NULL,

    prediction_date DATE NOT NULL,

    predicted_demand NUMERIC(15,2) NOT NULL,
    confidence NUMERIC(5,2),
    model_version VARCHAR(50),

    CONSTRAINT fk_prediction_phc
        FOREIGN KEY (phc_id)
        REFERENCES phcs(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_prediction_medicine
        FOREIGN KEY (medicine_id)
        REFERENCES medicines(id)
        ON DELETE CASCADE
);

-- ==========================================
-- ALERTS
-- ==========================================

CREATE TABLE alerts (
    id BIGSERIAL PRIMARY KEY,

    district_id BIGINT NOT NULL,
    medicine_id BIGINT NOT NULL,

    risk_level VARCHAR(20) NOT NULL,
    stockout_days INTEGER,

    message TEXT,

    status alert_status NOT NULL DEFAULT 'OPEN',

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_alert_district
        FOREIGN KEY (district_id)
        REFERENCES districts(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_alert_medicine
        FOREIGN KEY (medicine_id)
        REFERENCES medicines(id)
        ON DELETE CASCADE
);

-- ==========================================
-- REDISTRIBUTION RECOMMENDATIONS
-- ==========================================

CREATE TABLE redistribution_recommendations (
    id BIGSERIAL PRIMARY KEY,

    medicine_id BIGINT NOT NULL,

    source_district_id BIGINT NOT NULL,
    target_district_id BIGINT NOT NULL,

    quantity NUMERIC(15,2) NOT NULL,
    distance_km NUMERIC(10,2),

    priority INTEGER,
    status recommendation_status NOT NULL DEFAULT 'PENDING',

    reason TEXT,

    CONSTRAINT fk_rr_medicine
        FOREIGN KEY (medicine_id)
        REFERENCES medicines(id),

    CONSTRAINT fk_rr_source_district
        FOREIGN KEY (source_district_id)
        REFERENCES districts(id),

    CONSTRAINT fk_rr_target_district
        FOREIGN KEY (target_district_id)
        REFERENCES districts(id)
);

-- ==========================================
-- TRANSFERS
-- ==========================================

CREATE TABLE transfers (
    id BIGSERIAL PRIMARY KEY,

    recommendation_id BIGINT NOT NULL UNIQUE,

    status transfer_status NOT NULL DEFAULT 'PENDING',

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_transfer_recommendation
        FOREIGN KEY (recommendation_id)
        REFERENCES redistribution_recommendations(id)
        ON DELETE CASCADE
);

-- ==========================================
-- SIMULATION EVENTS
-- ==========================================

CREATE TABLE simulation_events (
    id BIGSERIAL PRIMARY KEY,

    event_type VARCHAR(100) NOT NULL,
    district_id BIGINT NOT NULL,

    severity INTEGER CHECK (severity BETWEEN 1 AND 10),

    active BOOLEAN NOT NULL DEFAULT TRUE,

    started_at TIMESTAMP NOT NULL,

    CONSTRAINT fk_simulation_district
        FOREIGN KEY (district_id)
        REFERENCES districts(id)
        ON DELETE CASCADE
);

-- ==========================================
-- INDEXES
-- ==========================================

CREATE INDEX idx_states_country
ON states(country_id);

CREATE INDEX idx_district_state
ON districts(state_id);

CREATE INDEX idx_phc_district
ON phcs(district_id);

CREATE INDEX idx_inventory_phc
ON inventory(phc_id);

CREATE INDEX idx_inventory_medicine
ON inventory(medicine_id);

CREATE INDEX idx_daily_metrics_date
ON daily_metrics(date);

CREATE INDEX idx_med_consumption_date
ON medicine_consumption(date);

CREATE INDEX idx_predictions_date
ON predictions(prediction_date);

CREATE INDEX idx_alerts_status
ON alerts(status);

CREATE INDEX idx_rr_status
ON redistribution_recommendations(status);

CREATE INDEX idx_transfers_status
ON transfers(status);

